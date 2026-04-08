"""
双教师蒸馏损失函数 (V4：几何/对比式蒸馏)

包含：
- Protein 蒸馏损失（对齐 ESM2）
- Molecular 蒸馏损失（对齐 GeminiMol/ChemBERTa）
- SMILES MLM 损失（自监督）
- 关系蒸馏（RKD：batch 内距离/角度）
- 对比式蒸馏（InfoNCE：batch 内 negatives）
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_embedding_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor
) -> torch.Tensor:
    """
    Cosine Embedding Loss

    计算学生和教师特征之间的余弦相似度损失
    Loss = 1 - cosine_similarity

    Args:
        student_features: [batch, dim]
        teacher_features: [batch, dim]

    Returns:
        scalar loss
    """
    cosine_sim = F.cosine_similarity(student_features, teacher_features, dim=1)
    loss = (1 - cosine_sim).mean()
    return loss


def _masked_offdiag(x: torch.Tensor) -> torch.Tensor:
    """提取方阵的非对角元素（展平）。"""
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Expected square matrix, got {tuple(x.shape)}")
    batch_size = x.shape[0]
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
    return x[mask]


def _normalize_relation_values(
    values: torch.Tensor,
    mode: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    if mode == "none":
        return values
    if mode == "mean":
        denom = values.mean().clamp_min(eps)
        return values / denom
    if mode == "median":
        denom = values.median().clamp_min(eps)
        return values / denom
    raise ValueError(f"Unsupported normalize mode: {mode}")


def rkd_distance_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    normalize: str = "mean",
    loss_type: str = "smooth_l1",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    RKD 距离蒸馏：对齐 batch 内样本两两距离（去除对角，按均值/中位数归一化）。
    """
    s = student.float()
    t = teacher.float()
    d_s = torch.cdist(s, s, p=2)
    d_t = torch.cdist(t, t, p=2)

    v_s = _masked_offdiag(d_s)
    v_t = _masked_offdiag(d_t)

    v_s = _normalize_relation_values(v_s, mode=normalize, eps=eps)
    v_t = _normalize_relation_values(v_t, mode=normalize, eps=eps)

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(v_s, v_t)
    if loss_type == "mse":
        return F.mse_loss(v_s, v_t)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _rkd_angle_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(batch_size, device=device)
    ii = idx[:, None, None]
    jj = idx[None, :, None]
    kk = idx[None, None, :]
    return (jj != ii) & (kk != ii) & (jj != kk)


def rkd_angle_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    loss_type: str = "smooth_l1",
    max_triplets: Optional[int] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    RKD 角度蒸馏：对齐三元组角度 cos( (x_i-x_j), (x_i-x_k) )。

    说明：全量三元组为 O(B^3)，B=16/32 可接受；可通过 max_triplets 采样降成本。
    """
    s = student.float()
    t = teacher.float()
    batch_size = s.shape[0]
    if batch_size < 3:
        return torch.tensor(0.0, device=s.device)

    s_diff = s[:, None, :] - s[None, :, :]  # [B, B, D]
    t_diff = t[:, None, :] - t[None, :, :]  # [B, B, D]
    s_diff = F.normalize(s_diff, dim=-1, eps=eps)
    t_diff = F.normalize(t_diff, dim=-1, eps=eps)

    # [B, B, B]，第 0 维为 anchor i，后两维为 (j, k)
    s_angle = torch.bmm(s_diff, s_diff.transpose(1, 2))
    t_angle = torch.bmm(t_diff, t_diff.transpose(1, 2))

    mask = _rkd_angle_mask(batch_size, device=s.device)
    s_vals = s_angle[mask]
    t_vals = t_angle[mask]

    if max_triplets is not None and max_triplets > 0 and s_vals.numel() > max_triplets:
        perm = torch.randperm(s_vals.numel(), device=s.device)[:max_triplets]
        s_vals = s_vals[perm]
        t_vals = t_vals[perm]

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(s_vals, t_vals)
    if loss_type == "mse":
        return F.mse_loss(s_vals, t_vals)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def rkd_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    distance_weight: float = 1.0,
    angle_weight: float = 0.0,
    distance_normalize: str = "mean",
    distance_loss_type: str = "smooth_l1",
    angle_loss_type: str = "smooth_l1",
    angle_max_triplets: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (total, dist, angle) 三个 loss。"""
    dist = rkd_distance_loss(
        student=student,
        teacher=teacher,
        normalize=distance_normalize,
        loss_type=distance_loss_type,
    )
    ang = torch.tensor(0.0, device=student.device)
    if angle_weight > 0:
        ang = rkd_angle_loss(
            student=student,
            teacher=teacher,
            loss_type=angle_loss_type,
            max_triplets=angle_max_triplets,
        )
    total = float(distance_weight) * dist + float(angle_weight) * ang
    return total, dist, ang


def contrastive_infonce_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    tau: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    对比式蒸馏（InfoNCE）：同 index 为正样本，其余为负样本（batch 内）。
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    s = F.normalize(student.float(), dim=1)
    t = F.normalize(teacher.float(), dim=1)
    logits = (s @ t.t()) / float(tau)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_st = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss_st
    loss_ts = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_st + loss_ts)


def _select_contrastive_feature(
    student_outputs: Dict[str, torch.Tensor],
    feature_key: str,
) -> torch.Tensor:
    if feature_key == "molecular":
        return student_outputs["molecular_proj"]
    if feature_key == "protein":
        return student_outputs["protein_proj"]
    if feature_key in {"smiles_repr", "smiles"}:
        return student_outputs["smiles_repr"]
    if feature_key == "contrastive_proj":
        if "contrastive_proj" not in student_outputs:
            raise KeyError("contrastive_proj not found in student_outputs")
        return student_outputs["contrastive_proj"]
    raise ValueError(f"Unsupported contrastive feature: {feature_key}")


def compute_dual_teacher_loss(
    student_outputs: Dict[str, torch.Tensor],
    teacher_esm2_features: Optional[torch.Tensor],
    teacher_chemberta_features: Optional[torch.Tensor],
    weights: Dict[str, float],
    mlm_labels: Optional[torch.Tensor] = None,
    vocab_size: int = 37,
    loss_cfg: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """
    双教师蒸馏损失（V4 版本：可选 RKD / InfoNCE）

    Args:
        student_outputs: 学生模型输出，包含：
            - 'protein_proj': [batch, 640]
            - 'molecular_proj': [batch, 768]
            - 'smiles_mlm_logits': [batch, seq_len, vocab_size] (可选)
        teacher_esm2_features: [batch, 640] ESM2 教师特征
        teacher_chemberta_features: [batch, 768] ChemBERTa 教师特征
        weights: 损失权重字典
            - 'protein': protein 蒸馏权重（主要）
            - 'molecular': molecular 蒸馏权重（辅助）
            - 'self': MLM 自监督权重（可选）
        mlm_labels: [batch, seq_len] MLM 标签（-100 表示忽略）
        vocab_size: SMILES 词表大小

    V4 新增（可选）：
    - losses['rkd']：关系蒸馏（距离/角度）
    - losses['contrastive']：对比式蒸馏（InfoNCE）

    Returns:
        loss_dict: 包含 total, protein, molecular, self, rkd, contrastive 损失
    """
    losses: Dict[str, torch.Tensor] = {}
    device = student_outputs['protein_proj'].device
    cfg = loss_cfg or {}

    # 1. Protein 蒸馏损失（主要，可关闭）
    protein_weight = weights.get('protein', 0.0)
    if protein_weight > 0 and teacher_esm2_features is not None:
        protein_mse = F.mse_loss(student_outputs['protein_proj'], teacher_esm2_features)
        protein_cosine = cosine_embedding_loss(student_outputs['protein_proj'], teacher_esm2_features)
        losses['protein'] = 0.7 * protein_mse + 0.3 * protein_cosine
    else:
        losses['protein'] = torch.tensor(0.0, device=device)

    # 2. Molecular 蒸馏损失（辅助，可单独保留）
    molecular_weight = weights.get('molecular', 0.0)
    if molecular_weight > 0 and teacher_chemberta_features is not None:
        molecular_mse = F.mse_loss(student_outputs['molecular_proj'], teacher_chemberta_features)
        molecular_cosine = cosine_embedding_loss(student_outputs['molecular_proj'], teacher_chemberta_features)
        losses['molecular'] = 0.7 * molecular_mse + 0.3 * molecular_cosine
    else:
        losses['molecular'] = torch.tensor(0.0, device=device)

    # 3. SMILES MLM 损失（自监督，可选）
    if mlm_labels is not None and 'smiles_mlm_logits' in student_outputs:
        mlm_logits = student_outputs['smiles_mlm_logits']
        # Flatten for cross entropy
        mlm_logits_flat = mlm_logits.view(-1, vocab_size)
        mlm_labels_flat = mlm_labels.view(-1)

        mlm_loss = F.cross_entropy(
            mlm_logits_flat,
            mlm_labels_flat,
            ignore_index=-100,
            reduction='mean'
        )
        losses['self'] = mlm_loss
    else:
        losses['self'] = torch.tensor(0.0, device=device)

    # 4. 关系蒸馏（RKD，默认只用于 molecular）
    rkd_weight = float(weights.get("rkd", 0.0))
    losses["rkd"] = torch.tensor(0.0, device=device)
    losses["rkd_dist"] = torch.tensor(0.0, device=device)
    losses["rkd_angle"] = torch.tensor(0.0, device=device)
    rkd_cfg = (cfg.get("rkd") or {}) if isinstance(cfg, dict) else {}
    if rkd_weight > 0 and bool(rkd_cfg.get("enabled", True)):
        apply_to = rkd_cfg.get("apply_to", ["molecular"])
        if isinstance(apply_to, str):
            apply_to = [apply_to]
        if "molecular" in apply_to and teacher_chemberta_features is not None:
            dist_w = float(rkd_cfg.get("distance_weight", 1.0))
            ang_w = float(rkd_cfg.get("angle_weight", 0.0))
            dist_norm = str(rkd_cfg.get("distance_normalize", "mean"))
            dist_loss_type = str(rkd_cfg.get("distance_loss_type", "smooth_l1"))
            ang_loss_type = str(rkd_cfg.get("angle_loss_type", "smooth_l1"))
            ang_max = rkd_cfg.get("angle_max_triplets", None)
            if ang_max is not None:
                ang_max = int(ang_max)
            total_rkd, dist_loss, ang_loss = rkd_loss(
                student=student_outputs["molecular_proj"],
                teacher=teacher_chemberta_features,
                distance_weight=dist_w,
                angle_weight=ang_w,
                distance_normalize=dist_norm,
                distance_loss_type=dist_loss_type,
                angle_loss_type=ang_loss_type,
                angle_max_triplets=ang_max,
            )
            losses["rkd"] = total_rkd
            losses["rkd_dist"] = dist_loss
            losses["rkd_angle"] = ang_loss

    # 5. 对比式蒸馏（InfoNCE，默认只用于 molecular）
    ctr_weight = float(weights.get("contrastive", 0.0))
    losses["contrastive"] = torch.tensor(0.0, device=device)
    ctr_cfg = (cfg.get("contrastive") or {}) if isinstance(cfg, dict) else {}
    if ctr_weight > 0 and bool(ctr_cfg.get("enabled", True)):
        apply_to = ctr_cfg.get("apply_to", ["molecular"])
        if isinstance(apply_to, str):
            apply_to = [apply_to]
        tau = float(ctr_cfg.get("tau", 0.07))
        symmetric = bool(ctr_cfg.get("symmetric", True))
        # 对比式蒸馏目前以 chemberta teacher 为目标（molecular/proj 或解耦分支）
        if teacher_chemberta_features is not None:
            ctr_count = 0
            for key in apply_to:
                if key in {"molecular", "smiles_repr", "smiles", "contrastive_proj"}:
                    student_feat = _select_contrastive_feature(student_outputs, key)
                    losses["contrastive"] = losses["contrastive"] + contrastive_infonce_loss(
                        student=student_feat,
                        teacher=teacher_chemberta_features,
                        tau=tau,
                        symmetric=symmetric,
                    )
                    ctr_count += 1
            if ctr_count > 1:
                losses["contrastive"] = losses["contrastive"] / float(ctr_count)

    # 6. 总损失（加权）
    total_loss = (
        weights.get('protein', 0.7) * losses['protein'] +
        weights.get('molecular', 0.15) * losses['molecular'] +
        weights.get('self', 0.15) * losses.get('self', torch.tensor(0.0)) +
        rkd_weight * losses.get("rkd", torch.tensor(0.0, device=device)) +
        ctr_weight * losses.get("contrastive", torch.tensor(0.0, device=device))
    )

    losses['total'] = total_loss

    return losses


def compute_task_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task_type: str = "regression"
) -> torch.Tensor:
    """
    下游任务损失（Phase 3 用）

    Args:
        predictions: [batch] or [batch, num_classes]
        labels: [batch]
        task_type: "regression" or "classification"

    Returns:
        scalar loss
    """
    if task_type == "regression":
        # MSE 损失
        loss = F.mse_loss(predictions, labels)
    elif task_type == "classification":
        # Cross entropy 损失
        loss = F.cross_entropy(predictions, labels)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return loss


def compute_phase3_loss(
    student_outputs: Dict[str, torch.Tensor],
    teacher_esm2_features: Optional[torch.Tensor],
    teacher_chemberta_features: Optional[torch.Tensor],
    task_labels: torch.Tensor,
    weights: Dict[str, float],
    task_type: str = "regression",
    use_distillation: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Phase 3 综合损失函数（任务微调 + 知识保留）

    结合：
    1. 任务损失（主要，权重0.80-0.90）
    2. 蒸馏损失（辅助，权重0.05-0.15，防止遗忘）

    Args:
        student_outputs: 学生模型输出，包含：
            - 'task_pred': [batch] or [batch, num_classes] 任务预测
            - 'protein_proj': [batch, 640] (可选)
            - 'molecular_proj': [batch, 768] (可选)
        teacher_esm2_features: [batch, 640] ESM2特征（可选）
        teacher_chemberta_features: [batch, 768] ChemBERTa特征（可选）
        task_labels: [batch] 任务标签
        weights: 损失权重字典
            - 'task': 任务损失权重（0.80-0.90）
            - 'protein': 蛋白质蒸馏权重（0.03-0.05）
            - 'molecular': 分子蒸馏权重（0.07-0.15）
        task_type: "regression" or "classification"
        use_distillation: 是否使用蒸馏损失

    Returns:
        loss_dict: 包含 total, task, protein, molecular 损失
    """
    losses = {}
    device = student_outputs['task_pred'].device

    # 1. 任务损失（主要）
    task_pred = student_outputs['task_pred'].squeeze(-1) if task_type == "regression" else student_outputs['task_pred']
    losses['task'] = compute_task_loss(task_pred, task_labels, task_type)

    # 2. 蒸馏损失（辅助，防止遗忘）
    if use_distillation:
        if teacher_esm2_features is not None and 'protein_proj' in student_outputs:
            protein_mse = F.mse_loss(student_outputs['protein_proj'], teacher_esm2_features)
            protein_cosine = cosine_embedding_loss(student_outputs['protein_proj'], teacher_esm2_features)
            losses['protein'] = 0.7 * protein_mse + 0.3 * protein_cosine
        else:
            losses['protein'] = torch.tensor(0.0, device=device)

        if teacher_chemberta_features is not None and 'molecular_proj' in student_outputs:
            molecular_mse = F.mse_loss(student_outputs['molecular_proj'], teacher_chemberta_features)
            molecular_cosine = cosine_embedding_loss(student_outputs['molecular_proj'], teacher_chemberta_features)
            losses['molecular'] = 0.7 * molecular_mse + 0.3 * molecular_cosine
        else:
            losses['molecular'] = torch.tensor(0.0, device=device)
    else:
        losses['protein'] = torch.tensor(0.0, device=device)
        losses['molecular'] = torch.tensor(0.0, device=device)

    # 3. 总损失（加权）
    total_loss = (
        weights.get('task', 0.80) * losses['task'] +
        weights.get('protein', 0.05) * losses['protein'] +
        weights.get('molecular', 0.15) * losses['molecular']
    )

    losses['total'] = total_loss

    return losses


def test_losses():
    """Test loss functions"""
    print("Testing Dual Teacher Loss...")

    batch_size = 4
    seq_len = 100
    vocab_size = 37

    # Prepare student outputs
    student_outputs = {
        'protein_proj': torch.randn(batch_size, 640),
        'molecular_proj': torch.randn(batch_size, 768),
        'smiles_mlm_logits': torch.randn(batch_size, seq_len, vocab_size)
    }

    # Prepare teacher features
    teacher_esm2 = torch.randn(batch_size, 640)
    teacher_chemberta = torch.randn(batch_size, 768)

    # Prepare MLM labels
    mlm_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    mlm_labels[mlm_labels < 10] = -100  # Some positions are ignored

    # Compute losses
    weights = {
        'protein': 0.70,
        'molecular': 0.15,
        'self': 0.15
    }

    losses = compute_dual_teacher_loss(
        student_outputs=student_outputs,
        teacher_esm2_features=teacher_esm2,
        teacher_chemberta_features=teacher_chemberta,
        weights=weights,
        mlm_labels=mlm_labels,
        vocab_size=vocab_size
    )

    print("\nLoss Results:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print(f"\nWeighted total = {weights['protein']} * {losses['protein'].item():.4f} + "
          f"{weights['molecular']} * {losses['molecular'].item():.4f} + "
          f"{weights['self']} * {losses['self'].item():.4f}")
    print(f"             = {losses['total'].item():.4f}")


if __name__ == "__main__":
    test_losses()
