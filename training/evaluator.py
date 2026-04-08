"""
评估器 (Phase 1 V4)

负责验证阶段的指标计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
import sys
from pathlib import Path

# Import metrics
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from metrics import compute_spearman_correlation, compute_linear_cka, compute_mse


class DualTeacherEvaluator:
    """
    双教师蒸馏评估器

    评估指标：
    - Protein Spearman: 学生 protein_proj 与 ESM2 特征的相关性
    - Molecular Spearman: 学生 molecular_proj 与 SMILES 教师特征的相关性（ChemBERTa/GeminiMol）
    - CKA: SMILES 特征与教师特征的 CKA 相似度
    - MSE: 投影特征与教师特征的 MSE
    """

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        teacher_cache_manager: Optional[object] = None,
        split_name: str = 'val',
        loss_weights: Optional[Dict[str, float]] = None,
        vocab_size: int = 0,
        use_mlm: bool = False,
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            model: 学生模型
            dataloader: 验证数据加载器
            teacher_cache_manager: 教师特征缓存管理器字典 {'esm2': cache, 'chemberta': cache}
            split_name: 数据集拆分名称

        Returns:
            metrics: 评估指标字典
        """
        model.eval()

        # 收集所有特征（为避免 teacher miss 导致维度不一致，按 teacher 可用性分别收集）
        protein_proj_list = []
        protein_smiles_repr_list = []
        teacher_esm2_list = []

        molecular_proj_list = []
        molecular_smiles_repr_list = []
        teacher_molecular_list = []

        total_loss = 0.0
        total_protein_loss = 0.0
        total_molecular_loss = 0.0
        total_self_loss = 0.0
        total_fusion_loss = 0.0
        total_task_loss = 0.0
        total_task_fp_loss = 0.0
        total_consistency_loss = 0.0
        num_batches = 0
        num_batches_protein = 0
        num_batches_molecular = 0
        num_batches_self = 0
        num_batches_fusion = 0
        num_batches_task = 0
        num_batches_task_fp = 0
        num_batches_consistency = 0

        def _infer_head_in_features(head: nn.Module) -> Optional[int]:
            if isinstance(head, nn.Linear):
                return int(head.in_features)
            if isinstance(head, nn.Sequential):
                for module in head.modules():
                    if isinstance(module, nn.Linear):
                        return int(module.in_features)
            in_features = getattr(head, "in_features", None)
            if in_features is None:
                return None
            return int(in_features)

        def _infer_task_feature_key(head: nn.Module, outputs_: Dict[str, torch.Tensor]) -> str:
            feature_key = getattr(head, "_task_feature_type", None)
            if isinstance(feature_key, str) and feature_key in outputs_:
                return feature_key

            input_dim = _infer_head_in_features(head)
            for key in ["molecular_proj", "smiles_repr"]:
                if key in outputs_ and outputs_[key].ndim == 2:
                    if input_dim is None or int(outputs_[key].shape[1]) == int(input_dim):
                        return key
            return "molecular_proj" if "molecular_proj" in outputs_ else "smiles_repr"

        def _compute_task_loss(
            head: nn.Module,
            outputs_: Dict[str, torch.Tensor],
            batch_: Dict,
            *,
            labels_key: str,
            mask_key: str,
            single_label_key: str,
        ) -> torch.Tensor:
            feature_key = _infer_task_feature_key(head, outputs_)
            x = outputs_[feature_key]
            pred = head(x)

            loss_type = str(getattr(head, "_task_loss_type", "mse")).lower()
            alpha = getattr(head, "_task_alpha", None)
            norm = getattr(head, "_task_norm", None)
            pos_weight = getattr(head, "_task_pos_weight", None)

            if labels_key in batch_:
                y = batch_[labels_key].to(self.device, non_blocking=True)
                mask = batch_.get(mask_key)
                if mask is None:
                    mask = torch.isfinite(y).to(dtype=torch.float32)
                else:
                    mask = mask.to(self.device, non_blocking=True)

                if pred.ndim == 1:
                    pred = pred.unsqueeze(-1)

                if loss_type == "bce":
                    y = y.to(dtype=torch.float32).clamp(0.0, 1.0)
                    mask = mask.to(dtype=torch.float32)

                    per_elem = F.binary_cross_entropy_with_logits(
                        pred.to(dtype=torch.float32),
                        y,
                        pos_weight=pos_weight,
                        reduction="none",
                    )
                    per_elem = per_elem * mask
                    denom = mask.sum(dim=0).clamp_min(1.0)
                    per_task = per_elem.sum(dim=0) / denom

                    if alpha is None:
                        return per_task.mean()
                    return (per_task * alpha).sum()

                if loss_type != "mse":
                    raise ValueError(f"Unsupported task supervision loss_type: {loss_type}")

                if norm is not None and norm.get("type") == "zscore":
                    mean = norm["mean"]
                    std = norm["std"]
                    y = (y - mean) / std

                se = (pred - y) ** 2
                se = se * mask
                denom = mask.sum(dim=0).clamp_min(1.0)
                per_task_mse = se.sum(dim=0) / denom

                if alpha is None:
                    return per_task_mse.mean()
                return (per_task_mse * alpha).sum()

            if single_label_key not in batch_:
                return torch.tensor(0.0, device=self.device)

            y1 = batch_[single_label_key].to(self.device, non_blocking=True)
            pred1 = pred.squeeze(-1)
            if loss_type == "bce":
                y1 = y1.to(dtype=torch.float32).clamp(0.0, 1.0)
                return F.binary_cross_entropy_with_logits(pred1.to(dtype=torch.float32), y1)
            return F.mse_loss(pred1, y1)

        for batch in dataloader:
            # 移动到设备
            smiles_input_ids = batch['smiles_input_ids'].to(self.device)
            smiles_attention_mask = batch['smiles_attention_mask'].to(self.device)
            graph_distances = batch.get("graph_distances")
            if graph_distances is not None:
                graph_distances = graph_distances.to(self.device, non_blocking=True)
            sequence_ids = batch['sequence_id']
            row_indices = batch.get('row_index')

            # 学生前向传播
            outputs = model(
                smiles_input_ids,
                smiles_attention_mask,
                graph_distances=graph_distances,
                return_all=bool(use_mlm),
            )

            # 加载教师特征（按 cache 的 id_keys 自动选择 sequence_id 或 row_index）
            teacher_esm2_batch = None
            teacher_molecular_batch = None
            if teacher_cache_manager:
                # ESM2 特征
                if 'esm2' in teacher_cache_manager:
                    esm2_features, missing = teacher_cache_manager['esm2'].get(split_name, sequence_ids)
                    if esm2_features is not None:
                        teacher_esm2_batch = esm2_features

                # SMILES 教师特征（ChemBERTa/GeminiMol）
                if 'chemberta' in teacher_cache_manager:
                    chem_cache = teacher_cache_manager['chemberta']
                    use_sequence_ids = hasattr(chem_cache, "id_keys") and any(
                        k in getattr(chem_cache, "id_keys", []) for k in ["sequence_ids", "sequence_id"]
                    )

                    if use_sequence_ids:
                        chem_ids = sequence_ids
                    else:
                        if row_indices is None:
                            chem_ids = []
                        else:
                            chem_ids = [str(idx.item()) if torch.is_tensor(idx) else str(idx) for idx in row_indices]

                    if chem_ids:
                        chem_features, missing = chem_cache.get(split_name, chem_ids)
                        if chem_features is not None:
                            teacher_molecular_batch = chem_features

            # 仅在 teacher 可用时，收集用于相关性/CKA 计算的对齐样本
            if teacher_esm2_batch is not None:
                protein_proj_list.append(outputs['protein_proj'].detach().cpu())
                protein_smiles_repr_list.append(outputs['smiles_repr'].detach().cpu())
                teacher_esm2_list.append(teacher_esm2_batch.detach().cpu())

            if teacher_molecular_batch is not None:
                molecular_proj_list.append(outputs['molecular_proj'].detach().cpu())
                molecular_smiles_repr_list.append(outputs['smiles_repr'].detach().cpu())
                teacher_molecular_list.append(teacher_molecular_batch.detach().cpu())

            # 计算验证损失（无MLM标签时自监督损失为0）
            if loss_weights:
                losses = {}
                task_weight = float(loss_weights.get("task", 0.0))
                task_fp_weight = float(loss_weights.get("task_fp", 0.0))
                consistency_weight = float(loss_weights.get("consistency", 0.0))

                if teacher_esm2_batch is not None:
                    teacher_esm2_tensor = teacher_esm2_batch.to(outputs['protein_proj'].device)
                    protein_mse = torch.nn.functional.mse_loss(outputs['protein_proj'], teacher_esm2_tensor)
                    protein_cos = 1 - torch.nn.functional.cosine_similarity(
                        outputs['protein_proj'],
                        teacher_esm2_tensor,
                        dim=1,
                    ).mean()
                    losses['protein'] = 0.7 * protein_mse + 0.3 * protein_cos
                    num_batches_protein += 1
                else:
                    losses['protein'] = torch.tensor(0.0, device=outputs['protein_proj'].device)

                if teacher_molecular_batch is not None:
                    teacher_molecular_tensor = teacher_molecular_batch.to(outputs['molecular_proj'].device)
                    molecular_mse = torch.nn.functional.mse_loss(outputs['molecular_proj'], teacher_molecular_tensor)
                    molecular_cos = 1 - torch.nn.functional.cosine_similarity(
                        outputs['molecular_proj'],
                        teacher_molecular_tensor,
                        dim=1,
                    ).mean()
                    losses['molecular'] = 0.7 * molecular_mse + 0.3 * molecular_cos
                    num_batches_molecular += 1
                else:
                    losses['molecular'] = torch.tensor(0.0, device=outputs['molecular_proj'].device)

                if use_mlm and 'smiles_mlm_logits' in outputs and 'smiles_mlm_labels' in batch:
                    logits = outputs['smiles_mlm_logits']
                    labels = batch['smiles_mlm_labels'].to(logits.device)
                    losses['self'] = torch.nn.functional.cross_entropy(
                        logits.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                    num_batches_self += 1
                else:
                    losses['self'] = torch.tensor(0.0, device=outputs['molecular_proj'].device)

                # Fusion residual distill (pretrain stacking) validation loss
                fusion_weight = float(loss_weights.get("fusion", 0.0))
                if (
                    fusion_weight > 0
                    and teacher_molecular_batch is not None
                    and "fusion_hs" in outputs
                    and "fusion_hp" in outputs
                ):
                    teacher_molecular_tensor = teacher_molecular_batch.to(outputs["fusion_hp"].device)
                    hs = outputs["fusion_hs"]
                    hp = outputs["fusion_hp"]
                    residual = teacher_molecular_tensor - hs
                    losses["fusion"] = torch.nn.functional.mse_loss(hp, residual)
                    num_batches_fusion += 1
                else:
                    losses["fusion"] = torch.tensor(0.0, device=outputs["molecular_proj"].device)

                # RDKit proxy / lightweight supervision validation loss
                if task_weight > 0:
                    head = getattr(model, "p4_task_head", None)
                    if head is not None:
                        losses["task"] = _compute_task_loss(
                            head,
                            outputs,
                            batch,
                            labels_key="task_labels",
                            mask_key="task_mask",
                            single_label_key="task_label",
                        )
                        num_batches_task += 1
                    else:
                        losses["task"] = torch.tensor(0.0, device=outputs["molecular_proj"].device)
                else:
                    losses["task"] = torch.tensor(0.0, device=outputs["molecular_proj"].device)

                # Optional aux proxy (e.g., fingerprint)
                if task_fp_weight > 0:
                    head = getattr(model, "p4_task_head_aux", None)
                    if head is not None:
                        losses["task_fp"] = _compute_task_loss(
                            head,
                            outputs,
                            batch,
                            labels_key="task_labels_aux",
                            mask_key="task_mask_aux",
                            single_label_key="task_label_aux",
                        )
                        num_batches_task_fp += 1
                    else:
                        losses["task_fp"] = torch.tensor(0.0, device=outputs["molecular_proj"].device)
                else:
                    losses["task_fp"] = torch.tensor(0.0, device=outputs["molecular_proj"].device)

                # Random-SMILES consistency validation loss (requires view2 in batch)
                if (
                    consistency_weight > 0
                    and "smiles_input_ids_view2" in batch
                    and "smiles_attention_mask_view2" in batch
                ):
                    view2_ids = batch["smiles_input_ids_view2"].to(self.device, non_blocking=True)
                    view2_mask = batch["smiles_attention_mask_view2"].to(self.device, non_blocking=True)
                    graph_distances_view2 = batch.get("graph_distances_view2")
                    if graph_distances_view2 is not None:
                        graph_distances_view2 = graph_distances_view2.to(self.device, non_blocking=True)

                    outputs_view2 = model(
                        view2_ids,
                        view2_mask,
                        graph_distances=graph_distances_view2,
                        return_all=False,
                    )

                    cfg = getattr(model, "_consistency_cfg", None) or {}
                    feature_key = str(cfg.get("apply_to") or "smiles_repr")
                    method = str(cfg.get("method") or "cosine").lower()
                    if feature_key not in outputs or feature_key not in outputs_view2:
                        losses["consistency"] = torch.tensor(0.0, device=outputs["molecular_proj"].device)
                    else:
                        z1 = outputs[feature_key]
                        z2 = outputs_view2[feature_key]
                        if method == "cosine":
                            losses["consistency"] = 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()
                        elif method == "mse":
                            losses["consistency"] = F.mse_loss(z1, z2)
                        elif method == "infonce":
                            from model.losses import contrastive_infonce_loss  # noqa: WPS433

                            tau = float(cfg.get("tau", 0.07))
                            symmetric = bool(cfg.get("symmetric", True))
                            losses["consistency"] = contrastive_infonce_loss(z1, z2, tau=tau, symmetric=symmetric)
                        else:
                            raise ValueError(f"Unsupported consistency.method: {method}")
                    num_batches_consistency += 1
                else:
                    losses["consistency"] = torch.tensor(0.0, device=outputs["molecular_proj"].device)

                total_loss += (
                    loss_weights.get('protein', 0.0) * losses['protein'].item() +
                    loss_weights.get('molecular', 0.0) * losses['molecular'].item() +
                    loss_weights.get('self', 0.0) * losses['self'].item() +
                    loss_weights.get('task', 0.0) * losses['task'].item() +
                    loss_weights.get('task_fp', 0.0) * losses['task_fp'].item() +
                    loss_weights.get('consistency', 0.0) * losses['consistency'].item() +
                    loss_weights.get('fusion', 0.0) * losses['fusion'].item()
                )
                total_protein_loss += losses['protein'].item()
                total_molecular_loss += losses['molecular'].item()
                total_self_loss += losses['self'].item()
                total_task_loss += losses["task"].item()
                total_task_fp_loss += losses["task_fp"].item()
                total_consistency_loss += losses["consistency"].item()
                total_fusion_loss += losses["fusion"].item()

            num_batches += 1

        metrics = {}

        # 计算 Protein 指标
        if teacher_esm2_list:
            protein_proj = torch.cat(protein_proj_list, dim=0)
            teacher_esm2 = torch.cat(teacher_esm2_list, dim=0)
            smiles_repr_protein = torch.cat(protein_smiles_repr_list, dim=0)

            # Spearman correlation
            protein_spearman, _ = compute_spearman_correlation(protein_proj, teacher_esm2)
            metrics['protein_spearman'] = protein_spearman

            # MSE
            protein_mse = compute_mse(protein_proj, teacher_esm2)
            metrics['protein_mse'] = protein_mse

            # CKA
            protein_cka = compute_linear_cka(protein_proj, teacher_esm2)  # Fixed: use projected features
            protein_cka_raw = compute_linear_cka(smiles_repr_protein, teacher_esm2)  # Before projection
            metrics['protein_cka'] = protein_cka
            metrics['protein_cka_raw'] = protein_cka_raw  # Raw CKA for analysis
            metrics['num_samples_protein'] = len(protein_proj)

        # 计算 Molecular 指标
        if teacher_molecular_list:
            molecular_proj = torch.cat(molecular_proj_list, dim=0)
            teacher_molecular = torch.cat(teacher_molecular_list, dim=0)
            smiles_repr_molecular = torch.cat(molecular_smiles_repr_list, dim=0)

            # Spearman correlation
            molecular_spearman, _ = compute_spearman_correlation(molecular_proj, teacher_molecular)
            metrics['molecular_spearman'] = molecular_spearman

            # MSE
            molecular_mse = compute_mse(molecular_proj, teacher_molecular)
            metrics['molecular_mse'] = molecular_mse

            # CKA
            molecular_cka = compute_linear_cka(smiles_repr_molecular, teacher_molecular)
            metrics['molecular_cka'] = molecular_cka
            metrics['num_samples_molecular'] = len(molecular_proj)

        # 计算平均损失（如果有）
        if num_batches > 0 and loss_weights:
            metrics['val_loss'] = total_loss / num_batches
            metrics['val_protein_loss'] = total_protein_loss / max(num_batches_protein, 1)
            metrics['val_molecular_loss'] = total_molecular_loss / max(num_batches_molecular, 1)
            metrics['val_self_loss'] = total_self_loss / max(num_batches_self, 1)
            if num_batches_task > 0:
                metrics["val_task_loss"] = total_task_loss / float(num_batches_task)
            if num_batches_task_fp > 0:
                metrics["val_task_fp_loss"] = total_task_fp_loss / float(num_batches_task_fp)
            if num_batches_consistency > 0:
                metrics["val_consistency_loss"] = total_consistency_loss / float(num_batches_consistency)
            if num_batches_fusion > 0:
                metrics["val_fusion_loss"] = total_fusion_loss / float(num_batches_fusion)
            if 'num_samples_protein' in metrics:
                metrics['num_samples'] = int(metrics['num_samples_protein'])

        return metrics

    def print_metrics(self, metrics: Dict[str, float], prefix: str = "Val"):
        """打印评估指标"""
        print(f"\n{prefix} Metrics:")
        print("="*70)

        # Protein metrics
        if 'protein_spearman' in metrics:
            print(f"Protein Alignment:")
            print(f"  Spearman: {metrics['protein_spearman']:.4f}")
            print(f"  MSE:      {metrics['protein_mse']:.4f}")
            print(f"  CKA:      {metrics['protein_cka']:.4f}")
        if 'protein_cka_raw' in metrics:
            print(f"  CKA (Raw):      {metrics['protein_cka_raw']:.4f}  # Before projection")

        # Molecular metrics
        if 'molecular_spearman' in metrics:
            print(f"\nMolecular Alignment:")
            print(f"  Spearman: {metrics['molecular_spearman']:.4f}")
            print(f"  MSE:      {metrics['molecular_mse']:.4f}")
            print(f"  CKA:      {metrics['molecular_cka']:.4f}")

        # Other metrics
        if 'val_loss' in metrics:
            print(f"\nValidation Loss: {metrics['val_loss']:.4f}")
            if 'val_protein_loss' in metrics:
                print(f"  Protein Loss:    {metrics['val_protein_loss']:.4f}")
            if 'val_molecular_loss' in metrics:
                print(f"  Molecular Loss:  {metrics['val_molecular_loss']:.4f}")
            if 'val_self_loss' in metrics:
                print(f"  MLM Loss:        {metrics['val_self_loss']:.4f}")
            if "val_task_loss" in metrics:
                print(f"  Proxy Loss:      {metrics['val_task_loss']:.4f}")
            if "val_task_fp_loss" in metrics:
                print(f"  Proxy(Aux) Loss: {metrics['val_task_fp_loss']:.4f}")
            if "val_consistency_loss" in metrics:
                print(f"  Consistency:     {metrics['val_consistency_loss']:.4f}")
            if "val_fusion_loss" in metrics:
                print(f"  Fusion Loss:     {metrics['val_fusion_loss']:.4f}")

        if 'num_samples' in metrics:
            print(f"Num Samples: {metrics['num_samples']}")

        print("="*70)


def test_evaluator():
    """Test evaluator"""
    print("Testing DualTeacherEvaluator...")

    # This would require a full model and dataloader
    # Just demonstrate the interface

    evaluator = DualTeacherEvaluator()

    # Dummy metrics
    metrics = {
        'protein_spearman': 0.75,
        'protein_mse': 0.32,
        'protein_cka': 0.68,
        'molecular_spearman': 0.82,
        'molecular_mse': 0.28,
        'molecular_cka': 0.71,
        'val_loss': 0.45,
        'num_samples': 10000
    }

    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    test_evaluator()
