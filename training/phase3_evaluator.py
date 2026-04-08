"""
Phase 3 评估器 - 任务微调评估

负责评估：
1. 回归任务性能（RMSE, R², MAE, Pearson, Spearman）
2. 知识保留度（可选，通过蒸馏指标）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
import sys
from pathlib import Path

# Import metrics
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from metrics import compute_spearman_correlation, compute_linear_cka, compute_mse


class Phase3Evaluator:
    """
    Phase 3 任务微调评估器

    评估指标：
    - 任务性能：RMSE, R², MAE, Pearson, Spearman
    - 知识保留（可选）：Protein/Molecular Spearman, CKA, MSE
    """

    def __init__(
        self,
        device: torch.device = torch.device('cpu'),
        denormalize_fn: Optional[callable] = None
    ):
        """
        Args:
            device: 计算设备
            denormalize_fn: 标签反归一化函数 (可选)
                例如: lambda x: x * std + mean
        """
        self.device = device
        self.denormalize_fn = denormalize_fn

    def _compute_regression_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算回归任务指标

        Args:
            predictions: [N] 预测值
            labels: [N] 真实标签

        Returns:
            metrics: 包含 RMSE, R², MAE, Pearson, Spearman
        """
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        # 1. RMSE (Root Mean Squared Error)
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)

        # 2. MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predictions - labels))

        # 3. R² (Coefficient of Determination)
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # 4. Pearson correlation
        pearson = np.corrcoef(predictions, labels)[0, 1]

        # 5. Spearman correlation (rank-based)
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(predictions, labels)

        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'pearson': float(pearson),
            'spearman': float(spearman)
        }

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        teacher_cache_manager: Optional[Dict] = None,
        split_name: str = 'val',
        evaluate_distillation: bool = False
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            model: 学生模型（带任务头）
            dataloader: 验证数据加载器
            teacher_cache_manager: 教师特征缓存管理器 {'esm2': cache, 'chemberta': cache}
            split_name: 数据集拆分名称 ('val' or 'test')
            evaluate_distillation: 是否评估蒸馏保留度

        Returns:
            metrics: 评估指标字典
        """
        model.eval()

        # 收集所有预测和标签
        all_predictions = []
        all_labels = []

        # 如果需要评估蒸馏，收集投影特征和教师特征
        if evaluate_distillation:
            all_protein_proj = []
            all_molecular_proj = []
            all_teacher_esm2 = []
            all_teacher_chemberta = []

        for batch in dataloader:
            # 移动到设备
            smiles_input_ids = batch['smiles_input_ids'].to(self.device)
            smiles_attention_mask = batch['smiles_attention_mask'].to(self.device)
            task_labels = batch['task_label'].to(self.device)

            # 学生前向传播
            outputs = model(smiles_input_ids, smiles_attention_mask, return_all=False)

            # 收集任务预测
            task_pred = outputs['task_pred'].squeeze(-1)  # [batch]
            all_predictions.append(task_pred.cpu())
            all_labels.append(task_labels.cpu())

            # 如果需要评估蒸馏
            if evaluate_distillation:
                all_protein_proj.append(outputs['protein_proj'].cpu())
                all_molecular_proj.append(outputs['molecular_proj'].cpu())

                # 加载教师特征
                if teacher_cache_manager:
                    if 'esm2' in teacher_cache_manager:
                        sequence_ids = batch['sequence_id']
                        esm2_features, _ = teacher_cache_manager['esm2'].get(split_name, sequence_ids)
                        if esm2_features is not None:
                            all_teacher_esm2.append(esm2_features.cpu())

                    if 'chemberta' in teacher_cache_manager:
                        row_indices = batch['row_index']
                        row_index_strs = [str(idx.item()) for idx in row_indices]
                        chemberta_features, _ = teacher_cache_manager['chemberta'].get(split_name, row_index_strs)
                        if chemberta_features is not None:
                            all_teacher_chemberta.append(chemberta_features.cpu())

        # 合并所有批次
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # 反归一化（如果需要）
        if self.denormalize_fn is not None:
            predictions_denorm = self.denormalize_fn(predictions)
            labels_denorm = self.denormalize_fn(labels)
        else:
            predictions_denorm = predictions
            labels_denorm = labels

        # 1. 计算任务性能指标（反归一化后）
        task_metrics = self._compute_regression_metrics(predictions_denorm, labels_denorm)
        metrics = {
            f'task_{k}': v for k, v in task_metrics.items()
        }

        # 也计算归一化空间的指标（用于对比）
        task_metrics_norm = self._compute_regression_metrics(predictions, labels)
        metrics.update({
            f'task_{k}_normalized': v for k, v in task_metrics_norm.items()
        })

        # 2. 计算蒸馏保留度指标（如果需要）
        if evaluate_distillation:
            protein_proj = torch.cat(all_protein_proj, dim=0)
            molecular_proj = torch.cat(all_molecular_proj, dim=0)

            # Protein 指标
            if len(all_teacher_esm2) > 0:
                teacher_esm2 = torch.cat(all_teacher_esm2, dim=0)
                protein_spearman, _ = compute_spearman_correlation(protein_proj, teacher_esm2)
                protein_mse = compute_mse(protein_proj, teacher_esm2)

                metrics['protein_spearman'] = protein_spearman
                metrics['protein_mse'] = protein_mse

            # Molecular 指标
            if len(all_teacher_chemberta) > 0:
                teacher_chemberta = torch.cat(all_teacher_chemberta, dim=0)
                molecular_spearman, _ = compute_spearman_correlation(molecular_proj, teacher_chemberta)
                molecular_mse = compute_mse(molecular_proj, teacher_chemberta)

                metrics['molecular_spearman'] = molecular_spearman
                metrics['molecular_mse'] = molecular_mse

        metrics['num_samples'] = len(predictions)

        return metrics

    def print_metrics(self, metrics: Dict[str, float], prefix: str = "Val"):
        """打印评估指标"""
        print(f"\n{prefix} Metrics:")
        print("="*70)

        # 任务性能（反归一化后）
        if 'task_rmse' in metrics:
            print(f"Task Performance (Denormalized):")
            print(f"  RMSE:     {metrics['task_rmse']:.4f}")
            print(f"  MAE:      {metrics['task_mae']:.4f}")
            print(f"  R²:       {metrics['task_r2']:.4f}")
            print(f"  Pearson:  {metrics['task_pearson']:.4f}")
            print(f"  Spearman: {metrics['task_spearman']:.4f}")

        # 任务性能（归一化空间）
        if 'task_rmse_normalized' in metrics:
            print(f"\nTask Performance (Normalized):")
            print(f"  RMSE:     {metrics['task_rmse_normalized']:.4f}")
            print(f"  R²:       {metrics['task_r2_normalized']:.4f}")

        # 蒸馏保留度
        if 'protein_spearman' in metrics or 'molecular_spearman' in metrics:
            print(f"\nKnowledge Retention:")
            if 'protein_spearman' in metrics:
                print(f"  Protein Spearman:   {metrics['protein_spearman']:.4f}")
                print(f"  Protein MSE:        {metrics['protein_mse']:.4f}")
            if 'molecular_spearman' in metrics:
                print(f"  Molecular Spearman: {metrics['molecular_spearman']:.4f}")
                print(f"  Molecular MSE:      {metrics['molecular_mse']:.4f}")

        print(f"\nSamples: {metrics.get('num_samples', 0)}")
        print("="*70)


def test_phase3_evaluator():
    """测试Phase 3评估器"""
    print("Testing Phase3Evaluator...")

    # 创建反归一化函数（示例：zscore反归一化）
    mean = -5.8720864991
    std = 1.0755504086
    denormalize_fn = lambda x: x * std + mean

    evaluator = Phase3Evaluator(
        device=torch.device('cpu'),
        denormalize_fn=denormalize_fn
    )

    # 模拟预测和标签
    predictions = torch.randn(100)  # 归一化空间
    labels = torch.randn(100)

    # 计算指标
    metrics = evaluator._compute_regression_metrics(
        denormalize_fn(predictions),
        denormalize_fn(labels)
    )

    print("\nRegression Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ Phase3Evaluator test passed!")


if __name__ == "__main__":
    test_phase3_evaluator()
