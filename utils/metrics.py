"""
评估指标计算 (Phase 1 V3)

包含：
- Spearman 相关性
- Linear CKA (Centered Kernel Alignment)
- Pearson 相关性
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Tuple, Optional


def compute_spearman_correlation(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Tuple[float, float]:
    """
    计算 Spearman 相关系数

    Args:
        pred: [batch, dim] 预测特征
        target: [batch, dim] 目标特征

    Returns:
        correlation: Spearman 相关系数
        p_value: p值
    """
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten if needed
    if pred.ndim > 1:
        pred = pred.flatten()
    if target.ndim > 1:
        target = target.flatten()

    # Compute Spearman correlation
    correlation, p_value = spearmanr(pred, target)

    return float(correlation), float(p_value)


def compute_pearson_correlation(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Tuple[float, float]:
    """
    计算 Pearson 相关系数

    Args:
        pred: [batch, dim] 预测特征
        target: [batch, dim] 目标特征

    Returns:
        correlation: Pearson 相关系数
        p_value: p值
    """
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten if needed
    if pred.ndim > 1:
        pred = pred.flatten()
    if target.ndim > 1:
        target = target.flatten()

    # Compute Pearson correlation
    correlation, p_value = pearsonr(pred, target)

    return float(correlation), float(p_value)


def compute_linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    debiased: bool = False
) -> float:
    """
    计算 Linear CKA (Centered Kernel Alignment)

    CKA 衡量两个特征表示的相似度，对线性变换不敏感

    Args:
        X: [batch, dim_x] 特征矩阵1
        Y: [batch, dim_y] 特征矩阵2
        debiased: 是否使用无偏估计

    Returns:
        cka_score: CKA 分数 [0, 1]
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    # Center the features
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute Gram matrices
    # K_X = X @ X.T, K_Y = Y @ Y.T
    # But we can use the identity: HSIC(K_X, K_Y) = tr(K_X @ K_Y) / n^2
    # For linear CKA: HSIC(X, Y) = ||X.T @ Y||_F^2 / n^2

    n = X.shape[0]

    # Compute HSIC
    hsic_xy = np.linalg.norm(X.T @ Y, ord='fro') ** 2 / (n ** 2)
    hsic_xx = np.linalg.norm(X.T @ X, ord='fro') ** 2 / (n ** 2)
    hsic_yy = np.linalg.norm(Y.T @ Y, ord='fro') ** 2 / (n ** 2)

    # CKA score
    cka_score = hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)

    return float(cka_score)


def compute_cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    计算余弦相似度

    Args:
        pred: 预测特征
        target: 目标特征
        dim: 计算维度

    Returns:
        cosine_sim: 余弦相似度
    """
    cosine_sim = torch.nn.functional.cosine_similarity(pred, target, dim=dim)
    return cosine_sim


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    计算 MSE

    Args:
        pred: 预测特征
        target: 目标特征

    Returns:
        mse: Mean Squared Error
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    return float(mse.item())


def test_metrics():
    """Test metrics"""
    print("Testing Metrics...")

    batch_size = 100
    dim_x = 640
    dim_y = 768

    # Generate random features
    X = torch.randn(batch_size, dim_x)
    Y = torch.randn(batch_size, dim_y)

    # Make them correlated
    Y_correlated = torch.randn(batch_size, dim_y)
    Y_correlated[:, :dim_x] = X + 0.1 * torch.randn(batch_size, dim_x)

    print("\n1. Spearman Correlation:")
    corr, p_val = compute_spearman_correlation(X, Y)
    print(f"   Random features: {corr:.4f} (p={p_val:.4f})")
    corr, p_val = compute_spearman_correlation(X, X)
    print(f"   Identical features: {corr:.4f} (p={p_val:.4f})")

    print("\n2. Pearson Correlation:")
    corr, p_val = compute_pearson_correlation(X, Y)
    print(f"   Random features: {corr:.4f} (p={p_val:.4f})")
    corr, p_val = compute_pearson_correlation(X, X)
    print(f"   Identical features: {corr:.4f} (p={p_val:.4f})")

    print("\n3. Linear CKA:")
    cka = compute_linear_cka(X, Y)
    print(f"   Random features: {cka:.4f}")
    cka = compute_linear_cka(X, X)
    print(f"   Identical features: {cka:.4f}")
    cka = compute_linear_cka(X, Y_correlated)
    print(f"   Correlated features: {cka:.4f}")

    print("\n4. Cosine Similarity:")
    cos_sim = compute_cosine_similarity(X, Y[:, :dim_x])
    print(f"   Mean cosine similarity: {cos_sim.mean():.4f}")

    print("\n5. MSE:")
    mse = compute_mse(X, Y[:, :dim_x])
    print(f"   MSE: {mse:.4f}")


if __name__ == "__main__":
    test_metrics()
