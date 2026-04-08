#!/usr/bin/env python3
"""
Phase 3 GPU 版头训练脚本（冻结编码器特征）。

目标：
- 读取 extract_phase3_features.py 生成的 train/val/test_features.npz
- 在 GPU 上训练线性 / Ridge / 小型 MLP 回归头
- 输出 test 集 RMSE / R2 等指标，便于与 LLM/手工特征做公平对比

示例：
  CUDA_VISIBLE_DEVICES=0 python -u scripts/train_phase3_heads.py \\
    --feature_dir features/phase3_mainline_molecular_proj \\
    --output_dir results/phase3_linear_heads/mainline_torch_ridge \\
    --head_type ridge \\
    --alpha_grid 1e-4 1e-3 1e-2 1e-1 \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 torch heads training (GPU)")
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="单个特征目录（含 train/val/test_features.npz）；若传 --feature_dirs 则作为第一个目录参与 concat",
    )
    parser.add_argument(
        "--feature_dirs",
        type=str,
        nargs="*",
        default=None,
        help="多个特征目录（含 train/val/test_features.npz），将按顺序 concat 到一起",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--head_type",
        type=str,
        choices=[
            "linear",
            "ridge",
            "mlp",
            "residual",
            "film",
            "moe",
            "gated_residual",
            "interaction",
            "residualize",
            "stacking",
            "residual_multi",
        ],
        default="ridge",
        help="头类型：linear / ridge / mlp / residual / film / moe / gated_residual / interaction / residualize / stacking / residual_multi",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-2,
        help="ridge 正则系数或 mlp weight_decay",
    )
    parser.add_argument(
        "--alpha_grid",
        type=float,
        nargs="*",
        default=None,
        help="ridge 超参网格（优先使用 grid 选最佳 alpha）",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="*",
        default=[512, 256],
        help="mlp 隐层维度列表",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20, help="mlp 早停耐心（val rmse）")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 123, 202],
        help="mlp 随机种子列表（默认 3 个快速筛）",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--gate_hidden",
        type=int,
        default=64,
        help="MoE/gated_residual gate 隐层维度（<=0 则使用线性 gate）",
    )
    parser.add_argument(
        "--gate_init_bias",
        type=float,
        default=-4.0,
        help="gated_residual gate 的最后一层 bias 初始化（越小越接近 0）",
    )
    parser.add_argument(
        "--interaction_dim",
        type=int,
        default=256,
        help="interaction 融合的交互维度 K（用于 (W1x_sr)⊙(W2x_aux)）",
    )
    parser.add_argument(
        "--residualize_alpha",
        type=float,
        default=1e-2,
        help="residualize: x_aux≈A x_sr 的 ridge 正则系数",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="保存 train/val/test 的 predictions_*.csv（需要 *_features.npz 内包含 ids）。",
    )
    return parser.parse_args()


def load_npz_with_ids(feature_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    npz_path = feature_dir / f"{split}_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    npz = np.load(npz_path, allow_pickle=True)
    feats = npz["features"].astype(np.float32)
    labels = npz["labels"].astype(np.float32)
    ids = npz["ids"] if "ids" in npz.files else None
    return feats, labels, ids


def load_concat_npz(feature_dirs: Sequence[Path], split: str) -> Tuple[np.ndarray, np.ndarray]:
    """按 feature_dirs 顺序加载同一 split，并在特征维度做 concat。要求 ids/labels 完全对齐。"""
    feats_list: List[np.ndarray] = []
    ref_labels: Optional[np.ndarray] = None
    ref_ids: Optional[np.ndarray] = None

    for d in feature_dirs:
        feats, labels, ids = load_npz_with_ids(d, split)
        if ref_labels is None:
            ref_labels = labels
            ref_ids = ids
        else:
            if ref_labels.shape != labels.shape or not np.allclose(ref_labels, labels, equal_nan=True):
                raise ValueError(f"Labels mismatch across feature dirs for split={split}: {d}")
            if (ref_ids is None) != (ids is None):
                raise ValueError(f"IDs presence mismatch across feature dirs for split={split}: {d}")
            if ref_ids is not None and ids is not None and not np.array_equal(ref_ids, ids):
                raise ValueError(f"IDs mismatch across feature dirs for split={split}: {d}")
        feats_list.append(feats)

    x = np.concatenate(feats_list, axis=1) if len(feats_list) > 1 else feats_list[0]
    assert ref_labels is not None
    return x, ref_labels


def load_multi_npz(
    feature_dirs: Sequence[Path], split: str
) -> Tuple[List[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """加载多个特征目录，返回按目录顺序排列的特征列表。要求 ids/labels 完全对齐。"""
    feats_list: List[np.ndarray] = []
    ref_labels: Optional[np.ndarray] = None
    ref_ids: Optional[np.ndarray] = None

    for d in feature_dirs:
        feats, labels, ids = load_npz_with_ids(d, split)
        if ref_labels is None:
            ref_labels = labels
            ref_ids = ids
        else:
            if ref_labels.shape != labels.shape or not np.allclose(ref_labels, labels, equal_nan=True):
                raise ValueError(f"Labels mismatch across feature dirs for split={split}: {d}")
            if (ref_ids is None) != (ids is None):
                raise ValueError(f"IDs presence mismatch across feature dirs for split={split}: {d}")
            if ref_ids is not None and ids is not None and not np.array_equal(ref_ids, ids):
                raise ValueError(f"IDs mismatch across feature dirs for split={split}: {d}")
        feats_list.append(feats)

    assert ref_labels is not None
    return feats_list, ref_labels, ref_ids


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 1.0)
    std = np.clip(std, 1e-6, None)
    x_filled = np.where(np.isfinite(x), x, mean)
    return (x_filled - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_filled = np.where(np.isfinite(x), x, mean)
    return (x_filled - mean) / std


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))

    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    pearson = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    spearman = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")

    return {"rmse": rmse, "r2": r2, "pearson": pearson, "spearman": spearman}


def _normalize_id(x: object) -> str:
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        if np.isfinite(xf) and xf.is_integer():
            return str(int(xf))
        return str(xf)
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return str(x)


def save_predictions_csv(path: Path, ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if ids is None:
        raise ValueError("save_predictions_csv requires ids (got None).")
    if len(ids) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError(f"predictions length mismatch: ids={len(ids)} y_true={len(y_true)} y_pred={len(y_pred)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "y_true", "y_pred"])
        for i in range(len(ids)):
            w.writerow([_normalize_id(ids[i]), float(y_true[i]), float(y_pred[i])])


def ridge_closed_form(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    alpha_grid: Sequence[float],
) -> Dict[str, float]:
    device = x_train.device
    n, d = x_train.shape
    ones_train = torch.ones((n, 1), device=device)
    x_train_aug = torch.cat([x_train, ones_train], dim=1)  # bias
    xtx = x_train_aug.T @ x_train_aug
    xty = x_train_aug.T @ y_train.unsqueeze(-1)
    eye = torch.eye(d + 1, device=device)

    ones_val = torch.ones((x_val.shape[0], 1), device=device)
    x_val_aug = torch.cat([x_val, ones_val], dim=1)

    y_val_mean = y_val.mean()
    ss_tot = torch.sum((y_val - y_val_mean) ** 2).clamp_min(1e-8)

    best = {"val_rmse": float("inf"), "val_r2": float("-inf"), "best_alpha": float(alpha_grid[0])}
    best_w: Optional[torch.Tensor] = None

    for alpha in alpha_grid:
        w = torch.linalg.solve(xtx + float(alpha) * eye, xty)
        pred = (x_val_aug @ w).squeeze(-1)
        mse = torch.mean((pred - y_val) ** 2)
        rmse = torch.sqrt(mse).item()

        ss_res = torch.sum((y_val - pred) ** 2)
        r2 = (1.0 - (ss_res / ss_tot)).item()

        if rmse < best["val_rmse"]:
            best["val_rmse"] = float(rmse)
            best["val_r2"] = float(r2)
            best["best_alpha"] = float(alpha)
            best_w = w.detach().clone()

    assert best_w is not None
    best["w"] = best_w
    return best


def train_linear_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    alpha: float,
    epochs: int,
    lr: float,
) -> Dict[str, torch.Tensor]:
    d = x_train.shape[1]
    model = nn.Linear(d, 1, bias=True).to(x_train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(alpha))
    best = {"val_rmse": float("inf"), "state_dict": None}

    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x_train).squeeze(-1)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val).squeeze(-1)
            rmse = torch.sqrt(torch.mean((pred_val - y_val) ** 2)).item()
        if rmse < best["val_rmse"]:
            best["val_rmse"] = float(rmse)
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best["state_dict"] is not None
    return {"state_dict": best["state_dict"]}


def build_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Module:
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        last = int(h)
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


def build_mlp_encoder(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Module:
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must be non-empty for encoder")
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        last = int(h)
    return nn.Sequential(*layers)


class ResidualFusionHead(nn.Module):
    def __init__(self, primary_dim: int, aux_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.primary = build_mlp(primary_dim, hidden_dims, dropout)
        self.aux = build_mlp(aux_dim, hidden_dims, dropout)
        self.aux_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_primary: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        return self.primary(x_primary) + self.aux_scale * self.aux(x_aux)


class FiLMFusionHead(nn.Module):
    def __init__(self, primary_dim: int, aux_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.encoder = build_mlp_encoder(primary_dim, hidden_dims, dropout)
        self.hidden_dim = int(hidden_dims[-1])
        self.film = nn.Linear(aux_dim, 2 * self.hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.post_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, x_primary: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_primary)
        gamma_beta = self.film(x_aux)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        h = torch.relu(h)
        h = self.post_dropout(h)
        return self.out(h)


class MoEFusionHead(nn.Module):
    def __init__(
        self,
        primary_dim: int,
        aux_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        gate_hidden: int,
    ) -> None:
        super().__init__()
        self.primary_expert = build_mlp(primary_dim, hidden_dims, dropout)
        self.aux_expert = build_mlp(aux_dim, hidden_dims, dropout)

        gate_layers: List[nn.Module] = []
        gate_input_dim = int(primary_dim + aux_dim)
        if int(gate_hidden) > 0:
            gate_layers.append(nn.Linear(gate_input_dim, int(gate_hidden)))
            gate_layers.append(nn.ReLU())
            gate_layers.append(nn.Linear(int(gate_hidden), 1))
        else:
            gate_layers.append(nn.Linear(gate_input_dim, 1))
        self.gate = nn.Sequential(*gate_layers)

        last_linear = None
        for m in reversed(self.gate):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, 2.0)

    def forward(self, x_primary: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        pred_primary = self.primary_expert(x_primary).squeeze(-1)
        pred_aux = self.aux_expert(x_aux).squeeze(-1)
        gate_in = torch.cat([x_primary, x_aux], dim=-1)
        w = torch.sigmoid(self.gate(gate_in)).squeeze(-1)
        pred = w * pred_primary + (1.0 - w) * pred_aux
        return pred.unsqueeze(-1)


class GatedResidualFusionHead(nn.Module):
    def __init__(
        self,
        primary_dim: int,
        aux_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        gate_hidden: int,
        gate_init_bias: float,
    ) -> None:
        super().__init__()
        self.primary = build_mlp(primary_dim, hidden_dims, dropout)
        self.aux = build_mlp(aux_dim, hidden_dims, dropout)

        gate_layers: List[nn.Module] = []
        gate_input_dim = int(primary_dim + aux_dim)
        if int(gate_hidden) > 0:
            gate_layers.append(nn.Linear(gate_input_dim, int(gate_hidden)))
            gate_layers.append(nn.ReLU())
            gate_layers.append(nn.Linear(int(gate_hidden), 1))
        else:
            gate_layers.append(nn.Linear(gate_input_dim, 1))
        self.gate = nn.Sequential(*gate_layers)

        last_linear = None
        for m in reversed(self.gate):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, float(gate_init_bias))

    def forward(self, x_primary: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        pred_primary = self.primary(x_primary).squeeze(-1)
        pred_aux = self.aux(x_aux).squeeze(-1)
        gate_in = torch.cat([x_primary, x_aux], dim=-1)
        w = torch.sigmoid(self.gate(gate_in)).squeeze(-1)
        pred = pred_primary + w * pred_aux
        return pred.unsqueeze(-1)


class InteractionFusionHead(nn.Module):
    def __init__(
        self,
        primary_dim: int,
        aux_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        interaction_dim: int,
    ) -> None:
        super().__init__()
        self.primary_dim = int(primary_dim)
        self.aux_dim = int(aux_dim)
        self.interaction_dim = int(interaction_dim)

        self.proj_primary = nn.Linear(self.primary_dim, self.interaction_dim)
        self.proj_aux = nn.Linear(self.aux_dim, self.interaction_dim)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        mlp_in_dim = int(self.primary_dim + self.aux_dim + self.interaction_dim)
        self.mlp = build_mlp(mlp_in_dim, hidden_dims, dropout)

    def forward(self, x_primary: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        h1 = self.proj_primary(x_primary)
        h2 = self.proj_aux(x_aux)
        inter = self.dropout(h1 * h2)
        z = torch.cat([x_primary, x_aux, inter], dim=-1)
        return self.mlp(z)


class MultiResidualFusionHead(nn.Module):
    def __init__(self, primary_dim: int, aux_dims: Sequence[int], hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.primary = build_mlp(primary_dim, hidden_dims, dropout)
        self.aux_dims = [int(d) for d in aux_dims]
        if len(self.aux_dims) < 1:
            raise ValueError("aux_dims must be non-empty for residual_multi")

        self.aux_mlps = nn.ModuleList([build_mlp(int(d), hidden_dims, dropout) for d in self.aux_dims])
        self.aux_scales = nn.Parameter(torch.zeros(len(self.aux_dims)))

    def forward(self, x_primary: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        pred = self.primary(x_primary)
        parts = torch.split(x_aux, self.aux_dims, dim=-1)
        if len(parts) != len(self.aux_mlps):
            raise ValueError("aux split mismatch")
        for i, (mlp, xi) in enumerate(zip(self.aux_mlps, parts)):
            pred = pred + self.aux_scales[i] * mlp(xi)
        return pred


def train_mlp_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dims: Sequence[int],
    dropout: float,
    alpha: float,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model = build_mlp(x_train.shape[1], hidden_dims, dropout).to(x_train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(alpha))

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

    best = {"val_rmse": float("inf"), "state_dict": None, "best_epoch": 0}
    bad = 0

    for epoch in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val).squeeze(-1)
            rmse = torch.sqrt(torch.mean((pred_val - y_val) ** 2)).item()

        if rmse < best["val_rmse"] - 1e-8:
            best["val_rmse"] = float(rmse)
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["best_epoch"] = int(epoch)
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    assert best["state_dict"] is not None
    return {"state_dict": best["state_dict"], "best_epoch": torch.tensor(best["best_epoch"])}


def train_fusion_head(
    model: nn.Module,
    x_train_primary: torch.Tensor,
    x_train_aux: torch.Tensor,
    y_train: torch.Tensor,
    x_val_primary: torch.Tensor,
    x_val_aux: torch.Tensor,
    y_val: torch.Tensor,
    alpha: float,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(alpha))

    train_ds = TensorDataset(x_train_primary, x_train_aux, y_train)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

    best = {"val_rmse": float("inf"), "state_dict": None, "best_epoch": 0}
    bad = 0

    for epoch in range(int(epochs)):
        model.train()
        for xb_primary, xb_aux, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb_primary, xb_aux).squeeze(-1)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_primary, x_val_aux).squeeze(-1)
            rmse = torch.sqrt(torch.mean((pred_val - y_val) ** 2)).item()

        if rmse < best["val_rmse"] - 1e-8:
            best["val_rmse"] = float(rmse)
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["best_epoch"] = int(epoch)
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    assert best["state_dict"] is not None
    return {"state_dict": best["state_dict"], "best_epoch": torch.tensor(best["best_epoch"])}


def fit_ridge_map_with_bias(x_primary: torch.Tensor, x_aux: torch.Tensor, alpha: float) -> torch.Tensor:
    """Fit a ridge map: x_aux ~= [x_primary, 1] @ A."""
    device = x_primary.device
    n, d = x_primary.shape
    ones = torch.ones((n, 1), device=device)
    x_aug = torch.cat([x_primary, ones], dim=1)  # [n, d+1]

    xtx = x_aug.T @ x_aug
    eye = torch.eye(d + 1, device=device)
    xty = x_aug.T @ x_aux
    return torch.linalg.solve(xtx + float(alpha) * eye, xty)


def apply_ridge_map_with_bias(x_primary: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    device = x_primary.device
    ones = torch.ones((x_primary.shape[0], 1), device=device)
    x_aug = torch.cat([x_primary, ones], dim=1)
    return x_aug @ a


def main() -> None:
    args = parse_args()

    feature_dirs: List[Path] = []
    feature_dirs.append(Path(args.feature_dir))
    if args.feature_dirs:
        feature_dirs.extend([Path(p) for p in args.feature_dirs])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[INFO] feature_dirs={feature_dirs}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] head_type={args.head_type}")
    print(f"[INFO] device={device}")

    x_train_list, y_train_np, train_ids = load_multi_npz(feature_dirs, "train")
    x_val_list, y_val_np, val_ids = load_multi_npz(feature_dirs, "val")
    x_test_list, y_test_np, test_ids = load_multi_npz(feature_dirs, "test")

    if args.save_predictions:
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            if ids is None:
                raise ValueError(
                    f"--save_predictions requires ids in {split_name}_features.npz, "
                    f"but ids is missing for split={split_name}."
                )

    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    x_train_std_list: List[np.ndarray] = []
    for x in x_train_list:
        x_std, mean, std = standardize_fit(x)
        x_train_std_list.append(x_std)
        means.append(mean)
        stds.append(std)

    x_val_std_list: List[np.ndarray] = []
    x_test_std_list: List[np.ndarray] = []
    for i in range(len(x_train_list)):
        x_val_std_list.append(standardize_apply(x_val_list[i], means[i], stds[i]))
        x_test_std_list.append(standardize_apply(x_test_list[i], means[i], stds[i]))

    x_train_np = np.concatenate(x_train_std_list, axis=1) if len(x_train_std_list) > 1 else x_train_std_list[0]
    x_val_np = np.concatenate(x_val_std_list, axis=1) if len(x_val_std_list) > 1 else x_val_std_list[0]
    x_test_np = np.concatenate(x_test_std_list, axis=1) if len(x_test_std_list) > 1 else x_test_std_list[0]

    x_train = torch.from_numpy(x_train_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)
    x_test = torch.from_numpy(x_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)

    if args.head_type == "ridge":
        alpha_grid = args.alpha_grid if args.alpha_grid is not None else [args.alpha]
        best = ridge_closed_form(x_train, y_train, x_val, y_val, alpha_grid)

        # 用 train+val 重训（闭式解）
        x_all = torch.cat([x_train, x_val], dim=0)
        y_all = torch.cat([y_train, y_val], dim=0)
        ones_all = torch.ones((x_all.shape[0], 1), device=device)
        x_all_aug = torch.cat([x_all, ones_all], dim=1)

        ones_test = torch.ones((x_test.shape[0], 1), device=device)
        x_test_aug = torch.cat([x_test, ones_test], dim=1)

        alpha = float(best["best_alpha"])
        n_all, d = x_all.shape
        eye = torch.eye(d + 1, device=device)
        w = torch.linalg.solve(x_all_aug.T @ x_all_aug + alpha * eye, x_all_aug.T @ y_all.unsqueeze(-1))

        ones_train = torch.ones((x_train.shape[0], 1), device=device)
        ones_val = torch.ones((x_val.shape[0], 1), device=device)
        x_train_aug = torch.cat([x_train, ones_train], dim=1)
        x_val_aug = torch.cat([x_val, ones_val], dim=1)

        pred_train = (x_train_aug @ w).squeeze(-1).detach().cpu().numpy()
        pred_val = (x_val_aug @ w).squeeze(-1).detach().cpu().numpy()
        pred_test = (x_test_aug @ w).squeeze(-1).detach().cpu().numpy()

        if args.save_predictions:
            save_predictions_csv(output_dir / "predictions_train.csv", train_ids, y_train_np, pred_train)
            save_predictions_csv(output_dir / "predictions_val.csv", val_ids, y_val_np, pred_val)
            save_predictions_csv(output_dir / "predictions_test.csv", test_ids, y_test_np, pred_test)
        metrics = compute_metrics(y_test.detach().cpu().numpy(), pred_test)
        metrics["best_alpha"] = float(best["best_alpha"])

        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[RESULT]", metrics)
        return

    if args.head_type == "linear":
        state = train_linear_head(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            alpha=float(args.alpha),
            epochs=int(args.epochs),
            lr=float(args.lr),
        )
        model = nn.Linear(x_train.shape[1], 1, bias=True).to(device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train).squeeze(-1).detach().cpu().numpy()
            pred_val = model(x_val).squeeze(-1).detach().cpu().numpy()
            pred_test = model(x_test).squeeze(-1).detach().cpu().numpy()
        if args.save_predictions:
            save_predictions_csv(output_dir / "predictions_train.csv", train_ids, y_train_np, pred_train)
            save_predictions_csv(output_dir / "predictions_val.csv", val_ids, y_val_np, pred_val)
            save_predictions_csv(output_dir / "predictions_test.csv", test_ids, y_test_np, pred_test)
        metrics = compute_metrics(y_test.detach().cpu().numpy(), pred_test)
        metrics["alpha"] = float(args.alpha)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[RESULT]", metrics)
        return

    if args.head_type == "mlp":
        # MLP：多 seed 复核（默认 3 个）
        best_all = {"rmse": float("inf"), "seed": None, "state_dict": None}
        for seed in args.seeds:
            state = train_mlp_head(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
                alpha=float(args.alpha),
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                patience=int(args.patience),
                seed=int(seed),
            )
            model = build_mlp(x_train.shape[1], args.hidden_dims, float(args.dropout)).to(device)
            model.load_state_dict(state["state_dict"])
            model.eval()
            with torch.no_grad():
                pred_val = model(x_val).squeeze(-1).detach().cpu().numpy()
            rmse = float(np.sqrt(np.mean((pred_val - y_val.detach().cpu().numpy()) ** 2)))
            if rmse < best_all["rmse"]:
                best_all["rmse"] = rmse
                best_all["seed"] = int(seed)
                best_all["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        assert best_all["state_dict"] is not None
        best_seed = int(best_all["seed"])
        model = build_mlp(x_train.shape[1], args.hidden_dims, float(args.dropout)).to(device)
        model.load_state_dict(best_all["state_dict"])
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train).squeeze(-1).detach().cpu().numpy()
            pred_val = model(x_val).squeeze(-1).detach().cpu().numpy()
            pred_test = model(x_test).squeeze(-1).detach().cpu().numpy()

        if args.save_predictions:
            save_predictions_csv(output_dir / "predictions_train.csv", train_ids, y_train_np, pred_train)
            save_predictions_csv(output_dir / "predictions_val.csv", val_ids, y_val_np, pred_val)
            save_predictions_csv(output_dir / "predictions_test.csv", test_ids, y_test_np, pred_test)

        metrics = compute_metrics(y_test.detach().cpu().numpy(), pred_test)
        metrics.update(
            {
                "alpha": float(args.alpha),
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "dropout": float(args.dropout),
                "hidden_dims": [int(x) for x in args.hidden_dims],
                "best_seed": best_seed,
            }
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[RESULT]", metrics)
        return

    if len(x_train_std_list) < 2:
        raise ValueError(f"head_type={args.head_type} requires >=2 feature dirs (primary + aux)")

    aux_dims = [int(x.shape[1]) for x in x_train_std_list[1:]]

    x_train_primary = torch.from_numpy(x_train_std_list[0]).to(device)
    x_val_primary = torch.from_numpy(x_val_std_list[0]).to(device)
    x_test_primary = torch.from_numpy(x_test_std_list[0]).to(device)

    x_train_aux = torch.from_numpy(np.concatenate(x_train_std_list[1:], axis=1)).to(device)
    x_val_aux = torch.from_numpy(np.concatenate(x_val_std_list[1:], axis=1)).to(device)
    x_test_aux = torch.from_numpy(np.concatenate(x_test_std_list[1:], axis=1)).to(device)

    if args.head_type == "stacking":
        best_all = {"rmse": float("inf"), "seed": None, "sr_state": None, "aux_state": None}
        for seed in args.seeds:
            # stage 1: sr -> y
            sr_state = train_mlp_head(
                x_train=x_train_primary,
                y_train=y_train,
                x_val=x_val_primary,
                y_val=y_val,
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
                alpha=float(args.alpha),
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                patience=int(args.patience),
                seed=int(seed),
            )
            sr_model = build_mlp(int(x_train_primary.shape[1]), args.hidden_dims, float(args.dropout)).to(device)
            sr_model.load_state_dict(sr_state["state_dict"])
            sr_model.eval()
            with torch.no_grad():
                pred_train_sr = sr_model(x_train_primary).squeeze(-1)
                pred_val_sr = sr_model(x_val_primary).squeeze(-1)

            res_train = (y_train - pred_train_sr).detach()
            res_val = (y_val - pred_val_sr).detach()

            # stage 2: aux -> residual
            aux_state = train_mlp_head(
                x_train=x_train_aux,
                y_train=res_train,
                x_val=x_val_aux,
                y_val=res_val,
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
                alpha=float(args.alpha),
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                patience=int(args.patience),
                seed=int(seed),
            )
            aux_model = build_mlp(int(x_train_aux.shape[1]), args.hidden_dims, float(args.dropout)).to(device)
            aux_model.load_state_dict(aux_state["state_dict"])
            aux_model.eval()
            with torch.no_grad():
                pred_val_aux = aux_model(x_val_aux).squeeze(-1)
                pred_val = (pred_val_sr + pred_val_aux).detach().cpu().numpy()

            rmse = float(np.sqrt(np.mean((pred_val - y_val.detach().cpu().numpy()) ** 2)))
            if rmse < best_all["rmse"]:
                best_all["rmse"] = rmse
                best_all["seed"] = int(seed)
                best_all["sr_state"] = {k: v.detach().cpu().clone() for k, v in sr_state["state_dict"].items()}
                best_all["aux_state"] = {k: v.detach().cpu().clone() for k, v in aux_state["state_dict"].items()}

        assert best_all["sr_state"] is not None and best_all["aux_state"] is not None
        best_seed = int(best_all["seed"])

        sr_model = build_mlp(int(x_train_primary.shape[1]), args.hidden_dims, float(args.dropout)).to(device)
        sr_model.load_state_dict(best_all["sr_state"])
        sr_model.eval()
        aux_model = build_mlp(int(x_train_aux.shape[1]), args.hidden_dims, float(args.dropout)).to(device)
        aux_model.load_state_dict(best_all["aux_state"])
        aux_model.eval()

        with torch.no_grad():
            pred_train = (sr_model(x_train_primary).squeeze(-1) + aux_model(x_train_aux).squeeze(-1)).detach().cpu().numpy()
            pred_val = (sr_model(x_val_primary).squeeze(-1) + aux_model(x_val_aux).squeeze(-1)).detach().cpu().numpy()
            pred_test = (sr_model(x_test_primary).squeeze(-1) + aux_model(x_test_aux).squeeze(-1)).detach().cpu().numpy()

        if args.save_predictions:
            save_predictions_csv(output_dir / "predictions_train.csv", train_ids, y_train_np, pred_train)
            save_predictions_csv(output_dir / "predictions_val.csv", val_ids, y_val_np, pred_val)
            save_predictions_csv(output_dir / "predictions_test.csv", test_ids, y_test_np, pred_test)

        metrics = compute_metrics(y_test.detach().cpu().numpy(), pred_test)
        metrics.update(
            {
                "alpha": float(args.alpha),
                "lr": float(args.lr),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "dropout": float(args.dropout),
                "hidden_dims": [int(x) for x in args.hidden_dims],
                "best_seed": best_seed,
                "head_type": str(args.head_type),
            }
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[RESULT]", metrics)
        return

    effective_head_type = str(args.head_type)
    residualize_a: Optional[torch.Tensor] = None
    if args.head_type == "residualize":
        residualize_alpha = float(args.residualize_alpha)
        residualize_a = fit_ridge_map_with_bias(x_train_primary, x_train_aux, residualize_alpha)
        x_train_aux = x_train_aux - apply_ridge_map_with_bias(x_train_primary, residualize_a)
        x_val_aux = x_val_aux - apply_ridge_map_with_bias(x_val_primary, residualize_a)
        x_test_aux = x_test_aux - apply_ridge_map_with_bias(x_test_primary, residualize_a)
        effective_head_type = "residual"

    def build_fusion_model() -> nn.Module:
        if effective_head_type == "residual":
            return ResidualFusionHead(
                primary_dim=int(x_train_primary.shape[1]),
                aux_dim=int(x_train_aux.shape[1]),
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
            )
        if effective_head_type == "film":
            return FiLMFusionHead(
                primary_dim=int(x_train_primary.shape[1]),
                aux_dim=int(x_train_aux.shape[1]),
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
            )
        if effective_head_type == "moe":
            return MoEFusionHead(
                primary_dim=int(x_train_primary.shape[1]),
                aux_dim=int(x_train_aux.shape[1]),
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
                gate_hidden=int(args.gate_hidden),
            )
        if effective_head_type == "gated_residual":
            return GatedResidualFusionHead(
                primary_dim=int(x_train_primary.shape[1]),
                aux_dim=int(x_train_aux.shape[1]),
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
                gate_hidden=int(args.gate_hidden),
                gate_init_bias=float(args.gate_init_bias),
            )
        if effective_head_type == "interaction":
            return InteractionFusionHead(
                primary_dim=int(x_train_primary.shape[1]),
                aux_dim=int(x_train_aux.shape[1]),
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
                interaction_dim=int(args.interaction_dim),
            )
        if effective_head_type == "residual_multi":
            return MultiResidualFusionHead(
                primary_dim=int(x_train_primary.shape[1]),
                aux_dims=aux_dims,
                hidden_dims=args.hidden_dims,
                dropout=float(args.dropout),
            )
        raise ValueError(f"Unknown head_type={args.head_type}")

    best_all = {"rmse": float("inf"), "seed": None, "state_dict": None}
    for seed in args.seeds:
        model = build_fusion_model().to(device)
        state = train_fusion_head(
            model=model,
            x_train_primary=x_train_primary,
            x_train_aux=x_train_aux,
            y_train=y_train,
            x_val_primary=x_val_primary,
            x_val_aux=x_val_aux,
            y_val=y_val,
            alpha=float(args.alpha),
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            patience=int(args.patience),
            seed=int(seed),
        )
        model = build_fusion_model().to(device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_primary, x_val_aux).squeeze(-1).detach().cpu().numpy()
        rmse = float(np.sqrt(np.mean((pred_val - y_val.detach().cpu().numpy()) ** 2)))
        if rmse < best_all["rmse"]:
            best_all["rmse"] = rmse
            best_all["seed"] = int(seed)
            best_all["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_all["state_dict"] is not None
    best_seed = int(best_all["seed"])
    model = build_fusion_model().to(device)
    model.load_state_dict(best_all["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_train = model(x_train_primary, x_train_aux).squeeze(-1).detach().cpu().numpy()
        pred_val = model(x_val_primary, x_val_aux).squeeze(-1).detach().cpu().numpy()
        pred_test = model(x_test_primary, x_test_aux).squeeze(-1).detach().cpu().numpy()

    if args.save_predictions:
        save_predictions_csv(output_dir / "predictions_train.csv", train_ids, y_train_np, pred_train)
        save_predictions_csv(output_dir / "predictions_val.csv", val_ids, y_val_np, pred_val)
        save_predictions_csv(output_dir / "predictions_test.csv", test_ids, y_test_np, pred_test)

    metrics = compute_metrics(y_test.detach().cpu().numpy(), pred_test)
    metrics.update(
        {
            "alpha": float(args.alpha),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "dropout": float(args.dropout),
            "hidden_dims": [int(x) for x in args.hidden_dims],
            "best_seed": best_seed,
            "head_type": str(args.head_type),
            "gate_hidden": int(args.gate_hidden) if str(args.head_type) in {"moe", "gated_residual"} else None,
            "gate_init_bias": float(args.gate_init_bias) if str(args.head_type) == "gated_residual" else None,
            "interaction_dim": int(args.interaction_dim) if str(args.head_type) == "interaction" else None,
            "residualize_alpha": float(args.residualize_alpha) if str(args.head_type) == "residualize" else None,
            "aux_dims": aux_dims if str(args.head_type) == "residual_multi" else None,
        }
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[RESULT]", metrics)
    return


if __name__ == "__main__":
    main()
