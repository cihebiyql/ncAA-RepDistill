#!/usr/bin/env python3
"""
Strict ncaa_cpp (Permeability regression) benchmark for frozen embeddings.

Implements two protocols:
- Protocol A (Ridge): StandardScaler(train fit) + Ridge(alpha_grid), select on val RMSE,
  retrain on train+val, report test metrics + bootstrap95% CI (seed=42, B=1000).
- Protocol B (MLP small): StandardScaler(train fit) + fixed MLP, early-stop on val RMSE,
  report test mean±std across 10 seeds.

Input: a feature_dir produced by scripts/extract_phase3_features.py containing:
  train_features.npz / val_features.npz / test_features.npz
Each NPZ must contain: features [N,D], labels [N], and optionally ids [N].
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict ncaa_cpp frozen-embedding benchmark (Ridge and/or MLP)")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory with *_features.npz")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for metrics JSON")
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["both", "ridge", "mlp"],
        default="both",
        help="Evaluation protocol to run: both (default), ridge only, or mlp only.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Write predictions_{train,val,test}.csv (requires ids in *_features.npz).",
    )

    parser.add_argument(
        "--alpha_grid",
        type=float,
        nargs="*",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        help="Ridge alpha grid (selected on val RMSE)",
    )
    parser.add_argument("--bootstrap_n", type=int, default=1000, help="Bootstrap iterations on test")
    parser.add_argument("--bootstrap_seed", type=int, default=42, help="Bootstrap RNG seed")

    parser.add_argument("--mlp_hidden_dims", type=int, nargs="*", default=[256], help="MLP hidden dims")
    parser.add_argument("--mlp_dropout", type=float, default=0.1, help="MLP dropout")
    parser.add_argument("--mlp_lr", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4, help="MLP weight decay (AdamW)")
    parser.add_argument("--mlp_epochs", type=int, default=400, help="MLP max epochs")
    parser.add_argument("--mlp_patience", type=int, default=40, help="MLP early-stop patience (val RMSE)")
    parser.add_argument("--mlp_batch_size", type=int, default=256, help="MLP batch size")
    parser.add_argument(
        "--mlp_seeds",
        type=int,
        nargs="*",
        default=[42, 123, 202, 314, 404, 456, 777, 1013, 1314, 2024],
        help="MLP seeds (Protocol B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for MLP training (e.g. cuda:0)",
    )
    return parser.parse_args()


def _load_npz(feature_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    path = feature_dir / f"{split}_features.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    npz = np.load(path, allow_pickle=True)
    x = npz["features"].astype(np.float32)
    y = npz["labels"].astype(np.float32)
    ids = npz["ids"] if "ids" in npz.files else None
    return x, y, ids


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    pearson = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    spearman = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


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


def _ridge_fit_weights(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Closed-form ridge with intercept NOT penalized (matches sklearn Ridge(fit_intercept=True)).
    x: [N, D], y: [N]
    returns w: [D+1, 1] for [x, 1] augmented.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    n, d = x.shape
    x_aug = np.concatenate([x, np.ones((n, 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    xty = x_aug.T @ y.reshape(-1, 1)

    penalty = np.eye(d + 1, dtype=x.dtype)
    penalty[-1, -1] = 0.0  # do not penalize intercept

    w = np.linalg.solve(xtx + float(alpha) * penalty, xty)
    return w


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    n = x.shape[0]
    x_aug = np.concatenate([x, np.ones((n, 1), dtype=x.dtype)], axis=1)
    return (x_aug @ w).reshape(-1)


@dataclass(frozen=True)
class RidgeResult:
    best_alpha: float
    val_rmse: float
    test_metrics: Dict[str, float]
    test_ci95: Dict[str, Tuple[float, float]]


def run_protocol_a_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha_grid: Sequence[float],
    bootstrap_n: int,
    bootstrap_seed: int,
) -> RidgeResult:
    best_alpha = float(alpha_grid[0])
    best_val_rmse = float("inf")

    for alpha in alpha_grid:
        w = _ridge_fit_weights(x_train, y_train, float(alpha))
        pred_val = _ridge_predict(x_val, w)
        rmse = float(np.sqrt(np.mean((pred_val - y_val.astype(np.float64)) ** 2)))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_alpha = float(alpha)

    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    w_all = _ridge_fit_weights(x_all, y_all, best_alpha)
    pred_test = _ridge_predict(x_test, w_all)
    test_metrics = compute_metrics(y_test, pred_test)
    test_metrics["best_alpha"] = float(best_alpha)

    rng = np.random.default_rng(int(bootstrap_seed))
    n_test = int(len(y_test))
    r2_list: List[float] = []
    rmse_list: List[float] = []
    mae_list: List[float] = []
    pearson_list: List[float] = []
    spearman_list: List[float] = []

    y_true = y_test.astype(np.float64)
    y_pred = pred_test.astype(np.float64)
    for _ in range(int(bootstrap_n)):
        idx = rng.integers(0, n_test, size=n_test)
        m = compute_metrics(y_true[idx], y_pred[idx])
        r2_list.append(float(m["r2"]))
        rmse_list.append(float(m["rmse"]))
        mae_list.append(float(m["mae"]))
        pearson_list.append(float(m["pearson"]))
        spearman_list.append(float(m["spearman"]))

    def _ci(values: Sequence[float]) -> Tuple[float, float]:
        lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5]).tolist()
        return float(lo), float(hi)

    test_ci95 = {
        "r2": _ci(r2_list),
        "rmse": _ci(rmse_list),
        "mae": _ci(mae_list),
        "pearson": _ci(pearson_list),
        "spearman": _ci(spearman_list),
    }

    return RidgeResult(
        best_alpha=float(best_alpha),
        val_rmse=float(best_val_rmse),
        test_metrics=test_metrics,
        test_ci95=test_ci95,
    )


def build_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Module:
    layers: List[nn.Module] = []
    last = int(input_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(last, int(h)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        last = int(h)
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class MlpSeedResult:
    seed: int
    best_epoch: int
    test_metrics: Dict[str, float]


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train_eval_mlp_one_seed(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test_np: np.ndarray,
    *,
    hidden_dims: Sequence[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    save_predictions_dir: Optional[Path] = None,
    train_ids: Optional[np.ndarray] = None,
    val_ids: Optional[np.ndarray] = None,
    test_ids: Optional[np.ndarray] = None,
    y_train_np: Optional[np.ndarray] = None,
    y_val_np: Optional[np.ndarray] = None,
) -> MlpSeedResult:
    _set_seed(int(seed))
    device = x_train.device

    model = build_mlp(int(x_train.shape[1]), hidden_dims, float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    best = {"val_rmse": float("inf"), "best_epoch": 0, "state_dict": None}
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
            val_rmse = torch.sqrt(torch.mean((pred_val - y_val) ** 2)).item()

        if val_rmse < best["val_rmse"] - 1e-8:
            best["val_rmse"] = float(val_rmse)
            best["best_epoch"] = int(epoch)
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    assert best["state_dict"] is not None
    model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_train = model(x_train).squeeze(-1).detach().cpu().numpy()
        pred_val = model(x_val).squeeze(-1).detach().cpu().numpy()
        pred_test = model(x_test).squeeze(-1).detach().cpu().numpy()

    if save_predictions_dir is not None:
        if train_ids is None or val_ids is None or test_ids is None:
            raise ValueError("save_predictions_dir provided but ids are missing.")
        if y_train_np is None or y_val_np is None:
            raise ValueError("save_predictions_dir provided but y_train_np/y_val_np are missing.")
        save_predictions_csv(save_predictions_dir / "predictions_train.csv", train_ids, y_train_np, pred_train)
        save_predictions_csv(save_predictions_dir / "predictions_val.csv", val_ids, y_val_np, pred_val)
        save_predictions_csv(save_predictions_dir / "predictions_test.csv", test_ids, y_test_np, pred_test)

    metrics = compute_metrics(y_test_np, pred_test)
    return MlpSeedResult(seed=int(seed), best_epoch=int(best["best_epoch"]), test_metrics=metrics)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return mean, std


def main() -> None:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_ids = _load_npz(feature_dir, "train")
    x_val, y_val, val_ids = _load_npz(feature_dir, "val")
    x_test, y_test, test_ids = _load_npz(feature_dir, "test")

    if args.save_predictions:
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            if ids is None:
                raise ValueError(
                    f"--save_predictions requires ids in {split_name}_features.npz, "
                    f"but ids is missing for split={split_name}."
                )

    d = int(x_train.shape[1])
    meta = {
        "feature_dir": str(feature_dir),
        "dims": d,
        "splits": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "has_ids": bool(train_ids is not None and val_ids is not None and test_ids is not None),
    }

    x_train_s, mean, std = standardize_fit(x_train)
    x_val_s = standardize_apply(x_val, mean, std)
    x_test_s = standardize_apply(x_test, mean, std)

    run_ridge = args.protocol in ("both", "ridge")
    run_mlp = args.protocol in ("both", "mlp")

    ridge: Optional[RidgeResult] = None
    if run_ridge:
        ridge = run_protocol_a_ridge(
            x_train_s,
            y_train,
            x_val_s,
            y_val,
            x_test_s,
            y_test,
            alpha_grid=args.alpha_grid,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
        )

        if args.save_predictions:
            best_alpha = float(ridge.best_alpha)
            x_all = np.concatenate([x_train_s, x_val_s], axis=0)
            y_all = np.concatenate([y_train, y_val], axis=0)
            w_all = _ridge_fit_weights(x_all, y_all, best_alpha)
            pred_train = _ridge_predict(x_train_s, w_all)
            pred_val = _ridge_predict(x_val_s, w_all)
            pred_test = _ridge_predict(x_test_s, w_all)
            pred_dir = output_dir / "protocol_a_ridge"
            save_predictions_csv(pred_dir / "predictions_train.csv", train_ids, y_train, pred_train)
            save_predictions_csv(pred_dir / "predictions_val.csv", val_ids, y_val, pred_val)
            save_predictions_csv(pred_dir / "predictions_test.csv", test_ids, y_test, pred_test)

    out: Dict[str, Any] = {"meta": meta}
    out["meta"]["protocol"] = str(args.protocol)

    if run_mlp:
        device = torch.device(str(args.device))
        x_train_t = torch.from_numpy(x_train_s).to(device)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
        x_val_t = torch.from_numpy(x_val_s).to(device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
        x_test_t = torch.from_numpy(x_test_s).to(device)

        mlp_seed_results: List[MlpSeedResult] = []
        for seed in args.mlp_seeds:
            pred_dir = None
            if args.save_predictions:
                pred_dir = output_dir / "protocol_b_mlp_small" / f"seed_{int(seed)}"
            r = train_eval_mlp_one_seed(
                x_train=x_train_t,
                y_train=y_train_t,
                x_val=x_val_t,
                y_val=y_val_t,
                x_test=x_test_t,
                y_test_np=y_test,
                hidden_dims=[int(x) for x in args.mlp_hidden_dims],
                dropout=float(args.mlp_dropout),
                lr=float(args.mlp_lr),
                weight_decay=float(args.mlp_weight_decay),
                epochs=int(args.mlp_epochs),
                batch_size=int(args.mlp_batch_size),
                patience=int(args.mlp_patience),
                seed=int(seed),
                save_predictions_dir=pred_dir,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                y_train_np=y_train,
                y_val_np=y_val,
            )
            mlp_seed_results.append(r)

        mlp_metrics_by_seed = {str(r.seed): r.test_metrics for r in mlp_seed_results}
        r2_list = [float(r.test_metrics["r2"]) for r in mlp_seed_results]
        rmse_list = [float(r.test_metrics["rmse"]) for r in mlp_seed_results]
        mae_list = [float(r.test_metrics["mae"]) for r in mlp_seed_results]
        pearson_list = [float(r.test_metrics["pearson"]) for r in mlp_seed_results]
        spearman_list = [float(r.test_metrics["spearman"]) for r in mlp_seed_results]

        mlp_mean = {
            "r2": _mean_std(r2_list)[0],
            "rmse": _mean_std(rmse_list)[0],
            "mae": _mean_std(mae_list)[0],
            "pearson": _mean_std(pearson_list)[0],
            "spearman": _mean_std(spearman_list)[0],
        }
        mlp_std = {
            "r2": _mean_std(r2_list)[1],
            "rmse": _mean_std(rmse_list)[1],
            "mae": _mean_std(mae_list)[1],
            "pearson": _mean_std(pearson_list)[1],
            "spearman": _mean_std(spearman_list)[1],
        }

        out["protocol_b_mlp_small"] = {
            "seeds": [int(s) for s in args.mlp_seeds],
            "hidden_dims": [int(x) for x in args.mlp_hidden_dims],
            "dropout": float(args.mlp_dropout),
            "lr": float(args.mlp_lr),
            "weight_decay": float(args.mlp_weight_decay),
            "epochs": int(args.mlp_epochs),
            "patience": int(args.mlp_patience),
            "batch_size": int(args.mlp_batch_size),
            "test_mean": mlp_mean,
            "test_std": mlp_std,
            "test_by_seed": mlp_metrics_by_seed,
            "best_epoch_by_seed": {str(r.seed): int(r.best_epoch) for r in mlp_seed_results},
        }

    if run_ridge:
        assert ridge is not None
        out["protocol_a_ridge"] = {
            "alpha_grid": [float(x) for x in args.alpha_grid],
            "selection_metric": "val_rmse",
            "best_alpha": float(ridge.best_alpha),
            "val_rmse_best_alpha": float(ridge.val_rmse),
            "test": ridge.test_metrics,
            "test_ci95_bootstrap": ridge.test_ci95,
            "bootstrap": {"n": int(args.bootstrap_n), "seed": int(args.bootstrap_seed)},
        }

    (output_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
