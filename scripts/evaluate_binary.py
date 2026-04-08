#!/usr/bin/env python3
"""
Strict binary classification benchmark for frozen embeddings.

Implements two protocols:
- Protocol A (LogisticRegression): StandardScaler(train fit) + LogisticRegression(C_grid),
  select on val MCC (threshold selected on val, MCC-max), retrain on train+val, report test
  metrics + bootstrap95% CI (seed=42, B=1000).
- Protocol B (MLP small): StandardScaler(train fit) + fixed MLP classifier, early-stop on
  val AUROC, report test mean±std across 10 seeds (threshold selected on val, MCC-max).

Input: a feature_dir containing:
  train_features.npz / val_features.npz / test_features.npz
Each NPZ must contain: features [N,D], labels [N] (0/1), and optionally ids [N].

Optionally, pass multiple feature dirs via --feature_dirs to concatenate features along dim=1.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict binary classification benchmark (LogReg + MLP)")
    p.add_argument("--feature_dir", type=str, required=True, help="Directory with *_features.npz")
    p.add_argument(
        "--feature_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Additional feature dirs (each with *_features.npz); concatenated to --feature_dir on dim=1.",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for metrics")
    p.add_argument(
        "--save_predictions",
        action="store_true",
        help="Write predictions_{train,val,test}.csv (requires ids in *_features.npz).",
    )

    p.add_argument(
        "--c_grid",
        type=float,
        nargs="*",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        help="LogisticRegression C grid (selected on val MCC)",
    )
    p.add_argument("--bootstrap_n", type=int, default=1000, help="Bootstrap iterations on test")
    p.add_argument("--bootstrap_seed", type=int, default=42, help="Bootstrap RNG seed")
    p.add_argument("--threshold_grid_size", type=int, default=1001, help="Threshold grid size in [0,1]")

    p.add_argument("--mlp_hidden_dims", type=int, nargs="*", default=[256], help="MLP hidden dims")
    p.add_argument("--mlp_dropout", type=float, default=0.1, help="MLP dropout")
    p.add_argument("--mlp_lr", type=float, default=1e-3, help="MLP learning rate")
    p.add_argument("--mlp_weight_decay", type=float, default=1e-4, help="MLP weight decay (AdamW)")
    p.add_argument("--mlp_epochs", type=int, default=400, help="MLP max epochs")
    p.add_argument("--mlp_patience", type=int, default=40, help="MLP early-stop patience (val AUROC)")
    p.add_argument("--mlp_batch_size", type=int, default=256, help="MLP batch size")
    p.add_argument(
        "--mlp_seeds",
        type=int,
        nargs="*",
        default=[42, 123, 202, 314, 404, 456, 777, 1013, 1314, 2024],
        help="MLP seeds (Protocol B)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for MLP training (e.g. cuda:0)",
    )
    return p.parse_args()


def _load_npz(feature_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    path = feature_dir / f"{split}_features.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    npz = np.load(path, allow_pickle=True)
    x = npz["features"].astype(np.float32)
    y = npz["labels"].astype(np.float32)
    ids = npz["ids"] if "ids" in npz.files else None
    return x, y, ids


def _load_concat_npz(feature_dirs: Sequence[Path], split: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    feats_list: List[np.ndarray] = []
    ref_y: Optional[np.ndarray] = None
    ref_ids: Optional[np.ndarray] = None

    for d in feature_dirs:
        x, y, ids = _load_npz(d, split)
        if ref_y is None:
            ref_y = y
            ref_ids = ids
        else:
            if ref_y.shape != y.shape or not np.allclose(ref_y, y, equal_nan=True):
                raise ValueError(f"Labels mismatch across feature dirs for split={split}: {d}")
            if (ref_ids is None) != (ids is None):
                raise ValueError(f"IDs presence mismatch across feature dirs for split={split}: {d}")
            if ref_ids is not None and ids is not None and not np.array_equal(ref_ids, ids):
                raise ValueError(f"IDs mismatch across feature dirs for split={split}: {d}")
        feats_list.append(x)

    if not feats_list:
        raise ValueError("feature_dirs is empty")
    x_concat = np.concatenate(feats_list, axis=1) if len(feats_list) > 1 else feats_list[0]
    assert ref_y is not None
    return x_concat, ref_y, ref_ids


def _to_binary_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    y = y.astype(np.float64)
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError(f"Labels must be binary 0/1, got unique={sorted(uniq)}")
    return y.astype(np.int64)


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


def save_predictions_csv(
    path: Path,
    ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> None:
    if ids is None:
        raise ValueError("save_predictions_csv requires ids (got None).")
    if len(ids) != len(y_true) or len(y_true) != len(y_score):
        raise ValueError(f"predictions length mismatch: ids={len(ids)} y_true={len(y_true)} y_score={len(y_score)}")

    y_true_i = _to_binary_labels(y_true)
    y_score_f = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score_f >= float(threshold)).astype(np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "y_true", "y_score", "y_pred"])
        for i in range(len(ids)):
            w.writerow([_normalize_id(ids[i]), int(y_true_i[i]), float(y_score_f[i]), int(y_pred[i])])


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    yt = y_true.astype(np.int64)
    yp = y_pred.astype(np.int64)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tp, tn, fp, fn


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(float(denom)))


def _f1(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return float(2 * tp / denom)


def _acc(tp: int, tn: int, fp: int, fn: int) -> float:
    total = tp + tn + fp + fn
    if total <= 0:
        return 0.0
    return float((tp + tn) / total)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_i = _to_binary_labels(y_true)
    y_score_f = np.asarray(y_score, dtype=np.float64)

    has_both = len(np.unique(y_true_i)) == 2
    auroc = float(roc_auc_score(y_true_i, y_score_f)) if has_both else float("nan")
    auprc = float(average_precision_score(y_true_i, y_score_f)) if has_both else float("nan")

    y_pred = (y_score_f >= float(threshold)).astype(np.int64)
    tp, tn, fp, fn = _confusion(y_true_i, y_pred)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "mcc": _mcc(tp, tn, fp, fn),
        "f1": _f1(tp, tn, fp, fn),
        "acc": _acc(tp, tn, fp, fn),
        "threshold": float(threshold),
    }


def select_threshold_mcc(y_true: np.ndarray, y_score: np.ndarray, *, grid_size: int) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    best_thr = 0.5
    best_mcc = -1.0
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}

    y_true_i = _to_binary_labels(y_true)
    y_score_f = np.asarray(y_score, dtype=np.float64)

    for thr in thresholds:
        y_pred = (y_score_f >= thr).astype(np.int64)
        tp, tn, fp, fn = _confusion(y_true_i, y_pred)
        mcc = _mcc(tp, tn, fp, fn)
        f1 = _f1(tp, tn, fp, fn)
        if (mcc > best_mcc) or (mcc == best_mcc and f1 > best_f1):
            best_mcc = mcc
            best_f1 = f1
            best_thr = float(thr)

    best_metrics = compute_metrics(y_true_i, y_score_f, best_thr)
    return best_thr, best_metrics


@dataclass(frozen=True)
class LogRegResult:
    best_c: float
    best_threshold: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    test_ci95: Dict[str, Tuple[float, float]]


def run_protocol_a_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    c_grid: Sequence[float],
    threshold_grid_size: int,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> Tuple[LogRegResult, List[Dict[str, float]]]:
    y_train_i = _to_binary_labels(y_train)
    y_val_i = _to_binary_labels(y_val)
    y_test_i = _to_binary_labels(y_test)

    best_c = float(c_grid[0])
    best_thr = 0.5
    best_val_mcc = -1.0
    best_val_metrics: Dict[str, float] = {}
    val_rows: List[Dict[str, float]] = []

    for c in c_grid:
        clf = LogisticRegression(
            C=float(c),
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            n_jobs=1,
        )
        clf.fit(x_train, y_train_i)
        val_score = clf.predict_proba(x_val)[:, 1]
        thr, metrics = select_threshold_mcc(y_val_i, val_score, grid_size=int(threshold_grid_size))
        val_rows.append({"C": float(c), **metrics})
        if float(metrics["mcc"]) > best_val_mcc:
            best_val_mcc = float(metrics["mcc"])
            best_c = float(c)
            best_thr = float(thr)
            best_val_metrics = dict(metrics)

    # retrain on train+val with best_c
    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train_i, y_val_i], axis=0)
    clf_all = LogisticRegression(C=best_c, penalty="l2", solver="liblinear", max_iter=1000, n_jobs=1)
    clf_all.fit(x_all, y_all)
    test_score = clf_all.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(y_test_i, test_score, best_thr)
    test_metrics["best_C"] = float(best_c)

    rng = np.random.default_rng(int(bootstrap_seed))
    n_test = int(len(y_test_i))
    auroc_list: List[float] = []
    auprc_list: List[float] = []
    mcc_list: List[float] = []
    f1_list: List[float] = []
    acc_list: List[float] = []

    for _ in range(int(bootstrap_n)):
        idx = rng.integers(0, n_test, size=n_test)
        m = compute_metrics(y_test_i[idx], test_score[idx], best_thr)
        auroc_list.append(float(m["auroc"]))
        auprc_list.append(float(m["auprc"]))
        mcc_list.append(float(m["mcc"]))
        f1_list.append(float(m["f1"]))
        acc_list.append(float(m["acc"]))

    def _ci(values: Sequence[float]) -> Tuple[float, float]:
        lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5]).tolist()
        return float(lo), float(hi)

    test_ci95 = {
        "auroc": _ci(auroc_list),
        "auprc": _ci(auprc_list),
        "mcc": _ci(mcc_list),
        "f1": _ci(f1_list),
        "acc": _ci(acc_list),
    }

    return (
        LogRegResult(
            best_c=float(best_c),
            best_threshold=float(best_thr),
            val_metrics=best_val_metrics,
            test_metrics=test_metrics,
            test_ci95=test_ci95,
        ),
        val_rows,
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


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(frozen=True)
class MlpSeedResult:
    seed: int
    best_epoch: int
    val_auroc: float
    best_threshold: float
    test_metrics: Dict[str, float]


def _batch_scores(model: nn.Module, x: torch.Tensor, *, batch_size: int, device: torch.device) -> np.ndarray:
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    scores: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device)).squeeze(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            scores.append(prob)
    return np.concatenate(scores, axis=0)


def train_eval_mlp_one_seed(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val_np: np.ndarray,
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
    device: torch.device,
    threshold_grid_size: int,
    save_predictions_dir: Optional[Path] = None,
    train_ids: Optional[np.ndarray] = None,
    val_ids: Optional[np.ndarray] = None,
    test_ids: Optional[np.ndarray] = None,
    y_train_np: Optional[np.ndarray] = None,
) -> MlpSeedResult:
    _set_seed(int(seed))
    model = build_mlp(int(x_train.shape[1]), hidden_dims=hidden_dims, dropout=float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)

    best_epoch = 0
    best_val_auroc = float("-inf")
    best_state: Dict[str, torch.Tensor] = {}
    patience_left = int(patience)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_score = _batch_scores(model, x_val, batch_size=int(batch_size), device=device)
        val_metrics = compute_metrics(y_val_np, val_score, 0.5)
        val_auroc = float(val_metrics["auroc"])

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state, strict=True)

    val_score = _batch_scores(model, x_val, batch_size=int(batch_size), device=device)
    best_thr, _ = select_threshold_mcc(y_val_np, val_score, grid_size=int(threshold_grid_size))

    test_score = _batch_scores(model, x_test, batch_size=int(batch_size), device=device)
    test_metrics = compute_metrics(y_test_np, test_score, best_thr)
    if save_predictions_dir is not None:
        if train_ids is None or val_ids is None or test_ids is None:
            raise ValueError("save_predictions_dir provided but ids are missing.")
        if y_train_np is None:
            raise ValueError("save_predictions_dir provided but y_train_np is missing.")
        train_score = _batch_scores(model, x_train, batch_size=int(batch_size), device=device)
        save_predictions_csv(
            save_predictions_dir / "predictions_train.csv",
            train_ids,
            y_train_np,
            train_score,
            threshold=float(best_thr),
        )
        save_predictions_csv(
            save_predictions_dir / "predictions_val.csv",
            val_ids,
            y_val_np,
            val_score,
            threshold=float(best_thr),
        )
        save_predictions_csv(
            save_predictions_dir / "predictions_test.csv",
            test_ids,
            y_test_np,
            test_score,
            threshold=float(best_thr),
        )
    return MlpSeedResult(
        seed=int(seed),
        best_epoch=int(best_epoch),
        val_auroc=float(best_val_auroc),
        best_threshold=float(best_thr),
        test_metrics=test_metrics,
    )


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float("nan")
    if arr.size <= 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1))


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    feature_dirs = [feature_dir] + [Path(p) for p in (args.feature_dirs or [])]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_ids = _load_concat_npz(feature_dirs, "train")
    x_val, y_val, val_ids = _load_concat_npz(feature_dirs, "val")
    x_test, y_test, test_ids = _load_concat_npz(feature_dirs, "test")

    y_train_i = _to_binary_labels(y_train)
    y_val_i = _to_binary_labels(y_val)
    y_test_i = _to_binary_labels(y_test)

    if args.save_predictions:
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            if ids is None:
                raise ValueError(
                    f"--save_predictions requires ids in {split_name}_features.npz, "
                    f"but ids is missing for split={split_name}."
                )

    x_train_s, mean, std = standardize_fit(x_train)
    x_val_s = standardize_apply(x_val, mean, std)
    x_test_s = standardize_apply(x_test, mean, std)

    # Protocol A
    logreg_dir = out_root / "logreg_tune"
    logreg_dir.mkdir(parents=True, exist_ok=True)
    logreg_result, val_rows = run_protocol_a_logreg(
        x_train_s,
        y_train_i,
        x_val_s,
        y_val_i,
        x_test_s,
        y_test_i,
        c_grid=args.c_grid,
        threshold_grid_size=int(args.threshold_grid_size),
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    if args.save_predictions:
        best_c = float(logreg_result.best_c)
        best_thr = float(logreg_result.best_threshold)
        x_all = np.concatenate([x_train_s, x_val_s], axis=0)
        y_all = np.concatenate([y_train_i, y_val_i], axis=0)
        clf_all = LogisticRegression(C=best_c, penalty="l2", solver="liblinear", max_iter=1000, n_jobs=1)
        clf_all.fit(x_all, y_all)
        train_score = clf_all.predict_proba(x_train_s)[:, 1]
        val_score = clf_all.predict_proba(x_val_s)[:, 1]
        test_score = clf_all.predict_proba(x_test_s)[:, 1]
        save_predictions_csv(logreg_dir / "predictions_train.csv", train_ids, y_train_i, train_score, threshold=best_thr)
        save_predictions_csv(logreg_dir / "predictions_val.csv", val_ids, y_val_i, val_score, threshold=best_thr)
        save_predictions_csv(logreg_dir / "predictions_test.csv", test_ids, y_test_i, test_score, threshold=best_thr)
    with (logreg_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_C": logreg_result.best_c,
                "best_threshold": logreg_result.best_threshold,
                "val_metrics": logreg_result.val_metrics,
                "test_metrics": logreg_result.test_metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    with (logreg_dir / "bootstrap_ci.json").open("w", encoding="utf-8") as f:
        json.dump(logreg_result.test_ci95, f, indent=2, sort_keys=True)
    with (logreg_dir / "val_table.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["C", "auroc", "auprc", "mcc", "f1", "acc", "threshold"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in val_rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    # Protocol B
    device = torch.device(str(args.device))
    mlp_dir = out_root / "mlp_small"
    mlp_dir.mkdir(parents=True, exist_ok=True)

    x_train_t = torch.from_numpy(x_train_s.astype(np.float32))
    y_train_t = torch.from_numpy(y_train_i.astype(np.float32))
    x_val_t = torch.from_numpy(x_val_s.astype(np.float32))
    x_test_t = torch.from_numpy(x_test_s.astype(np.float32))

    seed_results: List[Dict[str, float]] = []
    for seed in args.mlp_seeds:
        pred_dir = None
        if args.save_predictions:
            pred_dir = mlp_dir / f"seed_{int(seed)}"
        r = train_eval_mlp_one_seed(
            x_train=x_train_t,
            y_train=y_train_t,
            x_val=x_val_t,
            y_val_np=y_val_i,
            x_test=x_test_t,
            y_test_np=y_test_i,
            hidden_dims=args.mlp_hidden_dims,
            dropout=float(args.mlp_dropout),
            lr=float(args.mlp_lr),
            weight_decay=float(args.mlp_weight_decay),
            epochs=int(args.mlp_epochs),
            batch_size=int(args.mlp_batch_size),
            patience=int(args.mlp_patience),
            seed=int(seed),
            device=device,
            threshold_grid_size=int(args.threshold_grid_size),
            save_predictions_dir=pred_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            y_train_np=y_train_i,
        )
        seed_results.append(
            {
                "seed": float(r.seed),
                "best_epoch": float(r.best_epoch),
                "val_auroc": float(r.val_auroc),
                **{f"test_{k}": float(v) for k, v in r.test_metrics.items()},
            }
        )

    metrics = {}
    for k in ["auroc", "auprc", "mcc", "f1", "acc"]:
        vals = [float(row[f"test_{k}"]) for row in seed_results]
        mean, std = _mean_std(vals)
        metrics[k] = {"mean": mean, "std": std}

    with (mlp_dir / "metrics_all_seeds.json").open("w", encoding="utf-8") as f:
        json.dump(seed_results, f, indent=2, sort_keys=True)
    with (mlp_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # convenience: top-level summary
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol_A_logreg": {
                    "best_C": logreg_result.best_c,
                    "best_threshold": logreg_result.best_threshold,
                    "test_metrics": logreg_result.test_metrics,
                    "test_ci95": logreg_result.test_ci95,
                },
                "protocol_B_mlp_small": metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"[OK] wrote results -> {out_root}")


if __name__ == "__main__":
    main()
