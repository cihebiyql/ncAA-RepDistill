#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PHASE2_CONFIG="${PHASE2_CONFIG:-config/phase2/phase2_repro_sweep24_e0_b0.yaml}"
PHASE2_CKPT="${PHASE2_CKPT:-artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt}"

TRAIN_CSV="${TRAIN_CSV:-data/raw/downstream/ncaa_cpp_train.csv}"
VAL_CSV="${VAL_CSV:-data/raw/downstream/ncaa_cpp_val.csv}"
TEST_CSV="${TEST_CSV:-data/raw/downstream/ncaa_cpp_test.csv}"

FEATURE_DIR="${FEATURE_DIR:-features/repro_e0_b0_ncaa_cpp_smiles_repr}"
RESULT_DIR="${RESULT_DIR:-results/repro_e0_b0_ncaa_cpp_mlp_strict}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MLP_SEEDS="${MLP_SEEDS:-42 123 202 314 404 456 777 1013 1314 2024}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"

for p in "$PHASE2_CONFIG" "$PHASE2_CKPT" "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
  if [[ ! -f "$p" ]]; then
    echo "[FATAL] Missing file: $p" >&2
    exit 1
  fi
done

if [[ "$SKIP_EXTRACT" == "1" ]]; then
  echo "[1/2] SKIP_EXTRACT=1, skip feature extraction."
else
  echo "[1/2] Extracting ncaa_cpp smiles_repr features..."
  python -u scripts/extract_phase3_features.py \
    --config "$PHASE2_CONFIG" \
    --checkpoint "$PHASE2_CKPT" \
    --output_dir "$FEATURE_DIR" \
    --feature_type smiles_repr \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --test_csv "$TEST_CSV" \
    --label_col Permeability \
    --id_col CycPeptMPDB_ID \
    --smiles_col canonical_smiles \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --use_ema
fi

echo "[2/2] Running strict ncaa_cpp MLP protocol only..."
python -u scripts/evaluate_regression.py \
  --protocol mlp \
  --feature_dir "$FEATURE_DIR" \
  --output_dir "$RESULT_DIR" \
  --mlp_hidden_dims 256 \
  --mlp_dropout 0.1 \
  --mlp_lr 1e-3 \
  --mlp_weight_decay 1e-4 \
  --mlp_epochs 400 \
  --mlp_patience 40 \
  --mlp_batch_size 256 \
  --mlp_seeds $MLP_SEEDS \
  --device "$DEVICE" \
  --save_predictions

RESULT_DIR_ENV="$RESULT_DIR" python - <<'PY'
import json
import os
from pathlib import Path
metrics_path = Path(os.environ["RESULT_DIR_ENV"]) / "metrics.json"
if not metrics_path.exists():
    raise SystemExit(f"[FATAL] Missing metrics: {metrics_path}")
d = json.loads(metrics_path.read_text(encoding="utf-8"))
m = d["protocol_b_mlp_small"]["test_mean"]
s = d["protocol_b_mlp_small"]["test_std"]
print(f"[DONE] Protocol-B(MLP) R2={m['r2']:.10f} +/- {s['r2']:.10f}, Spearman={m['spearman']:.10f} +/- {s['spearman']:.10f}")
print(f"[DONE] Metrics: {metrics_path}")
PY
