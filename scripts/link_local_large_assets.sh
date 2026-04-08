#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PHASE1_ESM2_SRC="/root/private_data/aa_peptide/student/teacher/data/phase1_teacher_new/esm2"
PHASE1_CHEM_SRC="/root/private_data/aa_peptide/student/teacher/data/phase1_teacher_new/chemberta"
PHASE1_CKPT_SRC="${PHASE1_CKPT_SRC:-}"
PHASE2_BEST_CKPT_SRC="${PHASE2_BEST_CKPT_SRC:-}"

mkdir -p "$ROOT_DIR/data/teacher_cache/phase1" "$ROOT_DIR/artifacts/checkpoints"

if [ -d "$PHASE1_ESM2_SRC" ]; then
  rm -rf "$ROOT_DIR/data/teacher_cache/phase1/esm2"
  ln -s "$PHASE1_ESM2_SRC" "$ROOT_DIR/data/teacher_cache/phase1/esm2"
  echo "[linked] phase1 esm2 cache"
else
  echo "[skip] missing source: $PHASE1_ESM2_SRC"
fi

if [ -d "$PHASE1_CHEM_SRC" ]; then
  rm -rf "$ROOT_DIR/data/teacher_cache/phase1/chemberta"
  ln -s "$PHASE1_CHEM_SRC" "$ROOT_DIR/data/teacher_cache/phase1/chemberta"
  echo "[linked] phase1 chemberta cache"
else
  echo "[skip] missing source: $PHASE1_CHEM_SRC"
fi

if [ -n "$PHASE1_CKPT_SRC" ] && [ -f "$PHASE1_CKPT_SRC" ]; then
  ln -sf "$PHASE1_CKPT_SRC" "$ROOT_DIR/artifacts/checkpoints/phase1_mainline_epoch30.pt"
  echo "[linked] phase1 checkpoint"
else
  echo "[skip] set PHASE1_CKPT_SRC to link phase1 checkpoint"
fi

if [ -n "$PHASE2_BEST_CKPT_SRC" ] && [ -f "$PHASE2_BEST_CKPT_SRC" ]; then
  ln -sf "$PHASE2_BEST_CKPT_SRC" "$ROOT_DIR/artifacts/checkpoints/phase2_best_downstream_best.pt"
  echo "[linked] phase2 best checkpoint"
else
  echo "[skip] set PHASE2_BEST_CKPT_SRC to link phase2 checkpoint"
fi
