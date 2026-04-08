#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-all}"

run_phase1() {
  python -u scripts/train_phase1.py --config config/phase1/phase1_mainline.yaml
}

run_phase2_mainline() {
  python -u scripts/train_phase2.py --config config/phase2/phase2_mainline.yaml
}

run_phase2_best() {
  python -u scripts/train_phase2.py --config config/phase2/phase2_best_downstream.yaml
}

run_phase3_extract() {
  python -u scripts/extract_phase3_features.py \
    --config config/phase2/phase2_best_downstream.yaml \
    --checkpoint runs/phase2_best_downstream_e30/phase2_train/checkpoints/best.pt \
    --output_dir features/phase3_best_downstream_molecular_proj \
    --feature_type molecular_proj \
    --train_csv data/raw/downstream/ncaa_cpp_train.csv \
    --val_csv data/raw/downstream/ncaa_cpp_val.csv \
    --test_csv data/raw/downstream/ncaa_cpp_test.csv
}

run_prepare_binding_labels() {
  python -u scripts/prepare_binding_binary_labels.py \
    --input_dir data/raw/downstream \
    --output_dir data/prepared/downstream/binding_binary \
    --datasets aa_binding ncaa_binding \
    --splits train val test \
    --positive_threshold 6.3 \
    --negative_threshold 5.3
}

run_phase3_extract_aa_binding() {
  python -u scripts/extract_phase3_features.py \
    --config config/phase2/phase2_best_downstream.yaml \
    --checkpoint runs/phase2_best_downstream_e30/phase2_train/checkpoints/best.pt \
    --output_dir features/phase3_best_downstream_aa_binding_molecular_proj \
    --feature_type molecular_proj \
    --train_csv data/prepared/downstream/binding_binary/aa_binding_train_binary.csv \
    --val_csv data/prepared/downstream/binding_binary/aa_binding_val_binary.csv \
    --test_csv data/prepared/downstream/binding_binary/aa_binding_test_binary.csv \
    --label_col binding_label \
    --id_col complex_id
}

run_phase3_extract_ncaa_binding() {
  python -u scripts/extract_phase3_features.py \
    --config config/phase2/phase2_best_downstream.yaml \
    --checkpoint runs/phase2_best_downstream_e30/phase2_train/checkpoints/best.pt \
    --output_dir features/phase3_best_downstream_ncaa_binding_molecular_proj \
    --feature_type molecular_proj \
    --train_csv data/prepared/downstream/binding_binary/ncaa_binding_train_binary.csv \
    --val_csv data/prepared/downstream/binding_binary/ncaa_binding_val_binary.csv \
    --test_csv data/prepared/downstream/binding_binary/ncaa_binding_test_binary.csv \
    --label_col binding_label \
    --id_col complex_id
}

run_downstream() {
  python -u scripts/evaluate_regression.py \
    --feature_dir features/phase3_best_downstream_molecular_proj \
    --output_dir results/downstream/ncaa_cpp_strict
}

run_downstream_binary_aa() {
  python -u scripts/evaluate_binary.py \
    --feature_dir features/phase3_best_downstream_aa_binding_molecular_proj \
    --output_dir results/downstream/aa_binding_strict
}

run_downstream_binary_ncaa() {
  python -u scripts/evaluate_binary.py \
    --feature_dir features/phase3_best_downstream_ncaa_binding_molecular_proj \
    --output_dir results/downstream/ncaa_binding_strict
}

case "$STAGE" in
  phase1) run_phase1 ;;
  phase2_mainline) run_phase2_mainline ;;
  phase2_best) run_phase2_best ;;
  phase3) run_phase3_extract ;;
  prepare_binding_labels) run_prepare_binding_labels ;;
  phase3_aa_binding) run_phase3_extract_aa_binding ;;
  phase3_ncaa_binding) run_phase3_extract_ncaa_binding ;;
  downstream) run_downstream ;;
  downstream_binary_aa) run_downstream_binary_aa ;;
  downstream_binary_ncaa) run_downstream_binary_ncaa ;;
  all)
    run_phase1
    run_phase2_best
    run_phase3_extract
    run_downstream
    ;;
  *)
    echo "Usage: bash scripts/run_phase_pipeline.sh [phase1|phase2_mainline|phase2_best|phase3|prepare_binding_labels|phase3_aa_binding|phase3_ncaa_binding|downstream|downstream_binary_aa|downstream_binary_ncaa|all]"
    exit 1
    ;;
esac
