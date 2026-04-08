# ncAA-RepDistill

A unified and reproducible project for ncAA representation distillation with standardized naming across `phase1`, `phase2`, and `phase3`.

This repository restructures the full training/testing workflow into one clean layout for publication and GitHub sharing, without legacy experiment-name file prefixes.

## Standardized Structure

```text
ncAA-RepDistill/
  config/
    phase1/
    phase2/
    phase3/
    downstream/
  scripts/
    train_phase1.py
    train_phase2.py
    extract_phase3_features.py
    extract_single_smiles_feature.py
    train_phase3_heads.py
    evaluate_regression.py
    evaluate_binary.py
    prepare_binding_binary_labels.py
    prepare_rdkit2d_features.py
    prepare_rdkit6_features.py
    prepare_morgan_pca_features.py
    link_local_large_assets.sh
    run_phase_pipeline.sh
  model/
  training/
  data/
    raw/
    prepared/
    teacher_cache/
  artifacts/
  docs/
```

## Environment

```bash
conda create -n ncaa-repdistill python=3.10 -y
conda activate ncaa-repdistill
pip install -r requirements.txt
```

## Data And Asset Preparation

1. Use built-in datasets already placed under `data/raw` and `data/prepared`.
2. Link large local assets (Phase1 teacher caches / checkpoint):

```bash
bash scripts/link_local_large_assets.sh
```

3. If you need to regenerate pretrain descriptors:

```bash
python scripts/prepare_rdkit2d_features.py \
  --input data/raw/pretrain/uniref_lt30_c30/train_sequences.csv \
  --output data/prepared/pretrain/train_sequences_with_rdkit2d.csv \
  --smiles_column smiles --num_workers 24 --overwrite

python scripts/prepare_rdkit2d_features.py \
  --input data/raw/pretrain/uniref_lt30_c30/val_sequences.csv \
  --output data/prepared/pretrain/val_sequences_with_rdkit2d.csv \
  --smiles_column smiles --num_workers 24 --overwrite
```

4. Build binary labels for `aa_binding` / `ncaa_binding` (required before binary downstream eval):

```bash
python -u scripts/prepare_binding_binary_labels.py \
  --input_dir data/raw/downstream \
  --output_dir data/prepared/downstream/binding_binary \
  --datasets aa_binding ncaa_binding \
  --splits train val test \
  --positive_threshold 6.3 \
  --negative_threshold 5.3
```

## End-to-End Reproducibility

### Phase1

```bash
python -u scripts/train_phase1.py --config config/phase1/phase1_mainline.yaml
```

### Phase2

Mainline:

```bash
python -u scripts/train_phase2.py --config config/phase2/phase2_mainline.yaml
```

Best downstream setting:

```bash
python -u scripts/train_phase2.py --config config/phase2/phase2_best_downstream.yaml
```

### Phase3 Feature Extraction (ncaa_cpp)

```bash
python -u scripts/extract_phase3_features.py \
  --config config/phase2/phase2_best_downstream.yaml \
  --checkpoint runs/phase2_best_downstream_e30/phase2_train/checkpoints/best.pt \
  --output_dir features/phase3_best_downstream_molecular_proj \
  --feature_type molecular_proj \
  --train_csv data/raw/downstream/ncaa_cpp_train.csv \
  --val_csv data/raw/downstream/ncaa_cpp_val.csv \
  --test_csv data/raw/downstream/ncaa_cpp_test.csv
```

### Downstream Evaluation

Regression (`ncaa_cpp`, Spearman included):

```bash
python -u scripts/evaluate_regression.py \
  --feature_dir features/phase3_best_downstream_molecular_proj \
  --output_dir results/downstream/ncaa_cpp_strict
```

Run only MLP protocol (Protocol B):

```bash
python -u scripts/evaluate_regression.py \
  --protocol mlp \
  --feature_dir features/phase3_best_downstream_molecular_proj \
  --output_dir results/downstream/ncaa_cpp_strict_mlp_only
```

One-click reproduction for the strict `ncaa_cpp` MLP result tied to sweep24 `E0_B0`:

```bash
bash scripts/reproduce_ncaa_cpp_mlp_e0_b0.sh
```

If features are already extracted, you can skip extraction:

```bash
SKIP_EXTRACT=1 FEATURE_DIR=/path/to/ncaa_cpp_features bash scripts/reproduce_ncaa_cpp_mlp_e0_b0.sh
```

Bundled reproducibility assets:
- Config: `config/phase2/phase2_repro_sweep24_e0_b0.yaml`
- Checkpoint: `artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt`

Binary tasks (`aa_binding`, `ncaa_binding`):

```bash
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

python -u scripts/evaluate_binary.py \
  --feature_dir features/phase3_best_downstream_aa_binding_molecular_proj \
  --output_dir results/downstream/aa_binding_strict

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

python -u scripts/evaluate_binary.py \
  --feature_dir features/phase3_best_downstream_ncaa_binding_molecular_proj \
  --output_dir results/downstream/ncaa_binding_strict
```

Important:
- `scripts/evaluate_binary.py` requires `labels` in NPZ to be binary `0/1`.
- This repo provides a fixed binary protocol in `scripts/prepare_binding_binary_labels.py`.

### Single-SMILES Inference

```bash
python -u scripts/extract_single_smiles_feature.py \
  --config config/phase2/phase2_best_downstream.yaml \
  --checkpoint runs/phase2_best_downstream_e30/phase2_train/checkpoints/best.pt \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --feature_type molecular_proj \
  --output_npy features/single_smiles/aspirin_molecular_proj.npy \
  --output_json features/single_smiles/aspirin_molecular_proj.json
```

## Notes

- Detailed data inventory and large-asset policy: `docs/data_assets.md`
- Naming policy and migration notes: `docs/naming_and_migration.md`
- Reproducibility checklist: `docs/reproducibility_checklist.md`
- Binary-label protocol details: `docs/binary_label_protocol.md`
