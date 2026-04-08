# Progress Log

## 2026-04-07 07:52:02 UTC

- Copied Phase2 best checkpoint into standardized project path:
  - `runs/phase2_best_downstream_e30/phase2_train/checkpoints/best.pt`
- Added checkpoint shortcut symlink:
  - `artifacts/checkpoints/phase2_best_downstream_best.pt`
- Added single-SMILES inference script:
  - `scripts/extract_single_smiles_feature.py`
- Verified single-SMILES feature extraction on aspirin SMILES for:
  - `molecular_proj` (2048-d)
  - `fusion_hf` (2048-d)

## 2026-04-07 07:42:24 UTC

- Added reproducible binary-label preprocessing script:
  - `scripts/prepare_binding_binary_labels.py`
- Fixed fingerprint cache root resolution to stay inside repo:
  - `data/dataset.py` (`_resolve_repo_root`)
- Added binary-label protocol document:
  - `docs/binary_label_protocol.md`
- Generated labeled downstream datasets:
  - `data/prepared/downstream/binding_binary/*`
- Updated README with binary-label preprocessing and binary downstream commands.

## 2026-04-07 03:02:13 UTC

- Created standalone reproducibility project folder: `ncAA-RepDistill`.
- Migrated core code and renamed entry scripts to unified phase-based names.
- Consolidated Phase1/Phase2/Phase3 configs with standardized naming.
- Copied core training, adaptation, downstream, and extra benchmark datasets.
- Added documentation for data assets, migration policy, and reproducibility checklist.
