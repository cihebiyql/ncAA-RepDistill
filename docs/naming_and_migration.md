# Naming And Migration

Timestamp: 2026-04-07 03:02:13 UTC

## Unified Naming Rules

- Keep only stage-based names in user-facing entry points:
  - `phase1`, `phase2`, `phase3`
- Script names:
  - `train_phase1.py`
  - `train_phase2.py`
  - `extract_phase3_features.py`
  - `train_phase3_heads.py`
  - `evaluate_regression.py`
  - `evaluate_binary.py`
- Config names:
  - `config/phase1/phase1_mainline.yaml`
  - `config/phase2/phase2_mainline.yaml`
  - `config/phase2/phase2_auxdrop01_probe.yaml`
  - `config/phase2/phase2_best_downstream.yaml`

## Migration Summary

- Core runtime code moved into one project root:
  - `model/`, `training/`, `data/`, `utils/`, `scripts/`, `config/`
- Legacy script names were replaced by standardized phase names.
- Data paths in configs were rewritten to repository-relative paths.
