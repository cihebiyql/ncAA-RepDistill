# Reproducibility Checklist

Timestamp: 2026-04-07 03:02:13 UTC

- [x] Unified code structure (`phase1/phase2/phase3`) under one folder
- [x] Standardized script names (no legacy experiment prefix in entrypoints)
- [x] Standardized config names and relative data paths
- [x] Included core train/adapt/downstream/extra datasets in this folder
- [x] Added fixed binary-label preprocessing for aa_binding/ncaa_binding
- [x] Included Phase2 teacher cache used by adaptation
- [x] Added large-asset linking script for Phase1 teacher caches/checkpoints
- [x] Added README with end-to-end commands
- [x] Added requirements and Git LFS policy
