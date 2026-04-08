# Data Assets

Timestamp: 2026-04-07 07:42:24 UTC

## Included In Repository Folder

- `data/raw/pretrain/uniref_lt30_c30/`
  - `train_sequences.csv`
  - `val_sequences.csv`
  - `vocab_smiles.txt`
  - `vocab_sequence_std.txt`
- `data/prepared/pretrain/`
  - `train_sequences_with_rdkit2d.csv`
  - `val_sequences_with_rdkit2d.csv`
- `data/raw/adaptation/`
  - `ncaa_train_for_adaptation.csv`
  - `vocab_smiles.txt`
- `data/prepared/adaptation/`
  - `ncaa_train_for_adaptation_with_rdkit6.csv`
  - `ncaa_train_for_adaptation_with_rdkit2d.csv`
- `data/raw/phase2_split/`
  - `adapt_train.csv`
  - `adapt_val.csv`
- `data/raw/downstream/`
  - `ncaa_cpp_{train,val,test}.csv`
  - `aa_binding_{train,val,test}.csv`
  - `ncaa_binding_{train,val,test}.csv`
  - `vocab_smiles.txt`
- `data/prepared/downstream/binding_binary/`
  - `aa_binding_{train,val,test}_annotated.csv`
  - `aa_binding_{train,val,test}_binary.csv`
  - `ncaa_binding_{train,val,test}_annotated.csv`
  - `ncaa_binding_{train,val,test}_binary.csv`
  - `binding_label_summary.csv`
  - `binding_label_summary.json`
- `data/raw/extra/`
  - `ncaa_joint_{train,val,test}_invalid.csv`
  - `ncaa_xuamp_{train,val,test}.csv`
- `data/teacher_cache/phase2/geminimol_adapt/`
  - `train_features.pt`
  - `val_features.pt`
- `artifacts/pca/`
  - `morgan_pca128.joblib`
  - `morgan_pca128.meta.json`

## External / Large Assets (Linked Locally)

- Phase1 teacher caches are very large and are not copied as committed blobs:
  - `data/teacher_cache/phase1/esm2/`
  - `data/teacher_cache/phase1/chemberta/`
- Recommended local-link workflow:

```bash
bash scripts/link_local_large_assets.sh
```

## GitHub Upload Recommendation

- Enable Git LFS before first push:

```bash
git lfs install
git add .gitattributes
git add .
```
