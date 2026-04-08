# Binary Label Protocol (aa_binding / ncaa_binding)

Timestamp: 2026-04-07 04:32:38 UTC

This project builds binary labels from `affinity_measure` + `affinity` with a fixed, reproducible rule.

## Input assumptions

- `affinity_measure` is a string such as:
  - `kd=5um`
  - `ki<1nm`
  - `ic50>500um`
- `affinity` is the numeric p-scale value consistent with concentration:
  - `p = -log10(M)`

Supported metrics by default: `kd`, `ki`, `ic50`.

## Labeling rule

Default thresholds:
- Positive (`binding_label=1`): `p >= 6.3`  (affinity <= 0.5 uM)
- Negative (`binding_label=0`): `p <= 5.3`  (affinity >= 5 uM)
- Ambiguous (gray zone): `5.3 < p < 6.3`

Operator handling in `affinity_measure`:
- `=` or `~`: use parsed `p` directly.
- `<` or `<=`: parsed `p` is a lower bound of true `p`; only assign positive if bound already >= 6.3.
- `>` or `>=`: parsed `p` is an upper bound of true `p`; only assign negative if bound already <= 5.3.

Ambiguous rows are dropped in `*_binary.csv` by default.

## Reproducible command

```bash
python -u scripts/prepare_binding_binary_labels.py \
  --input_dir data/raw/downstream \
  --output_dir data/prepared/downstream/binding_binary \
  --datasets aa_binding ncaa_binding \
  --splits train val test \
  --positive_threshold 6.3 \
  --negative_threshold 5.3
```

## Outputs

For each dataset/split:
- `*_annotated.csv`: all rows with parse fields and `binding_label` (nullable).
- `*_binary.csv`: labeled rows only (`binding_label` in `{0,1}`).

Global summary:
- `binding_label_summary.csv`
- `binding_label_summary.json`
