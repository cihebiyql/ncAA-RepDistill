#!/usr/bin/env python3
"""
Build binary labels for binding datasets from affinity_measure + affinity.

Rule (default):
- Parse `affinity_measure` (e.g., kd=5um, ki<1nm, ic50>500um).
- Convert value+unit to molar and p-affinity: p = -log10(M).
- Use two-threshold labeling on p-affinity:
  - positive (1): p >= 6.3  (<= 0.5 uM)
  - negative (0): p <= 5.3  (>= 5 uM)
  - ambiguous: between (5.3, 6.3), dropped in *_binary.csv.
- For censored operators:
  - `<` / `<=`: treat parsed p as lower bound of true p, only assign positive if lower bound >= pos threshold.
  - `>` / `>=`: treat parsed p as upper bound of true p, only assign negative if upper bound <= neg threshold.
  - `=` / `~`: use parsed p directly with the two thresholds.

Outputs per dataset/split:
- *_annotated.csv: all rows + parse fields + nullable `binding_label`.
- *_binary.csv: labeled rows only (binding_label in {0,1}).
- binding_label_summary.json / binding_label_summary.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


MEASURE_PATTERN = re.compile(
    r"^\s*(?P<metric>[a-zA-Z0-9_]+)\s*(?P<op><=|>=|<|>|=|~)?\s*"
    r"(?P<value>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*"
    r"(?P<unit>fm|pm|nm|um|μm|µm|mm|m)?\s*$",
    re.IGNORECASE,
)

UNIT_TO_MOLAR = {
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    "um": 1e-6,
    "μm": 1e-6,
    "µm": 1e-6,
    "mm": 1e-3,
    "m": 1.0,
}

CERTAIN_OPERATORS = {"=", "~"}
UPPER_BOUND_OPERATORS = {"<", "<="}
LOWER_BOUND_OPERATORS = {">", ">="}


@dataclass(frozen=True)
class ParsedMeasure:
    metric: str
    operator: str
    value_raw: float
    unit: str
    molar: float
    p_affinity: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare binary labels for aa_binding/ncaa_binding")
    p.add_argument("--input_dir", type=str, default="data/raw/downstream", help="Input CSV directory")
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/prepared/downstream/binding_binary",
        help="Output directory for labeled CSVs",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["aa_binding", "ncaa_binding"],
        help="Dataset prefixes (expects <dataset>_<split>.csv)",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    p.add_argument("--measure_col", type=str, default="affinity_measure", help="Affinity-measure column")
    p.add_argument("--affinity_col", type=str, default="affinity", help="Numeric affinity column (p-scale)")
    p.add_argument(
        "--supported_measures",
        type=str,
        nargs="+",
        default=["kd", "ki", "ic50"],
        help="Supported metric names in affinity_measure",
    )
    p.add_argument(
        "--positive_threshold",
        type=float,
        default=6.3,
        help="Positive class threshold on p-affinity (>= threshold => label 1)",
    )
    p.add_argument(
        "--negative_threshold",
        type=float,
        default=5.3,
        help="Negative class threshold on p-affinity (<= threshold => label 0)",
    )
    p.add_argument(
        "--consistency_tolerance",
        type=float,
        default=0.05,
        help="Tolerance for |affinity - parsed_p_affinity| consistency check",
    )
    p.add_argument(
        "--keep_ambiguous",
        action="store_true",
        help="Keep ambiguous rows in *_binary.csv (default: drop ambiguous rows)",
    )
    return p.parse_args()


def parse_affinity_measure(raw_value: object) -> Optional[ParsedMeasure]:
    text = str(raw_value).strip().lower()
    text = text.replace("μ", "u").replace("µ", "u")
    m = MEASURE_PATTERN.match(text)
    if m is None:
        return None

    metric = str(m.group("metric")).lower()
    operator = str(m.group("op") or "=")
    value_raw = float(m.group("value"))
    unit = str(m.group("unit") or "m").lower()
    if unit not in UNIT_TO_MOLAR:
        return None

    molar = value_raw * UNIT_TO_MOLAR[unit]
    if molar <= 0:
        return None

    p_affinity = -math.log10(molar)
    return ParsedMeasure(
        metric=metric,
        operator=operator,
        value_raw=value_raw,
        unit=unit,
        molar=molar,
        p_affinity=p_affinity,
    )


def assign_label(
    parsed: ParsedMeasure,
    *,
    pos_thr: float,
    neg_thr: float,
    supported_measures: Sequence[str],
) -> Tuple[Optional[int], str, str]:
    if parsed.metric not in supported_measures:
        return None, "unsupported_measure", "unsupported_measure"

    p = parsed.p_affinity
    op = parsed.operator

    if op in CERTAIN_OPERATORS:
        if p >= pos_thr:
            return 1, "positive", "exact_or_approx_value_ge_pos_thr"
        if p <= neg_thr:
            return 0, "negative", "exact_or_approx_value_le_neg_thr"
        return None, "ambiguous", "exact_or_approx_value_in_gray_zone"

    if op in UPPER_BOUND_OPERATORS:
        # True p is >= parsed p.
        if p >= pos_thr:
            return 1, "positive", "upper_bound_still_ge_pos_thr"
        return None, "ambiguous", "upper_bound_not_strong_enough_for_positive"

    if op in LOWER_BOUND_OPERATORS:
        # True p is <= parsed p.
        if p <= neg_thr:
            return 0, "negative", "lower_bound_still_le_neg_thr"
        return None, "ambiguous", "lower_bound_not_strong_enough_for_negative"

    return None, "parse_error", "unknown_operator"


def process_split(
    df: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    measure_col: str,
    affinity_col: str,
    pos_thr: float,
    neg_thr: float,
    supported_measures: Sequence[str],
    consistency_tol: float,
    keep_ambiguous: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if measure_col not in df.columns:
        raise KeyError(f"{dataset}_{split}: missing measure_col '{measure_col}'")
    if affinity_col not in df.columns:
        raise KeyError(f"{dataset}_{split}: missing affinity_col '{affinity_col}'")

    affinity_from_col = pd.to_numeric(df[affinity_col], errors="coerce")

    parsed_list = [parse_affinity_measure(v) for v in df[measure_col].tolist()]

    metric_col: List[Optional[str]] = []
    op_col: List[Optional[str]] = []
    value_col: List[Optional[float]] = []
    unit_col: List[Optional[str]] = []
    molar_col: List[Optional[float]] = []
    p_from_measure_col: List[Optional[float]] = []
    parse_ok_col: List[bool] = []
    label_col: List[Optional[int]] = []
    label_status_col: List[str] = []
    label_rule_col: List[str] = []
    consistent_col: List[Optional[bool]] = []

    supported_set = {m.lower() for m in supported_measures}
    for idx, parsed in enumerate(parsed_list):
        if parsed is None:
            metric_col.append(None)
            op_col.append(None)
            value_col.append(None)
            unit_col.append(None)
            molar_col.append(None)
            p_from_measure_col.append(None)
            parse_ok_col.append(False)
            label_col.append(None)
            label_status_col.append("parse_error")
            label_rule_col.append("cannot_parse_affinity_measure")
            consistent_col.append(None)
            continue

        label, status, rule = assign_label(
            parsed,
            pos_thr=pos_thr,
            neg_thr=neg_thr,
            supported_measures=supported_set,
        )
        affinity_p = affinity_from_col.iloc[idx]
        if pd.isna(affinity_p):
            consistent = None
        else:
            consistent = abs(float(affinity_p) - parsed.p_affinity) <= consistency_tol

        metric_col.append(parsed.metric)
        op_col.append(parsed.operator)
        value_col.append(parsed.value_raw)
        unit_col.append(parsed.unit)
        molar_col.append(parsed.molar)
        p_from_measure_col.append(parsed.p_affinity)
        parse_ok_col.append(True)
        label_col.append(label)
        label_status_col.append(status)
        label_rule_col.append(rule)
        consistent_col.append(consistent)

    out = df.copy()
    out["affinity_p_from_column"] = affinity_from_col
    out["affinity_measure_metric"] = metric_col
    out["affinity_measure_operator"] = op_col
    out["affinity_measure_value"] = value_col
    out["affinity_measure_unit"] = unit_col
    out["affinity_molar_from_measure"] = molar_col
    out["affinity_p_from_measure"] = p_from_measure_col
    out["affinity_measure_parse_ok"] = parse_ok_col
    out["affinity_consistent_with_measure"] = consistent_col
    out["binding_label"] = pd.Series(label_col, dtype="Int64")
    out["binding_label_status"] = label_status_col
    out["binding_label_rule"] = label_rule_col

    if keep_ambiguous:
        binary = out.copy()
    else:
        binary = out[out["binding_label"].isin([0, 1])]

    n_total = int(len(out))
    n_parse_ok = int(out["affinity_measure_parse_ok"].sum())
    n_parse_fail = int((~out["affinity_measure_parse_ok"]).sum())
    n_supported = int(out["affinity_measure_metric"].isin(list(supported_set)).sum())
    n_unsupported = int(n_parse_ok - n_supported)
    n_consistency_checked = int(out["affinity_consistent_with_measure"].notna().sum())
    n_consistent = int(out["affinity_consistent_with_measure"].fillna(False).sum())
    n_inconsistent = int(n_consistency_checked - n_consistent)
    n_positive = int((out["binding_label"] == 1).sum())
    n_negative = int((out["binding_label"] == 0).sum())
    n_ambiguous = int((out["binding_label_status"] == "ambiguous").sum())
    n_labeled = int(n_positive + n_negative)

    summary = {
        "dataset": dataset,
        "split": split,
        "rows_total": n_total,
        "rows_parse_ok": n_parse_ok,
        "rows_parse_fail": n_parse_fail,
        "rows_supported_measure": n_supported,
        "rows_unsupported_measure": n_unsupported,
        "rows_consistency_checked": n_consistency_checked,
        "rows_consistent_with_affinity_col": n_consistent,
        "rows_inconsistent_with_affinity_col": n_inconsistent,
        "rows_positive": n_positive,
        "rows_negative": n_negative,
        "rows_ambiguous": n_ambiguous,
        "rows_labeled": n_labeled,
        "rows_binary_output": int(len(binary)),
        "labeled_rate": (n_labeled / n_total) if n_total > 0 else 0.0,
        "positive_rate_in_labeled": (n_positive / n_labeled) if n_labeled > 0 else 0.0,
        "negative_rate_in_labeled": (n_negative / n_labeled) if n_labeled > 0 else 0.0,
    }
    return out, binary, summary


def main() -> None:
    args = parse_args()
    if args.negative_threshold >= args.positive_threshold:
        raise ValueError(
            f"negative_threshold must be < positive_threshold, got "
            f"{args.negative_threshold} >= {args.positive_threshold}"
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: List[Dict[str, object]] = []
    for dataset in args.datasets:
        for split in args.splits:
            in_path = input_dir / f"{dataset}_{split}.csv"
            if not in_path.exists():
                raise FileNotFoundError(in_path)

            df = pd.read_csv(in_path)
            annotated, binary, summary = process_split(
                df,
                dataset=dataset,
                split=split,
                measure_col=args.measure_col,
                affinity_col=args.affinity_col,
                pos_thr=float(args.positive_threshold),
                neg_thr=float(args.negative_threshold),
                supported_measures=[m.lower() for m in args.supported_measures],
                consistency_tol=float(args.consistency_tolerance),
                keep_ambiguous=bool(args.keep_ambiguous),
            )
            all_summary.append(summary)

            annotated_path = output_dir / f"{dataset}_{split}_annotated.csv"
            binary_path = output_dir / f"{dataset}_{split}_binary.csv"
            annotated.to_csv(annotated_path, index=False)
            binary.to_csv(binary_path, index=False)

            print(
                f"[OK] {dataset}_{split}: total={summary['rows_total']}, "
                f"labeled={summary['rows_labeled']} (pos={summary['rows_positive']}, neg={summary['rows_negative']}), "
                f"ambiguous={summary['rows_ambiguous']}, parse_fail={summary['rows_parse_fail']}"
            )
            print(f"     -> {annotated_path}")
            print(f"     -> {binary_path}")

    meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "measure_col": args.measure_col,
        "affinity_col": args.affinity_col,
        "supported_measures": [m.lower() for m in args.supported_measures],
        "positive_threshold": float(args.positive_threshold),
        "negative_threshold": float(args.negative_threshold),
        "consistency_tolerance": float(args.consistency_tolerance),
        "keep_ambiguous_in_binary_output": bool(args.keep_ambiguous),
    }

    summary_df = pd.DataFrame(all_summary)
    summary_csv = output_dir / "binding_label_summary.csv"
    summary_json = output_dir / "binding_label_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "splits": all_summary}, f, indent=2, ensure_ascii=False)

    print(f"[OK] summary csv  -> {summary_csv}")
    print(f"[OK] summary json -> {summary_json}")


if __name__ == "__main__":
    main()
