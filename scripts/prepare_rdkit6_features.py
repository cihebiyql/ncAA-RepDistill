#!/usr/bin/env python3
"""
为 Phase2 适配数据生成 RDKit 2D 描述符（6 个）。

目标文件：
  data/ncaa_adaptation_v2/ncaa_train_for_adaptation.csv

新增列：
  - MolLogP
  - TPSA
  - qed
  - NumHDonors
  - NumHAcceptors
  - NumRotatableBonds

说明：
  - 该脚本用于统一的 Phase2 数据准备流程。
  - 默认输入/输出仍沿用仓库数据目录（data/），不重复拷贝数据集。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from rdkit import Chem  # type: ignore
from rdkit.Chem import Crippen, Lipinski, QED, rdMolDescriptors  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RDKit 2D descriptors for Phase2 adaptation CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/ncaa_adaptation_v2/ncaa_train_for_adaptation.csv",
        help="Input CSV path (relative to repo root).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ncaa_adaptation_v2/ncaa_train_for_adaptation_rdkit6.csv",
        help="Output CSV path (relative to repo root).",
    )
    parser.add_argument(
        "--smiles_column",
        type=str,
        default="canonical_smiles",
        help="SMILES column name to use first; will fallback to SMILES if missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if exists.",
    )
    return parser.parse_args()


def _compute_one(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "qed": float(QED.qed(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite to replace)")

    df = pd.read_csv(input_path)
    if args.smiles_column in df.columns:
        smiles_col = args.smiles_column
    elif "SMILES" in df.columns:
        smiles_col = "SMILES"
    else:
        raise KeyError(f"SMILES column not found: tried '{args.smiles_column}' and 'SMILES'")

    required = [
        "MolLogP",
        "TPSA",
        "qed",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
    ]

    total = len(df)
    ok = 0
    failed = 0
    rows = []

    for i, s in enumerate(df[smiles_col].astype(str).tolist()):
        if i % 500 == 0 and i > 0:
            print(f"[RDKit] processed {i}/{total} ...")
        try:
            out = _compute_one(s)
        except Exception:  # noqa: BLE001
            out = None
        if out is None:
            failed += 1
            rows.append({k: float("nan") for k in required})
        else:
            ok += 1
            rows.append(out)

    desc_df = pd.DataFrame(rows)
    for k in required:
        df[k] = desc_df[k].astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("")
    print("[RDKit] Done")
    print(f"  input : {input_path}")
    print(f"  output: {output_path}")
    print(f"  rows  : {total}")
    print(f"  ok    : {ok}")
    print(f"  failed: {failed}")
    for k in required:
        miss = float(df[k].isna().mean()) * 100.0
        print(f"  {k}: missing={miss:.2f}%")


if __name__ == "__main__":
    main()
