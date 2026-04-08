#!/usr/bin/env python3
"""
Compute rdkit2d_v1 (30D) descriptors for a CSV and write an augmented CSV.

This matches the 30D "RDKit 2D descriptors" family used in baseline experiments:
  - basic properties (10)
  - topological indices (12)
  - electronic properties (3)
  - simple graph counts (3)
  - FractionCSP3 (1)
  - MolMR (1)

Output columns (30):
  MolWt, MolLogP, TPSA, NumRotatableBonds, NumHDonors, NumHAcceptors,
  NumHeteroatoms, NumAromaticRings, NumSaturatedRings, NumAliphaticRings,
  BertzCT, Chi0, Chi1, Chi0n, Chi1n, Chi2n, Chi3n, Chi4n,
  HallKierAlpha, Kappa1, Kappa2, Kappa3,
  NumValenceElectrons, MaxPartialCharge, MinPartialCharge,
  NumAtoms, NumBonds, RingCount,
  FractionCSP3, MolMR

Notes:
  - Invalid SMILES -> all-NaN for these 30 columns (masked out by dataloader).
  - Non-finite descriptor values (nan/inf) are converted to NaN (masked out).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem  # type: ignore
from rdkit import RDLogger  # type: ignore
from rdkit.Chem import Descriptors  # type: ignore


DESCRIPTOR_COLUMNS: List[str] = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "BertzCT",
    "Chi0",
    "Chi1",
    "Chi0n",
    "Chi1n",
    "Chi2n",
    "Chi3n",
    "Chi4n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "NumValenceElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "NumAtoms",
    "NumBonds",
    "RingCount",
    "FractionCSP3",
    "MolMR",
]

_WORKER_FNS: Optional[List[Callable[[Chem.Mol], float]]] = None
_NAN_ROW: List[float] = [float("nan")] * len(DESCRIPTOR_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute rdkit2d_v1 (30D) for a CSV.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV path (relative to repo root).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path (relative to repo root).",
    )
    parser.add_argument(
        "--smiles_column",
        type=str,
        default="canonical_smiles",
        help="Preferred SMILES column; will fallback to SMILES/smiles if missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if exists.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=500,
        help="Print progress every N rows.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for RDKit descriptor computation (0=auto).",
    )
    parser.add_argument(
        "--mp_chunksize",
        type=int,
        default=200,
        help="Chunk size for multiprocessing imap (larger reduces overhead).",
    )
    return parser.parse_args()


def _pick_smiles_column(df: pd.DataFrame, preferred: str) -> str:
    candidates = [preferred, "SMILES", "smiles", "canonical_smiles"]
    for col in candidates:
        if col and col in df.columns:
            return col
    raise KeyError(f"SMILES column not found. Tried: {candidates}")


def _sanitize(value: Any) -> float:
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


def _descriptor_fns() -> Dict[str, Callable[[Chem.Mol], float]]:
    return {
        "MolWt": lambda m: _sanitize(Descriptors.MolWt(m)),
        "MolLogP": lambda m: _sanitize(Descriptors.MolLogP(m)),
        "TPSA": lambda m: _sanitize(Descriptors.TPSA(m)),
        "NumRotatableBonds": lambda m: _sanitize(Descriptors.NumRotatableBonds(m)),
        "NumHDonors": lambda m: _sanitize(Descriptors.NumHDonors(m)),
        "NumHAcceptors": lambda m: _sanitize(Descriptors.NumHAcceptors(m)),
        "NumHeteroatoms": lambda m: _sanitize(Descriptors.NumHeteroatoms(m)),
        "NumAromaticRings": lambda m: _sanitize(Descriptors.NumAromaticRings(m)),
        "NumSaturatedRings": lambda m: _sanitize(Descriptors.NumSaturatedRings(m)),
        "NumAliphaticRings": lambda m: _sanitize(Descriptors.NumAliphaticRings(m)),
        "BertzCT": lambda m: _sanitize(Descriptors.BertzCT(m)),
        "Chi0": lambda m: _sanitize(Descriptors.Chi0(m)),
        "Chi1": lambda m: _sanitize(Descriptors.Chi1(m)),
        "Chi0n": lambda m: _sanitize(Descriptors.Chi0n(m)),
        "Chi1n": lambda m: _sanitize(Descriptors.Chi1n(m)),
        "Chi2n": lambda m: _sanitize(Descriptors.Chi2n(m)),
        "Chi3n": lambda m: _sanitize(Descriptors.Chi3n(m)),
        "Chi4n": lambda m: _sanitize(Descriptors.Chi4n(m)),
        "HallKierAlpha": lambda m: _sanitize(Descriptors.HallKierAlpha(m)),
        "Kappa1": lambda m: _sanitize(Descriptors.Kappa1(m)),
        "Kappa2": lambda m: _sanitize(Descriptors.Kappa2(m)),
        "Kappa3": lambda m: _sanitize(Descriptors.Kappa3(m)),
        "NumValenceElectrons": lambda m: _sanitize(Descriptors.NumValenceElectrons(m)),
        "MaxPartialCharge": lambda m: _sanitize(Descriptors.MaxPartialCharge(m)),
        "MinPartialCharge": lambda m: _sanitize(Descriptors.MinPartialCharge(m)),
        "NumAtoms": lambda m: _sanitize(m.GetNumAtoms()),
        "NumBonds": lambda m: _sanitize(m.GetNumBonds()),
        "RingCount": lambda m: _sanitize(Descriptors.RingCount(m)),
        "FractionCSP3": lambda m: _sanitize(Descriptors.FractionCSP3(m)),
        "MolMR": lambda m: _sanitize(Descriptors.MolMR(m)),
    }


def _compute_one(smiles: str, fns: Dict[str, Callable[[Chem.Mol], float]]) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    out: Dict[str, float] = {}
    for k in DESCRIPTOR_COLUMNS:
        fn = fns.get(k)
        if fn is None:
            raise KeyError(f"Descriptor function missing: {k}")
        try:
            out[k] = fn(mol)
        except Exception:  # noqa: BLE001
            out[k] = float("nan")
    return out


def _nan_row() -> Dict[str, float]:
    return {k: float("nan") for k in DESCRIPTOR_COLUMNS}


def _init_worker() -> None:
    global _WORKER_FNS  # noqa: PLW0603
    RDLogger.DisableLog("rdApp.*")
    fns = _descriptor_fns()
    _WORKER_FNS = [fns[k] for k in DESCRIPTOR_COLUMNS]


def _compute_one_values(smiles: str) -> List[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _NAN_ROW

    fns = _WORKER_FNS
    if fns is None:
        fns = [_descriptor_fns()[k] for k in DESCRIPTOR_COLUMNS]

    out: List[float] = []
    for fn in fns:
        try:
            out.append(fn(mol))
        except Exception:  # noqa: BLE001
            out.append(float("nan"))
    return out


def _resolve_num_workers(n: int) -> int:
    if n <= 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, int(n))


def _resolve_mp_chunksize(n: int) -> int:
    return max(1, int(n))


def main() -> None:
    args = parse_args()

    RDLogger.DisableLog("rdApp.*")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite to replace)")

    df = pd.read_csv(input_path)
    smiles_col = _pick_smiles_column(df, str(args.smiles_column))

    total = len(df)
    ok = 0
    failed = 0

    smiles_list = df[smiles_col].astype(str).tolist()
    num_workers = _resolve_num_workers(int(args.num_workers))
    mp_chunksize = _resolve_mp_chunksize(int(args.mp_chunksize))
    if num_workers > 1:
        print(f"[rdkit2d_v1] multiprocessing enabled: num_workers={num_workers}, mp_chunksize={mp_chunksize}")

        ctx = mp.get_context("fork")
        rows_arr: List[List[float]] = []
        with ctx.Pool(processes=num_workers, initializer=_init_worker) as pool:
            for i, vals in enumerate(pool.imap(_compute_one_values, smiles_list, chunksize=mp_chunksize)):
                if args.progress_every > 0 and i > 0 and i % int(args.progress_every) == 0:
                    print(f"[rdkit2d_v1] processed {i}/{total} ...")
                if all(np.isnan(v) for v in vals):
                    failed += 1
                else:
                    ok += 1
                rows_arr.append(vals)
        desc_df = pd.DataFrame(np.asarray(rows_arr, dtype=np.float32), columns=DESCRIPTOR_COLUMNS)
    else:
        print("[rdkit2d_v1] multiprocessing disabled: num_workers=1")
        fns = _descriptor_fns()
        rows: List[Dict[str, float]] = []
        for i, s in enumerate(smiles_list):
            if args.progress_every > 0 and i > 0 and i % int(args.progress_every) == 0:
                print(f"[rdkit2d_v1] processed {i}/{total} ...")
            try:
                out = _compute_one(s, fns)
            except Exception:  # noqa: BLE001
                out = None
            if out is None:
                failed += 1
                rows.append(_nan_row())
            else:
                ok += 1
                rows.append(out)

        desc_df = pd.DataFrame(rows, columns=DESCRIPTOR_COLUMNS)
    for k in DESCRIPTOR_COLUMNS:
        df[k] = desc_df[k].astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("")
    print("[rdkit2d_v1] Done")
    print(f"  input : {input_path}")
    print(f"  output: {output_path}")
    print(f"  rows  : {total}")
    print(f"  ok    : {ok}")
    print(f"  failed: {failed}")
    for k in DESCRIPTOR_COLUMNS:
        miss = float(pd.isna(df[k]).mean()) * 100.0
        print(f"  {k}: missing={miss:.2f}%")


if __name__ == "__main__":
    main()
