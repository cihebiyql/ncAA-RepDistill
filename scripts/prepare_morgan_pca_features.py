#!/usr/bin/env python3
"""
Compute Morgan2048 fingerprints for a CSV and append PCA128 features.

Workflow:
  - Train: fit IncrementalPCA on Morgan2048 (binary) and write output CSV with
    columns morgan_pca_000..morgan_pca_127
  - Val/Test: load PCA model and transform only

Notes:
  - Invalid SMILES -> zero vector (kept, masked by downstream if needed)
  - Uses chunked processing to limit memory
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from rdkit import Chem  # type: ignore
from rdkit import RDLogger  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit import DataStructs  # type: ignore

from sklearn.decomposition import IncrementalPCA
import joblib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Morgan2048 -> PCA128 for CSV.")
    p.add_argument("--input", required=True, help="Input CSV path.")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--smiles_column", default="smiles", help="SMILES column name (fallback to SMILES/canonical_smiles).")
    p.add_argument("--radius", type=int, default=2, help="Morgan radius.")
    p.add_argument("--n_bits", type=int, default=2048, help="Morgan bits.")
    p.add_argument("--n_components", type=int, default=128, help="PCA components.")
    p.add_argument("--chunksize", type=int, default=4096, help="CSV chunk size.")
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Fingerprint multiprocessing workers (0=auto, 1=disable).",
    )
    p.add_argument("--mp_chunksize", type=int, default=200, help="Pool.imap chunksize.")
    p.add_argument("--fit", action="store_true", help="Fit PCA on input (train only).")
    p.add_argument("--pca_model", default=None, help="Path to PCA model (joblib).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")
    return p.parse_args()


def pick_smiles_col(df: pd.DataFrame, preferred: str) -> str:
    for col in (preferred, "SMILES", "smiles", "canonical_smiles"):
        if col in df.columns:
            return col
    raise KeyError("SMILES column not found in input CSV.")


def smiles_to_fp(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_fp_packed(smiles: str, radius: int, n_bits: int) -> bytes:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return bytes(n_bits // 8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
    return DataStructs.BitVectToBinaryText(fp)


def iter_chunks(csv_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(csv_path, chunksize=chunksize)


_WORKER_RADIUS: int | None = None
_WORKER_N_BITS: int | None = None


def _init_worker(radius: int, n_bits: int) -> None:
    global _WORKER_RADIUS, _WORKER_N_BITS
    _WORKER_RADIUS = radius
    _WORKER_N_BITS = n_bits
    RDLogger.DisableLog("rdApp.*")


def _smiles_to_fp_packed_worker(smiles: str) -> bytes:
    assert _WORKER_RADIUS is not None
    assert _WORKER_N_BITS is not None
    return smiles_to_fp_packed(smiles, _WORKER_RADIUS, _WORKER_N_BITS)


def build_fp_matrix(
    df: pd.DataFrame,
    smiles_col: str,
    radius: int,
    n_bits: int,
    pool: mp.pool.Pool | None,
    mp_chunksize: int,
) -> np.ndarray:
    smiles_list = df[smiles_col].astype(str).tolist()

    if pool is None:
        fps = np.zeros((len(df), n_bits), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            fps[i] = smiles_to_fp(smi, radius, n_bits)
        return fps

    if n_bits % 8 != 0:
        raise ValueError("--n_bits must be divisible by 8 for packed conversion.")

    packed_list = list(pool.imap(_smiles_to_fp_packed_worker, smiles_list, chunksize=mp_chunksize))
    packed_bytes = b"".join(packed_list)
    packed = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(len(smiles_list), n_bits // 8)
    bits = np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32)
    return bits


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"[MorganPCA] Output exists, skip: {output_path}")
        return

    if args.fit and not args.pca_model:
        raise ValueError("--fit requires --pca_model")

    model_path = Path(args.pca_model) if args.pca_model else None

    # Determine SMILES column from the first chunk
    first_chunk = next(iter_chunks(input_path, chunksize=1))
    smiles_col = pick_smiles_col(first_chunk, args.smiles_column)

    num_workers = args.num_workers if args.num_workers != 0 else (os.cpu_count() or 1)
    num_workers = max(1, int(num_workers))

    pool: mp.pool.Pool | None = None
    if num_workers > 1:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(args.radius, args.n_bits))
        print(f"[MorganPCA] multiprocessing enabled: num_workers={num_workers}, mp_chunksize={args.mp_chunksize}")

    try:
        if args.fit:
            print(f"[MorganPCA] Fitting IncrementalPCA on {input_path}")
            ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.chunksize)
            total = 0
            for chunk in iter_chunks(input_path, chunksize=args.chunksize):
                fp_mat = build_fp_matrix(
                    chunk,
                    smiles_col,
                    args.radius,
                    args.n_bits,
                    pool=pool,
                    mp_chunksize=args.mp_chunksize,
                )
                ipca.partial_fit(fp_mat)
                total += len(chunk)
                if total % (args.chunksize * 5) == 0:
                    print(f"[MorganPCA] fitted {total} rows...")

            joblib.dump(ipca, model_path)
            meta = {
                "input": str(input_path),
                "smiles_col": smiles_col,
                "radius": args.radius,
                "n_bits": args.n_bits,
                "n_components": args.n_components,
                "explained_variance_ratio_sum": float(ipca.explained_variance_ratio_.sum()),
            }
            meta_path = model_path.with_suffix(".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[MorganPCA] Saved PCA model: {model_path}")

        if model_path is None or not model_path.exists():
            raise FileNotFoundError("PCA model not found; use --fit to create it.")

        ipca = joblib.load(model_path)
        cols = [f"morgan_pca_{i:03d}" for i in range(args.n_components)]

        print(f"[MorganPCA] Transforming and writing to {output_path}")
        first = True
        for chunk in iter_chunks(input_path, chunksize=args.chunksize):
            fp_mat = build_fp_matrix(
                chunk,
                smiles_col,
                args.radius,
                args.n_bits,
                pool=pool,
                mp_chunksize=args.mp_chunksize,
            )
            proj = ipca.transform(fp_mat).astype(np.float32)
            proj_df = pd.DataFrame(proj, columns=cols)
            out_df = pd.concat([chunk.reset_index(drop=True), proj_df], axis=1)
            out_df.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
            first = False

        print("[MorganPCA] Done")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
