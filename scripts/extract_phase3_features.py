#!/usr/bin/env python3
"""
Phase 3 feature extraction script (frozen encoder).

Features:
- Load Phase 2 student model (SingleInputDualTeacher)
- Freeze encoder for inference only
- Extract features for ncaa_cpp_{train,val,test}.csv (default), or custom split CSVs:
  - smiles_repr: CLS token embedding [N, D]
  - smiles_mean_pool: masked mean pooling [N, D]
  - protein_proj: projection to ESM2 space [N, E]
  - molecular_proj: projection to molecular teacher space [N, M]
  - fusion_hf: fused representation (pretrain fusion) [N, M]
- Save features + labels to NPZ

Example:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/extract_phase3_features.py \\
    --config config/phase2/phase2_mainline.yaml \\
    --checkpoint runs/phase2_mainline/phase2_train/checkpoints/best.pt \\
    --output_dir features/phase3_mainline_molecular_proj \\
    --feature_type molecular_proj \\
    --data_dir data/raw/downstream \\
    --batch_size 64 \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 feature extraction (frozen encoder)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Phase 2 config path (for vocab + max_length)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Phase 2 checkpoint path (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["smiles_repr", "smiles_mean_pool", "protein_proj", "molecular_proj", "fusion_hf"],
        default="molecular_proj",
        help="Feature type: smiles_repr / smiles_mean_pool / protein_proj / molecular_proj / fusion_hf",
    )
    parser.add_argument(
        "--long_smiles_strategy",
        type=str,
        choices=["truncate", "chunk_mean", "seq_segment_mean"],
        default="truncate",
        help=(
            "How to handle SMILES longer than smiles_max_length. "
            "'truncate' keeps current behavior. "
            "'chunk_mean' splits long SMILES into windows and mean-pools window embeddings. "
            "'seq_segment_mean' uses sequence segmentation (RDKit MolFromSequence per segment) and pools segment embeddings."
        ),
    )
    parser.add_argument(
        "--sequence_col",
        type=str,
        default="sequence",
        help="Sequence column name (used only when --long_smiles_strategy=seq_segment_mean).",
    )
    parser.add_argument(
        "--sequence_segment_len",
        type=int,
        default=30,
        help="Segment length (AA residues) for seq_segment_mean.",
    )
    parser.add_argument(
        "--sequence_segment_stride",
        type=int,
        default=30,
        help="Stride (AA residues) for seq_segment_mean (default non-overlap).",
    )
    parser.add_argument(
        "--sequence_segment_weight",
        type=str,
        choices=["uniform", "segment_len"],
        default="segment_len",
        help="Weighting scheme when pooling segment embeddings for seq_segment_mean.",
    )
    parser.add_argument(
        "--chunk_payload",
        type=int,
        default=0,
        help="Override chunk payload length (characters) for chunk_mean; default uses max_length-2.",
    )
    parser.add_argument(
        "--chunk_stride",
        type=int,
        default=0,
        help="Override chunk stride length (characters) for chunk_mean; default uses payload (no overlap).",
    )
    parser.add_argument(
        "--chunk_pool_weight",
        type=str,
        choices=["uniform", "window_len"],
        default="uniform",
        help="Weighting scheme when pooling window embeddings for chunk_mean.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/downstream",
        help="Directory containing ncaa_cpp_{train,val,test}.csv (used when --{train,val,test}_csv are not provided)",
    )
    parser.add_argument("--train_csv", type=str, default=None, help="Override train split CSV path")
    parser.add_argument("--val_csv", type=str, default=None, help="Override val split CSV path")
    parser.add_argument("--test_csv", type=str, default=None, help="Override test split CSV path")
    parser.add_argument("--label_col", type=str, default="Permeability", help="Label column name in split CSV")
    parser.add_argument(
        "--id_col",
        type=str,
        default="CycPeptMPDB_ID",
        help="ID column name in split CSV (if missing, will fallback to np.arange(N))",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default=None,
        help="Override SMILES column name (default prefers canonical_smiles when present)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device (e.g. cuda:0)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use ema_state_dict from checkpoint if available",
    )
    return parser.parse_args()


def load_model_and_vocab(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool = False,
):
    """Load SingleInputDualTeacher model and SMILES vocab."""
    from pathlib import Path as _Path
    import sys as _sys

    project_root = _Path(__file__).parent.parent
    _sys.path.insert(0, str(project_root))

    from model.single_input_dual_teacher import SingleInputDualTeacher  # noqa: WPS433
    from data.vocab_manager import VocabManager  # noqa: WPS433

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    smiles_cfg = cfg["student"]["smiles_encoder"]
    proj_cfg = cfg["student"]["projection"]
    vocab_path = cfg["data"]["paths"]["vocab_smiles"]
    graph_attention_cfg = cfg.get("student", {}).get("graph_attention") or {}
    contrastive_proj_cfg = cfg.get("student", {}).get("contrastive_proj") or {}
    fusion_cfg = cfg.get("fusion") or {}
    if not isinstance(fusion_cfg, dict):
        fusion_cfg = {}

    vocab_mgr = VocabManager(vocab_path)

    model = SingleInputDualTeacher(
        smiles_vocab_size=int(smiles_cfg["vocab_size"]),
        smiles_embed_dim=smiles_cfg["embed_dim"],
        smiles_num_layers=smiles_cfg["num_layers"],
        smiles_num_heads=smiles_cfg["num_heads"],
        smiles_max_length=smiles_cfg["max_length"],
        esm2_dim=proj_cfg["esm2_dim"],
        chemberta_dim=proj_cfg["chemberta_dim"],
        graph_attention=graph_attention_cfg,
        contrastive_proj_cfg=contrastive_proj_cfg,
        fusion_cfg=fusion_cfg,
        use_smiles_mlm=bool(cfg["student"].get("use_smiles_mlm", True)),
        dropout=float(smiles_cfg.get("dropout", 0.1)),
        use_gradient_checkpointing=False,
    )
    model.to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if use_ema and "ema_state_dict" in ckpt:
        print("[INFO] Using ema_state_dict from checkpoint")
        state_dict = ckpt["ema_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    return model, vocab_mgr, int(smiles_cfg["max_length"]), graph_attention_cfg


def build_loader_for_split(
    data_path: Path,
    smiles_vocab_manager,
    smiles_max_length: int,
    batch_size: int,
    num_workers: int,
    split_name: str,
    graph_attention_cfg: dict,
    label_col: str,
    smiles_col: Optional[str],
):
    from data.dataloader import collate_and_trim_smiles_batch  # noqa: WPS433
    from data.dataset import SingleInputDataset  # noqa: WPS433

    dataset = SingleInputDataset(
        data_path=str(data_path),
        smiles_vocab_manager=smiles_vocab_manager,
        smiles_max_length=smiles_max_length,
        apply_smiles_masking=False,
        mask_prob=0.0,
        random_smiles_config=None,
        consistency_config=None,
        graph_attention_config=graph_attention_cfg,
        label_column=str(label_col),
        normalization_config=None,
        split_name=split_name,
    )

    if smiles_col is not None:
        if smiles_col not in dataset.df.columns:
            raise KeyError(f"smiles_col='{smiles_col}' not found in dataset columns: {data_path}")
        dataset.smiles_col = str(smiles_col)
        print(f"[INFO] Using smiles column override: {dataset.smiles_col} ({data_path.name})")
    else:
        # Benchmark protocol expects canonical_smiles when available.
        if "canonical_smiles" in dataset.df.columns:
            dataset.smiles_col = "canonical_smiles"
            print(f"[INFO] Using smiles column: canonical_smiles ({data_path.name})")
        else:
            print(f"[INFO] Using smiles column: {dataset.smiles_col} ({data_path.name})")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_and_trim_smiles_batch,
    )

    return loader, dataset


def _iter_smiles_windows(
    smiles: str,
    *,
    max_length: int,
    payload: Optional[int] = None,
    stride: Optional[int] = None,
) -> Iterable[Tuple[str, int]]:
    """
    Yield SMILES windows (as raw substrings) and their token lengths (char-level).

    Notes:
    - Tokenization is char-level in this project; char length approximates token length.
    - We only use this for long-SMILES pooling; windows are encoded with [CLS]/[SEP] later.
    """
    if max_length <= 2:
        raise ValueError(f"max_length must be > 2, got {max_length}")
    payload_len = int(payload) if payload is not None and int(payload) > 0 else (max_length - 2)
    if payload_len <= 0:
        raise ValueError(f"Invalid payload length computed from max_length={max_length}, payload={payload}")

    stride_len = int(stride) if stride is not None and int(stride) > 0 else payload_len
    if stride_len <= 0:
        raise ValueError(f"Invalid stride length: {stride}")

    s = str(smiles)
    n = len(s)
    if n <= payload_len:
        yield s, n
        return

    for start in range(0, n, stride_len):
        end = min(start + payload_len, n)
        window = s[start:end]
        yield window, len(window)


def _encode_windows(
    windows: List[str],
    *,
    vocab_mgr,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Encode windows to (input_ids, attention_mask, window_lens).

    - input_ids: [K, L]
    - attention_mask: [K, L]
    - window_lens: K lengths excluding special tokens and padding (char-level).
    """
    input_ids_list: List[List[int]] = []
    attn_list: List[List[int]] = []
    lens: List[int] = []

    for w in windows:
        ids = vocab_mgr.encode(
            list(w),
            add_special_tokens=True,
            max_length=int(max_length),
            padding=True,
            truncation=True,
        )
        attn = [1 if int(tid) != int(vocab_mgr.pad_token_id) else 0 for tid in ids]
        input_ids_list.append(ids)
        attn_list.append(attn)
        lens.append(len(w))

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attn_list, dtype=torch.long)
    return input_ids, attention_mask, lens


@torch.no_grad()
def extract_features_for_split_long_smiles(
    *,
    model: torch.nn.Module,
    dataset,
    vocab_mgr,
    device: torch.device,
    feature_type: Literal["smiles_repr", "smiles_mean_pool", "protein_proj", "molecular_proj", "fusion_hf"],
    label_col: str,
    id_col: str,
    smiles_max_length: int,
    batch_size: int,
    long_smiles_strategy: Literal["chunk_mean", "seq_segment_mean"],
    sequence_col: Optional[str] = None,
    sequence_segment_len: int = 30,
    sequence_segment_stride: int = 30,
    sequence_segment_weight: Literal["uniform", "segment_len"] = "segment_len",
    chunk_payload: int = 0,
    chunk_stride: int = 0,
    chunk_pool_weight: Literal["uniform", "window_len"] = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features for a split, pooling over windows for long SMILES.

    Implementation detail:
    - We intentionally do NOT use graph_distances in this mode, because window substrings are
      not guaranteed to be valid SMILES (RDKit parse may fail). This is primarily intended
      for very long peptide/protein-derived SMILES (e.g., aa_sol).
    """
    if long_smiles_strategy not in {"chunk_mean", "seq_segment_mean"}:
        raise ValueError(f"Unsupported long_smiles_strategy: {long_smiles_strategy}")

    df = dataset.df
    if label_col not in df.columns:
        raise KeyError(f"label_col='{label_col}' not found in dataset columns: {dataset.data_path}")

    if not getattr(dataset, "smiles_col", None):
        raise RuntimeError("dataset.smiles_col is not set; build_loader_for_split must run first.")
    smiles_col = str(dataset.smiles_col)
    if smiles_col not in df.columns:
        raise KeyError(f"smiles_col='{smiles_col}' not found in dataset columns: {dataset.data_path}")

    labels = df[label_col].to_numpy().astype(np.float32)
    if id_col and id_col in df.columns:
        ids = df[id_col].to_numpy()
    else:
        ids = np.arange(len(df))

    if long_smiles_strategy == "seq_segment_mean":
        if not sequence_col:
            raise ValueError("sequence_col must be set for long_smiles_strategy=seq_segment_mean")
        if int(sequence_segment_len) <= 0:
            raise ValueError(f"sequence_segment_len must be >0, got {sequence_segment_len}")
        if int(sequence_segment_stride) <= 0:
            raise ValueError(f"sequence_segment_stride must be >0, got {sequence_segment_stride}")
        if str(sequence_col) not in df.columns:
            raise KeyError(f"sequence_col='{sequence_col}' not found in dataset columns: {dataset.data_path}")

        # For seq_segment_mean we stream segments in mini-batches to avoid storing huge graph distance tensors.
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("seq_segment_mean requires RDKit, but rdkit import failed.") from exc

        peptide_bond = Chem.MolFromSmarts("C(=O)N")

        def _standardize_mol(mol):
            try:
                mol = rdMolStandardize.Cleanup(mol)
                mol = rdMolStandardize.FragmentParent(mol)
                mol = rdMolStandardize.ChargeParent(mol)
                mol = rdMolStandardize.Normalizer().normalize(mol)
                mol = rdMolStandardize.Reionizer().reionize(mol)
            except Exception:  # noqa: BLE001
                return mol, "MolStandardize_failed"
            return mol, ""

        def _sequence_to_smiles(seq: str) -> Tuple[str, str]:
            seq = str(seq).strip()
            if not seq:
                return "", "empty_sequence"
            mol = Chem.MolFromSequence(seq, sanitize=True)
            if mol is None:
                return "", "MolFromSequence_failed"
            mol, reason = _standardize_mol(mol)
            if reason:
                return "", reason
            if len(seq) > 1 and peptide_bond is not None and not mol.HasSubstructMatch(peptide_bond):
                return "", "no_peptide_bond_substructure"
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            except Exception:  # noqa: BLE001
                return "", "MolToSmiles_failed"
            if not smiles:
                return "", "empty_smiles"
            return smiles, ""

        smiles_encoder = getattr(model, "smiles_encoder", None)
        if smiles_encoder is None:
            raise RuntimeError("model.smiles_encoder is missing; cannot extract features.")
        graph_attention_enabled = bool(getattr(smiles_encoder, "graph_attention_enabled", False))

        dim: Optional[int] = None
        pooled_sum: Optional[np.ndarray] = None
        pooled_wsum = np.zeros((len(df),), dtype=np.float32)

        batch_smiles: List[str] = []
        batch_owner: List[int] = []
        batch_weight: List[float] = []

        def _flush_batch() -> None:
            nonlocal dim, pooled_sum
            if not batch_smiles:
                return

            input_ids, attn, _ = _encode_windows(batch_smiles, vocab_mgr=vocab_mgr, max_length=int(smiles_max_length))
            input_ids = input_ids.to(device)
            attn = attn.to(device)

            graph_distances = None
            if graph_attention_enabled:
                dist_list = []
                for s in batch_smiles:
                    # Use dataset's graph-distance implementation for valid SMILES segments.
                    dist = dataset._get_graph_distances(s)  # noqa: SLF001
                    dist_list.append(dist)
                graph_np = np.stack(dist_list, axis=0)
                graph_distances = torch.from_numpy(graph_np).to(device=device, dtype=torch.long, non_blocking=True)

            smiles_output = model.smiles_encoder(  # type: ignore[attr-defined]
                input_ids,
                attn,
                graph_distances=graph_distances,
            )
            smiles_cls = smiles_output[:, 0, :]

            if feature_type == "smiles_repr":
                feats = smiles_cls
            elif feature_type == "smiles_mean_pool":
                mask = attn.to(dtype=smiles_output.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                feats = (smiles_output * mask).sum(dim=1) / denom
            elif feature_type == "protein_proj":
                feats = model.protein_proj(smiles_cls)  # type: ignore[attr-defined]
            elif feature_type == "molecular_proj":
                feats = model.molecular_proj(smiles_cls)  # type: ignore[attr-defined]
            else:
                if not getattr(model, "fusion_enabled", False):
                    raise RuntimeError("Requested feature_type=fusion_hf but model.fusion_enabled is False.")

                protein_proj = model.protein_proj(smiles_cls)  # type: ignore[attr-defined]

                hs_source = str(getattr(model, "fusion_cfg", {}).get("hs_source", "molecular_proj")).strip().lower()
                if hs_source == "molecular_proj":
                    hs = model.molecular_proj(smiles_cls)  # type: ignore[attr-defined]
                else:
                    hs_layer = getattr(model, "fusion_smiles_to_target", None)
                    if hs_layer is None:
                        raise RuntimeError("fusion_smiles_to_target is not initialized.")
                    hs = hs_layer(smiles_cls)

                hp_layer = getattr(model, "fusion_prot_to_target", None)
                if hp_layer is None:
                    raise RuntimeError("fusion_prot_to_target is not initialized.")
                hp = hp_layer(protein_proj)
                feats = hs + hp

            feats_np = feats.detach().cpu().numpy().astype(np.float32)
            if dim is None:
                dim = int(feats_np.shape[1])
                pooled_sum = np.zeros((len(df), dim), dtype=np.float32)

            assert pooled_sum is not None
            for j, owner in enumerate(batch_owner):
                w = float(batch_weight[j])
                pooled_sum[owner] += feats_np[j] * w
                pooled_wsum[owner] += w

            batch_smiles.clear()
            batch_owner.clear()
            batch_weight.clear()

        for i, seq in enumerate(df[str(sequence_col)].tolist()):
            seq_s = str(seq).strip() if seq is not None else ""
            if not seq_s:
                continue

            seg_len = int(sequence_segment_len)
            stride = int(sequence_segment_stride)
            # Non-overlapping by default; allow overlap via stride < seg_len.
            for start in range(0, len(seq_s), stride):
                seg = seq_s[start : start + seg_len]
                if not seg:
                    continue
                smiles_seg, reason = _sequence_to_smiles(seg)
                if reason:
                    continue

                batch_smiles.append(smiles_seg)
                batch_owner.append(int(i))
                if sequence_segment_weight == "uniform":
                    batch_weight.append(1.0)
                else:
                    batch_weight.append(float(len(seg)))

                if len(batch_smiles) >= int(batch_size):
                    _flush_batch()

        _flush_batch()

        if pooled_sum is None or dim is None:
            raise RuntimeError("seq_segment_mean produced no valid segment features.")

        # Avoid divide-by-zero; if any sample has no segments, fallback to 1.0 and keep zeros.
        denom = np.clip(pooled_wsum, 1e-6, None).astype(np.float32)
        pooled = pooled_sum / denom[:, None]
        return pooled, labels, ids

    # chunk_mean fallback: Flatten all windows across samples for efficient batching.
    flat_windows: List[str] = []
    flat_window_lens: List[int] = []
    flat_owner: List[int] = []

    for i, s in enumerate(df[smiles_col].tolist()):
        if s is None or (isinstance(s, float) and np.isnan(s)):  # type: ignore[arg-type]
            raise KeyError(f"SMILES is missing at row={i} (col={smiles_col})")
        smiles = str(s)
        for w, wlen in _iter_smiles_windows(
            smiles,
            max_length=int(smiles_max_length),
            payload=int(chunk_payload) if int(chunk_payload) > 0 else None,
            stride=int(chunk_stride) if int(chunk_stride) > 0 else None,
        ):
            flat_windows.append(w)
            flat_window_lens.append(int(wlen))
            flat_owner.append(int(i))

    if not flat_windows:
        raise RuntimeError("No SMILES windows generated; check input data.")

    window_input_ids, window_attn, _ = _encode_windows(
        flat_windows,
        vocab_mgr=vocab_mgr,
        max_length=int(smiles_max_length),
    )

    smiles_encoder = getattr(model, "smiles_encoder", None)
    if smiles_encoder is None:
        raise RuntimeError("model.smiles_encoder is missing; cannot extract features.")

    graph_attention_enabled = bool(getattr(smiles_encoder, "graph_attention_enabled", False))
    neutral_graph_idx: Optional[int] = None
    if graph_attention_enabled:
        graph_bias_embed = getattr(smiles_encoder, "graph_bias_embed", None)
        if graph_bias_embed is None:
            raise RuntimeError("graph_attention is enabled but smiles_encoder.graph_bias_embed is missing.")
        with torch.no_grad():
            w = graph_bias_embed.weight.detach()
            if w.ndim != 2 or w.shape[0] < 3:
                raise RuntimeError(f"Unexpected graph_bias_embed.weight shape: {tuple(w.shape)}")
            # Choose a neutral-ish index from {1..K-1} (exclude 0 which is used for padding/unmapped pairs).
            norms = w[1:].norm(dim=1)
            neutral_graph_idx = int(norms.argmin().item()) + 1

    # Batch over windows.
    feats_windows: List[np.ndarray] = []
    n = window_input_ids.shape[0]
    for start in tqdm(range(0, n, int(batch_size)), desc=f"Extract {feature_type} (chunk_mean)"):
        end = min(start + int(batch_size), n)
        smiles_input_ids = window_input_ids[start:end].to(device)
        smiles_attention_mask = window_attn[start:end].to(device)

        graph_distances = None
        if graph_attention_enabled:
            assert neutral_graph_idx is not None
            bsz, seq_len = smiles_input_ids.shape
            graph_distances = torch.zeros((bsz, seq_len, seq_len), dtype=torch.long, device=device)
            # Fill the valid-token block with a neutral-ish distance index.
            # Padding pairs remain 0.
            for i in range(bsz):
                lv = int(smiles_attention_mask[i].sum().item())
                if lv > 0:
                    graph_distances[i, :lv, :lv] = int(neutral_graph_idx)

        smiles_output = model.smiles_encoder(  # type: ignore[attr-defined]
            smiles_input_ids,
            smiles_attention_mask,
            graph_distances=graph_distances,
        )
        smiles_cls = smiles_output[:, 0, :]

        if feature_type == "smiles_repr":
            feats = smiles_cls
        elif feature_type == "smiles_mean_pool":
            mask = smiles_attention_mask.to(dtype=smiles_output.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            feats = (smiles_output * mask).sum(dim=1) / denom
        elif feature_type == "protein_proj":
            feats = model.protein_proj(smiles_cls)  # type: ignore[attr-defined]
        elif feature_type == "molecular_proj":
            feats = model.molecular_proj(smiles_cls)  # type: ignore[attr-defined]
        else:
            if not getattr(model, "fusion_enabled", False):
                raise RuntimeError("Requested feature_type=fusion_hf but model.fusion_enabled is False.")

            protein_proj = model.protein_proj(smiles_cls)  # type: ignore[attr-defined]

            hs_source = str(getattr(model, "fusion_cfg", {}).get("hs_source", "molecular_proj")).strip().lower()
            if hs_source == "molecular_proj":
                hs = model.molecular_proj(smiles_cls)  # type: ignore[attr-defined]
            else:
                hs_layer = getattr(model, "fusion_smiles_to_target", None)
                if hs_layer is None:
                    raise RuntimeError("fusion_smiles_to_target is not initialized.")
                hs = hs_layer(smiles_cls)

            hp_layer = getattr(model, "fusion_prot_to_target", None)
            if hp_layer is None:
                raise RuntimeError("fusion_prot_to_target is not initialized.")
            hp = hp_layer(protein_proj)
            feats = hs + hp

        feats_windows.append(feats.cpu().numpy())

    window_features = np.concatenate(feats_windows, axis=0)
    dim = int(window_features.shape[1])

    # Pool windows per sample (mean).
    pooled = np.zeros((len(df), dim), dtype=np.float32)
    counts = np.zeros((len(df),), dtype=np.float32)
    for w_idx, owner in enumerate(flat_owner):
        if chunk_pool_weight == "window_len":
            w = float(flat_window_lens[w_idx])
        else:
            w = 1.0
        pooled[owner] += window_features[w_idx].astype(np.float32) * w
        counts[owner] += w
    if np.any(counts <= 0):
        raise RuntimeError("Some samples have zero windows; this should be impossible.")
    pooled = pooled / counts[:, None].astype(np.float32)

    assert pooled.shape[0] == labels.shape[0], "Feature/label count mismatch"
    return pooled, labels, ids


@torch.no_grad()
def extract_features_for_split(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset,
    device: torch.device,
    feature_type: Literal["smiles_repr", "smiles_mean_pool", "protein_proj", "molecular_proj", "fusion_hf"],
    label_col: str,
    id_col: str,
):
    """Extract features for a single split (train/val/test)."""
    features_list = []

    for batch in tqdm(dataloader, desc=f"Extract {feature_type}"):
        smiles_input_ids = batch["smiles_input_ids"].to(device)
        smiles_attention_mask = batch["smiles_attention_mask"].to(device)
        graph_distances = batch.get("graph_distances")
        if graph_distances is not None:
            graph_distances = graph_distances.to(device)

        smiles_output = model.smiles_encoder(  # type: ignore[attr-defined]
            smiles_input_ids,
            smiles_attention_mask,
            graph_distances=graph_distances,
        )
        smiles_cls = smiles_output[:, 0, :]

        if feature_type == "smiles_repr":
            feats = smiles_cls
        elif feature_type == "smiles_mean_pool":
            mask = smiles_attention_mask.to(dtype=smiles_output.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            feats = (smiles_output * mask).sum(dim=1) / denom
        elif feature_type == "protein_proj":
            feats = model.protein_proj(smiles_cls)  # type: ignore[attr-defined]
        elif feature_type == "molecular_proj":
            feats = model.molecular_proj(smiles_cls)  # type: ignore[attr-defined]
        else:
            if not getattr(model, "fusion_enabled", False):
                raise RuntimeError("Requested feature_type=fusion_hf but model.fusion_enabled is False.")

            protein_proj = model.protein_proj(smiles_cls)  # type: ignore[attr-defined]

            hs_source = str(getattr(model, "fusion_cfg", {}).get("hs_source", "molecular_proj")).strip().lower()
            if hs_source == "molecular_proj":
                hs = model.molecular_proj(smiles_cls)  # type: ignore[attr-defined]
            else:
                hs_layer = getattr(model, "fusion_smiles_to_target", None)
                if hs_layer is None:
                    raise RuntimeError("fusion_smiles_to_target is not initialized.")
                hs = hs_layer(smiles_cls)

            hp_layer = getattr(model, "fusion_prot_to_target", None)
            if hp_layer is None:
                raise RuntimeError("fusion_prot_to_target is not initialized.")
            hp = hp_layer(protein_proj)
            feats = hs + hp

        features_list.append(feats.cpu().numpy())

    features = np.concatenate(features_list, axis=0)

    df = dataset.df
    if label_col not in df.columns:
        raise KeyError(f"label_col='{label_col}' not found in dataset columns: {dataset.data_path}")
    labels = df[label_col].to_numpy().astype(np.float32)

    if id_col and id_col in df.columns:
        ids = df[id_col].to_numpy()
    else:
        ids = np.arange(len(df))

    assert features.shape[0] == labels.shape[0], "Feature/label count mismatch"
    return features, labels, ids


def save_npz(output_dir: Path, split: str, features: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{split}_features.npz", features=features, labels=labels, ids=ids)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv
    use_explicit_splits = any(p is not None for p in (train_csv, val_csv, test_csv))
    if use_explicit_splits and not all(p is not None for p in (train_csv, val_csv, test_csv)):
        raise ValueError("If providing explicit split paths, you must set --train_csv --val_csv --test_csv together.")

    label_col = str(args.label_col)
    id_col = str(args.id_col)
    smiles_col = args.smiles_col

    device = torch.device(args.device)

    model, vocab_mgr, smiles_max_length, graph_attention_cfg = load_model_and_vocab(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=args.use_ema,
    )

    if use_explicit_splits:
        split_paths = {
            "train": Path(str(train_csv)),
            "val": Path(str(val_csv)),
            "test": Path(str(test_csv)),
        }
    else:
        split_paths = {split: data_dir / f"ncaa_cpp_{split}.csv" for split in ("train", "val", "test")}

    for split in ["train", "val", "test"]:
        split_path = split_paths[split]
        if not split_path.exists():
            print(f"[WARN] split not found, skip: {split_path}")
            continue

        loader, dataset = build_loader_for_split(
            data_path=split_path,
            smiles_vocab_manager=vocab_mgr,
            smiles_max_length=smiles_max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_name=f"phase3_{split}",
            graph_attention_cfg=graph_attention_cfg,
            label_col=label_col,
            smiles_col=smiles_col,
        )

        if args.long_smiles_strategy == "truncate":
            features, labels, ids = extract_features_for_split(
                model=model,
                dataloader=loader,
                dataset=dataset,
                device=device,
                feature_type=args.feature_type,
                label_col=label_col,
                id_col=id_col,
            )
        else:
            features, labels, ids = extract_features_for_split_long_smiles(
                model=model,
                dataset=dataset,
                vocab_mgr=vocab_mgr,
                device=device,
                feature_type=args.feature_type,
                label_col=label_col,
                id_col=id_col,
                smiles_max_length=smiles_max_length,
                batch_size=args.batch_size,
                long_smiles_strategy=args.long_smiles_strategy,
                sequence_col=args.sequence_col,
                sequence_segment_len=args.sequence_segment_len,
                sequence_segment_stride=args.sequence_segment_stride,
                sequence_segment_weight=args.sequence_segment_weight,
                chunk_payload=args.chunk_payload,
                chunk_stride=args.chunk_stride,
                chunk_pool_weight=args.chunk_pool_weight,
            )
        save_npz(output_dir, split, features, labels, ids)
        print(f"[INFO] Saved {split} features to: {output_dir}")


if __name__ == "__main__":
    main()
