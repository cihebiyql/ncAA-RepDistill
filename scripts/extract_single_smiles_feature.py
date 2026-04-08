#!/usr/bin/env python3
"""
Extract one feature vector from a single SMILES string.

Example:
  python -u scripts/extract_single_smiles_feature.py \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --feature_type molecular_proj \
    --output_npy features/single_smiles/aspirin_molecular_proj.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract feature vector for one SMILES")
    p.add_argument(
        "--config",
        type=str,
        default="config/phase2/phase2_best_downstream.yaml",
        help="Phase2 config path",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="runs/phase2_best_downstream_e30/phase2_train/checkpoints/best.pt",
        help="Checkpoint path (.pt)",
    )
    p.add_argument("--smiles", type=str, required=True, help="Input SMILES string")
    p.add_argument(
        "--feature_type",
        type=str,
        default="molecular_proj",
        choices=["smiles_repr", "smiles_mean_pool", "protein_proj", "molecular_proj", "fusion_hf"],
        help="Feature key in model outputs",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    p.add_argument("--use_ema", action="store_true", help="Use ema_state_dict if present in checkpoint")
    p.add_argument(
        "--print_first_n",
        type=int,
        default=16,
        help="Print first N dimensions for quick check",
    )
    p.add_argument("--output_npy", type=str, default=None, help="Optional output .npy path")
    p.add_argument("--output_json", type=str, default=None, help="Optional output metadata .json path")
    return p.parse_args()


def load_model_and_vocab(config_path: Path, checkpoint_path: Path, device: torch.device, use_ema: bool):
    import sys

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from model.single_input_dual_teacher import SingleInputDualTeacher  # noqa: WPS433
    from data.vocab_manager import VocabManager  # noqa: WPS433

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    smiles_cfg = cfg["student"]["smiles_encoder"]
    proj_cfg = cfg["student"]["projection"]
    graph_attention_cfg = cfg.get("student", {}).get("graph_attention") or {}
    contrastive_proj_cfg = cfg.get("student", {}).get("contrastive_proj") or {}
    fusion_cfg = cfg.get("fusion") or {}
    if not isinstance(fusion_cfg, dict):
        fusion_cfg = {}

    vocab_path = cfg["data"]["paths"]["vocab_smiles"]
    vocab_mgr = VocabManager(vocab_path)

    model = SingleInputDualTeacher(
        smiles_vocab_size=int(smiles_cfg["vocab_size"]),
        smiles_embed_dim=int(smiles_cfg["embed_dim"]),
        smiles_num_layers=int(smiles_cfg["num_layers"]),
        smiles_num_heads=int(smiles_cfg["num_heads"]),
        smiles_max_length=int(smiles_cfg["max_length"]),
        esm2_dim=int(proj_cfg["esm2_dim"]),
        chemberta_dim=int(proj_cfg["chemberta_dim"]),
        graph_attention=graph_attention_cfg,
        contrastive_proj_cfg=contrastive_proj_cfg,
        fusion_cfg=fusion_cfg,
        use_smiles_mlm=bool(cfg["student"].get("use_smiles_mlm", True)),
        dropout=float(smiles_cfg.get("dropout", 0.1)),
        use_gradient_checkpointing=False,
    ).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict: Dict[str, torch.Tensor] = ckpt.get("model_state_dict", ckpt)
    if use_ema and "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
        print("[INFO] using ema_state_dict")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    max_length = int(smiles_cfg["max_length"])
    return model, vocab_mgr, max_length, graph_attention_cfg


def build_model_inputs(
    smiles: str,
    *,
    vocab_mgr,
    max_length: int,
    graph_attention_cfg: Dict,
):
    import sys

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from data.dataset import SingleInputDataset  # noqa: WPS433

    with tempfile.TemporaryDirectory(prefix="single_smiles_") as td:
        tmp_csv = Path(td) / "input.csv"
        pd.DataFrame({"canonical_smiles": [smiles]}).to_csv(tmp_csv, index=False)

        dataset = SingleInputDataset(
            data_path=str(tmp_csv),
            smiles_vocab_manager=vocab_mgr,
            smiles_max_length=max_length,
            apply_smiles_masking=False,
            mask_prob=0.0,
            random_smiles_config=None,
            consistency_config=None,
            graph_attention_config=graph_attention_cfg,
            label_column=None,
            normalization_config=None,
            split_name="single_smiles_infer",
        )
        item = dataset[0]

    input_ids = item["smiles_input_ids"].unsqueeze(0)  # [1, L]
    attn_mask = item["smiles_attention_mask"].unsqueeze(0)  # [1, L]
    graph_dist = item.get("graph_distances")
    if graph_dist is not None:
        graph_dist = graph_dist.unsqueeze(0)  # [1, L, L]
    return input_ids, attn_mask, graph_dist


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model, vocab_mgr, max_length, graph_attention_cfg = load_model_and_vocab(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=bool(args.use_ema),
    )
    input_ids, attn_mask, graph_dist = build_model_inputs(
        args.smiles,
        vocab_mgr=vocab_mgr,
        max_length=max_length,
        graph_attention_cfg=graph_attention_cfg,
    )
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    if graph_dist is not None:
        graph_dist = graph_dist.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            attn_mask,
            graph_distances=graph_dist,
            return_all=False,
        )
    if args.feature_type not in outputs:
        raise KeyError(f"feature_type={args.feature_type} not found in model outputs: {list(outputs.keys())}")

    feat = outputs[args.feature_type].detach().float().cpu().numpy()[0]
    print(f"[OK] feature_type={args.feature_type}, dim={feat.shape[0]}")
    print(f"[OK] first_{args.print_first_n}={feat[: args.print_first_n].tolist()}")

    if args.output_npy is not None:
        out_npy = Path(args.output_npy)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, feat)
        print(f"[OK] saved npy: {out_npy}")

    if args.output_json is not None:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "smiles": args.smiles,
            "feature_type": args.feature_type,
            "dim": int(feat.shape[0]),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "first_values": feat[: args.print_first_n].tolist(),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[OK] saved json: {out_json}")


if __name__ == "__main__":
    main()
