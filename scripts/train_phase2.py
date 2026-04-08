#!/usr/bin/env python3
"""
Phase 2 training script (ncAA adaptation).

Usage:
  python -u scripts/train_phase2.py --config config/phase2/phase2_mainline.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Optional

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Training (ncAA adaptation)")
    parser.add_argument("--config", type=str, required=True, help="Path to Phase 2 config file")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume Phase 2 training from (override training.resume_from)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from training.trainer import DualTeacherTrainer  # noqa: WPS433

    print(f"Loading config from: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 70)
    print(f"Run ID      : {config['run_id']}")
    print(f"Description : {config.get('description', '')}")
    print(f"GPU         : {config['device']['gpu_id']}")
    print("=" * 70 + "\n")

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        print(f"Using GPU: {config['device']['gpu_id']}")
    else:
        print("Warning: CUDA not available, using CPU")

    if args.resume:
        print(f"\n[Override] Resuming from checkpoint: {args.resume}")
        config.setdefault("training", {})
        config["training"]["resume_from"] = args.resume

    def _handle_sigterm(_signum: int, _frame: Optional[FrameType]) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    trainer = DualTeacherTrainer(config)
    try:
        final_metrics = trainer.train()
        print("\nFinal Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
        print("Checkpoint saved.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n\nTraining failed with error: {exc}")
        import traceback

        traceback.print_exc()
        print("\nSaving checkpoint...")
        trainer.save_checkpoint("error")
        print("Checkpoint saved.")
        raise


if __name__ == "__main__":
    main()
