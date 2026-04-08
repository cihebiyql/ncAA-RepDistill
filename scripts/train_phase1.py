#!/usr/bin/env python3
"""
Phase 1 主训练脚本（Graph-Augmented + Consistency + Distill）。

用法:
    python scripts/train_phase1.py --config config/phase1/phase1_mainline.yaml
"""

import argparse
import os
import sys
import signal
from pathlib import Path
from types import FrameType
from typing import Optional

import torch
import yaml
import torch.multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.trainer import DualTeacherTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.resume:
        config.setdefault("training", {})
        config["training"]["resume_from"] = args.resume

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

    sharing_strategy = os.environ.get("MP_SHARING_STRATEGY")
    if sharing_strategy:
        try:
            mp.set_sharing_strategy(sharing_strategy)
            print(f"[MP] sharing_strategy={sharing_strategy}")
        except Exception as e:  # noqa: BLE001
            print(f"[MP] set_sharing_strategy failed: {e}")

    trainer = DualTeacherTrainer(config)

    def _handle_sigterm(_signum: int, _frame: Optional[FrameType]) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

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
    except Exception as e:  # noqa: BLE001
        print(f"\n\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        print("\nSaving checkpoint...")
        trainer.save_checkpoint("error")
        print("Checkpoint saved.")
        raise


if __name__ == "__main__":
    main()
