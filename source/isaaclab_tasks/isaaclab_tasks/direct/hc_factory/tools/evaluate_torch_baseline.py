#!/usr/bin/env python3
"""Evaluate a trained B3-B5 PyTorch baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factory_baselines import evaluate_torch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    metrics = evaluate_torch_checkpoint(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split_name=args.split,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
