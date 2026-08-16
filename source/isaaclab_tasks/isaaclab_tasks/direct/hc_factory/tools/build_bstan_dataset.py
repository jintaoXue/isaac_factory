#!/usr/bin/env python3
"""Build fixed-shape BSTAN tensors from shared canonical benchmark tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bstan_baseline.dataset import build_bstan_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--derived_dir_name", default="canonical_factory_bn_v1")
    parser.add_argument("--window_size", type=float, default=30.0)
    parser.add_argument("--stride", type=float, default=30.0)
    parser.add_argument("--input_windows", type=int, default=4)
    parser.add_argument("--horizon", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[6]
    result = build_bstan_dataset(
        run_dirs=args.run_dirs,
        out_dir=args.out_dir,
        derived_dir_name=args.derived_dir_name,
        window_size=args.window_size,
        stride=args.stride,
        input_windows=args.input_windows,
        horizon=args.horizon,
        seed=args.seed,
        repo_root=repo_root,
    )
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "dataset_dir": str(args.out_dir.resolve()),
                "dataset_contract": manifest["dataset_contract"],
                "label_version": manifest["label_version"],
                "total_samples": manifest["total_samples"],
                "positive_samples": manifest["positive_samples"],
                "positive_rate": manifest["positive_rate"],
                "sample_counts": manifest["sample_counts"],
                "episode_counts": manifest["episode_counts"],
                "nodes": len(manifest["node_ids"]),
                "features": len(manifest["feature_names"]),
                "validation": manifest["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
