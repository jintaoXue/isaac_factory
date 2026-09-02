#!/usr/bin/env python3
"""Train the B2 XGBoost factory bottleneck baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factory_baselines import B2XGBoostConfig, train_b2_xgboost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--near_remain_windows", type=int, default=60)
    parser.add_argument("--negative_cell_ratio", type=float, default=4.0)
    parser.add_argument("--empty_sample_negative_cells", type=int, default=32)
    parser.add_argument("--prediction_cell_chunk_size", type=int, default=65536)
    parser.add_argument("--hot_eval_threshold", type=float, default=0.55)
    parser.add_argument("--event_report_threshold", type=float, default=0.65)
    args = parser.parse_args()
    config = B2XGBoostConfig(
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        n_jobs=args.n_jobs,
        near_remain_windows=args.near_remain_windows,
        negative_cell_ratio=args.negative_cell_ratio,
        empty_sample_negative_cells=args.empty_sample_negative_cells,
        prediction_cell_chunk_size=args.prediction_cell_chunk_size,
        hot_eval_threshold=args.hot_eval_threshold,
        event_report_threshold=args.event_report_threshold,
    )
    summary = train_b2_xgboost(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
