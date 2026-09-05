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
    parser.add_argument("--training_profile", default="baseline_fair_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=3.0)
    parser.add_argument("--reg_lambda", type=float, default=5.0)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--near_remain_windows", type=int, default=60)
    parser.add_argument("--negative_cell_ratio", type=float, default=4.0)
    parser.add_argument("--hot_scale_pos_weight", type=float, default=4.0)
    parser.add_argument("--event_will_scale_pos_weight", type=float, default=4.0)
    parser.add_argument("--empty_sample_negative_cells", type=int, default=32)
    parser.add_argument("--prediction_cell_chunk_size", type=int, default=65536)
    parser.add_argument("--hot_eval_threshold", type=float, default=0.55)
    parser.add_argument("--event_report_threshold", type=float, default=0.68)
    parser.add_argument("--report_threshold_sweep", type=float, nargs="+")
    parser.add_argument("--report_threshold_min_precision", type=float, default=0.80)
    parser.add_argument("--checkpoint_min_report_recall", type=float, default=0.35)
    parser.add_argument("--validation_only", action="store_true")
    args = parser.parse_args()
    config = B2XGBoostConfig(
        training_profile=args.training_profile,
        evaluate_test=not args.validation_only,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        n_jobs=args.n_jobs,
        near_remain_windows=args.near_remain_windows,
        negative_cell_ratio=args.negative_cell_ratio,
        hot_scale_pos_weight=args.hot_scale_pos_weight,
        event_will_scale_pos_weight=args.event_will_scale_pos_weight,
        empty_sample_negative_cells=args.empty_sample_negative_cells,
        prediction_cell_chunk_size=args.prediction_cell_chunk_size,
        hot_eval_threshold=args.hot_eval_threshold,
        event_report_threshold=args.event_report_threshold,
        report_threshold_sweep=(
            tuple(args.report_threshold_sweep)
            if args.report_threshold_sweep
            else B2XGBoostConfig.report_threshold_sweep
        ),
        report_threshold_min_precision=args.report_threshold_min_precision,
        checkpoint_min_report_recall=args.checkpoint_min_report_recall,
    )
    summary = train_b2_xgboost(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
