#!/usr/bin/env python3
"""Train the B4 dense GCN-GRU baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factory_baselines import (
    MultiTaskLossConfig,
    TorchTrainConfig,
    train_torch_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--training_profile", default="baseline_fair_v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--lr_min", type=float, default=1.0e-6)
    parser.add_argument("--lr_schedule", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gcn_hidden", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--prediction_horizon", type=float, default=180.0)
    parser.add_argument("--hot_eval_threshold", type=float, default=0.55)
    parser.add_argument("--event_report_threshold", type=float, default=0.68)
    parser.add_argument("--report_threshold_sweep", type=float, nargs="+")
    parser.add_argument("--checkpoint_min_report_precision", type=float, default=0.80)
    parser.add_argument("--checkpoint_min_report_recall", type=float, default=0.35)
    args = parser.parse_args()
    summary = train_torch_baseline(
        model_kind="b4_gcn_gru",
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_overrides={
            "gcn_hidden": args.gcn_hidden,
            "gru_hidden": args.gru_hidden,
            "gru_layers": args.gru_layers,
            "dropout": args.dropout,
        },
        train_config=TorchTrainConfig(
            training_profile=args.training_profile,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_epochs=args.min_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_min=args.lr_min,
            lr_schedule=args.lr_schedule,
            gradient_clip_norm=args.gradient_clip_norm,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
            hot_eval_threshold=args.hot_eval_threshold,
            event_report_threshold=args.event_report_threshold,
            report_threshold_sweep=(
                tuple(args.report_threshold_sweep)
                if args.report_threshold_sweep
                else TorchTrainConfig.report_threshold_sweep
            ),
            checkpoint_min_report_precision=args.checkpoint_min_report_precision,
            checkpoint_min_report_recall=args.checkpoint_min_report_recall,
        ),
        loss_config=MultiTaskLossConfig(prediction_horizon=args.prediction_horizon),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
