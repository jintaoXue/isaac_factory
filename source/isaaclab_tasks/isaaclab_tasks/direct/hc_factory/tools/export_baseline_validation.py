#!/usr/bin/env python3
"""Export completed validation results and provenance without opening test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def export_validation(model_root: Path) -> dict:
    records = []
    for path in sorted(model_root.glob("tuning/*/candidate_*/seed*/run_summary.json")):
        run = path.parent
        with path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary["status"] != "validation_completed":
            continue
        metrics_path = run / "metrics_validation.json"
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        with (run / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        if summary["baseline_id"] == "B2":
            configuration = config["config"]
            training = configuration
            metadata = config
        elif summary["baseline_id"] in {"B3", "B4", "B5"}:
            configuration = {key: config[key] for key in ("model", "loss", "training")}
            training = config["training"]
            metadata = config["metadata"]
        else:
            raise ValueError(f"Unknown baseline: {summary['baseline_id']}")
        if training["evaluate_test"]:
            raise ValueError(f"Tuning run enables test evaluation: {run}")
        records.append({
            "run": run.relative_to(model_root).as_posix(),
            "summary": {key: summary[key] for key in (
                "baseline_id", "model_name", "model_kind", "training_profile", "seed",
                "best_epoch", "epochs_trained", "checkpoint_constraint_met",
                "event_report_threshold", "elapsed_seconds", "trainable_parameter_count",
            ) if key in summary},
            "configuration": configuration,
            "provenance": {key: metadata[key] for key in (
                "git_commit", "dataset_manifest_sha256", "dataset_version",
                "dataset_contract", "label_version", "torch_version",
            ) if key in metadata},
            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "validation": metrics,
        })
    return {
        "snapshot_version": "baseline_validation_snapshot_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "test_metrics_opened": False,
        "completed_runs": len(records),
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_root.is_dir():
        raise NotADirectoryError(args.model_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = export_validation(args.model_root)
    if not result["runs"]:
        raise ValueError("No completed validation-only runs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {result['completed_runs']} completed validation runs: {args.output}")


if __name__ == "__main__":
    main()
