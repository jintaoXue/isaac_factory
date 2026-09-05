#!/usr/bin/env python3
"""Rank validation-only baseline candidates without consulting test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key)
    return default if value is None else float(value)


def summarize_candidate(
    candidate_dir: Path,
    expected_seeds: set[int] | None = None,
) -> dict[str, Any]:
    runs = []
    for seed_dir in sorted(candidate_dir.glob("seed*")):
        summary_path = seed_dir / "run_summary.json"
        metrics_path = seed_dir / "metrics.json"
        if not summary_path.is_file() or not metrics_path.is_file():
            continue
        summary = _read_json(summary_path)
        metrics = _read_json(metrics_path)
        if summary.get("status") != "validation_completed":
            raise ValueError(f"{seed_dir}: expected validation_completed status")
        if "test" in metrics or any(key.startswith("test_") for key in summary):
            raise ValueError(f"{seed_dir}: tuning candidate contains test metrics")
        validation = metrics["validation"]
        report = validation["station_report"]
        remain = validation["remain"]
        will = validation.get("event_will", {})
        runs.append(
            {
                "seed": int(summary.get("seed", seed_dir.name.removeprefix("seed"))),
                "profile": summary.get("training_profile"),
                "constraint_met": bool(summary.get("checkpoint_constraint_met")),
                "report_precision": _value(report, "report_precision"),
                "report_recall": _value(report, "report_recall"),
                "report_f1": _value(report, "report_f1"),
                "who_f1": _value(report, "who_f1"),
                "ongoing_recall": _value(report, "report_recall_ongoing"),
                "upcoming_recall": _value(report, "report_recall_upcoming"),
                "hot_f1": _value(remain, "hot_f1"),
                "hot_ap": _value(remain, "hot_ap"),
                "will_ap": _value(will, "pr_auc", default=-1.0),
                "elapsed_seconds": _value(summary, "elapsed_seconds"),
            }
        )
    if not runs:
        raise FileNotFoundError(f"No completed validation runs under {candidate_dir}")
    observed_seeds = {int(run["seed"]) for run in runs}
    if expected_seeds is not None and observed_seeds != expected_seeds:
        raise ValueError(
            f"{candidate_dir}: expected seeds {sorted(expected_seeds)}, "
            f"received {sorted(observed_seeds)}"
        )

    aggregate = {
        "candidate": candidate_dir.name,
        "run_count": len(runs),
        "seeds": ",".join(str(run["seed"]) for run in runs),
        "all_constraints_met": all(run["constraint_met"] for run in runs),
        "constraint_rate": _mean([float(run["constraint_met"]) for run in runs]),
        "profile": runs[0]["profile"],
    }
    for key in (
        "report_precision",
        "report_recall",
        "report_f1",
        "who_f1",
        "ongoing_recall",
        "upcoming_recall",
        "hot_f1",
        "hot_ap",
        "will_ap",
        "elapsed_seconds",
    ):
        values = [float(run[key]) for run in runs]
        aggregate[f"{key}_mean"] = _mean(values)
        aggregate[f"{key}_std"] = _std(values)
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
    aggregate["report_f1_robust"] = (
        aggregate["report_f1_mean"] - aggregate["report_f1_std"]
    )
    aggregate["runs"] = runs
    return aggregate


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(candidate["all_constraints_met"]),
        float(candidate["report_f1_robust"]),
        float(candidate["report_f1_mean"]),
        float(candidate["upcoming_recall_mean"]),
        float(candidate["report_precision_mean"]),
        float(candidate["hot_f1_mean"]),
        float(candidate["will_ap_mean"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning_dir", type=Path, required=True)
    parser.add_argument("--expected_seeds", type=int, nargs="+")
    args = parser.parse_args()

    tuning_dir = args.tuning_dir.resolve()
    expected_seeds = set(args.expected_seeds) if args.expected_seeds else None
    candidates = [
        summarize_candidate(path, expected_seeds)
        for path in sorted(tuning_dir.glob("candidate_*"))
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No candidate directories under {tuning_dir}")
    candidates.sort(key=candidate_rank, reverse=True)

    csv_rows = [
        {key: value for key, value in row.items() if key != "runs"}
        for row in candidates
    ]
    with (tuning_dir / "tuning_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    selection = {
        "status": "validation_selection_completed",
        "selection_split": "validation",
        "test_evaluated": False,
        "rank_policy": [
            "all_constraints_met",
            "report_f1_mean_minus_std",
            "report_f1_mean",
            "upcoming_recall_mean",
            "report_precision_mean",
            "hot_f1_mean",
            "will_ap_mean",
        ],
        "selected_candidate": candidates[0]["candidate"],
        "selected_profile": candidates[0]["profile"],
        "candidates": candidates,
    }
    with (tuning_dir / "selection.json").open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

    print("Validation-only candidate ranking")
    for index, candidate in enumerate(candidates, 1):
        print(
            f"{index}. {candidate['candidate']} "
            f"report_f1={candidate['report_f1_mean']:.4f}"
            f"+/-{candidate['report_f1_std']:.4f} "
            f"robust={candidate['report_f1_robust']:.4f} "
            f"P={candidate['report_precision_mean']:.4f} "
            f"R={candidate['report_recall_mean']:.4f} "
            f"hot_f1={candidate['hot_f1_mean']:.4f} "
            f"upcoming_r={candidate['upcoming_recall_mean']:.4f} "
            f"feasible={int(candidate['all_constraints_met'])}"
        )
    print(f"Selected: {candidates[0]['candidate']}")
    print(f"Outputs: {tuning_dir}")


if __name__ == "__main__":
    main()
