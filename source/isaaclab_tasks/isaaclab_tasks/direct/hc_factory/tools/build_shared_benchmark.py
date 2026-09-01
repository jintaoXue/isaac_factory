#!/usr/bin/env python3
"""Build shared dev_tyx-derived tables and the B2-B5 tensor dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_bottleneck_data import audit_env_dir, build_report, discover_env_dirs
from bn_agg.pipeline import process_env_dir
from factory_baselines import build_factory_baseline_dataset
from factory_bn_shared.contract import (
    DERIVED_CONTRACT_VERSION,
    DERIVED_SOURCE_BRANCH,
    DERIVED_SOURCE_COMMIT,
    SHARED_DERIVED_DIR,
    SHARED_LABEL_VERSION,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--window_size", type=float, default=60.0)
    parser.add_argument("--input_windows", type=int, default=30)
    parser.add_argument("--horizon", type=float, default=180.0)
    parser.add_argument("--score_threshold", type=float, default=0.55)
    parser.add_argument("--min_event_windows", type=int, default=8)
    parser.add_argument("--occupancy_horizon_windows", type=int, default=15)
    parser.add_argument("--hot_gap_windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_accepted_episodes", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run_dirs = [path.expanduser().absolute() for path in args.run_dirs]
    pairs = discover_env_dirs(run_dirs)
    audit_rows = [audit_env_dir(run_dir, env_dir) for run_dir, env_dir in pairs]
    report = build_report(audit_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rejected = [row for row in audit_rows if not row["accepted"]]
    if rejected and args.strict:
        detail = "\n".join(
            f"  {row['env_dir']}: {'; '.join(row['errors'])}" for row in rejected
        )
        raise SystemExit(
            f"Rejected {len(rejected)}/{len(audit_rows)} episodes:\n{detail}"
        )
    accepted = [row for row in audit_rows if row["accepted"]]
    if (
        args.expected_accepted_episodes is not None
        and len(accepted) != args.expected_accepted_episodes
    ):
        raise SystemExit(
            "Accepted episode count mismatch: "
            f"expected {args.expected_accepted_episodes}, got {len(accepted)}"
        )
    if len(accepted) < 3:
        raise SystemExit(
            f"At least 3 accepted episodes are required, got {len(accepted)}"
        )

    derived_summaries = []
    for (run_dir, env_dir), audit in zip(pairs, audit_rows):
        if not audit["accepted"]:
            continue
        derived_dir = run_dir / SHARED_DERIVED_DIR / env_dir.relative_to(run_dir)
        print(f"[shared bn_agg] {env_dir} -> {derived_dir}")
        summary = process_env_dir(
            env_dir=env_dir,
            out_dir=derived_dir,
            window_sizes=[args.window_size],
            horizon=args.horizon,
            score_threshold=args.score_threshold,
            min_event_windows=args.min_event_windows,
            closed_windows_only=True,
            label_mode="supervised",
        )
        metadata = {
            "derived_contract_version": DERIVED_CONTRACT_VERSION,
            "label_version": SHARED_LABEL_VERSION,
            "derived_source_branch": DERIVED_SOURCE_BRANCH,
            "derived_source_commit": DERIVED_SOURCE_COMMIT,
            "raw_contract_version": audit["raw_contract_version"],
            "raw_episode_sha256": audit["raw_episode_sha256"],
            "scenario_id": audit["scenario_id"],
            "window_size_s": args.window_size,
            "stride_s": args.window_size,
            "horizon_s": args.horizon,
            "score_threshold": args.score_threshold,
            "min_event_windows": args.min_event_windows,
            "closed_windows_only": True,
            "target_mode": "unsupervised_operational_occupancy",
            "episode_end_s": audit["episode_end_s"],
        }
        (derived_dir / "shared_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        derived_summaries.append(summary)

    result = build_factory_baseline_dataset(
        run_dirs=run_dirs,
        out_dir=args.out_dir,
        derived_dir_name=SHARED_DERIVED_DIR,
        window_size=args.window_size,
        stride=args.window_size,
        input_windows=args.input_windows,
        horizon=args.horizon,
        max_remain_windows=args.occupancy_horizon_windows,
        hot_min_windows=args.min_event_windows,
        hot_gap_windows=args.hot_gap_windows,
        seed=args.seed,
        repo_root=Path(__file__).resolve().parents[6],
        allowed_group_ids={
            f"{row['run_id']}:env_{int(row['env_id']):02d}:episode_{int(row['episode_id']):02d}"
            for row in accepted
        },
    )
    manifest = result["manifest"]
    summary = {
        "raw_audit": report,
        "derived_episodes": derived_summaries,
        "dataset": {
            key: manifest[key]
            for key in (
                "dataset_contract",
                "dataset_version",
                "label_version",
                "total_samples",
                "event_positive_samples",
                "event_positive_rate",
                "sample_counts",
                "episode_counts",
            )
        },
    }
    (args.out_dir / "shared_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["dataset"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
