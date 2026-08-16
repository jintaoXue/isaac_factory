#!/usr/bin/env python3
"""Audit tyx-v0.3 raw runs and build canonical BSTAN benchmark tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_bottleneck_data import audit_env_dir, build_report, discover_env_dirs
from bstan_baseline.dataset import build_bstan_dataset
from build_bottleneck_features import process_env_dir
from canonical_factory_bn.contract import CANONICAL_DERIVED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--window_size", type=float, default=30.0)
    parser.add_argument("--stride", type=float, default=30.0)
    parser.add_argument("--input_windows", type=int, default=4)
    parser.add_argument("--horizon", type=float, default=120.0)
    parser.add_argument("--score_threshold", type=float, default=0.50)
    parser.add_argument("--min_event_windows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of excluding rejected raw episodes.",
    )
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.run_dirs]
    pairs = discover_env_dirs(run_dirs)
    audit_rows = [audit_env_dir(run_dir, env_dir) for run_dir, env_dir in pairs]
    report = build_report(audit_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rejected = [row for row in audit_rows if not row["accepted"]]
    if rejected and args.strict:
        details = "\n".join(
            f"  {row['env_dir']}: {'; '.join(row['errors'])}" for row in rejected
        )
        raise SystemExit(
            f"Canonical build rejected {len(rejected)}/{len(audit_rows)} episodes:\n"
            f"{details}"
        )
    accepted_rows = [row for row in audit_rows if row["accepted"]]
    if len(accepted_rows) < 3:
        raise SystemExit(
            f"At least 3 accepted episodes are required, got {len(accepted_rows)}"
        )
    if rejected:
        print(f"[audit] excluding {len(rejected)} rejected episodes")

    summaries = []
    accepted_env_dirs = {row["env_dir"] for row in accepted_rows}
    for run_dir, env_dir in pairs:
        if str(env_dir) not in accepted_env_dirs:
            continue
        out_dir = run_dir / CANONICAL_DERIVED_DIR / env_dir.relative_to(run_dir)
        print(f"[canonical] {env_dir} -> {out_dir}")
        summaries.append(
            process_env_dir(
                env_dir=env_dir,
                out_dir=out_dir,
                window_sizes=[args.window_size],
                stride=args.stride,
                horizon=args.horizon,
                score_threshold=args.score_threshold,
                min_event_windows=args.min_event_windows,
            )
        )

    result = build_bstan_dataset(
        run_dirs=run_dirs,
        out_dir=args.out_dir,
        derived_dir_name=CANONICAL_DERIVED_DIR,
        window_size=args.window_size,
        stride=args.stride,
        input_windows=args.input_windows,
        horizon=args.horizon,
        seed=args.seed,
        repo_root=Path(__file__).resolve().parents[6],
        allowed_group_ids={
            (
                f"{row['run_id']}:env_{int(row['env_id']):02d}:"
                f"episode_{int(row['episode_id']):02d}"
            )
            for row in accepted_rows
        },
    )
    manifest = result["manifest"]
    summary = {
        "raw_audit": report,
        "canonical_episodes": summaries,
        "dataset": {
            "output_dir": str(args.out_dir.resolve()),
            "dataset_contract": manifest["dataset_contract"],
            "dataset_version": manifest["dataset_version"],
            "label_version": manifest["label_version"],
            "total_samples": manifest["total_samples"],
            "positive_samples": manifest["positive_samples"],
            "positive_rate": manifest["positive_rate"],
            "sample_counts": manifest["sample_counts"],
            "episode_counts": manifest["episode_counts"],
        },
    }
    (args.out_dir / "canonical_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["dataset"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
