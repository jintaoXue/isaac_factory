#!/usr/bin/env python3
"""Audit strict tyx-v0.3 raw episodes before canonical aggregation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from canonical_factory_bn.contract import (
    CANONICAL_CONTRACT_VERSION,
    RAW_CONTRACT_VERSION,
    audit_raw_episode,
)


AUDIT_VERSION = "canonical_raw_audit_v1"


def discover_env_dirs(run_dirs: Iterable[Path]) -> list[tuple[Path, Path]]:
    discovered = []
    seen = set()
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).resolve()
        for env_dir in sorted(run_dir.glob("episode_*/env_*")):
            key = str(env_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            discovered.append((run_dir, env_dir.resolve()))
    return discovered


def audit_env_dir(run_dir: Path, env_dir: Path) -> dict[str, Any]:
    row = audit_raw_episode(env_dir)
    return {
        **row,
        "run_dir": str(Path(run_dir).resolve()),
        "lifecycle_event": "PROVEN_COMPLETE" if row["accepted"] else "REJECTED",
        "trainable": row["accepted"],
    }


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario_rows[str(row["scenario_id"])].append(row)

    scenarios = {}
    for scenario_id, items in sorted(scenario_rows.items()):
        events = [event for item in items for event in item["runtime_events"]]
        scenarios[scenario_id] = {
            "attempted": len(items),
            "accepted": sum(bool(item["accepted"]) for item in items),
            "rejected": sum(not bool(item["accepted"]) for item in items),
            "runtime_event_count": len(events),
            "runtime_event_types": dict(
                sorted(Counter(str(event["type"]) for event in events).items())
            ),
            "runtime_event_targets": dict(
                sorted(Counter(str(event["target"]) for event in events).items())
            ),
        }

    attempted = len(rows)
    accepted = sum(bool(row["accepted"]) for row in rows)
    return {
        "audit_version": AUDIT_VERSION,
        "raw_contract_version": RAW_CONTRACT_VERSION,
        "canonical_contract_version": CANONICAL_CONTRACT_VERSION,
        "status": "passed" if attempted > 0 and accepted == attempted else "failed",
        "attempted_episodes": attempted,
        "accepted_episodes": accepted,
        "rejected_episodes": attempted - accepted,
        "scenarios": scenarios,
        "episodes": rows,
    }


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "env_id",
        "episode_id",
        "scenario_id",
        "collector_version",
        "expected_jobs",
        "completed_jobs",
        "episode_end_step",
        "resource_count",
        "runtime_event_count",
        "accepted",
        "errors",
        "warnings",
        "env_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "errors": json.dumps(row["errors"], ensure_ascii=False),
                    "warnings": json.dumps(row["warnings"], ensure_ascii=False),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any episode is rejected.",
    )
    args = parser.parse_args()

    pairs = discover_env_dirs(args.run_dirs)
    rows = [audit_env_dir(run_dir, env_dir) for run_dir, env_dir in pairs]
    report = build_report(rows)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_episode_csv(out_dir / "episode_quality.csv", rows)

    print(
        f"status={report['status']} attempted={report['attempted_episodes']} "
        f"trainable={report['accepted_episodes']} "
        f"rejected={report['rejected_episodes']}"
    )
    print(f"outputs={out_dir}")
    if args.strict and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
