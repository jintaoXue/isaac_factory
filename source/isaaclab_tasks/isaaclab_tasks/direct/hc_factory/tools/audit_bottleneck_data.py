#!/usr/bin/env python3
"""Audit raw bottleneck runs before Phase B dataset generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


AUDIT_VERSION = "bottleneck_raw_audit_v1"
COLLECTOR_VERSION = "v0.6"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def discover_env_dirs(run_dirs: Iterable[Path]) -> list[tuple[Path, Path]]:
    discovered: list[tuple[Path, Path]] = []
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).resolve()
        candidates = sorted(run_dir.glob("episode_*/env_*"))
        discovered.extend((run_dir, path) for path in candidates if path.is_dir())
    return discovered


def _expected_jobs(config: dict[str, str], override: int | None) -> int | None:
    if override is not None:
        return override
    order = _json_dict(config.get("product_order"))
    values = [_int(value) for value in order.values()]
    return sum(value for value in values if value is not None) if values else None


def _monotonic(rows: list[dict[str, Any]], field: str) -> bool:
    values = [_int(row.get(field)) for row in rows]
    present = [value for value in values if value is not None]
    return present == sorted(present)


def _audit_events(
    rows: list[dict[str, str]],
    disturbance_dim: str,
    episode_end_step: int | None,
    resource_ids: set[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    event_summaries: list[dict[str, Any]] = []
    phases = defaultdict(lambda: defaultdict(list))
    for row in rows:
        phase = str(row.get("event_phase") or "").upper()
        if phase in {"START", "END"}:
            phases[row.get("disturbance_id") or "missing_id"][phase].append(row)

    for event_id, event_phases in sorted(phases.items()):
        starts = event_phases["START"]
        ends = event_phases["END"]
        if len(starts) != 1:
            errors.append(f"event:{event_id}:start_count={len(starts)}")
        if len(ends) != 1:
            errors.append(f"event:{event_id}:end_count={len(ends)}")
        if len(starts) != 1 or len(ends) != 1:
            continue
        start_row, end_row = starts[0], ends[0]
        start = _int(start_row.get("actual_start_time_step"))
        end = _int(end_row.get("actual_end_time_step"))
        target = start_row.get("actual_target_resource_id") or ""
        end_target = end_row.get("actual_target_resource_id") or ""
        if start is None or end is None or end < start:
            errors.append(f"event:{event_id}:invalid_interval={start}:{end}")
        if episode_end_step is not None and end is not None and end > episode_end_step:
            errors.append(
                f"event:{event_id}:ends_after_episode={end}>{episode_end_step}"
            )
        if not target:
            errors.append(f"event:{event_id}:missing_actual_target")
        elif (
            disturbance_dim in {"machine", "human", "logistics"}
            and target not in resource_ids
        ):
            errors.append(f"event:{event_id}:target_not_in_resource_catalog={target}")
        if end_target != target:
            errors.append(f"event:{event_id}:target_changed={target}:{end_target}")
        event_summaries.append(
            {
                "event_id": event_id,
                "start_step": start,
                "end_step": end,
                "duration_steps": end - start
                if start is not None and end is not None
                else None,
                "target": target,
                "type": start_row.get("disturbance_type") or "",
            }
        )

    runtime_rows = [
        row
        for row in rows
        if str(row.get("event_phase") or "").upper() in {"START", "END"}
    ]
    if disturbance_dim == "none" and runtime_rows:
        errors.append("none_scenario_has_runtime_events")
    if disturbance_dim != "none" and not phases:
        errors.append("disturbed_scenario_missing_runtime_event")
    if rows and any(not row.get("event_phase") for row in rows):
        errors.append("disturbance_row_missing_event_phase")
    return errors, warnings, event_summaries


def audit_env_dir(
    run_dir: Path,
    env_dir: Path,
    expected_completed_jobs: int | None = None,
) -> dict[str, Any]:
    config_rows = _read_csv(env_dir / "episode_config.csv")
    lifecycle = _read_csv(env_dir / "episode_lifecycle.csv")
    disturbance = _read_csv(env_dir / "disturbance_log.csv")
    resource_events = _read_jsonl(env_dir / "resource_event_log.jsonl")
    config = config_rows[0] if config_rows else {}

    errors: list[str] = []
    warnings: list[str] = []
    if len(config_rows) != 1:
        errors.append(f"episode_config_count={len(config_rows)}")
    if config.get("collector_version") != COLLECTOR_VERSION:
        errors.append(
            f"collector_version={config.get('collector_version')!r},"
            f" expected={COLLECTOR_VERSION!r}"
        )

    start_rows = [
        row for row in lifecycle if str(row.get("event", "")).upper() == "START"
    ]
    end_rows = [row for row in lifecycle if str(row.get("event", "")).upper() == "END"]
    aborted_rows = [
        row for row in lifecycle if str(row.get("event", "")).upper() == "ABORTED"
    ]
    if len(start_rows) != 1:
        errors.append(f"lifecycle_start_count={len(start_rows)}")
    if len(end_rows) != 1:
        errors.append(f"lifecycle_end_count={len(end_rows)}")
    if aborted_rows:
        errors.append(
            f"episode_aborted={aborted_rows[-1].get('termination_reason') or 'unknown'}"
        )

    end_row = end_rows[-1] if end_rows else {}
    episode_end_step = _int(end_row.get("time_step"))
    expected_jobs = _expected_jobs(config, expected_completed_jobs)
    completed_jobs = _int(end_row.get("completed_jobs"), 0) or 0
    if end_rows and _int(end_row.get("production_done"), 0) != 1:
        errors.append("production_done=0")
    if expected_jobs is None:
        warnings.append("expected_completed_jobs_unknown")
    elif completed_jobs != expected_jobs:
        errors.append(f"completed_jobs={completed_jobs}/{expected_jobs}")

    resource_ids = {
        str(row.get("resource_id")) for row in resource_events if row.get("resource_id")
    }
    init_ids = {
        str(row.get("resource_id"))
        for row in resource_events
        if row.get("resource_id")
        and _int(row.get("time_step")) == 0
        and str(row.get("from_state", "")).upper() == "INIT"
    }
    missing_init = sorted(resource_ids - init_ids)
    if missing_init:
        errors.append(f"resources_missing_init={','.join(missing_init)}")
    if not resource_ids:
        errors.append("resource_events_empty")
    if not _monotonic(resource_events, "time_step"):
        errors.append("resource_event_timestamps_not_monotonic")
    if not _monotonic(lifecycle, "time_step"):
        errors.append("lifecycle_timestamps_not_monotonic")

    disturbance_dim = str(config.get("disturbance_dim") or "none")
    event_errors, event_warnings, event_summaries = _audit_events(
        disturbance, disturbance_dim, episode_end_step, resource_ids
    )
    errors.extend(event_errors)
    warnings.extend(event_warnings)

    episode_id = _int(config.get("episode_id"))
    env_id = _int(config.get("env_id"))
    return {
        "run_dir": str(run_dir),
        "env_dir": str(env_dir),
        "run_id": config.get("run_id") or run_dir.name,
        "env_id": env_id,
        "episode_id": episode_id,
        "collector_version": config.get("collector_version") or "unknown",
        "scenario_id": config.get("scenario_id") or "unknown",
        "disturbance_dim": disturbance_dim,
        "disturbance_intensity": _float(config.get("disturbance_intensity")),
        "expected_jobs": expected_jobs,
        "completed_jobs": completed_jobs,
        "episode_end_step": episode_end_step,
        "lifecycle_event": "ABORTED"
        if aborted_rows
        else ("END" if end_rows else "INCOMPLETE"),
        "resource_count": len(resource_ids),
        "runtime_event_count": len(event_summaries),
        "runtime_events": event_summaries,
        "trainable": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _scenario_key(row: dict[str, Any]) -> str:
    return f"{row['disturbance_dim']}@{row['disturbance_intensity']:g}"


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[_scenario_key(row)].append(row)

    scenarios = {}
    for scenario, items in sorted(by_scenario.items()):
        events = [event for item in items for event in item["runtime_events"]]
        targets = Counter(event["target"] for event in events)
        starts = {event["start_step"] for event in events}
        durations = {event["duration_steps"] for event in events}
        attempted = len(items)
        completed = sum(item["lifecycle_event"] == "END" for item in items)
        aborted = sum(item["lifecycle_event"] == "ABORTED" for item in items)
        scenario_warnings = []
        if attempted >= 3 and len(events) >= 3:
            if len(starts) < 2:
                scenario_warnings.append("runtime_event_start_has_no_variation")
            if len(durations) < 2:
                scenario_warnings.append("runtime_event_duration_has_no_variation")
            if len(targets) > 1 and max(targets.values()) / len(events) > 0.5:
                scenario_warnings.append("runtime_event_target_share_above_0.5")
        scenarios[scenario] = {
            "attempted_episodes": attempted,
            "trainable_episodes": sum(item["trainable"] for item in items),
            "completed_episodes": completed,
            "aborted_episodes": aborted,
            "completion_rate": completed / attempted if attempted else 0.0,
            "aborted_rate": aborted / attempted if attempted else 0.0,
            "runtime_event_count": len(events),
            "runtime_event_targets": dict(sorted(targets.items())),
            "runtime_event_start_steps": sorted(
                value for value in starts if value is not None
            ),
            "runtime_event_duration_steps": sorted(
                value for value in durations if value is not None
            ),
            "warnings": scenario_warnings,
        }

    return {
        "audit_version": AUDIT_VERSION,
        "status": "passed"
        if rows and all(row["trainable"] for row in rows)
        else "failed",
        "attempted_episodes": len(rows),
        "trainable_episodes": sum(row["trainable"] for row in rows),
        "rejected_episodes": sum(not row["trainable"] for row in rows),
        "scenarios": scenarios,
        "episodes": rows,
    }


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "env_id",
        "episode_id",
        "collector_version",
        "scenario_id",
        "disturbance_dim",
        "disturbance_intensity",
        "lifecycle_event",
        "expected_jobs",
        "completed_jobs",
        "episode_end_step",
        "resource_count",
        "runtime_event_count",
        "trainable",
        "errors",
        "warnings",
        "env_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "errors": ";".join(row["errors"]),
                    "warnings": ";".join(row["warnings"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--expected_completed_jobs", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    env_dirs = discover_env_dirs(args.run_dirs)
    if not env_dirs:
        raise SystemExit("No episode_*/env_* or env_* directories found")
    rows = [
        audit_env_dir(run_dir, env_dir, args.expected_completed_jobs)
        for run_dir, env_dir in env_dirs
    ]
    report = build_report(rows)
    out_dir = (args.out_dir or (args.run_dirs[0] / "quality_audit_v1")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_episode_csv(out_dir / "data_quality_episodes.csv", rows)
    print(
        f"status={report['status']} attempted={report['attempted_episodes']} "
        f"trainable={report['trainable_episodes']} rejected={report['rejected_episodes']}"
    )
    print(f"outputs={out_dir}")
    if args.strict and report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
