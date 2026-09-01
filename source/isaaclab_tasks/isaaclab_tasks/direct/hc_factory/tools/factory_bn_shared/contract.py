"""Strict raw and shared-derived contracts for bottleneck benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RAW_COLLECTOR_VERSION = "v0.3"
RAW_CONTRACT_VERSION = "tyx_raw_v0.3"
DERIVED_CONTRACT_VERSION = "tyx_bn_agg_unsupervised_v2"
SHARED_LABEL_VERSION = "factory_ops_hot_v1"
SHARED_DERIVED_DIR = "shared_bn_agg_unsupervised_v2"
DERIVED_SOURCE_BRANCH = "dev_tyx"
DERIVED_SOURCE_COMMIT = "7b2fc02"

REQUIRED_RAW_FILES = (
    "episode_config.csv",
    "disturbance_log.csv",
    "resource_event_log.jsonl",
    "job_trace.csv",
    "buffer_event_log.csv",
    "route_transport_task.csv",
    "material_inventory_log.csv",
)

RUNTIME_DISTURBANCE_TYPES = frozenset(
    {
        "machine_failure",
        "human_unavailable",
        "transport_delay",
        "material_shortage",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _time_step(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = as_int(row.get(key))
        if value is not None:
            return value
    return None


def _expected_jobs(config: dict[str, Any]) -> int | None:
    product_order = json_dict(config.get("product_order"))
    counts = [as_int(value) for value in product_order.values()]
    if not counts or any(value is None or value < 0 for value in counts):
        return None
    return sum(value for value in counts if value is not None)


def _scenario_id(config: dict[str, Any]) -> str:
    dim = str(config.get("disturbance_dim") or "none")
    intensity = as_float(config.get("disturbance_intensity"))
    humans = sum(
        as_int(value, 0) or 0
        for value in json_dict(config.get("human_config")).values()
    )
    robots = sum(
        as_int(value, 0) or 0
        for value in json_dict(config.get("robot_config")).values()
    )
    gantries = json_dict(config.get("gantry_config")).get("active_gantry_indices", [])
    gantry_count = len(gantries) if isinstance(gantries, list) else 0
    order = json_dict(config.get("product_order"))
    order_payload = json.dumps(order, sort_keys=True, separators=(",", ":"))
    order_hash = hashlib.sha1(order_payload.encode("utf-8")).hexdigest()[:8]
    return (
        f"{dim}_i{intensity:g}_h{humans}_r{robots}"
        f"_g{gantry_count}_order{order_hash}"
    )


def paired_disturbance_intervals(
    rows: list[dict[str, Any]], logic_dt: float, episode_end_s: float
) -> list[dict[str, Any]]:
    """Parse completed v0.3 runtime intervals without treating them as labels."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        disturbance_type = str(row.get("disturbance_type") or "")
        if disturbance_type not in RUNTIME_DISTURBANCE_TYPES:
            continue
        event_id = str(row.get("disturbance_id") or "")
        if not event_id:
            raise ValueError("Runtime disturbance is missing disturbance_id")
        grouped[event_id].append(row)

    intervals = []
    for event_id, event_rows in sorted(grouped.items()):
        starts = [
            _time_step(row, "start_time_step", "actual_start_time_step")
            for row in event_rows
        ]
        starts = [value for value in starts if value is not None]
        ends = [
            _time_step(row, "end_time_step", "actual_end_time_step")
            for row in event_rows
        ]
        ends = [value for value in ends if value is not None]
        targets = {
            str(
                row.get("target_resource_id")
                or row.get("actual_target_resource_id")
                or ""
            )
            for row in event_rows
            if row.get("target_resource_id") or row.get("actual_target_resource_id")
        }
        if not starts or not ends:
            raise ValueError(f"Unpaired runtime disturbance: {event_id}")
        if len(targets) != 1:
            raise ValueError(f"Disturbance target mismatch: {event_id}")
        start_s = min(starts) * logic_dt
        end_s = max(ends) * logic_dt
        if end_s <= start_s or end_s > episode_end_s + 1e-9:
            raise ValueError(
                f"Invalid disturbance interval: {event_id}={start_s}:{end_s}"
            )
        first = event_rows[0]
        intervals.append(
            {
                "event_id": event_id,
                "start": start_s,
                "end": end_s,
                "type": str(first.get("disturbance_type") or ""),
                "target": next(iter(targets)),
            }
        )
    return intervals


def _monotonic(rows: list[dict[str, Any]], key: str) -> bool:
    values = [as_int(row.get(key)) for row in rows]
    filtered = [value for value in values if value is not None]
    return all(current >= previous for previous, current in zip(filtered, filtered[1:]))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_job_tasks(job_rows: list[dict[str, Any]]) -> list[str]:
    events: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in job_rows:
        job_id = str(row.get("job_id") or "")
        task = str(row.get("task") or "")
        event = str(row.get("event") or "")
        if job_id and task and event:
            events[(job_id, task)].add(event)
    return sorted(
        f"{job_id}:{task}"
        for (job_id, task), task_events in events.items()
        if "job_selected" in task_events
        and "departure" not in task_events
        and "stage_complete" not in task_events
    )


def _open_transport_tasks(rows: list[dict[str, Any]]) -> list[str]:
    final_status: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            final_status[task_id] = str(row.get("status") or "")
    return sorted(
        task_id for task_id, status in final_status.items() if status != "completed"
    )


def audit_raw_episode(env_dir: Path) -> dict[str, Any]:
    """Validate one tyx-v0.3 episode and prove its usable right boundary."""
    env_dir = Path(env_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    missing_files = [
        name for name in REQUIRED_RAW_FILES if not (env_dir / name).is_file()
    ]
    if missing_files:
        errors.append(f"missing_files={','.join(missing_files)}")

    config_rows = read_csv(env_dir / "episode_config.csv")
    if len(config_rows) != 1:
        errors.append(f"episode_config_count={len(config_rows)}")
    config = config_rows[0] if config_rows else {}
    if config.get("collector_version") != RAW_COLLECTOR_VERSION:
        errors.append(
            f"collector_version={config.get('collector_version')!r},"
            f" expected={RAW_COLLECTOR_VERSION!r}"
        )

    logic_dt = as_float(config.get("logic_dt"))
    if logic_dt <= 0:
        errors.append(f"invalid_logic_dt={config.get('logic_dt')!r}")
        logic_dt = 1.0

    resource_rows = read_jsonl(env_dir / "resource_event_log.jsonl")
    job_rows = read_csv(env_dir / "job_trace.csv")
    buffer_rows = read_csv(env_dir / "buffer_event_log.csv")
    transport_rows = read_csv(env_dir / "route_transport_task.csv")
    material_rows = read_csv(env_dir / "material_inventory_log.csv")
    disturbance_rows = read_csv(env_dir / "disturbance_log.csv")

    if not resource_rows:
        errors.append("resource_events_empty")
    if not job_rows:
        errors.append("job_trace_empty")
    if not buffer_rows:
        errors.append("buffer_events_empty")
    if not _monotonic(resource_rows, "time_step"):
        errors.append("resource_event_timestamps_not_monotonic")

    expected_jobs = _expected_jobs(config)
    completed_job_ids = {
        str(row.get("job_id"))
        for row in job_rows
        if row.get("job_id") not in (None, "")
        and str(row.get("event") or "") == "stage_complete"
    }
    completed_jobs = len(completed_job_ids)
    if expected_jobs is None:
        errors.append("expected_jobs_unknown")
    elif completed_jobs != expected_jobs:
        errors.append(f"completed_jobs={completed_jobs}/{expected_jobs}")

    open_job_tasks = _open_job_tasks(job_rows)
    if open_job_tasks:
        errors.append(f"open_job_tasks={','.join(open_job_tasks[:10])}")
    open_transport_tasks = _open_transport_tasks(transport_rows)
    if open_transport_tasks:
        errors.append(f"open_transport_tasks={','.join(open_transport_tasks[:10])}")

    observed_steps: list[int] = []
    for rows, keys in (
        (resource_rows, ("time_step",)),
        (job_rows, ("time_step", "departure_time_step")),
        (buffer_rows, ("time_step",)),
        (material_rows, ("time_step",)),
        (disturbance_rows, ("end_time_step", "start_time_step")),
    ):
        for row in rows:
            value = _time_step(row, *keys)
            if value is not None:
                observed_steps.append(value)
    for row in transport_rows:
        values = [
            _time_step(
                row,
                "transport_end_time_step",
                "dropoff_time_step",
                "transport_start_time_step",
                "request_time_step",
            )
        ]
        observed_steps.extend(value for value in values if value is not None)
    episode_end_step = max(observed_steps) if observed_steps else None
    if episode_end_step is None or episode_end_step <= 0:
        errors.append("episode_end_unavailable")

    disturbance_intervals = []
    if episode_end_step is not None:
        try:
            disturbance_intervals = paired_disturbance_intervals(
                disturbance_rows, logic_dt, episode_end_step * logic_dt
            )
        except ValueError as exc:
            errors.append(str(exc))

    resource_ids = sorted(
        {str(row.get("resource_id")) for row in resource_rows if row.get("resource_id")}
    )
    first_resource_step = min(
        (as_int(row.get("time_step"), 0) or 0 for row in resource_rows),
        default=None,
    )
    first_buffer_step = min(
        (as_int(row.get("time_step"), 0) or 0 for row in buffer_rows),
        default=None,
    )
    first_material_step = min(
        (as_int(row.get("time_step"), 0) or 0 for row in material_rows),
        default=None,
    )
    if first_resource_step not in (None, 0):
        warnings.append(
            f"resource_initial_state_unobserved_until={first_resource_step}"
        )
    if first_buffer_step not in (None, 0):
        warnings.append(f"buffer_initial_state_unobserved_until={first_buffer_step}")
    if first_material_step not in (None, 0):
        warnings.append(
            f"material_initial_state_unobserved_until={first_material_step}"
        )

    raw_file_sha256 = {
        name: _file_sha256(env_dir / name)
        for name in REQUIRED_RAW_FILES
        if (env_dir / name).is_file()
    }
    episode_digest = hashlib.sha256()
    for name, digest in sorted(raw_file_sha256.items()):
        episode_digest.update(f"{name}:{digest}\n".encode("utf-8"))

    return {
        "raw_contract_version": RAW_CONTRACT_VERSION,
        "derived_contract_version": DERIVED_CONTRACT_VERSION,
        "env_dir": str(env_dir),
        "run_id": config.get("run_id") or env_dir.parents[1].name,
        "env_id": as_int(config.get("env_id"), 0),
        "episode_id": as_int(config.get("episode_id"), 0),
        "collector_version": config.get("collector_version") or "unknown",
        "scenario_id": _scenario_id(config),
        "logic_dt": logic_dt,
        "expected_jobs": expected_jobs,
        "completed_jobs": completed_jobs,
        "completion_evidence": "stage_complete_unique_job_count",
        "episode_end_step": episode_end_step,
        "episode_end_s": (
            episode_end_step * logic_dt if episode_end_step is not None else None
        ),
        "episode_end_evidence": "max_observed_raw_timestamp",
        "resource_count": len(resource_ids),
        "resource_ids": resource_ids,
        "runtime_event_count": len(disturbance_intervals),
        "runtime_events": disturbance_intervals,
        "raw_file_sha256": raw_file_sha256,
        "raw_episode_sha256": episode_digest.hexdigest(),
        "first_resource_step": first_resource_step,
        "first_buffer_step": first_buffer_step,
        "first_material_step": first_material_step,
        "accepted": not errors,
        "errors": errors,
        "warnings": warnings,
    }
