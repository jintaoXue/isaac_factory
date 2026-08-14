"""Order / job cycle-time KPIs from job_trace."""

from __future__ import annotations

import statistics
from collections import defaultdict

from .io_util import _f

def build_job_kpis(
    job_rows: list[dict[str, str]],
    run_id: str,
    env_id: int,
    episode_id: int | None = None,
) -> tuple[list[dict], dict]:
    """Per-job start/complete/cycle from job_trace + order-level throughput KPIs.

    Definitions (logic seconds, logic_dt=1 → same as time_step):
      - start: first ``job_selected`` for this job_id
      - complete: ``stage_complete`` (product enters progress["finished"]);
        fallback: last ``process_end`` on paint_rust_proof if stage_complete missing
      - cycle_time_s: complete - start (flow / sojourn time of one pipe)
      - order_makespan_s: max(complete) among completed jobs (from episode t=0)
    """
    by_job: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in job_rows:
        jid = r.get("job_id", "")
        if jid == "":
            continue
        by_job[jid].append(r)

    kpi_rows: list[dict] = []
    for jid in sorted(by_job.keys(), key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x)):
        evs = by_job[jid]
        product_type = next((e.get("product_type") or "" for e in evs if e.get("product_type")), "")
        starts = [
            _f(e.get("logic_time_s"), _f(e.get("time_step")))
            for e in evs
            if e.get("event") == "job_selected"
        ]
        completes = [
            _f(e.get("logic_time_s"), _f(e.get("time_step")))
            for e in evs
            if e.get("event") == "stage_complete"
        ]
        paint_ends = [
            _f(e.get("logic_time_s"), _f(e.get("time_step")))
            for e in evs
            if e.get("event") == "process_end" and e.get("task") == "paint_rust_proof"
        ]
        start_s = min(starts) if starts else None
        if completes:
            complete_s = min(completes)
            complete_source = "stage_complete"
        elif paint_ends:
            complete_s = min(paint_ends)
            complete_source = "paint_process_end"
        else:
            complete_s = None
            complete_source = ""

        cycle = None
        if start_s is not None and complete_s is not None:
            cycle = complete_s - start_s

        kpi_rows.append(
            {
                "run_id": run_id,
                "env_id": env_id,
                "episode_id": "" if episode_id is None else episode_id,
                "job_id": jid,
                "product_type": product_type,
                "start_s": "" if start_s is None else round(start_s, 3),
                "complete_s": "" if complete_s is None else round(complete_s, 3),
                "cycle_time_s": "" if cycle is None else round(cycle, 3),
                "completed": 1 if complete_s is not None else 0,
                "complete_source": complete_source,
            }
        )

    completed = [r for r in kpi_rows if r["completed"] == 1 and r["cycle_time_s"] != ""]
    cycles = [float(r["cycle_time_s"]) for r in completed]
    complete_times = [float(r["complete_s"]) for r in completed]
    start_times = [float(r["start_s"]) for r in completed if r["start_s"] != ""]

    order_summary = {
        "n_jobs": len(kpi_rows),
        "n_completed": len(completed),
        "n_incomplete": len(kpi_rows) - len(completed),
        "order_makespan_s": round(max(complete_times), 3) if complete_times else None,
        "first_job_start_s": round(min(start_times), 3) if start_times else None,
        "last_job_complete_s": round(max(complete_times), 3) if complete_times else None,
        "mean_cycle_time_s": round(statistics.mean(cycles), 3) if cycles else None,
        "median_cycle_time_s": round(statistics.median(cycles), 3) if cycles else None,
        "std_cycle_time_s": round(statistics.pstdev(cycles), 3) if len(cycles) >= 2 else (0.0 if cycles else None),
        "min_cycle_time_s": round(min(cycles), 3) if cycles else None,
        "max_cycle_time_s": round(max(cycles), 3) if cycles else None,
        "throughput_jobs_per_hour": (
            round(len(completed) / (max(complete_times) / 3600.0), 4)
            if complete_times and max(complete_times) > 0
            else None
        ),
    }
    return kpi_rows, order_summary
