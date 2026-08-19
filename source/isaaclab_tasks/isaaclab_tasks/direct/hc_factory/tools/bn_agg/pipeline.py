"""Orchestrate Stage-C for one env dir and the CLI."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from .constants import DEFAULT_MIN_EVENT_WINDOWS, DEFAULT_SCORE_THRESHOLD
from .features import add_bottleneck_scores, compute_window_features
from .io_util import _derived_out_dir, _discover_env_dirs, _f, _i, _read_csv, _read_jsonl, _write_csv
from .kpi import build_job_kpis
from .labels import build_labels_and_events, parse_disturbance_l2_intervals
from .timelines import build_timelines

def _max_timestamp(events: list, job_rows: list) -> float:
    times = []
    for e in events:
        times.append(_f(e.get("logic_time_s"), _f(e.get("time_step"))))
    for r in job_rows:
        times.append(_f(r.get("logic_time_s"), _f(r.get("time_step"))))
    return max(times) if times else 0.0


def process_env_dir(
    env_dir: Path,
    out_dir: Path,
    window_sizes: list[float],
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
    *,
    closed_windows_only: bool = False,
) -> dict:
    events = _read_jsonl(env_dir / "resource_event_log.jsonl")
    job_rows = _read_csv(env_dir / "job_trace.csv")
    buffer_rows = _read_csv(env_dir / "buffer_event_log.csv")
    transport_rows = _read_csv(env_dir / "route_transport_task.csv")
    material_rows = _read_csv(env_dir / "material_inventory_log.csv")
    ep_rows = _read_csv(env_dir / "episode_config.csv")
    disturbance_rows = _read_csv(env_dir / "disturbance_log.csv")

    run_id = ep_rows[0]["run_id"] if ep_rows else env_dir.parent.name
    env_id = _i(ep_rows[0].get("env_id"), 0) if ep_rows else 0
    episode_id = _i(ep_rows[0].get("episode_id"), None) if ep_rows else None
    if episode_id is None:
        # Infer from path episode_XX/env_YY
        for part in env_dir.parts:
            if part.startswith("episode_"):
                try:
                    episode_id = int(part.split("_", 1)[1])
                except ValueError:
                    pass
                break

    episode_end = _max_timestamp(events, job_rows)
    if episode_end <= 0:
        raise RuntimeError(f"No usable timestamps in {env_dir}")

    as_of_s = episode_end if closed_windows_only else None
    timelines = build_timelines(events, episode_end)
    dist_intervals = parse_disturbance_l2_intervals(
        disturbance_rows, open_end_s=as_of_s
    )

    all_features: list[dict] = []
    for ws in window_sizes:
        feats = compute_window_features(
            timelines=dict(timelines),  # copy ids; buffers may be added
            job_rows=job_rows,
            buffer_rows=buffer_rows,
            transport_rows=transport_rows,
            material_rows=material_rows,
            window_size=ws,
            episode_end=episode_end,
            run_id=run_id,
            env_id=env_id if env_id is not None else 0,
            disturbance_intervals=dist_intervals,
            closed_windows_only=closed_windows_only,
        )
        all_features.extend(feats)

    all_features = add_bottleneck_scores(all_features)
    labels, event_rows = build_labels_and_events(
        all_features,
        horizon,
        score_threshold,
        min_event_windows,
        as_of_s=as_of_s,
    )
    job_kpi_rows, order_kpi = build_job_kpis(
        job_rows,
        run_id=run_id,
        env_id=env_id if env_id is not None else 0,
        episode_id=episode_id,
    )

    _write_csv(out_dir / "window_feature_table.csv", all_features)
    _write_csv(out_dir / "bottleneck_label.csv", labels)
    _write_csv(out_dir / "bottleneck_event.csv", event_rows)
    _write_csv(
        out_dir / "job_kpi.csv",
        job_kpi_rows,
        fieldnames=[
            "run_id",
            "env_id",
            "episode_id",
            "job_id",
            "product_type",
            "start_s",
            "complete_s",
            "cycle_time_s",
            "completed",
            "complete_source",
        ],
    )

    # Summary stats
    top_nodes = []
    for ws in window_sizes:
        ws_labels = [l for l in labels if l["window_size_s"] == ws]
        hot = [l for l in ws_labels if l["is_bottleneck_window"]]
        node_counts = defaultdict(int)
        for l in hot:
            node_counts[l["bottleneck_node_t"]] += 1
        top = sorted(node_counts.items(), key=lambda x: -x[1])[:5]
        top_nodes.append({"window_size_s": ws, "hot_windows": len(hot), "top_nodes": top})

    summary = {
        "run_id": run_id,
        "env_id": env_id,
        "episode_id": episode_id,
        "episode_end_s": episode_end,
        "n_resources": len(timelines),
        "n_feature_rows": len(all_features),
        "n_label_rows": len(labels),
        "n_events": len(event_rows),
        "n_job_kpi_rows": len(job_kpi_rows),
        "order_kpi": order_kpi,
        "window_sizes": window_sizes,
        "horizon_s": horizon,
        "score_threshold": score_threshold,
        "min_event_windows": min_event_windows,
        "n_disturbance_l2": len(dist_intervals),
        # Sanity: L2 is input context only; this must stay 0 after 2026-08-19.
        "n_events_from_disturbance": sum(
            1
            for e in event_rows
            if e.get("event_source") == "disturbance_log"
        ),
        "closed_windows_only": closed_windows_only,
        "as_of_s": as_of_s,
        "per_window_size": top_nodes,
        "will_bottleneck_rate": {
            str(ws): (
                sum(1 for l in labels if l["window_size_s"] == ws and l["will_bottleneck"] == 1)
                / max(sum(1 for l in labels if l["window_size_s"] == ws), 1)
            )
            for ws in window_sizes
        },
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _run_once(
    run_dir: Path,
    out_root: Path,
    env_id: int | None,
    window_sizes: list[float],
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
    closed_windows_only: bool,
) -> list[dict]:
    env_dirs = _discover_env_dirs(run_dir, env_id)
    if not env_dirs:
        return []

    summaries = []
    for env_dir in env_dirs:
        out_dir = _derived_out_dir(out_root, run_dir, env_dir)
        print(f"[build] {env_dir} → {out_dir}")
        summary = process_env_dir(
            env_dir=env_dir,
            out_dir=out_dir,
            window_sizes=window_sizes,
            horizon=horizon,
            score_threshold=score_threshold,
            min_event_windows=min_event_windows,
            closed_windows_only=closed_windows_only,
        )
        summaries.append(summary)
        print(
            f"  episode_end={summary['episode_end_s']:.0f}s  "
            f"features={summary['n_feature_rows']}  "
            f"labels={summary['n_label_rows']}  "
            f"events={summary['n_events']}"
            + ("  [closed windows]" if closed_windows_only else "")
        )
        okpi = summary.get("order_kpi") or {}
        if okpi.get("n_completed"):
            print(
                f"  jobs={okpi.get('n_completed')}/{okpi.get('n_jobs')}  "
                f"makespan={okpi.get('order_makespan_s')}s  "
                f"mean_cycle={okpi.get('mean_cycle_time_s')}s  "
                f"throughput={okpi.get('throughput_jobs_per_hour')}/h"
            )
        for ps in summary["per_window_size"]:
            print(f"  ws={ps['window_size_s']}: hot_windows={ps['hot_windows']} top={ps['top_nodes'][:3]}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bottleneck features & labels (offline, or --follow while Isaac is collecting)"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to run dir containing env_XX/ subdirs",
    )
    parser.add_argument(
        "--env_id",
        type=int,
        default=None,
        help="Only process this env id (default: all env_*)",
    )
    parser.add_argument("--window_sizes", type=str, default="30,60", help="Comma-separated logic seconds")
    parser.add_argument("--horizon", type=float, default=180.0, help="Future horizon H (logic seconds)")
    parser.add_argument("--score_threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument(
        "--min_event_windows",
        type=int,
        default=DEFAULT_MIN_EVENT_WINDOWS,
        help="Min consecutive turning-point windows to emit a score event (default 1).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <run_dir>/derived)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Watch a live run: re-aggregate closed 60s windows as logs grow. Ctrl-C to stop.",
    )
    parser.add_argument("--poll", type=float, default=5.0, help="--follow poll interval in seconds")
    parser.add_argument(
        "--closed-windows",
        action="store_true",
        help="One-shot: only emit fully closed windows (no short last window). Implied by --follow.",
    )
    args = parser.parse_args()

    window_sizes = [float(x) for x in args.window_sizes.split(",") if x.strip()]
    run_dir = args.run_dir.resolve()
    out_root = (args.out_dir or (run_dir / "derived")).resolve()
    closed = bool(args.follow or args.closed_windows)

    if args.follow:
        print(f"[follow] {run_dir}  poll={args.poll}s  closed windows only  Ctrl-C to stop")
        last_sig = None
        try:
            while True:
                env_dirs = _discover_env_dirs(run_dir, args.env_id)
                sig = tuple(
                    (str(d), (d / "resource_event_log.jsonl").stat().st_size if (d / "resource_event_log.jsonl").exists() else 0)
                    for d in env_dirs
                )
                if sig and sig != last_sig:
                    try:
                        summaries = _run_once(
                            run_dir,
                            out_root,
                            args.env_id,
                            window_sizes,
                            args.horizon,
                            args.score_threshold,
                            args.min_event_windows,
                            closed_windows_only=True,
                        )
                        (out_root / "all_env_summary.json").write_text(
                            json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        last_sig = sig
                    except RuntimeError as exc:
                        print(f"[follow] skip: {exc}")
                time.sleep(max(args.poll, 0.5))
        except KeyboardInterrupt:
            print("[follow] stopped")
            return

    summaries = _run_once(
        run_dir,
        out_root,
        args.env_id,
        window_sizes,
        args.horizon,
        args.score_threshold,
        args.min_event_windows,
        closed_windows_only=closed,
    )
    if not summaries:
        raise SystemExit(f"No env_* directories under {run_dir} (checked flat and episode_*/ layouts)")
    (out_root / "all_env_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] outputs under {out_root}")


if __name__ == "__main__":
    main()
