"""Orchestrate Stage-C for one env dir and the CLI."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .constants import DEFAULT_MIN_EVENT_WINDOWS, DEFAULT_SCORE_THRESHOLD
from .features import add_bottleneck_scores, compute_window_features
from .io_util import _derived_out_dir, _discover_env_dirs, _f, _i, _read_csv, _read_jsonl, _write_csv
from .kpi import build_job_kpis
from .labels import build_labels_and_events, parse_disturbance_l2_intervals
from .timelines import build_timelines

def process_env_dir(
    env_dir: Path,
    out_dir: Path,
    window_sizes: list[float],
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
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

    times = []
    for e in events:
        times.append(_f(e.get("logic_time_s"), _f(e.get("time_step"))))
    for r in job_rows:
        times.append(_f(r.get("logic_time_s"), _f(r.get("time_step"))))
    episode_end = max(times) if times else 0.0
    if episode_end <= 0:
        raise RuntimeError(f"No usable timestamps in {env_dir}")

    timelines = build_timelines(events, episode_end)
    dist_intervals = parse_disturbance_l2_intervals(disturbance_rows)

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
        )
        all_features.extend(feats)

    all_features = add_bottleneck_scores(all_features)
    labels, event_rows = build_labels_and_events(
        all_features,
        horizon,
        score_threshold,
        min_event_windows,
        disturbance_rows=disturbance_rows,
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
        "n_events_from_disturbance": sum(
            1
            for e in event_rows
            if e.get("event_source") == "disturbance_log"
        ),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline bottleneck features & labels")
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
    args = parser.parse_args()

    window_sizes = [float(x) for x in args.window_sizes.split(",") if x.strip()]
    run_dir = args.run_dir.resolve()
    out_root = (args.out_dir or (run_dir / "derived")).resolve()

    env_dirs = _discover_env_dirs(run_dir, args.env_id)
    if not env_dirs:
        raise SystemExit(f"No env_* directories under {run_dir} (checked flat and episode_*/ layouts)")

    summaries = []
    for env_dir in env_dirs:
        out_dir = _derived_out_dir(out_root, run_dir, env_dir)
        print(f"[build] {env_dir} → {out_dir}")
        summary = process_env_dir(
            env_dir=env_dir,
            out_dir=out_dir,
            window_sizes=window_sizes,
            horizon=args.horizon,
            score_threshold=args.score_threshold,
            min_event_windows=args.min_event_windows,
        )
        summaries.append(summary)
        print(
            f"  episode_end={summary['episode_end_s']:.0f}s  "
            f"features={summary['n_feature_rows']}  "
            f"labels={summary['n_label_rows']}  "
            f"events={summary['n_events']}"
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

    (out_root / "all_env_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] outputs under {out_root}")


if __name__ == "__main__":
    main()
