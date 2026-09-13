"""Audit A.1 event support under explicit hot/event duration contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from factory_bn.dataset import build_dataloaders
from factory_bn.remain import node_event_targets


CONTRACTS = (
    ("hot8_event5", 8, 5, "legacy"),
    ("hot5_event5_legacy", 5, 5, "legacy"),
    ("hot5_event5_close", 5, 5, "close_then_filter"),
    ("hot10_event10", 10, 10, "legacy"),
)


def _count(loader, *, event_min: int, max_start: int, resource_types: list[str]) -> dict:
    starts = np.zeros(max_start + 1, dtype=np.int64)
    by_type = {name: 0 for name in sorted(set(resource_types))}
    total = ongoing = upcoming = 0
    for batch in loader:
        will, start, _ = node_event_targets(
            batch["y_hot"].numpy(),
            min_windows=event_min,
            remain_mask=batch["remain_mask"].numpy(),
            occ_node_mask=batch["occ_node_mask"].numpy(),
            max_start_windows=max_start,
            hist_last_hot=batch["hist_last_hot"].numpy(),
            ongoing_min_windows=1,
        )
        pos = will > 0.5
        last = batch["hist_last_hot"].numpy() > 0.5
        total += int(pos.sum())
        ongoing += int((pos & last).sum())
        upcoming += int((pos & ~last).sum())
        for k in range(max_start + 1):
            starts[k] += int((pos & ~last & (start == k)).sum())
        for ni, name in enumerate(resource_types):
            by_type[name] += int(pos[:, ni].sum())
    return {
        "total": total,
        "ongoing": ongoing,
        "upcoming": upcoming,
        "upcoming_start_counts": starts.tolist(),
        "by_resource_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="raw_data/dense_i1")
    parser.add_argument("--max_start", type=int, default=5)
    parser.add_argument("--occupancy_horizon", type=int, default=15)
    parser.add_argument("--min_episode_jobs_total", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    result: dict[str, dict] = {}
    for name, hot_min, event_min, order in CONTRACTS:
        train, val, test, feature = build_dataloaders(
            data_dir,
            input_window=30,
            output_window=1,
            horizon_s=180,
            batch_size=args.batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42,
            max_hist_events=8,
            remain_to_jobs_done=True,
            max_remain_windows=args.occupancy_horizon,
            occupancy_horizon_windows=args.occupancy_horizon,
            hot_min_windows=hot_min,
            hot_gap_windows=1,
            hot_smoothing_order=order,
            min_episode_jobs_total=args.min_episode_jobs_total,
            train_mode="unsupervised",
        )
        resource_types = [str(x) for x in feature["resource_types"]]
        split_result = {
            split: _count(
                loader,
                event_min=event_min,
                max_start=args.max_start,
                resource_types=resource_types,
            )
            for split, loader in (("train", train), ("val", val), ("test", test))
        }
        result[name] = {
            "episode_filter": {
                "source": feature["n_source_episodes"],
                "retained": feature["n_filtered_episodes"],
                "excluded": feature["excluded_episodes"],
            },
            **split_result,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
