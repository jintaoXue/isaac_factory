#!/usr/bin/env python3
"""Count unique onset support and retrospective onset conditions on train/validation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from factory_baselines.evaluation import event_rule_kwargs
from factory_bn_shared.bundle import file_hash
from factory_bn_shared.remain import (
    first_done_index, node_event_targets, occupancy_node_mask, ops_hot_mask, pack_remain_target,
)


def episode_support(features, scores, hot, done, anchors, occ, node_ids):
    """Deduplicate by (episode, node, absolute onset), never by selected model scores."""
    counts = Counter()
    events = defaultdict(list)
    per_node = Counter()
    # These are descriptive observed channels; zero does not mean no predictive information.
    cue_indices = [0, 6, 7, 11, 12, 13, 14, 15, 18]
    for t in anchors:
        _, y_hot, remain_mask, _ = pack_remain_target(
            scores, hot, t=t, done_ti=done, max_remain_windows=15, occupancy_horizon_windows=15,
        )
        will, start, duration = node_event_targets(
            y_hot, remain_mask=remain_mask, occ_node_mask=occ, hist_last_hot=hot[t-1],
            **event_rule_kwargs(8),
        )
        valid = np.asarray(occ) > .5
        positive = (will > .5) & valid
        upcoming = positive & (start > 0)
        counts.update(samples=1, ongoing_targets=int((positive & (start == 0)).sum()),
                      upcoming_targets=int(upcoming.sum()), negative_targets=int((~positive & valid).sum()),
                      positive_start_zero_hist_cold=int((positive & (start == 0) & (hot[t-1] <= .5)).sum()))
        for node in np.flatnonzero(upcoming):
            onset = t + int(start[node])
            key = (int(node), onset)
            per_node[node_ids[node]] += 1
            local_before = bool(features[t-1, node, 18] > 0)
            any_before = bool((features[t-1, :, 18] > 0).any())
            local_during = bool((features[t:onset+1, node, 18] > 0).any())
            zero_cues = bool(np.all(np.abs(features[t-5:t, node][:, cue_indices]) <= 1e-6))
            description = {
                "first_future_position": t, "start_index": int(start[node]),
                "target_duration_windows": float(duration[node]),
                "local_disturbance_observed_at_anchor": local_before,
                "any_node_disturbance_observed_at_anchor": any_before,
                "new_local_disturbance_by_onset": not local_before and local_during,
                "last_five_local_cue_channels_all_zero": zero_cues,
            }
            events[key].append(description)
            counts.update({name: int(description[name]) for name in (
                "local_disturbance_observed_at_anchor", "any_node_disturbance_observed_at_anchor",
                "new_local_disturbance_by_onset", "last_five_local_cue_channels_all_zero")})
    rows = [{"resource_id": node_ids[node], "onset_position": onset, "anchor_support": len(values),
             "anchors": sorted(values, key=lambda r: r["first_future_position"])}
            for (node, onset), values in sorted(events.items())]
    return {"counts": dict(counts), "unique_upcoming_onsets": len(rows),
            "upcoming_targets_by_node": dict(per_node), "upcoming_onsets": rows}


def diagnose(root: Path) -> dict:
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    splits = json.loads((root / "split_manifest.json").read_text())
    if manifest["dataset_version"] != "factory_baseline_dataset_v6" or manifest["input_windows"] != 30:
        raise ValueError("Expected the existing dense v6 contract")
    hashes = {name: file_hash(root / name) for name in (
        "dataset_manifest.json", "split_manifest.json", "model_sample_index.csv", "episodes.npz")}
    if hashes["episodes.npz"] != manifest["shared_bundle_alignment"]["bundle_sha256"]:
        raise ValueError("The common export has changed")
    sources = {r["group_id"]: r for r in manifest["source_episodes"]}
    groups = defaultdict(list)
    with (root / "model_sample_index.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in {"train", "validation"}:
                groups[row["group_id"]].append(row)
    episodes = []
    indices = defaultdict(list)
    with np.load(root / "episodes.npz", allow_pickle=False) as bundle:
        node_ids = manifest["node_ids"]
        ids = bundle["resource_ids"].tolist()
        if set(ids) != set(node_ids): raise ValueError("Node inventories differ")
        order = [ids.index(node) for node in node_ids]
        for group, rows in groups.items():
            split_name = rows[0]["split"]
            if group not in splits[split_name]["group_ids"] or any(r["split"] != split_name for r in rows):
                raise ValueError("Sample split identity mismatch")
            source = sources[group]
            name = source["main_episode_name"]
            features = bundle[name + "_features"][:, order]
            scores = bundle[name + "_scores"][:, order]
            windows = bundle[name + "_windows"]
            positions = {int(w): i for i, w in enumerate(windows)}
            anchors = [positions[int(r["anchor_window_index"])] + 1 for r in rows]
            indices[split_name].extend(int(r["sample_index"]) for r in rows)
            hot = ops_hot_mask(features, window_size_s=60, min_hot_windows=8, gap_windows=1)
            result = episode_support(features, scores, hot, first_done_index(bundle[name + "_jobs_remaining"]),
                                     anchors, occupancy_node_mask(features), node_ids)
            result.update(group_id=group, split=split_name, main_episode_name=name,
                          scenario_id=source["scenario_id"], source_prefix=re.split(r"__?episode_", name)[0])
            for event in result["upcoming_onsets"]:
                event["onset_window_index"] = int(windows[event["onset_position"]])
            episodes.append(result)
    summaries = {}
    for split_name in ("train", "validation"):
        if sorted(indices[split_name]) != sorted(splits[split_name]["sample_indices"]):
            raise ValueError("Incomplete or duplicate sample coverage")
        selected = [row for row in episodes if row["split"] == split_name]
        totals, prefixes, nodes = Counter(), {}, Counter()
        for row in selected:
            totals.update(row["counts"])
            nodes.update(row["upcoming_targets_by_node"])
            p = prefixes.setdefault(row["source_prefix"], Counter())
            p.update(episodes=1, samples=row["counts"]["samples"],
                     upcoming_targets=row["counts"].get("upcoming_targets", 0),
                     unique_upcoming_onsets=row["unique_upcoming_onsets"])
        expected = {"train": (4191, 595, 297152), "validation": (950, 145, 67331)}[split_name]
        if tuple(totals[k] for k in ("ongoing_targets", "upcoming_targets", "negative_targets")) != expected:
            raise ValueError("Reconstructed target support differs from frozen diagnostics")
        summaries[split_name] = {
            "episodes": len(selected), "counts": dict(totals),
            "episodes_with_upcoming": sum(row["unique_upcoming_onsets"] > 0 for row in selected),
            "unique_upcoming_onsets": sum(row["unique_upcoming_onsets"] for row in selected),
            "upcoming_targets_by_node": dict(nodes),
            "source_prefix_support": {key: dict(value) for key, value in prefixes.items()},
        }
    return {"status": "completed", "dataset_files_sha256": hashes, "summaries": summaries,
            "episodes": episodes, "test_evaluated": False, "model_training": False,
            "scope": "Retrospective train/validation labels and observed conditions; no prediction threshold selected.",
            "limitations": ["Unique onsets within an episode can still be statistically dependent.",
                            "Zero local cue channels or an unobserved disturbance do not establish unpredictability.",
                            "Whole-episode smoothed hot is retained only as the existing target definition."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_commit", help="Pinned Git source when this standalone diagnostic is streamed on stdin")
    args = parser.parse_args()
    root, output = args.dataset_dir.resolve(), args.output.resolve()
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if repo.name != "BSTAN_isaac_factory" or branch != "dev_xwt" or not root.is_relative_to(repo):
        raise ValueError("Run only in existing BSTAN_isaac_factory/dev_xwt")
    if not root.is_dir() or output.parent != root or output.exists():
        raise ValueError("Use a new result file in the existing benchmark directory")
    result = diagnose(root)
    result["worktree_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["code_commit"] = args.source_commit or result["worktree_commit"]
    source_path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/diagnose_dense_event_support.py"
    source = subprocess.check_output(["git", "show", f"{result['code_commit']}:{source_path}"])
    result["diagnostic_source_sha256"] = hashlib.sha256(source).hexdigest()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: {k: v for k, v in summary.items() if k not in {"source_prefix_support", "upcoming_targets_by_node"}}
                      for key, summary in result["summaries"].items()}, indent=2))


if __name__ == "__main__":
    main()
