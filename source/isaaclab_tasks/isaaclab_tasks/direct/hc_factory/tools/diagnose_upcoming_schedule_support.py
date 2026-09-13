#!/usr/bin/env python3
"""Join frozen train/validation onsets to recorded runtime disturbance starts."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def match_future_runtime(onset, anchor, runtime, window_seconds=60):
    """Use actual runtime resource IDs and half-open observation windows."""
    if window_seconds <= 0 or anchor["start_index"] not in (1, 2):
        raise ValueError("Expected the existing positive upcoming start contract")
    begin = (onset["onset_window_index"] - anchor["start_index"]) * window_seconds
    end = (onset["onset_window_index"] + 1) * window_seconds
    if begin < 0:
        raise ValueError("Negative first forecast time")
    matches = [event for event in runtime if event["target"] == onset["resource_id"]
               and begin <= event["start"] < end]
    return dict(first_future_time_s=begin, onset_window_end_s=end,
        matched_runtime_event_ids=[event["event_id"] for event in matches],
        matched_runtime_types=sorted({event["type"] for event in matches}),
        matched_future_local_runtime_start=bool(matches),
        no_prior_recorded_runtime_start=not any(event["start"] < begin for event in runtime))


def summarize_support_cells(episodes):
    """Separate distinct onsets from overlapping window-node targets."""
    levels = {
        "node": lambda ep, event: (event["resource_id"],),
        "scenario": lambda ep, event: (ep["scenario_id"],),
        "node_scenario": lambda ep, event: (event["resource_id"], ep["scenario_id"]),
    }
    result = {}
    for name, key in levels.items():
        train = Counter(key(ep, event) for ep in episodes if ep["split"] == "train"
                        for event in ep["upcoming_onsets"])
        groups = defaultdict(lambda: dict(validation_unique_onsets=0, validation_window_targets=0))
        for ep in episodes:
            if ep["split"] != "validation":
                continue
            for event in ep["upcoming_onsets"]:
                cell = key(ep, event)
                groups[cell]["validation_unique_onsets"] += 1
                groups[cell]["validation_window_targets"] += len(event["anchors"])
        rows = [dict(cell=list(cell), train_unique_onsets=train[cell], **values)
                for cell, values in sorted(groups.items())]
        result[name] = dict(cells=rows, coverage=[dict(maximum_train_unique_onsets=limit,
            validation_unique_onsets=sum(row["validation_unique_onsets"] for row in rows if row["train_unique_onsets"] <= limit),
            validation_window_targets=sum(row["validation_window_targets"] for row in rows if row["train_unique_onsets"] <= limit))
            for limit in (0, 1, 2, 3, 5, 10)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
    tools_dir = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
    d = tools_dir.parent / "output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    output = d / "baseline_upcoming_schedule_support20260913.json"
    assert d.is_dir() and not output.exists()
    source_path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/diagnose_upcoming_schedule_support.py"
    source = subprocess.check_output(["git", "show", args.source_commit + ":" + source_path])
    manifest_path = d / "dataset_manifest.json"
    assert sha(manifest_path) == "e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f"
    manifest = json.loads(manifest_path.read_text())
    support_path = d / "dense_event_support20260912.json"
    assert sha(support_path) == "56d2dc07169d12e420ed1d2877bcf3290d2cd53d0abdd317ea6581845b32cd93"
    support = json.loads(support_path.read_text())
    audit_path = Path(manifest["cohort_audit"]["path"])
    assert sha(audit_path) == manifest["cohort_audit"]["sha256"]
    audit = json.loads(audit_path.read_text())
    raw = {f"{row['run_id']}:env_{row['env_id']:02d}:episode_{row['episode_id']:02d}": row for row in audit["episodes"]}
    source_episodes = {row["group_id"]: row for row in manifest["source_episodes"]}
    splits = json.loads((d / "split_manifest.json").read_text())
    episodes = support["episodes"]
    assert {ep["split"] for ep in episodes} == {"train", "validation"}
    for split, expected in (("train", 138), ("validation", 30)):
        selected = [ep["group_id"] for ep in episodes if ep["split"] == split]
        assert len(selected) == expected and sorted(selected) == sorted(splits[split]["group_ids"])
    generator_path = tools_dir.parent / "env_asset_cfg/cfg_disturbance.py"
    generator_hash = sha(generator_path)
    spec = importlib.util.spec_from_file_location("schedule_support_generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    from factory_bn_shared.contract import paired_disturbance_intervals
    raw_hashes, rows, plans = {}, [], []
    counts = {split: Counter() for split in ("train", "validation")}
    for ep in episodes:
        entry = raw[ep["group_id"]]
        assert entry["accepted"] and entry["raw_episode_sha256"] == source_episodes[ep["group_id"]]["raw_episode_sha256"]
        assert ep["scenario_id"] == entry["scenario_id"]
        directory = Path(entry["env_dir"])
        for name in ("episode_config.csv", "disturbance_log.csv"):
            path = directory / name
            expected = entry["raw_file_sha256"][name]
            assert sha(path) == expected
            raw_hashes[str(path)] = expected
        with (directory / "episode_config.csv").open(newline="") as stream:
            configs = list(csv.DictReader(stream))
        assert len(configs) == 1
        config = configs[0]
        with (directory / "disturbance_log.csv").open(newline="") as stream:
            runtime = paired_disturbance_intervals(list(csv.DictReader(stream)), float(config["logic_dt"]), entry["episode_end_s"])
        assert runtime == entry["runtime_events"]
        applied = json.loads(config["disturbance_applied"])
        mode = applied.get("event_schedule_mode")
        actual_plan = applied.get("event_schedule") or []
        reproduced = None
        if mode == "resample_per_episode":
            generator.RuntimeDisturbanceCfg["applied"] = applied
            reconstructed = generator.episode_l2_schedule(config["disturbance_dim"], float(config["disturbance_intensity"]),
                int(config["seed"]), int(config["env_id"]), int(config["episode_id"]))
            reproduced = reconstructed == actual_plan
        plans.append(dict(group_id=ep["group_id"], split=ep["split"], schedule_mode=mode,
            planned_event_count=len(actual_plan), runtime_event_count=len(runtime),
            current_generator_reproduces_saved_plan=reproduced))
        summary = counts[ep["split"]]
        summary["episodes"] += 1
        summary["unique_upcoming_onsets"] += len(ep["upcoming_onsets"])
        for event in ep["upcoming_onsets"]:
            any_match = False
            for anchor in event["anchors"]:
                match = match_future_runtime(event, anchor, runtime)
                row = dict(group_id=ep["group_id"], split=ep["split"], resource_id=event["resource_id"],
                    onset_window_index=event["onset_window_index"], start_index=anchor["start_index"],
                    legacy_new_local_disturbance_by_onset=anchor["new_local_disturbance_by_onset"], **match)
                rows.append(row)
                summary["upcoming_window_targets"] += 1
                summary["matched_future_local_runtime_start_targets"] += int(match["matched_future_local_runtime_start"])
                summary["matched_and_no_prior_recorded_runtime_start_targets"] += int(match["matched_future_local_runtime_start"] and match["no_prior_recorded_runtime_start"])
                summary["legacy_new_local_disturbance_by_onset_targets"] += int(anchor["new_local_disturbance_by_onset"])
                summary["legacy_new_local_without_matching_runtime_start_targets"] += int(anchor["new_local_disturbance_by_onset"] and not match["matched_future_local_runtime_start"])
                any_match |= match["matched_future_local_runtime_start"]
            summary["unique_upcoming_onsets_with_matching_runtime_start"] += int(any_match)
    for split, expected in (("train", (299, 595)), ("validation", (73, 145))):
        assert (counts[split]["unique_upcoming_onsets"], counts[split]["upcoming_window_targets"]) == expected
    for path, expected in raw_hashes.items():
        assert sha(Path(path)) == expected
    assert sha(manifest_path) == "e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f"
    assert sha(generator_path) == generator_hash
    record = dict(status="frozen_train_validation_runtime_schedule_and_support_audit_completed",
        source_commit=args.source_commit, source_sha256=hashlib.sha256(source).hexdigest(),
        runtime_source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        manifest_sha256=sha(manifest_path), support_sha256=sha(support_path), cohort_audit_sha256=sha(audit_path),
        generator_source_sha256=generator_hash, raw_config_and_disturbance_file_sha256=raw_hashes,
        summaries={split: dict(values) for split, values in counts.items()}, support_cells=summarize_support_cells(episodes),
        episode_plan_reconstruction=plans, upcoming_targets=rows, test_evaluated=False,
        model_training=False, main_repository_modified=False,
        limitations=["Retrospective alignment with an external event is not proof that it caused the target.",
            "No prior recorded runtime start does not exclude earlier quality holds or other predictive history.",
            "Private seeded schedules can have dependencies on configuration and earlier events; no information-theoretic recall ceiling is estimated.",
            "Only configuration and disturbance raw files were reread; other raw-file identities rely on the frozen audited cohort.",
            "No planned future schedules, identities or future labels enter a model or alter a report."])
    with output.open("x") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print("SUMMARIES", json.dumps(record["summaries"]), flush=True)
    print("COVERAGE", json.dumps({name: value["coverage"] for name, value in record["support_cells"].items()}), flush=True)
    print("PLANS", Counter((row["schedule_mode"], row["current_generator_reproduces_saved_plan"]) for row in plans), flush=True)
    print("SCHEDULE_SUPPORT_COMPLETE", output.stat().st_size, sha(output), flush=True)


if __name__ == "__main__":
    main()
