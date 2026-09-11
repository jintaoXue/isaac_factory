#!/usr/bin/env python3
"""Audit, freeze splits and rebuild in place without creating experiment folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from collections import defaultdict

import numpy as np

from audit_bottleneck_data import audit_env_dir, build_report, discover_env_dirs
from bn_agg.pipeline import process_env_dir
from factory_baselines.artifacts import archive_files
from factory_baselines.dataset import FactoryBaselineTensorDataset, build_factory_baseline_dataset, _split_groups, _stable_group_number
from factory_baselines.evaluation import event_rule_kwargs
from factory_bn_shared.bundle import file_hash
from factory_bn_shared.contract import DERIVED_CONTRACT_VERSION, RAW_CONTRACT_VERSION, SHARED_LABEL_VERSION
from factory_bn.export_dataset import export_runs
from factory_bn_shared.remain import first_done_index, node_event_targets, occupancy_node_mask, ops_hot_mask, pack_remain_target


DATASET_FILES = [
    "dataset.pt", "dataset_manifest.json", "split_manifest.json", "normalization.json",
    "node_catalog.csv", "graph_edge_table.csv", "model_sample_index.csv",
    "episodes.npz", "meta.json", "node_map.json",
]


def identity(row: dict) -> str:
    return f"{row['run_id']}:env_{int(row['env_id']):02d}:episode_{int(row['episode_id']):02d}"


def extend_splits(old: dict, accepted: list[dict], seed: int) -> dict[str, list[str]]:
    result = {name: list(value["group_ids"]) for name, value in old.items()}
    if set(result) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split names")
    existing = [group for values in result.values() for group in values]
    available = [identity(row) for row in accepted]
    if len(set(existing)) != len(existing) or len(set(available)) != len(available):
        raise ValueError("Duplicate physical episode across cohorts or splits")
    if not set(existing).issubset(available):
        raise ValueError("An old split episode no longer passes quality audit")
    additions = {identity(row): Path(row["run_dir"]).name for row in accepted
                 if identity(row) not in existing}
    if additions:
        extra = _split_groups({key: [] for key in additions}, additions, seed)
        for name in result:
            result[name].extend(extra[name])
    return {name: sorted(values) for name, values in result.items()}


def verify_export_alignment(result: dict, root: Path) -> dict:
    """Check every sample against the common export before any model training."""
    payload, manifest = result["payload"], result["manifest"]
    normalization = json.loads((root / "normalization.json").read_text())
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    names = {row["group_id"]: row["main_episode_name"] for row in manifest["source_episodes"]}
    by_group = defaultdict(list)
    for row in result["sample_rows"]:
        by_group[row["group_id"]].append(row)
    dataset = FactoryBaselineTensorDataset(payload)
    with np.load(root / "episodes.npz", allow_pickle=False) as bundle:
        node_ids = bundle["resource_ids"].tolist()
        if set(node_ids) != set(manifest["node_ids"]):
            raise ValueError("Canonical/baseline node inventories differ")
        order = [node_ids.index(node) for node in manifest["node_ids"]]
        for group, rows in by_group.items():
            name = names[group]
            features = bundle[name + "_features"][:, order]
            scores = bundle[name + "_scores"][:, order]
            cause = bundle[name + "_cause"]
            jobs = bundle[name + "_jobs_remaining"]
            total = bundle[name + "_jobs_total"][0]
            windows = bundle[name + "_windows"]
            starts = bundle[name + "_window_start_s"]
            hot = ops_hot_mask(features, window_size_s=60, min_hot_windows=8, gap_windows=1)
            occ = occupancy_node_mask(features)
            done = first_done_index(jobs)
            for row in rows:
                sample = dataset[int(row["sample_index"])]
                t = int(sample["target_start_position"])
                history = len(json.loads(row["input_window_indices"]))
                np.testing.assert_array_equal(json.loads(row["input_window_indices"]), windows[t-history:t])
                np.testing.assert_allclose([float(row["anchor_time_s"]), float(row["first_future_start_s"])], starts[[t-1, t]])
                actual = sample["x"].numpy().copy()
                actual[..., :len(mean)] = actual[..., :len(mean)] * std + mean
                observed = sample["observation_mask"].numpy()
                np.testing.assert_allclose(actual[observed], features[t-history:t][observed], rtol=1e-5, atol=5e-4)
                score, target_hot, mask, length = pack_remain_target(
                    scores, hot, t=t, done_ti=done, max_remain_windows=15, occupancy_horizon_windows=15,
                )
                target_will, start, dur = node_event_targets(
                    target_hot, remain_mask=mask, occ_node_mask=occ,
                    hist_last_hot=hot[t-1], **event_rule_kwargs(8),
                )
                expected = dict(y_score=score, y_hot=target_hot, remain_mask=mask,
                                occ_node_mask=occ,
                                hist_last_hot=hot[t-1], event_will=target_will, event_start=start,
                                event_duration=dur, jobs_remaining=jobs[t-1], jobs_total=total,
                                y_cause=cause[t-1], target_remain_len=length)
                for key, value in expected.items():
                    np.testing.assert_allclose(sample[key].numpy(), value, rtol=1e-5, atol=1e-4, err_msg=f"{group}/{t}/{key}")
                if int(row["event_will_any"]) != int((target_will > .5).any()):
                    raise ValueError("Sample index occurrence differs from dense targets")
            print(f"[aligned] {name} samples={len(rows)}", flush=True)
    return {"status": "passed", "samples": len(dataset), "episodes": len(by_group),
            "scope": "exported inputs, targets, masks, history anchors; no model/test metrics",
            "bundle_sha256": file_hash(root / "episodes.npz")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--additional_run_dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--archive_tag", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.benchmark_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Reuse an existing benchmark directory")
    if not args.archive_tag or not all(c.isalnum() or c in "_-" for c in args.archive_tag):
        raise ValueError("archive_tag must contain only letters, numbers, _ or -")
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != "dev_xwt":
        raise ValueError("This rebuild must run on dev_xwt")
    if not root.is_relative_to(repo) or repo.name != "BSTAN_isaac_factory":
        raise ValueError("Server outputs must stay inside BSTAN_isaac_factory")
    manifest_path = root / "dataset_manifest.json"
    old = json.loads(manifest_path.read_text())
    old_split = json.loads((root / "split_manifest.json").read_text())
    run_dirs = [Path(path).absolute() for path in old["source_run_directories"]]
    run_dirs.extend(path.absolute() for path in args.additional_run_dirs)
    if len({path.name for path in run_dirs}) != len(run_dirs):
        raise ValueError("Run aliases must be unique")
    for path in run_dirs:
        if not discover_env_dirs([path]):
            raise FileNotFoundError(f"Raw run contains no episodes: {path}")
    audit_path = root / f"raw_expansion_{args.archive_tag}.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    rows = []
    for run, env in discover_env_dirs(run_dirs):
        row = audit_env_dir(run, env)
        rows.append(row)
        print(f"[audit] {run.name}/{env.parent.name} accepted={row['accepted']}", flush=True)
    accepted = [row for row in rows if row["accepted"]]
    by_id = {identity(row): row for row in accepted}
    for previous in old["source_episodes"]:
        current = by_id.get(previous["group_id"])
        if current is None or current["raw_episode_sha256"] != previous["raw_episode_sha256"]:
            raise ValueError(f"Old raw identity/hash changed: {previous['group_id']}")
    splits = extend_splits(old_split, accepted, int(old["seed"]))
    report = build_report(rows)
    report.update(
        old_manifest_sha256=file_hash(manifest_path),
        preserved_split_episode_count=sum(len(v["group_ids"]) for v in old_split.values()),
        frozen_splits=splits,
        selection_policy="raw integrity/completion/no-deadlock only; no model metrics",
        main_dense_i1_identity_match="unverified; not the documented 204 cohort",
    )
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("AUDIT", len(rows), "accepted", len(accepted), "splits", {k: len(v) for k, v in splits.items()}, flush=True)
    if not args.apply:
        return
    prepared, groups = [], []
    for row in accepted:
        env, run = Path(row["env_dir"]), Path(row["run_dir"])
        with (env / "episode_config.csv").open(newline="") as stream:
            config = next(csv.DictReader(stream))
        print(f"[derive in memory] {run.name}/{env.parent.name}", flush=True)
        tables = process_env_dir(env, None, [60.0], 180.0, .55, 8, closed_windows_only=False)
        group_id = identity(row)
        groups.append({
            "group_id": group_id, "group_number": _stable_group_number(group_id),
            "run_dir": run, "run_name": run.name, "raw_dir": env,
            "run_id": row["run_id"], "env_id": row["env_id"], "episode_id": row["episode_id"],
            "scenario_id": row["scenario_id"], "collector_version": row["collector_version"],
            "derived_contract_version": DERIVED_CONTRACT_VERSION, "label_version": SHARED_LABEL_VERSION,
            "raw_contract_version": RAW_CONTRACT_VERSION, "raw_episode_sha256": row["raw_episode_sha256"],
            "config": config, "feature_rows": tables["features"], "job_kpi_rows": tables["job_kpi"],
        })
        name = run.name + "__" + env.parent.name
        prepared.append((name, tables["features"], {int(r["window_index"]): r for r in tables["labels"]},
                         sorted(tables["events"], key=lambda r: r["start_s"]), tables["job_kpi"]))
    for row in accepted:
        current = audit_env_dir(Path(row["run_dir"]), Path(row["env_dir"]))
        if not current["accepted"] or current["raw_episode_sha256"] != row["raw_episode_sha256"]:
            raise RuntimeError(f"Raw changed during aggregation: {row['env_dir']}")
    # Preserve the exact pre-build dataset and index files, not only summary numbers.
    saved = archive_files(root, DATASET_FILES, f"dataset_before_{args.archive_tag}.zip")
    print("ARCHIVE", saved, flush=True)
    export_runs(run_dirs, root, window_size=60.0, write_atomic=False, prepared_episodes=prepared)
    result = build_factory_baseline_dataset(
        run_dirs, root, root, root, seed=int(old["seed"]), repo_root=repo,
        allowed_group_ids=set(by_id), episode_groups=groups, frozen_split_groups=splits,
    )
    new = result["manifest"]
    new["validation"] = "pending_shared_bundle_alignment"
    (root / "dataset_manifest.json").write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
    new["shared_bundle_alignment"] = verify_export_alignment(result, root)
    new["validation"] = "passed"
    new["cohort_audit"] = {"path": str(audit_path), "sha256": file_hash(audit_path)}
    new["prior_dataset_archive"] = str(saved)
    new["main_dense_i1_identity_match"] = False
    new["comparison_scope"] = "shared new cohort; main model must be reevaluated on this same bundle/split"
    (root / "dataset_manifest.json").write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: new[k] for k in ("dataset_version", "total_samples", "event_positive_samples", "episode_counts")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
