#!/usr/bin/env python3
"""Freeze a label-blind, nested 54-episode data-quantity diagnostic plan."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

RUNTIME = "90a41a5b9c44e625f8c80083a8a950cd50f5eb13"
SALT = "baseline_data_quantity54_20260913_v1"
RELATIVE = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/prepare_baseline_data_quantity.py"
PARENT_PLAN_SHA = "459945152e82ae529dd50bcb7b6427951e3b9204f4ea4ef0fdb71b9ac0ab4e2a"
CAPACITY_VERIFICATION_SHA = "c600344f63566090541ae8c788f2055d48eb0ffc64aa2663a553175fcbdffbb6"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(identifier):
    return hashlib.sha256((SALT + ":" + identifier).encode()).hexdigest(), identifier


def nested_episode_plan(parent, rows):
    names = {"fit", "heldout", "original_validation"}
    if set(parent["groups"]) != names or set(parent["sample_indices"]) != names:
        raise ValueError("Expected the existing three-view parent plan")
    groups = {k: set(v) for k, v in parent["groups"].items()}
    indices = {k: set(v) for k, v in parent["sample_indices"].items()}
    if any(len(groups[k]) != len(parent["groups"][k]) or len(indices[k]) != len(parent["sample_indices"][k]) for k in names):
        raise ValueError("Duplicate parent identity")
    if any(groups[a] & groups[b] or indices[a] & indices[b] for a in names for b in names if a != b):
        raise ValueError("Parent views overlap")
    membership = {group: name for name in names for group in groups[name]}
    samples, by_group = {}, defaultdict(list)
    for row in rows:
        group, sample = row["group_id"], int(row["sample_index"])
        if row["split"] not in ("train", "validation") or group not in membership:
            raise ValueError("Only the unchanged original train/validation identities are allowed")
        if sample in samples:
            raise ValueError("Duplicate sample identity")
        view = membership[group]
        if sample not in indices[view] or (row["split"] == "validation") != (view == "original_validation"):
            raise ValueError("Sample/episode identity crosses a parent split")
        if ":env_" not in group:
            raise ValueError("Unknown raw-run identifier")
        samples[sample] = group
        by_group[group].append(sample)
    if set(samples) != set.union(*indices.values()) or set(by_group) != set(membership):
        raise ValueError("Sample coverage differs from the complete frozen parent plan")
    by_run = defaultdict(list)
    for group in groups["fit"]:
        by_run[group.split(":env_", 1)[0]].append(group)
    if any(len(values) < 2 for values in by_run.values()):
        raise ValueError("Each original fitting raw run must retain representation in both halves")
    target = (len(groups["fit"]) + 1) // 2
    quotas = {run: len(values) // 2 for run, values in by_run.items()}
    odd_runs = sorted((run for run, values in by_run.items() if len(values) % 2), key=rank)
    extra = target - sum(quotas.values())
    for run in odd_runs[:extra]:
        quotas[run] += 1
    if not 0 <= extra <= len(odd_runs) or sum(quotas.values()) != target:
        raise ValueError("Invalid exact-half largest-remainder allocation")
    retained = set()
    for run, values in by_run.items():
        retained.update(sorted(values, key=rank)[:quotas[run]])
    unused = groups["fit"] - retained
    new_groups = dict(fit=sorted(retained), unused_parent_fit=sorted(unused),
        heldout=sorted(groups["heldout"]), original_validation=sorted(groups["original_validation"]))
    new_indices = {key: sorted(i for group in values for i in by_group[group]) for key, values in new_groups.items()}
    return dict(salt=SALT, selection="exact_ceil_half_within_raw_run_largest_remainder_identifier_hash_no_labels",
        groups=new_groups, sample_indices=new_indices,
        raw_run_strata=[dict(raw_run=run, parent_fit_episodes=len(by_run[run]), retained_episodes=quotas[run],
            unused_episodes=len(by_run[run])-quotas[run]) for run in sorted(by_run)],
        normalization_views=dict(fit=new_indices["fit"], heldout=sorted(new_indices["unused_parent_fit"]+new_indices["heldout"]),
            original_validation=new_indices["original_validation"]),
        evaluation_views={key:new_indices[key] for key in ("fit","heldout","original_validation")},
        parent_fitting_samples_per_reference_interval=len(indices["fit"]), reference_intervals=60,
        checkpoint_reference_intervals=[10,30,60], fixed_report_threshold=.70,
        preprocessing_fit_scope="retained_54_only", unused_parent_fit_used_for_training=False,
        sampler_contract="uniform shuffled fitting-window cycles truncated to the original fitting sample count each reference interval",
        independent_episode_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    output = d / "baseline_data_quantity54_plan20260913.json"
    assert not output.exists(), "Reuse the frozen plan; never reselect after viewing labels or predictions"
    p = d / "baseline_episode_holdout_plan20260913.json"
    assert sha(p) == PARENT_PLAN_SHA
    assert sha(d / "baseline_gru_capacity_final_verification20260913.json") == CAPACITY_VERIFICATION_SHA
    parent = json.loads(p.read_text())
    batch = json.loads((d / "baseline_dense_gru_capacity_metrics_20260913.json").read_text())
    assert sha(d / "baseline_dense_gru_capacity_metrics_20260913.json") == "5fe983e0b0e0031afcaf58b92bc85d12a750bf4ce4cb5a67845f81f0bbc15011"

    def guard():
        assert subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip() == RUNTIME
        assert subprocess.check_output(["git","branch","--show-current"],text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip()
        assert sha(p) == PARENT_PLAN_SHA and sha(d / "model_sample_index.csv") == parent["sample_index_sha256"]
        assert sha(d / "dataset_manifest.json") == parent["dataset_manifest_sha256"]
        for name,h in batch["runtime_source_sha256"].items():
            assert sha(repo / name) == h
        for name,expected in batch["dataset_files_stat"].items():
            st=(d/name).stat();assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns)==expected
        for run in batch["runs"]:
            out=d/f"models/tuning/{run['model']}_representation_v1/candidate_history/seed{run['seed']}"
            for name,h in run["training"]["files_sha256"].items():
                assert sha(out/name)==h

    guard()
    with (d / "model_sample_index.csv").open(newline="") as stream:
        rows=[{key:row[key] for key in ("sample_index","group_id","split")} for row in csv.DictReader(stream) if row["split"] in ("train","validation")]
    plan=nested_episode_plan(parent["plan"],rows)
    assert [len(plan["groups"][k]) for k in ("fit","unused_parent_fit","heldout","original_validation")]==[54,53,31,30]
    assert len(plan["raw_run_strata"])==21 and plan["parent_fitting_samples_per_reference_interval"]==18095
    assert plan["sample_indices"]["heldout"]==parent["plan"]["sample_indices"]["heldout"]
    assert plan["sample_indices"]["original_validation"]==parent["plan"]["sample_indices"]["original_validation"]
    guard()
    source=subprocess.check_output(["git","show",args.source_commit+":"+RELATIVE])
    record=dict(status="label_blind_nested54_identifiers_frozen_no_labels_preflight_or_training",
        source_commit=args.source_commit,source_sha256=hashlib.sha256(source).hexdigest(),runtime_commit=RUNTIME,
        parent_plan_sha256=PARENT_PLAN_SHA,dataset_manifest_sha256=parent["dataset_manifest_sha256"],
        sample_index_sha256=parent["sample_index_sha256"],plan=plan,
        counts={key:dict(episodes=len(plan["groups"][key]),samples=len(ids)) for key,ids in plan["sample_indices"].items()},
        reference_optimizer_updates=dict(b4=45240,b5=67860),labels_read=False,model_forward=False,model_training=False,
        original_208_split_modified=False,test_evaluated=False,goal_met=False,
        next_required="Implement and verify update-matched two-model training, fit-only preprocessing, and reuse original 107-fit diagnostic caches; do not start old training scripts on this plan.")
    with output.open("x") as stream:
        json.dump(record,stream,indent=2);stream.write("\n")
    print("DATA_QUANTITY54_PLAN",output.stat().st_size,sha(output),flush=True)
    print(json.dumps(record["counts"],sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
