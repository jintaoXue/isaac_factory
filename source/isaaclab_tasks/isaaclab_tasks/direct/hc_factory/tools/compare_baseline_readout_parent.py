#!/usr/bin/env python3
"""Reuse twelve near-parent diagnostics; fill only four missing B5 last views."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


RUNTIME = "ee838f59d2bdf2893a35ec00acae1308f9dd0e07"
NEAR_SHA = "e71c84ded25a2b5fe861b45ecc12e2a0941193043a526654a9d8327a9d7a4b27"
PREFLIGHT_SHA = "a357eb599f05e5e3b715ab547c6f1d1ae3ce90ff02305e437d36b39bed3c0f9a"
TOOL_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def validate_diagnostic(result, run, checkpoint, split, manifest, archive_sha=None):
    """Old loose files and new archive members have distinct provenance fields."""
    assert result["split"] == split and not result["test_evaluated"]
    assert result["sample_count"] == (23859 if split == "train" else 5439)
    assert result["dataset_manifest_sha256"] == manifest
    assert result["epoch"] == run["record"]["summary"]["best_epoch" if checkpoint == "best" else "epochs_trained"]
    member_sha = run["files_sha256"][checkpoint + ".pt"]
    if archive_sha is None:
        assert result["checkpoint_archive_member"] is None
        assert result["checkpoint_file_sha256"] == member_sha
    else:
        assert result["checkpoint_archive_member"] == checkpoint + ".pt"
        assert result["checkpoint_file_sha256"] == archive_sha
        assert result["checkpoint_member_sha256"] == member_sha
    assert [result["groups"][k]["count"] for k in ("ongoing", "upcoming", "negative")] == (
        [4191, 595, 297152] if split == "train" else [950, 145, 67331])
    canonical = next(r for r in result["thresholds"] if r["threshold"] == result["saved_report_threshold"])
    if checkpoint == "best" and split == "validation":
        original = run["metrics"]["station_report"]
        assert result["saved_report_threshold"] == original["report_threshold_used"]
        for key in original:
            if key.startswith(("n_", "who_", "report_")) and key != "report_threshold_used":
                assert abs(canonical[key] - original[key]) < 1e-10, key
    return result["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    tools = repo / TOOL_PATH; d = tools.parent / "output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    final = d / "baseline_readout_dropout_parent_comparison20260913.json"; assert not final.exists()
    near_path = d / "baseline_dense_near_metrics_20260912.json"
    pf_path = d / "baseline_readout_dropout_preflight20260913.json"
    assert sha(near_path) == NEAR_SHA and sha(pf_path) == PREFLIGHT_SHA
    near, pf = json.loads(near_path.read_text()), json.loads(pf_path.read_text())
    driver = subprocess.check_output(["git", "show", args.source_commit + ":" + TOOL_PATH + Path("compare_baseline_readout_parent.py").name])
    diagnostic_path = tools / "diagnose_baseline_events.py"
    diagnostic_sha = hashlib.sha256(subprocess.check_output(["git", "show", RUNTIME + ":" + TOOL_PATH + diagnostic_path.name])).hexdigest()
    parent_results, existing_sha = {}, {}
    for run in near["runs"]:
        model, seed = run["model"], run["seed"]
        for checkpoint in ("best", "last"):
            if model == "b5" and checkpoint == "last": continue
            for split in ("train", "validation"):
                tag = "headprobe20260912" if checkpoint == "best" else "nearpreclast20260912"
                p = d / f"{model}_seed{seed}_{split}_diagnostics_{tag}.json"
                result = json.loads(p.read_text())
                validate_diagnostic(result, run, checkpoint, split, pf["manifest_sha256"])
                existing_sha[p.name] = sha(p)
                parent_results[model, seed, checkpoint, split] = dict(file=p.name, file_sha256=sha(p), reused=True, result=result)
    assert len(existing_sha) == 12

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        assert sha(diagnostic_path) == diagnostic_sha
        for name, h in existing_sha.items(): assert sha(d / name) == h
        for name, value in pf["dataset_files_stat"].items():
            st = (d / name).stat(); assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == value

    guard()
    print("REUSED_TWELVE_PARENT_DIAGNOSTICS_WAITING_FOR_EXISTING_BATCH_610573", flush=True)
    deadline = time.monotonic() + 4 * 3600
    while True:
        pid, dead, status = subprocess.check_output(["tmux", "display-message", "-p", "-t", "baseline_dense_v6:0.0", "#{pane_pid}|#{pane_dead}|#{pane_dead_status}"], text=True).strip().split("|")
        assert pid == "610573", "Do not follow an unrelated replacement job"
        if dead == "1":
            assert status == "0", "Existing batch failed; never restart training here"
            break
        assert dead == "0" and time.monotonic() < deadline
        time.sleep(15)
    guard()
    candidate_path = d / "baseline_dense_readout_dropout_metrics_20260913.json"
    candidate = json.loads(candidate_path.read_text()); candidate_sha = sha(candidate_path)
    assert candidate["status"] == "four_training_runs_and_sixteen_frozen_diagnostics_completed_and_verified"
    assert candidate["source_commit"] == RUNTIME and not candidate["test_evaluated"]
    assert len(candidate["runs"]) == 4
    for run in near["runs"]:
        if run["model"] != "b5": continue
        seed = run["seed"]; archive = d / f"models/tuning/b5_representation_v1/candidate_history/seed{seed}/model_before_onsetaux20260912.zip"
        archive_sha = sha(archive)
        anchor = next(r for r in pf["parent_archives"] if r["model"] == "B5" and r["seed"] == seed)
        assert archive_sha == anchor["sha256"]
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name, h in run["files_sha256"].items(): assert hashlib.sha256(z.read(name)).hexdigest() == h
        for split in ("validation", "train"):
            guard(); p = d / f"baseline_near_parent_b5s{seed}_last_{split}20260913.json"
            reused = p.exists()
            if not reused:
                print("FILL_MISSING_PARENT_DIAG", seed, split, flush=True)
                subprocess.run([sys.executable, "-B", "-u", str(diagnostic_path), "--dataset_dir", str(d),
                    "--checkpoint", str(archive), "--archive_member", "last.pt", "--output", str(p), "--split", split,
                    "--device", "cuda:0", "--batch_size", "16", "--threads", "2", "--source_commit", RUNTIME], check=True)
            result = json.loads(p.read_text())
            ap = validate_diagnostic(result, run, "last", split, pf["manifest_sha256"], archive_sha)
            assert result["diagnostic_source_commit"] == RUNTIME and result["diagnostic_source_sha256"] == diagnostic_sha
            assert sha(archive) == archive_sha
            parent_results["b5", seed, "last", split] = dict(file=p.name, file_sha256=sha(p), reused=reused, result=result)
            print("PARENT_DIAG_VERIFIED", seed, split, "AP", ap, flush=True)
    comparisons = []
    for run in candidate["runs"]:
        model, seed = run["model"], run["seed"]
        parent = next(r for r in near["runs"] if r["model"] == model and r["seed"] == seed)
        out = d / f"models/tuning/{model}_representation_v1/candidate_history/seed{seed}"
        for name, h in run["training"]["files_sha256"].items(): assert sha(out / name) == h
        aps = []
        for row in run["diagnostics"]:
            assert sha(d / row["file"]) == row["file_sha256"]
            view = parent_results[model, seed, row["checkpoint"], row["split"]]
            assert sha(d / view["file"]) == view["file_sha256"]
            aps.append(dict(checkpoint=row["checkpoint"], split=row["split"],
                parent_AP=view["result"]["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"],
                candidate_AP=row["result"]["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"]))
        comparisons.append(dict(model=model, seed=seed, parent_metrics=parent["metrics"]["station_report"],
            candidate_metrics=run["training"]["metrics"]["station_report"], ranking_AP=aps))
    guard(); assert sha(candidate_path) == candidate_sha and len(parent_results) == 16
    record = dict(status="matched_near_parent_comparison_complete_twelve_old_views_reused_four_missing_views_filled",
        runtime_commit=RUNTIME, source_commit=args.source_commit, source_sha256=hashlib.sha256(driver).hexdigest(),
        diagnostic_source_sha256=diagnostic_sha, parent_near_sha256=NEAR_SHA, candidate_result_sha256=candidate_sha,
        parent_diagnostics=list(parent_results.values()), comparisons=comparisons,
        parent_model_retrained=False, test_evaluated=False, goal_met=False)
    with final.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    print("PARENT_COMPARISON_COMPLETE", final.stat().st_size, sha(final), flush=True)


if __name__ == "__main__": main()
