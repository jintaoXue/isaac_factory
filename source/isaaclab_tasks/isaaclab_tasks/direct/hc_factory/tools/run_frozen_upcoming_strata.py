#!/usr/bin/env python3
"""Read existing negative-ablation checkpoints for new retrospective strata only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RUNTIME = "979c680fb4d1919ae760cdfb0038f69fb7cc6708"
REFERENCES = {
    "baseline_dense_vector_gat_metrics_20260913.json": "e90919aeaae88f5bb924324bfc7d36451957f028f7dec0c554ba2e9b7f3285ae",
    "baseline_dense_joint_onset_metrics_20260913.json": "c35d4d46a2a45d40f48381662278ea1655c0a1e0a02b8ec053a7018767dcae19",
    "baseline_upcoming_schedule_final_verification20260913.json": "17bb7349324fe5747736a89c0d9bc84eac542206808b7cae53bba1643c0c1612",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_canonical(actual, expected):
    """Include structured false-alarm counts; keep continuous timing MAE out of this check."""
    keys = [key for key in expected if key.startswith(("n_", "who_", "report_")) and key != "report_threshold_used"]
    assert keys
    for key in keys:
        if isinstance(expected[key], (int, float)):
            assert abs(actual[key] - expected[key]) < 1e-10, (key, actual[key], expected[key])
        else:
            assert actual[key] == expected[key], (key, actual[key], expected[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--driver_commit", required=True)
    parser.add_argument("--model", choices=("b4", "b5"), required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    driver_source = subprocess.check_output(["git", "show", args.driver_commit + ":source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/run_frozen_upcoming_strata.py"])
    assert Path(__file__).read_bytes() == driver_source
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    output = d / f"baseline_schedule_strata_{args.model}_complete20260913.json"
    assert d.is_dir() and not output.exists()
    references = {}
    for name, h in REFERENCES.items():
        assert sha(d / name) == h
        references[name] = json.loads((d / name).read_text())
    vector = references["baseline_dense_vector_gat_metrics_20260913.json"]
    joint = references["baseline_dense_joint_onset_metrics_20260913.json"]
    audit = references["baseline_upcoming_schedule_final_verification20260913.json"]
    all_runs = vector["runs"] + [row for row in joint["runs"] if row["model"] == "b4"]
    runs = sorted([row for row in all_runs if row["model"] == args.model], key=lambda row: row["seed"])
    assert [row["seed"] for row in runs] == [42, 43]
    source_path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/diagnose_baseline_events.py"
    source = subprocess.check_output(["git", "show", args.source_commit + ":" + source_path])
    source_hash = hashlib.sha256(source).hexdigest()
    test_path = d / "baseline_schedule_strata_server_tests20260913.json"
    tests = json.loads(test_path.read_text())
    assert tests["source_commit"] == args.source_commit and tests["diagnostic_source_sha256"] == source_hash
    assert tests["tests_passed"] == 3 and not tests["test_evaluated"]
    driver_tests_path = d / "baseline_schedule_strata_resume_server_tests20260913.json"
    driver_tests = json.loads(driver_tests_path.read_text())
    assert driver_tests["driver_source_commit"] == args.driver_commit
    assert driver_tests["driver_source_sha256"] == hashlib.sha256(driver_source).hexdigest()
    assert driver_tests["tests_passed"] == 1 and not driver_tests["test_evaluated"]

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, h in REFERENCES.items():
            assert sha(d / name) == h
        for name, value in audit["dataset_files_stat"].items():
            stat = (d / name).stat()
            assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value
        for name, h in audit["current_model_files_sha256"].items():
            assert sha(d / name) == h

    guard()
    results = []
    for run in runs:
        out = d / f"models/tuning/{args.model}_representation_v1/candidate_history/seed{run['seed']}"
        for checkpoint, split in (("best", "validation"), ("best", "train"), ("last", "validation"), ("last", "train")):
            guard()
            original = next(row for row in run["diagnostics"] if row["checkpoint"] == checkpoint and row["split"] == split)
            assert sha(d / original["file"]) == original["file_sha256"]
            reference = original["result"]
            assert json.loads((d / original["file"]).read_text()) == reference
            threshold = reference["saved_report_threshold"]
            path = d / f"baseline_schedule_strata_{args.model}s{run['seed']}_{checkpoint}_{split}20260913.json"
            # Match each completed diagnostic's device and batch size for numerical comparability.
            device, batch = ("cpu", 32) if args.model == "b4" else ("cuda:0", 16)
            reused = path.exists()
            if not path.exists():
                print("START_STRATA", args.model, run["seed"], checkpoint, split, device, batch, flush=True)
                subprocess.run([sys.executable, "-B", "-", "--dataset_dir", str(d), "--checkpoint", str(out / (checkpoint + ".pt")),
                    "--output", str(path), "--split", split, "--device", device, "--batch_size", str(batch), "--threads", "2",
                    "--thresholds", str(threshold), "--source_commit", args.source_commit, "--inspect_schedule_strata"], input=source, check=True)
            report = json.loads(path.read_text())
            assert report["diagnostic_source_commit"] == args.source_commit and report["diagnostic_source_sha256"] == source_hash
            for key in ("epoch", "sample_count", "checkpoint_file_sha256", "checkpoint", "checkpoint_archive_member", "dataset_manifest_sha256", "saved_report_threshold", "split"):
                assert report[key] == reference[key], (key, report[key], reference[key])
            assert not report["test_evaluated"]
            actual = next(row for row in report["thresholds"] if row["threshold"] == threshold)
            expected = next(row for row in reference["thresholds"] if row["threshold"] == threshold)
            verify_canonical(actual, expected)
            for name in ("ongoing", "upcoming", "negative"):
                assert report["groups"][name]["count"] == reference["groups"][name]["count"]
            old_ap = reference["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"]
            new_ap = report["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"]
            assert abs(old_ap - new_ap) <= 1e-8, (old_ap, new_ap)
            strata = report["schedule_strata"]
            assert strata["summary"]["window_targets"] == (595 if split == "train" else 145)
            assert strata["summary"]["unique_onsets"] == (299 if split == "train" else 73)
            guard()
            results.append(dict(seed=run["seed"], checkpoint=checkpoint, split=split, file=path.name, file_sha256=sha(path),
                reused_existing_output=reused,
                original_file=original["file"], original_file_sha256=original["file_sha256"], original_scores_counts_and_AP_reproduced=True,
                checkpoint_file_sha256=report["checkpoint_file_sha256"], saved_threshold=threshold,
                overall_upcoming_AP=new_ap, summary=strata["summary"], groups=strata["groups"],
                compatibility_upcoming_vs_negative=strata["compatibility_upcoming_vs_negative"]))
            print("STRATA_VERIFIED", args.model, run["seed"], checkpoint, split, json.dumps(strata["groups"]), flush=True)
    guard()
    record = dict(status="eight_frozen_schedule_strata_diagnostics_completed_and_verified", model=args.model,
        source_commit=args.source_commit, diagnostic_source_sha256=source_hash, runtime_source_commit=RUNTIME,
        driver_source_commit=args.driver_commit, driver_source_sha256=hashlib.sha256(driver_source).hexdigest(),
        references_sha256=REFERENCES, server_tests_sha256=sha(test_path), runs=results,
        driver_tests_sha256=sha(driver_tests_path),
        current_model_files_unchanged=28, dataset_files_stat_unchanged=6,
        model_training=False, test_evaluated=False, main_repository_modified=False, goal_met=False)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print("MODEL_STRATA_COMPLETE", args.model, output.stat().st_size, sha(output), flush=True)


if __name__ == "__main__":
    main()
