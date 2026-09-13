#!/usr/bin/env python3
"""Run the four registered frozen B5 diagnostics in existing server directories."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

RUNTIME = "979c680fb4d1919ae760cdfb0038f69fb7cc6708"
DIAGNOSTIC = "0953fb3f10a53a1e998e3686770c5591bfc2eb97"
DIAGNOSTIC_SHA = "5ac68a4105bc908843da17d1cff812b6d6c06e8158a56c33a4dbf97158fef43d"
PREFLIGHT_SHA = "e5f86076c68359f6887f8875507ca256db9199d09ebe12fe1882602aa27de811"
JOINT_SHA = "c35d4d46a2a45d40f48381662278ea1655c0a1e0a02b8ec053a7018767dcae19"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    out = d / f"models/tuning/b5_representation_v1/candidate_history/seed{args.seed}"
    output = d / f"baseline_vector_gat_b5s{args.seed}_diagnostics20260913.json"
    assert out.is_dir() and not output.exists()
    preflight = d / "baseline_vector_gat_preflight20260913.json"
    joint = d / "baseline_dense_joint_onset_metrics_20260913.json"
    assert sha(preflight) == PREFLIGHT_SHA and sha(joint) == JOINT_SHA
    pf, prior = json.loads(preflight.read_text()), json.loads(joint.read_text())
    snapshot = d / f"baseline_dense_vector_gat_b5s{args.seed}_training20260913.json"
    training_sha = sha(snapshot)
    training = json.loads(snapshot.read_text())
    assert training["status"] == "training_completed_and_current_files_and_prior_archive_verified"
    assert training["source_commit"] == RUNTIME and training["B4_files_unchanged"]
    assert training["config"]["model"]["gat_score_mode"] == "vector_additive"
    assert training["summary"]["trainable_parameter_count"] == 285982
    assert training["test_evaluated"] is False
    server_tests = d / "baseline_vector_gat_ranking_server_tests20260913.json"
    tests = json.loads(server_tests.read_text())
    assert tests["status"] == "four_server_tests_and_full_shape_CUDA_observer_passed"
    assert tests["diagnostic_source_commit"] == DIAGNOSTIC and tests["diagnostic_source_sha256"] == DIAGNOSTIC_SHA
    source_path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/diagnose_baseline_events.py"
    source = subprocess.check_output(["git", "show", DIAGNOSTIC + ":" + source_path])
    assert hashlib.sha256(source).hexdigest() == DIAGNOSTIC_SHA

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        assert sha(snapshot) == training_sha and sha(preflight) == PREFLIGHT_SHA and sha(joint) == JOINT_SHA
        for name, value in pf["runtime_source_sha256"].items():
            assert sha(repo / name) == value
        for name, value in pf["dataset_files_stat"].items():
            stat = (d / name).stat()
            assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value
        for name, value in training["files_sha256"].items():
            assert sha(out / name) == value
        for run in prior["runs"]:
            if run["model"] != "b4":
                continue
            frozen = d / f"models/tuning/b4_representation_v1/candidate_history/seed{run['seed']}"
            for name, value in run["training"]["files_sha256"].items():
                assert sha(frozen / name) == value

    guard()
    archive_path = out / "model_before_vectorgat20260913.zip"
    assert sha(archive_path) == training["prior_archive_sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("archive_manifest.json"))
        assert manifest == training["prior_archive_manifest"]
        assert set(archive.namelist()) == set(manifest) | {"archive_manifest.json"}
        for name, value in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == value
    results = []
    for checkpoint, split in (("best", "validation"), ("best", "train"), ("last", "validation"), ("last", "train")):
        guard()
        path = d / f"baseline_vector_gat_b5s{args.seed}_{checkpoint}_{split}20260913.json"
        if not path.exists():
            print("START", args.seed, checkpoint, split, flush=True)
            subprocess.run([sys.executable, "-B", "-", "--dataset_dir", str(d),
                "--checkpoint", str(out / (checkpoint + ".pt")), "--output", str(path),
                "--split", split, "--device", "cuda:0", "--batch_size", "16", "--threads", "2",
                "--inspect_gat_ranking", "--source_commit", DIAGNOSTIC], input=source, check=True)
        # An existing result is reused only after all provenance/count checks.
        result = json.loads(path.read_text())
        expected_epoch = training["summary"]["best_epoch" if checkpoint == "best" else "epochs_trained"]
        assert result["epoch"] == expected_epoch and result["split"] == split
        assert result["sample_count"] == (23859 if split == "train" else 5439)
        assert result["checkpoint_file_sha256"] == training["files_sha256"][checkpoint + ".pt"]
        assert result["checkpoint_archive_member"] is None
        assert result["checkpoint"] == str(out / (checkpoint + ".pt"))
        assert result["diagnostic_source_commit"] == DIAGNOSTIC and result["diagnostic_source_sha256"] == DIAGNOSTIC_SHA
        assert result["dataset_manifest_sha256"] == pf["manifest_sha256"]
        assert result["test_evaluated"] is False
        assert [result["groups"][k]["count"] for k in ("ongoing", "upcoming", "negative")] == (
            [4191, 595, 297152] if split == "train" else [950, 145, 67331])
        observation = result["gat_ranking_observation"]
        assert observation["version"] == "factory_frozen_gat_common_neighbor_top_v1"
        assert observation["score_mode"] == "vector_additive"
        for layer in observation["layers"].values():
            for name, row in layer["groups"].items():
                assert row["count"] == result["groups"][name]["count"]
        threshold = result["saved_report_threshold"]
        canonical = next(row for row in result["thresholds"] if row["threshold"] == threshold)
        checked = checkpoint == "best" and split == "validation"
        if checked:
            original = training["metrics"]["station_report"]
            assert threshold == original["report_threshold_used"] == training["summary"]["event_report_threshold"]
            keys = [key for key in original if key.startswith(("n_", "who_", "report_")) and key != "report_threshold_used"]
            assert keys
            for key in keys:
                assert abs(canonical[key] - original[key]) < 1e-10, (key, canonical.get(key), original[key])
        guard()
        results.append(dict(checkpoint=checkpoint, split=split, file=path.name, file_sha256=sha(path),
            epoch=result["epoch"], sample_count=result["sample_count"], saved_threshold=threshold,
            original_canonical_score_and_count_keys_checked=checked,
            canonical_comparison_scope="Count and P/R/F1/who/ongoing/upcoming keys; continuous MAE is not asserted bitwise. Saved best threshold is checked separately.", result=result))
        print("VERIFIED", checkpoint, split, "AP", result["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"],
              "UP", canonical["n_matched_who_upcoming"], "GAT", json.dumps({name: row["groups"]["upcoming"] for name, row in observation["layers"].items()}), flush=True)
    guard()
    record = dict(status="four_frozen_vector_gat_diagnostics_completed_and_verified", model="B5", seed=args.seed,
        training_source_commit=RUNTIME, diagnostic_source_commit=DIAGNOSTIC,
        diagnostic_source_sha256=DIAGNOSTIC_SHA, training_snapshot_sha256=training_sha,
        server_tests_sha256=sha(server_tests), diagnostics=results,
        B4_files_unchanged=True, test_evaluated=False)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print("SEED_DIAGNOSTICS_COMPLETE", args.seed, output.stat().st_size, sha(output), flush=True)


if __name__ == "__main__":
    main()
