#!/usr/bin/env python3
"""Continue the already running vector-GAT batch; never restart training."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

RUNTIME = "979c680fb4d1919ae760cdfb0038f69fb7cc6708"
DIAGNOSTIC = "0953fb3f10a53a1e998e3686770c5591bfc2eb97"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    repo = Path.cwd()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    final = d / "baseline_dense_vector_gat_metrics_20260913.json"
    driver = d / "baseline_vector_gat_frozen_driver20260913.py"
    assert not final.exists()
    assert sha(driver) == "9e8fd5c37e51476de8d16851d01c1522d0e6ce4ff735d5db76f54fd091833d37"
    preflight_path = d / "baseline_vector_gat_preflight20260913.json"
    joint_path = d / "baseline_dense_joint_onset_metrics_20260913.json"
    assert sha(preflight_path) == "e5f86076c68359f6887f8875507ca256db9199d09ebe12fe1882602aa27de811"
    assert sha(joint_path) == "c35d4d46a2a45d40f48381662278ea1655c0a1e0a02b8ec053a7018767dcae19"
    pf, joint = json.loads(preflight_path.read_text()), json.loads(joint_path.read_text())
    assert sha(d / "baseline_vector_gat_b5s42_diagnostics20260913.json") == "02c92f54bf9361efdf0a59ce1277c075fdbc1704f44c9d50658f5c7f4385b256"
    assert sha(d / "baseline_vector_gat_b5s42_verification20260913.json") == "ee18d021d94092f5d3c9c15de8f9f1d7e7b023a3d152d6d58023f0f144c5dc6f"

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, h in pf["runtime_source_sha256"].items():
            assert sha(repo / name) == h
        for name, s in pf["dataset_files_stat"].items():
            st = (d / name).stat()
            assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == s

    guard()
    training_path = d / "baseline_dense_vector_gat_training20260913.json"
    print("WAITING_FOR_EXISTING_SEED43_TRAINING_NO_RESTART", flush=True)
    deadline = time.monotonic() + 7200
    while True:
        guard()
        dead, exit_code = subprocess.check_output(["tmux", "display-message", "-p", "-t", "baseline_dense_v6", "#{pane_dead}:#{pane_dead_status}"], text=True).strip().split(":")
        if dead == "1":
            assert exit_code == "0" and training_path.exists(), "Existing training did not finish successfully"
            break
        assert dead == "0" and time.monotonic() < deadline, "Training wait exceeded its fixed two-hour bound"
        time.sleep(15)
    training_aggregate = json.loads(training_path.read_text())
    assert training_aggregate["status"] == "two_B5_training_runs_completed"
    assert training_aggregate["source_commit"] == RUNTIME and not training_aggregate["test_evaluated"]
    assert {(r["model"], r["seed"]) for r in training_aggregate["runs"]} == {("B5", 42), ("B5", 43)}
    assert not (d / "baseline_vector_gat_b5s43_diagnostics20260913.json").exists()
    print("SEED43_TRAINING_TERMINAL_START_FOUR_FROZEN_DIAGNOSTICS", flush=True)
    subprocess.run([sys.executable, "-B", str(driver), "--seed", "43"], check=True)
    guard()
    runs = []
    for seed in (42, 43):
        tpath = d / f"baseline_dense_vector_gat_b5s{seed}_training20260913.json"
        dpath = d / f"baseline_vector_gat_b5s{seed}_diagnostics20260913.json"
        training, diagnostic = json.loads(tpath.read_text()), json.loads(dpath.read_text())
        assert training["source_commit"] == RUNTIME and training["test_evaluated"] is False
        assert diagnostic["status"] == "four_frozen_vector_gat_diagnostics_completed_and_verified"
        assert diagnostic["training_snapshot_sha256"] == sha(tpath)
        assert diagnostic["diagnostic_source_commit"] == DIAGNOSTIC and not diagnostic["test_evaluated"]
        registered = next(row for row in training_aggregate["runs"] if row["seed"] == seed)
        assert registered["snapshot_sha256"] == sha(tpath)
        out = d / f"models/tuning/b5_representation_v1/candidate_history/seed{seed}"
        for name, h in training["files_sha256"].items():
            assert sha(out / name) == h
        assert len(diagnostic["diagnostics"]) == 4
        assert {(row["checkpoint"], row["split"]) for row in diagnostic["diagnostics"]} == {
            (checkpoint, split) for checkpoint in ("best", "last") for split in ("train", "validation")}
        for row in diagnostic["diagnostics"]:
            path = d / row["file"]
            assert sha(path) == row["file_sha256"] and json.loads(path.read_text()) == row["result"]
            assert row["result"]["checkpoint_file_sha256"] == training["files_sha256"][row["checkpoint"] + ".pt"]
            assert row["result"]["epoch"] == training["summary"]["best_epoch" if row["checkpoint"] == "best" else "epochs_trained"]
        archive_path = out / "model_before_vectorgat20260913.zip"
        assert sha(archive_path) == training["prior_archive_sha256"]
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.testzip() is None
            manifest = json.loads(archive.read("archive_manifest.json"))
            assert manifest == training["prior_archive_manifest"]
            assert set(archive.namelist()) == set(manifest) | {"archive_manifest.json"}
            for name, h in manifest.items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == h
        runs.append(dict(model="b5", seed=seed, training_file=tpath.name, training_file_sha256=sha(tpath),
            training=training, diagnostic_file=dpath.name, diagnostic_file_sha256=sha(dpath),
            diagnostics=diagnostic["diagnostics"]))
    for run in joint["runs"]:
        if run["model"] == "b4":
            out = d / f"models/tuning/b4_representation_v1/candidate_history/seed{run['seed']}"
            for name, h in run["training"]["files_sha256"].items():
                assert sha(out / name) == h
    print("VERIFYING_SIX_FROZEN_DATA_FULL_HASHES", flush=True)
    for name, h in pf["dataset_files_sha256"].items():
        assert sha(d / name) == h
    guard()
    record = dict(status="two_training_runs_and_eight_frozen_diagnostics_completed_and_verified",
        training_source_commit=RUNTIME, diagnostic_source_commit=DIAGNOSTIC,
        preflight_sha256=sha(preflight_path), training_aggregate_sha256=sha(training_path),
        dataset_manifest_sha256=pf["manifest_sha256"], dataset_files_stat=pf["dataset_files_stat"],
        dataset_files_sha256=pf["dataset_files_sha256"], runs=runs,
        B4_files_unchanged=True, training_pane_exit_code=0,
        diagnostic_subprocess_exit_codes=[0, 0], diagnostic_pane_exit_code=None,
        diagnostic_pane_note="Final independent check of this pane's exit status must be done after the driver exits.",
        diagnostics_completed=True, test_evaluated=False, main_repository_modified=False, goal_met=False)
    with final.open("x") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print("VECTOR_BATCH_COMPLETE", final.stat().st_size, sha(final), flush=True)


if __name__ == "__main__":
    main()
