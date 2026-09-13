#!/usr/bin/env python3
"""Independently verify the completed GRU32 batch using immutable files only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


RUNTIME = "90a41a5b9c44e625f8c80083a8a950cd50f5eb13"
COMPARISON_SOURCE = "182ab933bb0a1be0ee25522811bff2d658fedc03"
RELATIVE = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/verify_baseline_gru_capacity_results.py"
EXPECTED = {
    "baseline_gru_capacity_preflight20260913.json": "4256633509d19c0539fc56055febb8b15d347e553e890e4633dc701387d5e510",
    "baseline_dense_gru_capacity_metrics_20260913.json": "5fe983e0b0e0031afcaf58b92bc85d12a750bf4ce4cb5a67845f81f0bbc15011",
    "baseline_gru_capacity_parent_comparison20260913.json": "f1e7ed4904a03ad38d2dd44ed8f558e94e03e074894c83fd4c1f407c3be2e791",
    "baseline_gru_capacity_comparison_wait_complete20260913.json": "08c40413abde5fea1a232cf892f1586694f25bf7f904e6c2bbbb3c9eb918e3b2",
    "baseline_readout_dropout_parent_comparison20260913.json": "73dbbb0f81f43ac9422e1fb914247113a4936a9983694ea96648a291127b459c",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def stats(result):
    matches = [row for row in result["thresholds"] if row["threshold"] == result["saved_report_threshold"]]
    assert len(matches) == 1
    row = matches[0]
    n = row["n_true_upcoming"]
    hits = n * row["report_recall_upcoming"]
    assert abs(hits - round(hits)) < 1e-8
    assert round(hits) == row["n_matched_who_upcoming"] - row["upcoming_timing_misses"]
    return dict(upcoming_AP=result["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"],
        upcoming_probability_median=result["groups"]["upcoming"]["will_q10_q50_q90"][1],
        upcoming_report_hits=round(hits), upcoming_true=n, upcoming_recall=row["report_recall_upcoming"],
        upcoming_station_hits=row["n_matched_who_upcoming"], upcoming_timing_misses=row["upcoming_timing_misses"],
        precision=row["report_precision"], recall=row["report_recall"], f1=row["report_f1"],
        saved_threshold=row["threshold"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    target = d / "baseline_gru_capacity_final_verification20260913.json"
    assert not target.exists()
    for name, expected in EXPECTED.items():
        assert sha(d / name) == expected, name
    pf = read(d / "baseline_gru_capacity_preflight20260913.json")
    batch = read(d / "baseline_dense_gru_capacity_metrics_20260913.json")
    comparison = read(d / "baseline_gru_capacity_parent_comparison20260913.json")
    parent = read(d / "baseline_readout_dropout_parent_comparison20260913.json")
    complete = read(d / "baseline_gru_capacity_comparison_wait_complete20260913.json")
    assert batch["status"] == "four_training_runs_and_sixteen_frozen_diagnostics_completed_and_verified"
    assert comparison["status"] == "sixteen_matched_cached_near_parent_comparisons_verified"
    assert batch["source_commit"] == comparison["runtime_commit"] == complete["runtime_commit"] == RUNTIME
    assert comparison["source_commit"] == complete["comparison_source_commit"] == COMPARISON_SOURCE
    assert complete["candidate_sha256"] == comparison["candidate_summary_sha256"] == EXPECTED["baseline_dense_gru_capacity_metrics_20260913.json"]
    assert complete["comparison_sha256"] == EXPECTED["baseline_gru_capacity_parent_comparison20260913.json"]
    assert all(not item["test_evaluated"] for item in (batch, comparison, parent, complete, pf))
    assert not comparison["model_forward"] and not comparison["model_training"] and not parent["parent_model_retrained"]
    states = {}
    for name, pid in (("baseline_dense_v6:0.0", "783323"), ("baseline_dense_diag:0.0", "840337")):
        state = subprocess.check_output(["tmux", "display-message", "-p", "-t", name,
            "#{pane_pid}|#{pane_dead}|#{pane_dead_status}"], text=True).strip()
        actual_pid, dead, exit_status = state.split("|")
        assert actual_pid == pid and dead == "1"
        if exit_status != "0":
            assert exit_status == ""
            fields = (Path("/proc") / pid / "stat").read_text().rsplit(")", 1)[1].split()
            assert fields[0] == "Z" and fields[49] == "0"
        states[name] = state

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, expected in EXPECTED.items():
            assert sha(d / name) == expected
        assert batch["runtime_source_sha256"] == pf["runtime_source_sha256"]
        for name, expected in batch["runtime_source_sha256"].items():
            assert sha(repo / name) == expected
        assert batch["dataset_files_stat"] == pf["dataset_files_stat"]
        for name, expected in batch["dataset_files_stat"].items():
            st = (d / name).stat()
            assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == expected
        assert sha(d / "dataset_manifest.json") == batch["dataset_manifest_sha256"] == pf["manifest_sha256"]

    guard()
    cases = {(m, s) for m in ("b4", "b5") for s in (42, 43)}
    assert len(batch["runs"]) == 4 and {(r["model"], r["seed"]) for r in batch["runs"]} == cases
    parent_views = {v["file"]: v for v in parent["parent_diagnostics"]}
    assert len(parent_views) == 16
    rows = {(v["model"], v["seed"], v["checkpoint"], v["split"]): v for v in comparison["comparisons"]}
    assert len(comparison["comparisons"]) == len(rows) == 16
    verified_views, runs, files, old_members = [], [], {}, 0
    for run in batch["runs"]:
        model, seed = run["model"], run["seed"]
        out = d / f"models/tuning/{model}_representation_v1/candidate_history/seed{seed}"
        training = run["training"]
        assert read(d / run["training_file"]) == training
        assert sha(d / run["training_file"]) == run["training_file_sha256"]
        expected = next(x for x in pf["checks"] if x["model"].lower() == model and x["seed"] == seed)
        config, summary, control = read(out / "config.json"), read(out / "run_summary.json"), read(out / "dense_control_grucapacity20260913.json")
        assert config == training["config"] and summary == training["summary"] and control == training["control"]
        assert config["model"] == expected["model_config"] and config["model"]["gru_hidden"] == 32
        assert config["training"] == expected["training_config"] and config["loss"] == expected["loss_config"]
        assert config["metadata"]["git_commit"] == RUNTIME and config["metadata"]["input_feature_contract"] == pf["input_feature_contract"]
        assert summary["trainable_parameter_count"] == expected["parameter_count"]
        assert summary["status"] == control["status"] == "validation_completed" and control["summary"] == summary
        assert control["initialization"] == "from_scratch" and control["variant"] == "gru_capacity32" and not control["test_evaluated"]
        metrics = read(out / "metrics.json")
        assert set(metrics) == {"validation"} and metrics["validation"] == training["metrics"]
        history = list(csv.DictReader((out / "history.csv").read_text().splitlines()))
        assert [int(h["epoch"]) for h in history] == list(range(1, summary["epochs_trained"] + 1))
        assert 0 < summary["best_epoch"] <= summary["epochs_trained"] <= 60
        assert len(training["files_sha256"]) == 12
        for name, expected_hash in training["files_sha256"].items():
            assert sha(out / name) == expected_hash
            files[str((out / name).relative_to(d))] = expected_hash
        archive = d / training["prior_archive_file"]
        assert sha(archive) == training["prior_archive_sha256"]
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read("archive_manifest.json"))
            assert manifest == training["prior_archive_manifest"] and len(manifest) == 11
            assert set(z.namelist()) == set(manifest) | {"archive_manifest.json"} and z.testzip() is None
            for name, expected_hash in manifest.items():
                assert hashlib.sha256(z.read(name)).hexdigest() == expected_hash
                assert pf["current_model_files_sha256"][str((out / name).relative_to(d))] == expected_hash
                old_members += 1
            prefix = str(out.relative_to(d)) + "/"
            for name, expected_hash in pf["current_model_files_sha256"].items():
                if name.startswith(prefix) and name[len(prefix):] not in manifest:
                    assert sha(d / name) == expected_hash
        assert len(run["diagnostics"]) == 4
        for view in run["diagnostics"]:
            checkpoint, split = view["checkpoint"], view["split"]
            key = model, seed, checkpoint, split
            record = rows[key]
            result = read(d / view["file"])
            assert result == view["result"] and sha(d / view["file"]) == view["file_sha256"] == record["candidate_file_sha256"]
            assert record["candidate_file"] == view["file"]
            assert result["epoch"] == summary["best_epoch" if checkpoint == "best" else "epochs_trained"]
            assert result["checkpoint_file_sha256"] == training["files_sha256"][checkpoint + ".pt"]
            assert result["checkpoint"] == str(out / (checkpoint + ".pt")) and result["checkpoint_archive_member"] is None
            assert result["diagnostic_source_commit"] == RUNTIME
            if model == "b5" and checkpoint == "last":
                expected_parent = f"baseline_near_parent_b5s{seed}_last_{split}20260913.json"
            else:
                tag = "headprobe20260912" if checkpoint == "best" else "nearpreclast20260912"
                expected_parent = f"{model}_seed{seed}_{split}_diagnostics_{tag}.json"
            assert record["parent_file"] == expected_parent
            original = parent_views[record["parent_file"]]
            assert original["file_sha256"] == record["parent_file_sha256"] == sha(d / original["file"])
            before = read(d / original["file"])
            assert before == original["result"]
            for item in (before, result):
                assert item["split"] == split and not item["test_evaluated"]
                assert item["dataset_manifest_sha256"] == pf["manifest_sha256"]
                assert item["sample_count"] == (23859 if split == "train" else 5439)
                assert [item["groups"][g]["count"] for g in ("ongoing", "upcoming", "negative")] == ([4191,595,297152] if split == "train" else [950,145,67331])
            for side, item in (("parent", before), ("candidate", result)):
                assert stats(item) == record[side]
            for metric, delta in record["candidate_minus_parent"].items():
                assert record["candidate"][metric] - record["parent"][metric] == delta
            if checkpoint == "best" and split == "validation":
                canonical = next(v for v in result["thresholds"] if v["threshold"] == result["saved_report_threshold"])
                report = training["metrics"]["station_report"]
                assert report["report_threshold_used"] == result["saved_report_threshold"] == summary["event_report_threshold"]
                for name, value in report.items():
                    if name.startswith(("n_", "who_", "report_")) and name != "report_threshold_used":
                        assert abs(canonical[name] - value) < 1e-10
            verified_views.append(record)
        report = training["metrics"]["station_report"]
        runs.append(dict(model=model, seed=seed, best_epoch=summary["best_epoch"], epochs_trained=summary["epochs_trained"],
            parameter_count=summary["trainable_parameter_count"], formal_report=report,
            training_file=run["training_file"], training_file_sha256=run["training_file_sha256"]))
    assert len(files) == 48 and old_members == 44
    assert {(v["model"],v["seed"],v["checkpoint"],v["split"]) for v in verified_views} == {(m,s,c,v) for m,s in cases for c in ("best","last") for v in ("train","validation")}
    for archive in pf["parent_archives"]:
        assert sha(d / archive["file"]) == archive["sha256"]
        with zipfile.ZipFile(d / archive["file"]) as z:
            for name, expected_hash in archive["verified_parent_files"].items():
                assert hashlib.sha256(z.read(name)).hexdigest() == expected_hash
    guard()
    source = subprocess.check_output(["git", "show", args.source_commit + ":" + RELATIVE])
    output = dict(status="four_training_sixteen_candidate_and_sixteen_parent_views_archives_and_exit_states_independently_verified",
        source_commit=args.source_commit, source_sha256=hashlib.sha256(source).hexdigest(), runtime_commit=RUNTIME,
        pane_exit_states=states, artifacts_sha256=EXPECTED, runs=runs, comparisons=verified_views,
        current_model_files_verified=len(files), prior_archive_members_verified=old_members,
        parent_archives_verified=len(pf["parent_archives"]), runtime_files_verified=len(batch["runtime_source_sha256"]),
        dataset_stat_verified=len(batch["dataset_files_stat"]), model_forward=False, model_training=False,
        new_threshold_selection=False, test_evaluated=False, goal_met=False)
    with target.open("x") as stream:
        json.dump(output, stream, indent=2); stream.write("\n")
    print("GRU_CAPACITY_FINAL_INDEPENDENT_VERIFICATION", target.stat().st_size, sha(target), flush=True)
    for run in runs:
        report = run["formal_report"]
        print("FORMAL_VERIFIED", run["model"], run["seed"], run["best_epoch"], run["epochs_trained"],
              "UP", round(report["report_recall_upcoming"] * report["n_true_upcoming"]), flush=True)


if __name__ == "__main__":
    main()
