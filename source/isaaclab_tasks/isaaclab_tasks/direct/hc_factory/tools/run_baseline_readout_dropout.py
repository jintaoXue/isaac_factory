#!/usr/bin/env python3
"""Run the registered four readout-dropout controls, then frozen diagnostics.

Training never resumes implicitly. A diagnostic-only continuation reuses
verified training snapshots and completed diagnostic files without retraining.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from preflight_baseline_readout_dropout import sha
from train_dense_baseline_control import TRAINING_FILES


TAG = "readoutdrop20260913"
CASES = [(model, seed) for model in ("B4", "B5") for seed in (42, 43)]


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2); stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--preflight_sha256", required=True)
    parser.add_argument("--phase", choices=("all", "diagnose"), default="all")
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    tools = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
    d = tools.parent / "output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    preflight_path = d / "baseline_readout_dropout_preflight20260913.json"
    assert sha(preflight_path) == args.preflight_sha256
    pf = json.loads(preflight_path.read_text())
    assert pf["status"] == "four_readout_dropout_real_208_preflight_cases_passed"
    assert pf["source_commit"] == args.source_commit and len(pf["checks"]) == 4
    assert not pf["test_evaluated"] and not pf["model_training_launched"]
    sources = {**pf["runtime_source_sha256"], **{str(p.relative_to(repo)): sha(p) for p in
        (Path(__file__).resolve(), tools / "diagnose_baseline_events.py")}}
    final = d / "baseline_dense_readout_dropout_metrics_20260913.json"
    assert not final.exists()

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        assert sha(preflight_path) == args.preflight_sha256
        for name, h in sources.items(): assert sha(repo / name) == h
        for name, value in pf["dataset_files_stat"].items():
            stat = (d / name).stat(); assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value

    def paths(model, seed):
        out = d / f"models/tuning/{model.lower()}_representation_v1/candidate_history/seed{seed}"
        snap = d / f"baseline_readout_dropout_{model.lower()}s{seed}_training20260913.json"
        return out, snap

    def check_archive(out, expected=None):
        archive = out / f"model_before_{TAG}.zip"
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            manifest = json.loads(z.read("archive_manifest.json"))
            assert set(z.namelist()) == set(manifest) | {"archive_manifest.json"}
            if expected is not None: assert manifest == expected
            import hashlib
            for name, h in manifest.items(): assert hashlib.sha256(z.read(name)).hexdigest() == h
            # Prior control records are intentionally retained outside the ZIP.
            prefix = str(out.relative_to(d)) + "/"
            for name, h in pf["current_model_files_sha256"].items():
                if name.startswith(prefix):
                    relative = name[len(prefix):]
                    if relative in TRAINING_FILES: assert manifest[relative] == h
                    else: assert sha(out / relative) == h
        return dict(prior_archive_file=str(archive.relative_to(d)), prior_archive_sha256=sha(archive), prior_archive_manifest=manifest)

    guard()
    if args.phase == "all":
        # Check every destination and every old model before the first archive.
        for model, seed in CASES:
            out, snap = paths(model, seed)
            assert out.is_dir() and not snap.exists()
            assert not (out / f"dense_control_{TAG}.json").exists()
            assert not (out / f"model_before_{TAG}.zip").exists()
        for name, h in pf["current_model_files_sha256"].items(): assert sha(d / name) == h
        for model, seed in CASES:
            guard(); out, snap = paths(model, seed)
            old = {name: sha(out / name) for name in TRAINING_FILES if (out / name).exists()}
            print("START_READOUT_TRAIN", model, seed, flush=True)
            subprocess.run([sys.executable, "-B", "-u", str(tools / "train_dense_baseline_control.py"),
                "--model", model, "--dataset_dir", str(d), "--output_dir", str(out), "--seed", str(seed),
                "--variant", "readout_dropout", "--archive_tag", TAG, "--device", "cuda:0"], check=True)
            guard()
            config = json.loads((out / "config.json").read_text())
            summary = json.loads((out / "run_summary.json").read_text())
            control = json.loads((out / f"dense_control_{TAG}.json").read_text())
            metrics = json.loads((out / "metrics.json").read_text())
            expected = next(row for row in pf["checks"] if row["model"] == model and row["seed"] == seed)
            assert config["model"] == expected["model_config"]
            assert config["training"] == expected["training_config"] and config["loss"] == expected["loss_config"]
            assert config["metadata"]["git_commit"] == args.source_commit
            assert config["metadata"]["input_feature_contract"] == pf["input_feature_contract"]
            assert summary["status"] == control["status"] == "validation_completed"
            assert control["summary"] == summary and control["initialization"] == "from_scratch"
            assert control["variant"] == "readout_dropout" and not control["test_evaluated"]
            assert summary["trainable_parameter_count"] == expected["parameter_count"]
            assert summary["seed"] == seed and 0 < summary["best_epoch"] <= summary["epochs_trained"] <= 60
            assert set(metrics) == {"validation"}
            assert not any((out / name).exists() for name in TRAINING_FILES if "_test." in name)
            files = {name: sha(out / name) for name in TRAINING_FILES + [f"dense_control_{TAG}.json"] if (out / name).exists()}
            record = dict(status="training_completed_and_files_and_prior_archive_verified", model=model, seed=seed,
                source_commit=args.source_commit, preflight_sha256=args.preflight_sha256, files_sha256=files,
                config=config, summary=summary, control=control, metrics=metrics["validation"],
                test_evaluated=False, **check_archive(out, old))
            write_new(snap, record)
            sr = record["metrics"]["station_report"]
            print("READOUT_TRAIN_COMPLETE", model, seed, "best", summary["best_epoch"], "epochs", summary["epochs_trained"],
                  "UP", sr["n_matched_who_upcoming"], "F1", sr["report_f1"], "snapshot", sha(snap), flush=True)

    runs = []
    for model, seed in CASES:
        guard(); out, snap = paths(model, seed)
        training = json.loads(snap.read_text())
        assert training["source_commit"] == args.source_commit and training["preflight_sha256"] == args.preflight_sha256
        assert training["status"] == "training_completed_and_files_and_prior_archive_verified"
        assert not training["test_evaluated"]
        assert check_archive(out, training["prior_archive_manifest"])["prior_archive_sha256"] == training["prior_archive_sha256"]
        results = []
        for checkpoint, split in (("best", "validation"), ("best", "train"), ("last", "validation"), ("last", "train")):
            guard()
            for name, h in training["files_sha256"].items(): assert sha(out / name) == h
            path = d / f"baseline_readout_dropout_{model.lower()}s{seed}_{checkpoint}_{split}20260913.json"
            if not path.exists():
                print("START_READOUT_DIAG", model, seed, checkpoint, split, flush=True)
                subprocess.run([sys.executable, "-B", "-u", str(tools / "diagnose_baseline_events.py"),
                    "--dataset_dir", str(d), "--checkpoint", str(out / (checkpoint + ".pt")), "--output", str(path),
                    "--split", split, "--device", "cuda:0", "--batch_size", "24" if model == "B4" else "16",
                    "--threads", "2", "--source_commit", args.source_commit], check=True)
            result = json.loads(path.read_text())
            assert result["epoch"] == training["summary"]["best_epoch" if checkpoint == "best" else "epochs_trained"]
            assert result["split"] == split and result["sample_count"] == (23859 if split == "train" else 5439)
            assert result["checkpoint_file_sha256"] == training["files_sha256"][checkpoint + ".pt"]
            assert result["checkpoint"] == str(out / (checkpoint + ".pt")) and result["checkpoint_archive_member"] is None
            assert result["diagnostic_source_commit"] == args.source_commit
            assert result["diagnostic_source_sha256"] == sources[str((tools / "diagnose_baseline_events.py").relative_to(repo))]
            assert result["dataset_manifest_sha256"] == pf["manifest_sha256"] and not result["test_evaluated"]
            assert [result["groups"][key]["count"] for key in ("ongoing", "upcoming", "negative")] == (
                [4191, 595, 297152] if split == "train" else [950, 145, 67331])
            canonical = next(row for row in result["thresholds"] if row["threshold"] == result["saved_report_threshold"])
            if checkpoint == "best" and split == "validation":
                original = training["metrics"]["station_report"]
                assert result["saved_report_threshold"] == original["report_threshold_used"] == training["summary"]["event_report_threshold"]
                for key in original:
                    if key.startswith(("n_", "who_", "report_")) and key != "report_threshold_used":
                        assert abs(canonical[key] - original[key]) < 1e-10, key
            results.append(dict(checkpoint=checkpoint, split=split, file=path.name, file_sha256=sha(path), result=result))
            print("READOUT_DIAG_VERIFIED", model, seed, checkpoint, split,
                  "AP", result["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"],
                  "UP", canonical["n_matched_who_upcoming"], flush=True)
        runs.append(dict(model=model.lower(), seed=seed, training_file=snap.name, training_file_sha256=sha(snap),
                         training=training, diagnostics=results))
    guard()
    for run in runs:
        out, snap = paths(run["model"], run["seed"])
        assert sha(snap) == run["training_file_sha256"]
        for name, h in run["training"]["files_sha256"].items(): assert sha(out / name) == h
        for row in run["diagnostics"]: assert sha(d / row["file"]) == row["file_sha256"]
    write_new(final, dict(status="four_training_runs_and_sixteen_frozen_diagnostics_completed_and_verified",
        source_commit=args.source_commit, runtime_source_sha256=sources, preflight_sha256=args.preflight_sha256,
        dataset_manifest_sha256=pf["manifest_sha256"], dataset_files_stat=pf["dataset_files_stat"], runs=runs,
        test_evaluated=False, main_repository_modified=False, goal_met=False))
    print("READOUT_BATCH_COMPLETE", final.stat().st_size, sha(final), flush=True)


if __name__ == "__main__": main()
