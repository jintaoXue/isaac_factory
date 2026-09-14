#!/usr/bin/env python3
"""Complete only B2/B3 Start5/10/15 in existing directories, preserving B4/B5."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import default_collate

from factory_baselines import protocol_20260913 as protocol
from factory_baselines.artifacts import archive_files
from factory_baselines.b2_xgboost import B2XGBoostConfig, train_b2_xgboost, _event_training_data, _require_xgboost
from factory_baselines.b3_lstm import B3Lstm, B3ModelConfig
from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.torch_losses import MultiTaskLossConfig, compute_multitask_loss
from factory_baselines.torch_trainer import TorchTrainConfig, train_torch_baseline, _model_inputs, _move_batch
from preflight_baseline_matched_protocol import SOURCE_MANIFEST, sha
from train_dense_baseline_control import TRAINING_FILES
from verify_baseline_matched_results import station_counts, verify_prediction_rows, selected_epoch, close

ROOT = Path("/home/sci/work/BSTAN_isaac_factory")
REL = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory"
DATASET = ROOT / REL / "output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
TAG = "remaining_matched20260914"
PLAN = TAG + "_plan.json"
FINAL = TAG + "_results.json"
B45_FINAL = "baseline_matched_protocol_training20260913_final_verification.json"
TASKS = [(model, cap) for cap in (5, 10, 15) for model in ("B2", "B3")]
HEADS = ("cause", "remain_len", "remain_hot", "remain_score", "event_will", "event_start", "event_duration")
FILES = list(dict.fromkeys(TRAINING_FILES + ["initial.pt"] + [name + ".json" for name in HEADS]))
DEFAULT_OUTPUTS = {
    "B2": "models/tuning/b2_v5_confirmation_v1/candidate_c5_event_w12/seed42",
    "B3": "models/tuning/b3_v5_confirmation_v1/candidate_c0_incumbent/seed42",
}

def write(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

def require_runtime(source):
    if Path.cwd().resolve() != ROOT or git("branch", "--show-current") != "dev_xwt":
        raise ValueError("Use only the authorized server baseline checkout on dev_xwt")
    if git("rev-parse", "HEAD") != source or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Runtime changed or tracked files are dirty; do not train")
    if sha(DATASET / "dataset_manifest.json") != SOURCE_MANIFEST:
        raise ValueError("This queue is only for the fixed 208-episode dataset")

def b2_config(cap):
    return B2XGBoostConfig(
        training_profile=f"{TAG}_b2_start{cap}", evaluate_test=False, evaluate_train=True,
        evaluation_protocol=protocol.VERSION, event_max_start_windows=cap,
        seed=42, n_jobs=4, near_remain_windows=20, event_will_scale_pos_weight=12.,
        report_threshold_sweep=protocol.THRESHOLDS,
    )

def b3_config(cap, device, causes):
    training = TorchTrainConfig(
        training_profile=f"{TAG}_b3_start{cap}", evaluate_test=False, evaluate_train=True,
        evaluation_protocol=protocol.VERSION, event_max_start_windows=cap, seed=42,
        device=device, batch_size=32, learning_rate=3e-4, weight_decay=.001,
        max_epochs=100, patience=40, min_epochs=12, event_oversample_factor=4.,
        report_threshold_sweep=protocol.THRESHOLDS,
    )
    model = dict(lstm_hidden=128, lstm_layers=1, node_hidden=128, node_embedding=32, dropout=.25)
    loss = MultiTaskLossConfig(
        near_remain_windows=20, event_partition="history", lambda_remain_len=.5,
        event_will_upcoming_pos_weight={5: 9., 10: 10., 15: 11.}[cap],
        event_will_fp_weight=2.5, remain_progress_weight_floor=.25,
        remain_progress_weight_power=1.5,
        cause_ignored_ids=tuple(i for i, name in enumerate(causes) if name not in protocol.CAUSE_CLASSES),
    )
    return training, model, loss

def completed_b45():
    data = json.loads((DATASET / B45_FINAL).read_text())
    expected = [(m, cap) for cap in (5, 10, 15) for m in ("B4", "B5")]
    if (data["status"] != "six_matched_tasks_independently_verified" or data["test_evaluated"]
            or [(r["model"], r["max_start"]) for r in data["results"]] != expected):
        raise ValueError("B4/B5 must have six completed verified results")
    batch = DATASET / "baseline_matched_protocol_training20260913_results.json"
    if sha(batch) != data["batch_result_sha256"]:
        raise ValueError("The completed B4/B5 batch result changed")
    return sha(DATASET / B45_FINAL)

def check_stats(stats):
    for name, expected in stats.items():
        p = DATASET / name
        s = p.stat()
        if [s.st_size, s.st_mtime_ns] != expected:
            raise ValueError(f"Protected old/data artifact changed: {name}")

def archive_hashes(directory, archive_name):
    names = [name for name in FILES if (directory / name).exists()]
    if not names:
        return None
    path = archive_files(directory, names, archive_name)
    return {"path": str(path), "sha256": sha(path)}

def preflight(device, original):
    """Only new model checks; reuse the completed common label-count preflight."""
    classifier, regressor = _require_xgboost()
    import xgboost
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    c = classifier(n_estimators=2, max_depth=1, n_jobs=2, tree_method="hist", random_state=42)
    c.fit(x, np.arange(8) % 2)
    g = regressor(n_estimators=2, max_depth=1, n_jobs=2, tree_method="hist", random_state=42)
    g.fit(x, np.arange(8, dtype=np.float32))
    if not np.isfinite(c.predict_proba(x)).all() or not np.isfinite(g.predict(x)).all():
        raise ValueError("XGBoost synthetic runtime check failed")
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise ValueError("B3 requires the server CUDA environment for this registered queue")
    payload, manifest = load_shared_dataset(DATASET)
    if tuple(payload["x"].shape[1:]) != (30, 38, 27):
        raise ValueError("Unexpected shared input shape")
    checks = []
    for cap in (5, 10, 15):
        view, _ = protocol.protocol_view(payload, manifest, cap)
        dataset = FactoryBaselineTensorDataset(view, view["split_indices"]["train"].tolist())
        pos = original["tasks"][str(cap)]["splits"]["train"]["first_upcoming_position"]
        positions = sorted({0, 1, 2, 3, pos if pos is not None else 0})
        samples = [dataset[i] for i in positions]
        indices = [int(v["sample_index"]) for v in samples]
        event = _event_training_data(view, indices)
        upcoming = (event["will"] > 0) & ~event["ongoing"]
        expected = sum(int(((v["event_will"] > .5) & (v["hist_last_hot"] <= .5)
                            & (v["occ_node_mask"] > .5)).sum()) for v in samples)
        if int(upcoming.sum()) != expected or not np.isfinite(event["features"]).all():
            raise ValueError("B2 upcoming training rows disagree with the common labels")
        training, overrides, loss = b3_config(cap, device, manifest["cause_classes"])
        model = B3Lstm(B3ModelConfig(
            input_dim=27, global_dim=int(payload["global_features"].shape[-1]),
            num_nodes=38, max_remain_windows=20, num_causes=len(manifest["cause_classes"]), **overrides,
        )).to(device)
        batch = _move_batch(default_collate(samples), torch.device(device))
        output = model(**_model_inputs(batch, model))
        value, _ = compute_multitask_loss(output, batch, loss)
        value.backward()
        if (output["remain_hot_logit"].shape[1] != 20 or not torch.isfinite(value)
                or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)):
            raise ValueError("B3 real-data forward/backward is invalid")
        checks.append({"max_start": cap, "sample_indices": indices,
                       "b2_labels_passed": True, "b3_loss_and_gradients_finite": True})
        print("REMAINING_PREFLIGHT_PASSED", cap, flush=True)
        del model, batch, output, event
        gc.collect(); torch.cuda.empty_cache()
    return {"checks": checks, "xgboost_version": xgboost.__version__, "torch_version": str(torch.__version__),
            "actual_training_optimizer_steps": 0, "test_indexed": False,
            "b2_synthetic_dependency_check": "two boosting rounds on eight synthetic rows only"}

def prepare(source, device, paths):
    require_runtime(source)
    if (DATASET / PLAN).exists() or (DATASET / FINAL).exists():
        raise FileExistsError("This queue is already registered; use --mode run/status, never duplicate it")
    b45_hash = completed_b45()
    manifest = json.loads((DATASET / "dataset_manifest.json").read_text())
    old_plan = json.loads((DATASET / "baseline_matched_protocol_training20260913_plan.json").read_text())
    data_stats = {name: stat for name, stat in old_plan["protected_files_stat_before_training"].items()
                  if not name.startswith("models/")}
    check_stats(data_stats)
    for model, directory in paths.items():
        if not directory.is_dir() or not directory.is_relative_to(DATASET / "models"):
            raise ValueError(f"{model}: choose an EXISTING output directory inside the benchmark: {directory}")
        if directory.is_symlink():
            raise ValueError("Do not use symlink output directories")
        config = directory / "config.json"
        if config.exists():
            saved = json.loads(config.read_text())
            if saved.get("baseline_id", saved.get("metadata", {}).get("baseline_id")) not in (None, model):
                raise ValueError("Output directory contains a different model")
    if (paths["B2"].is_relative_to(paths["B3"]) or paths["B3"].is_relative_to(paths["B2"])
            or any("b4" in str(p).lower() or "b5" in str(p).lower() for p in paths.values())):
        raise ValueError("B2/B3 output directories must be distinct and must not be B4/B5 directories")
    tasks = []
    for model, cap in TASKS:
        directory = paths[model]
        record = directory / f"{TAG}_s{cap}.json"
        archive = directory / f"model_before_{TAG}_s{cap}.zip"
        if record.exists() or archive.exists():
            raise FileExistsError("An existing stage must be inspected, not restarted")
        training, overrides, loss = b3_config(cap, device, manifest["cause_classes"])
        tasks.append({"model": model, "max_start": cap, "output_dir": str(directory),
                      "record": str(record), "archive": str(archive),
                      "training": asdict(b2_config(cap)) if model == "B2" else asdict(training),
                      "model_overrides": {} if model == "B2" else overrides,
                      "loss": None if model == "B2" else asdict(loss)})
    protected = {}
    for p in DATASET.rglob("*"):
        if (p.is_file() and not (p.parent in paths.values() and p.name in FILES)
                and not (p.parent == DATASET and p.name.startswith(TAG))):
            s = p.stat()
            protected[str(p.relative_to(DATASET))] = [s.st_size, s.st_mtime_ns]
    prior = json.loads((DATASET / "baseline_matched_protocol_preflight20260913.json").read_text())
    if sha(DATASET / "baseline_matched_protocol_preflight20260913.json") != "532d3b715d95106d54ba1fae01dc8bbc0e5d6995cfdda4ff2be2216b51281904":
        raise ValueError("Common completed preflight changed")
    checks = preflight(device, prior)
    check_stats(protected)
    plan = {"status": "registered", "source_commit": source, "source_manifest_sha256": SOURCE_MANIFEST,
            "reference_commit": protocol.REFERENCE_COMMIT, "tasks": tasks, "preflight": checks,
            "b45_final_sha256": b45_hash, "protected_stats": protected,
            "label_counts": {cap: item["splits"] for cap, item in prior["tasks"].items()},
            "test_evaluated": False, "new_directories": False,
            "initialization": {"B2": "independent fixed 500-round fits per head per task",
                               "B3": "random Start5, own selected Start5 -> Start10 -> Start15; no old-dataset parent"},
            "comparison_limits": "Same fixed input history and evaluation rules; model-native features/losses/budgets differ. B3 has no near summary projection. B2 is not assigned neural epochs or transferred trees."}
    write(DATASET / PLAN, plan, exclusive=True)
    print("REMAINING_SIX_REGISTERED", flush=True)
    return plan

def verify_output(task, summary, source, counts):
    directory = Path(task["output_dir"])
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    if summary != json.loads((directory / "run_summary.json").read_text()) or summary["status"] != "validation_completed":
        raise ValueError("Incomplete training/export")
    if set(metrics) != {"train", "validation"} or config.get("training", config.get("config")) != task["training"]:
        raise ValueError("Configuration or evaluated split differs from registration")
    cap = task["max_start"]
    if summary["evaluation_contract"] != protocol.evaluation_contract(cap):
        raise ValueError("Wrong summary task contract")
    splits = json.loads((DATASET / "split_manifest.json").read_text())
    classes = json.loads((DATASET / "dataset_manifest.json").read_text())["cause_classes"]
    scores = {}
    for split, values in metrics.items():
        if (values != json.loads((directory / f"metrics_{split}.json").read_text())
                or values["evaluation_contract"] != {**protocol.evaluation_contract(cap), "window_size_s": 60.}
                or values["sample_count"] != counts[split]["samples"]):
            raise ValueError("Export differs from completed common protocol/support")
        scores[split] = station_counts(values["station_report"], counts[split])
        close(scores[split]["threshold"], summary["event_report_threshold"])
        with (directory / f"predictions_{split}.csv").open(newline="") as stream:
            confusion = verify_prediction_rows(list(csv.DictReader(stream)),
                set(splits[split]["sample_indices"]), split, values, classes)
        with (directory / f"confusion_matrix_{split}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                for predicted in classes:
                    if int(row[f"predicted__{predicted}"]) != confusion[row["target_cause"], predicted]:
                        raise ValueError("Cause confusion differs from prediction rows")
        remain = values["remain"]
        if remain["remain_len_mae_primary"] != remain.get("remain_len_mae_middle_weighted"):
            raise ValueError("Wrong remaining-time primary metric")
    if task["model"] == "B3":
        if config["loss"] != task["loss"] or config["metadata"]["git_commit"] != source:
            raise ValueError("B3 configuration/source changed")
        for k, v in task["model_overrides"].items():
            if config["model"][k] != v:
                raise ValueError("B3 backbone changed")
        history = list(csv.DictReader((directory / "history.csv").open(newline="")))
        if [int(h["epoch"]) for h in history] != list(range(1, summary["epochs_trained"] + 1)):
            raise ValueError("Incomplete B3 history")
        chosen, feasible = selected_epoch(history, task["training"])
        if int(chosen["epoch"]) != summary["best_epoch"] or feasible != summary["checkpoint_constraint_met"]:
            raise ValueError("B3 selected epoch differs from validation rule")
        for name, epoch in (("best.pt", summary["best_epoch"]), ("last.pt", summary["epochs_trained"])):
            ckpt = torch.load(directory / name, map_location="cpu", weights_only=False)
            if (ckpt["epoch"] != epoch or ckpt["model_kind"] != "b3_lstm"
                    or ckpt["metadata"]["git_commit"] != source
                    or ckpt["model_config"] != config["model"]
                    or ckpt["train_config"] != config["training"]
                    or any(not torch.isfinite(x).all() for x in ckpt["model_state_dict"].values())):
                raise ValueError("B3 saved weight identity/values are invalid")
        for key in ("precision", "recall", "f1"):
            close(metrics["validation"]["station_report"][f"will15_{key}"], chosen[f"validation_primary_{key}"])
        budget = summary["training_budget"]
        if budget["stage_optimizer_steps"] != summary["epochs_trained"] * math.ceil(counts["train"]["samples"] / 32):
            raise ValueError("B3 stage update accounting differs")
    else:
        for meta in config["models"].values():
            if meta["path"]:
                json.loads((directory / meta["path"]).read_text())
    files = {name: sha(directory / name) for name in FILES if (directory / name).exists()}
    if any("test" in name for name in files):
        raise ValueError("Unexpected test artifacts in new stage")
    return {"status": "validation_completed", "source_commit": source, "model": task["model"],
            "max_start": cap, "summary": summary, "metrics": metrics, "scores": scores,
            "artifact_sha256": files, "test_evaluated": False, "verification":
            "Saved identities/history, metric counts and split CSV coverage; cause confusion/unweighted MAE independently recomputed. Weighted MAE uses saved phase statistics."}

def run(source, device):
    require_runtime(source)
    plan = json.loads((DATASET / PLAN).read_text())
    if plan["source_commit"] != source or plan["status"] not in {"registered", "running", "failed"}:
        raise ValueError("Inspect existing completed/foreign queue")
    if [(t["model"], t["max_start"]) for t in plan["tasks"]] != TASKS:
        raise ValueError("Registered task identities/order changed")
    causes = json.loads((DATASET / "dataset_manifest.json").read_text())["cause_classes"]
    for task in plan["tasks"]:
        directory = Path(task["output_dir"]).resolve()
        cap, model = task["max_start"], task["model"]
        if (not directory.is_dir() or not directory.is_relative_to(DATASET / "models")
                or any(m in str(directory).lower() for m in ("b4", "b5"))
                or Path(task["record"]) != directory / f"{TAG}_s{cap}.json"
                or Path(task["archive"]) != directory / f"model_before_{TAG}_s{cap}.zip"):
            raise ValueError("Registered paths no longer satisfy the authorized output scope")
        training, overrides, loss = b3_config(cap, device, causes)
        expected = {"training": asdict(b2_config(cap)) if model == "B2" else asdict(training),
                    "model_overrides": {} if model == "B2" else overrides,
                    "loss": None if model == "B2" else asdict(loss)}
        if any(task[key] != json.loads(json.dumps(value)) for key, value in expected.items()):
            raise ValueError("Registered configuration changed; no unregistered tuning")
    if completed_b45() != plan["b45_final_sha256"]:
        raise ValueError("B4/B5 final result changed")
    check_stats(plan["protected_stats"])
    records = []
    plan["status"] = "running"
    write(DATASET / PLAN, plan)
    for task in plan["tasks"]:
        record_path, directory = Path(task["record"]), Path(task["output_dir"])
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["status"] != "validation_completed" or record["source_commit"] != source:
                raise ValueError("Partial stage exists: inspect it; automatic retraining is forbidden")
            # At resume, earlier task files may already be in the next cap archive.
            next_path = directory / f"model_before_{TAG}_s{task['max_start'] + 5}.zip"
            from verify_baseline_matched_results import artifact_snapshot
            files, _ = artifact_snapshot(directory, next_path if task["max_start"] < 15 else None, record["artifact_sha256"])
            if ((record["model"], record["max_start"]) != (task["model"], task["max_start"])
                    or record["test_evaluated"] or json.loads(files["metrics.json"]) != record["metrics"]
                    or json.loads(files["run_summary.json"]) != record["summary"]):
                raise ValueError("Completed record differs from preserved stage artifacts")
            records.append(record)
            print("REUSING_COMPLETED", task["model"], task["max_start"], flush=True)
            continue
        require_runtime(source); check_stats(plan["protected_stats"])
        archive_path = Path(task["archive"])
        if archive_path.exists():
            raise ValueError("Archive without stage record: inspect without overwriting")
        record = {"status": "starting", "source_commit": source, "model": task["model"],
                  "max_start": task["max_start"], "test_evaluated": False}
        write(record_path, record, exclusive=True)
        try:
            archived = archive_hashes(directory, archive_path.name)
            record["prior_archive"] = archived
            write(record_path, record)
            print("REMAINING_STAGE_START", task["model"], task["max_start"], flush=True)
            if task["model"] == "B2":
                summary = train_b2_xgboost(DATASET, directory, B2XGBoostConfig(**task["training"]))
            else:
                summary = train_torch_baseline(
                    model_kind="b3_lstm", dataset_dir=DATASET, output_dir=directory,
                    train_config=TorchTrainConfig(**task["training"]),
                    model_overrides=task["model_overrides"], loss_config=MultiTaskLossConfig(**task["loss"]),
                    warm_start_archive=archive_path if task["max_start"] > 5 else None,
                )
            record = {**verify_output(task, summary, source, plan["label_counts"][str(task["max_start"])]),
                      "prior_archive": archived}
            write(record_path, record)
            records.append(record)
            print("REMAINING_STAGE_COMPLETE", task["model"], task["max_start"],
                  json.dumps(record["scores"]["validation"]), flush=True)
        except Exception as error:
            record.update(status="failed", error=str(error))
            write(record_path, record)
            plan.update(status="failed", failed_stage=[task["model"], task["max_start"]])
            write(DATASET / PLAN, plan)
            raise
        gc.collect(); torch.cuda.empty_cache()
    if [(r["model"], r["max_start"]) for r in records] != TASKS:
        raise ValueError("Six-stage result is incomplete")
    check_stats(plan["protected_stats"])
    if completed_b45() != plan["b45_final_sha256"]:
        raise ValueError("B4/B5 preservation check failed")
    for record in records:
        prior = record["prior_archive"]
        if prior and sha(Path(prior["path"])) != prior["sha256"]:
            raise ValueError("Retained previous weights archive changed")
    write(DATASET / FINAL, {"status": "six_remaining_tasks_verified",
          "source_commit": source, "test_evaluated": False, "b45_final_sha256": plan["b45_final_sha256"],
          "results": records}, exclusive=True)
    plan["status"] = "completed"; write(DATASET / PLAN, plan)
    print("B2_B3_SIX_TASKS_VERIFIED", sha(DATASET / FINAL), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--mode", choices=("all", "prepare", "run", "status"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--b2_output_dir", type=Path, default=DATASET / DEFAULT_OUTPUTS["B2"])
    parser.add_argument("--b3_output_dir", type=Path, default=DATASET / DEFAULT_OUTPUTS["B3"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "status":
        for name in (PLAN, FINAL):
            p = DATASET / name
            print(name, json.loads(p.read_text())["status"] if p.exists() else "not_created")
    else:
        if args.mode in ("all", "prepare"):
            prepare(args.source_commit, args.device, {"B2": args.b2_output_dir.resolve(), "B3": args.b3_output_dir.resolve()})
        if args.mode in ("all", "run"):
            run(args.source_commit, args.device)
