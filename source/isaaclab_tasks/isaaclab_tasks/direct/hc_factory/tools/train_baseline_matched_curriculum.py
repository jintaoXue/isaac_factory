#!/usr/bin/env python3
"""Register and run six validation-only B4/B5 tasks in existing model directories."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import subprocess

import torch

from factory_baselines import protocol_20260913 as protocol
from factory_baselines.artifacts import archive_files
from factory_baselines.matched_warm_start import load_matched_archive_parent
from factory_baselines.torch_trainer import _model_spec, train_torch_baseline
from preflight_baseline_matched_protocol import SOURCE_MANIFEST, sha
from train_dense_baseline_control import TRAINING_FILES, dense_configuration


TAG = "matched20260913"
PLAN = "baseline_matched_protocol_training20260913_plan.json"
RESULT = "baseline_matched_protocol_training20260913_results.json"
PREFLIGHT_SHA = "532d3b715d95106d54ba1fae01dc8bbc0e5d6995cfdda4ff2be2216b51281904"
FILES = [*TRAINING_FILES, "initial.pt"]


def configuration(model, max_start, device="cuda:0"):
    protocol.evaluation_contract(max_start)
    training, overrides, loss = dense_configuration(model, "near_precursor", 42, device)
    training = replace(
        training, evaluation_protocol=protocol.VERSION, event_max_start_windows=max_start,
        training_profile=f"{TAG}_{model.lower()}_start{max_start}_min8",
        max_epochs=100, patience=40, evaluate_train=True,
        event_oversample_factor=4.0, event_oversample_target="any_event",
        report_threshold_sweep=protocol.THRESHOLDS,
    )
    loss = replace(loss, event_will_upcoming_pos_weight={5: 9., 10: 10., 15: 11.}[max_start],
                   event_will_fp_weight=2.5, lambda_remain_len=.5,
                   near_remain_windows=20, event_partition="history",
                   remain_progress_weight_floor=.25, remain_progress_weight_power=1.5)
    return training, overrides, loss


def output_directory(dataset, model):
    return dataset / f"models/tuning/{model.lower()}_representation_v1/candidate_history/seed42"


def write_json(path, value, *, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def runtime(dataset, source_commit):
    repo = Path.cwd().resolve()
    if repo != Path("/home/sci/work/BSTAN_isaac_factory"):
        raise ValueError("Use only the authorized baseline server checkout")
    if not dataset.is_dir() or not dataset.is_relative_to(repo):
        raise ValueError("Use the existing benchmark directory")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "dev_xwt":
        raise ValueError("Wrong branch")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != source_commit:
        raise ValueError("Runtime source changed; do not pull during training")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Tracked source is dirty")
    if sha(dataset / "dataset_manifest.json") != SOURCE_MANIFEST:
        raise ValueError("Wrong source dataset")


def check_stats(dataset, stats):
    for name, expected in stats.items():
        path = dataset / name
        if [path.stat().st_size, path.stat().st_mtime_ns] != expected:
            raise ValueError(f"Protected artifact changed: {name}")


def prepare(dataset, source_commit, device):
    if (dataset / PLAN).exists() or (dataset / RESULT).exists():
        raise FileExistsError("Read the existing registration instead of repeating it")
    preflight_path = dataset / "baseline_matched_protocol_preflight20260913.json"
    if sha(preflight_path) != PREFLIGHT_SHA:
        raise ValueError("The previously completed preflight has changed")
    preflight = json.loads(preflight_path.read_text())
    check_stats(dataset, preflight["protected_files_stat"])
    protected_stats = dict(preflight["protected_files_stat"])
    for name in ("episodes.npz", "model_sample_index.csv", "node_catalog.csv", "graph_edge_table.csv"):
        path = dataset / name
        protected_stats[name] = [path.stat().st_size, path.stat().st_mtime_ns]
    for name, field in (("episodes.npz", "bundle_sha256"), ("model_sample_index.csv", "sample_index_sha256")):
        if sha(dataset / name) != preflight["input_contract"][field]:
            raise ValueError("The completed preflight input source changed")
    parents, tasks = {}, []
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    for model, kind in (("B4", "b4_gcn_gru"), ("B5", "b5_gat_gru")):
        directory = output_directory(dataset, model)
        if not directory.is_dir():
            raise ValueError("Do not create output directories")
        training, overrides, _ = configuration(model, 5, device)
        cls, config_cls, _, _ = _model_spec(kind)
        model_config = config_cls(input_dim=27, global_dim=0, num_nodes=38,
                                 num_causes=len(manifest["cause_classes"]), max_remain_windows=20, **overrides)
        torch.manual_seed(42)
        candidate = cls(model_config)
        _, parent = load_matched_archive_parent(
            directory / "model_before_onsetaux20260912.zip",
            model_kind=kind, model_config=model_config.to_dict(), config_class=config_cls,
            initial_state=candidate.state_dict(), seed=42, dataset_manifest_sha256=SOURCE_MANIFEST,
            train_sample_count=23859, max_start=5,
        )
        parents[model] = parent
    for cap in (5, 10, 15):
        for model in ("B4", "B5"):
            training, overrides, loss = configuration(model, cap, device)
            # Freeze the actual four supported cause IDs before any optimizer step.
            loss = replace(loss, cause_ignored_ids=tuple(i for i, name in enumerate(manifest["cause_classes"])
                                                       if name not in protocol.CAUSE_CLASSES))
            directory = output_directory(dataset, model)
            archive = directory / f"model_before_{TAG}_s{cap}.zip"
            record = directory / f"{TAG}_s{cap}.json"
            if archive.exists() or record.exists():
                raise FileExistsError("A registered stage already has artifacts; do not repeat it")
            tasks.append({"model": model, "max_start": cap, "training": asdict(training),
                          "model_overrides": overrides, "loss": asdict(loss),
                          "output_dir": str(directory), "record": str(record), "archive": str(archive)})
    plan = {
        "status": "registered", "source_commit": source_commit, "reference_commit": protocol.REFERENCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST, "preflight_sha256": PREFLIGHT_SHA,
        "tasks": tasks, "parents": parents, "protected_files_stat_before_training": protected_stats,
        "preflight_label_counts": {cap: item["splits"] for cap, item in preflight["tasks"].items()},
        "test_evaluated": False, "new_directory_created": False,
        "initialization": "each_own_completed_near_start2_best_then_start5_then_start10_then_start15",
        "head_transition": "15_to_20_reinitialize_only_three_shape_changed_horizon_tensors",
        "training_comparison_limit": "Own parent pretraining and backbone learning rates/batch sizes differ; record all executed budgets. No claim of a pure architecture-only experiment.",
        "preserved_model_differences": "No main prefix/hazard/occupancy-union/state embedding; retain baseline loss paths without main-only hard-negative/far-start weighting paths.",
        "task_boundary": "Start15 with 20 grids and Min8 has no positive upcoming start index above 12, exactly as pinned main reference.",
        "final_train_evaluation": "every original training window once, selected best and threshold fixed on validation",
        "checkpoint_candidates": "trained_epochs_only_as_main_runner_no_extra_epoch_zero_candidate",
    }
    write_json(dataset / PLAN, plan, exclusive=True)
    print("MATCHED_TRAINING_REGISTERED", sha(dataset / PLAN), flush=True)
    return plan


def run(dataset, source_commit, device):
    plan = json.loads((dataset / PLAN).read_text())
    if plan["status"] != "registered" or plan["source_commit"] != source_commit or (dataset / RESULT).exists():
        raise ValueError("This queue has already started or changed; inspect it instead of restarting")
    check_stats(dataset, plan["protected_files_stat_before_training"])
    source_stats = {name: stat for name, stat in plan["protected_files_stat_before_training"].items()
                    if not name.startswith("models/")}
    plan["status"] = "running"
    write_json(dataset / PLAN, plan)
    completed = []
    for task in plan["tasks"]:
        runtime(dataset, source_commit)
        check_stats(dataset, source_stats)
        model, cap = task["model"], task["max_start"]
        directory, archive, record_path = (Path(task[k]) for k in ("output_dir", "archive", "record"))
        if archive.exists() or record_path.exists():
            raise FileExistsError("Partial/existing stage must be inspected, never restarted automatically")
        training, overrides, loss = configuration(model, cap, device)
        loss = replace(loss, cause_ignored_ids=tuple(task["loss"]["cause_ignored_ids"]))
        if json.loads(json.dumps(asdict(training))) != task["training"] or json.loads(json.dumps(asdict(loss))) != task["loss"]:
            raise ValueError("Runtime configuration differs from the preregistration")
        if cap == 5:
            parent = Path(plan["parents"][model]["archive"])
            if sha(parent) != plan["parents"][model]["archive_sha256"]:
                raise ValueError("Registered near parent changed")
        else:
            previous = next(item for item in completed if item["model"] == model and item["max_start"] == {10: 5, 15: 10}[cap])
            if sha(directory / "best.pt") != previous["artifact_sha256"]["best.pt"]:
                raise ValueError("Previous stage selected checkpoint changed")
            parent = archive
        record = {"status": "starting", "model": model, "max_start": cap, "source_commit": source_commit,
                  "parent_archive": str(parent), "test_evaluated": False}
        write_json(record_path, record, exclusive=True)
        try:
            archive_files(directory, FILES, archive.name)
            record["prior_artifact_archive_sha256"] = sha(archive)
            write_json(record_path, record)
            print("MATCHED_STAGE_START", model, cap, flush=True)
            summary = train_torch_baseline(
                model_kind="b4_gcn_gru" if model == "B4" else "b5_gat_gru",
                dataset_dir=dataset, output_dir=directory, model_overrides=overrides,
                train_config=training, loss_config=loss, warm_start_archive=parent,
            )
            metrics = json.loads((directory / "metrics.json").read_text())
            if set(metrics) != {"train", "validation"} or summary["status"] != "validation_completed":
                raise ValueError("Unexpected evaluated splits or incomplete run")
            for split, values in metrics.items():
                counts = plan["preflight_label_counts"][str(cap)][split]
                if (values["sample_count"] != counts["samples"]
                        or values["station_report"]["n_true_upcoming"] != counts["upcoming"]
                        or values["station_report"]["n_true_ongoing"] != counts["ongoing"]
                        or values["evaluation_contract"] != protocol.evaluation_contract(cap)):
                    raise ValueError("Final evaluation disagrees with the registered task/support")
            record.update(status="validation_completed", summary=summary, metrics=metrics,
                          artifact_sha256={name: sha(directory / name) for name in FILES if (directory / name).exists()})
            write_json(record_path, record)
            completed.append(record)
            print("MATCHED_STAGE_COMPLETE", model, cap, json.dumps(metrics["validation"]["station_report"]), flush=True)
        except Exception as error:
            record.update(status="failed", error=str(error))
            write_json(record_path, record)
            plan.update(status="failed", failed_stage=[model, cap])
            write_json(dataset / PLAN, plan)
            raise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    check_stats(dataset, source_stats)
    write_json(dataset / RESULT, {"status": "completed", "source_commit": source_commit,
                                 "tasks": completed, "test_evaluated": False}, exclusive=True)
    plan["status"] = "completed"
    write_json(dataset / PLAN, plan)
    print("MATCHED_CURRICULUM_COMPLETE", sha(dataset / RESULT), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = args.dataset_dir.resolve()
    runtime(dataset, args.source_commit)
    (prepare if args.mode == "prepare" else run)(dataset, args.source_commit, args.device)
