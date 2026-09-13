#!/usr/bin/env python3
"""Verify one completed matched stage from saved artifacts, without model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile


RUNTIME = "6289761c11d09e3570899b89858486b29e9f4c83"
REFERENCE = "7b2ab393e838c1b54a78b2f125d41e0066d53ce0"
VERSION = "factory_dense_i1_eval_tyx_7b2ab39_v1"
SOURCE_MANIFEST = "e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f"
RELATIVE = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/verify_baseline_matched_results.py"
CAUSES = {"transport_delay", "material_shortage", "starved_upstream", "queue_buildup"}


class SnapshotNotReady(RuntimeError):
    """A stage is unfinished or its completed files are transitioning into a ZIP."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(actual, expected, *, tolerance=1e-7):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=tolerance):
        raise ValueError(f"Numeric mismatch: {actual} != {expected}")


def count(value):
    integer = int(value)
    if not math.isfinite(value) or value != integer or integer < 0:
        raise ValueError("Expected a finite non-negative integer count")
    return integer


def selected_epoch(history, training):
    """Reconstruct first-improvement F1 selection after both feasibility gates."""
    fallback, feasible = None, None
    for row in history:
        score = float(row["validation_primary_f1"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid selection score")
        if fallback is None or score > float(fallback["validation_primary_f1"]) + 1e-6:
            fallback = row
        allowed = (float(row["validation_primary_precision"]) >= training["checkpoint_min_report_precision"]
                   and float(row["validation_primary_recall"]) >= training["checkpoint_min_report_recall"])
        if allowed and (feasible is None or score > float(feasible["validation_primary_f1"]) + 1e-6):
            feasible = row
    if fallback is None:
        raise ValueError("No trained epoch candidates")
    return feasible or fallback, feasible is not None


def station_counts(report, expected):
    """Recompute metrics and split missed upcoming alarms into score/start misses."""
    pred, true = (count(report[key]) for key in ("n_pred_who", "n_true_who"))
    if true != expected["ongoing"] + expected["upcoming"]:
        raise ValueError("Positive denominator differs from the preflight")
    if pred > true + expected["negative"]:
        raise ValueError("More predictions than eligible observations")
    for prefix, key in (("who", "n_matched_who"), ("report", "n_matched_report")):
        matched = count(report[key])
        if matched > min(pred, true):
            raise ValueError("Impossible true-positive count")
        precision = matched / pred if pred else 0.
        recall = matched / true if true else 0.
        f1 = 2 * matched / (pred + true) if pred + true else 0.
        for name, value in (("precision", precision), ("recall", recall), ("f1", f1)):
            close(report[f"{prefix}_{name}"], value)
            if prefix == "who":
                close(report[f"will15_{name}"], value)
    groups = {}
    for group in ("ongoing", "upcoming"):
        n = count(report[f"n_true_{group}"])
        who = count(report[f"n_matched_who_{group}"])
        strict = count(report[f"n_matched_report_{group}"])
        if n != expected[group] or not 0 <= strict <= who <= n:
            raise ValueError("Invalid ongoing/upcoming support or matched counts")
        close(report[f"who_recall_{group}"], who / n if n else 0.)
        close(report[f"report_recall_{group}"], strict / n if n else 0.)
        groups[group] = {"true": n, "who_hits": who, "strict_hits": strict,
                         "score_misses": n - who, "start_tolerance_misses": who - strict,
                         "who_recall": who / n if n else 0., "strict_recall": strict / n if n else 0.}
    if sum(v["who_hits"] for v in groups.values()) != report["n_matched_who"]:
        raise ValueError("Group who hits do not sum to overall hits")
    if sum(v["strict_hits"] for v in groups.values()) != report["n_matched_report"]:
        raise ValueError("Group strict hits do not sum to overall hits")
    previous_hits = 0
    for tolerance in (1, 2, 3):
        raw_hits = float(report[f"report_recall_at_{tolerance}"]) * true
        hits = round(raw_hits)
        close(raw_hits, hits)
        if not previous_hits <= hits <= report["n_matched_who"]:
            raise ValueError("Tolerance-specific hits are not nested")
        close(report[f"report_precision_at_{tolerance}"], hits / pred if pred else 0.)
        close(report[f"report_f1_at_{tolerance}"], 2 * hits / (pred + true) if pred + true else 0.)
        previous_hits = hits
    if previous_hits != report["n_matched_report"]:
        raise ValueError("Strict default does not match tolerance three")
    bins = {}
    for name in ("start_le_5", "start_6_10", "start_gt_10"):
        n = count(report[f"n_true_upcoming_{name}"])
        raw_hits = n * float(report[f"report_recall_upcoming_{name}"])
        hits = round(raw_hits); close(raw_hits, hits)
        if not 0 <= hits <= n:
            raise ValueError("Invalid start-bin matches")
        bins[name] = {"true": n, "strict_hits": hits, "strict_recall": hits / n if n else 0.}
    if (sum(v["true"] for v in bins.values()) != groups["upcoming"]["true"]
            or sum(v["strict_hits"] for v in bins.values()) != groups["upcoming"]["strict_hits"]):
        raise ValueError("Start bins do not partition upcoming results")
    return {"will15_f1": report["will15_f1"], "will15_precision": report["will15_precision"],
            "will15_recall": report["will15_recall"], "strict_f1_at_1_2_3": [report[f"report_f1_at_{i}"] for i in (1, 2, 3)],
            "groups": groups, "start_bins": bins, "threshold": report["report_threshold_used"]}


def verify_prediction_rows(rows, expected_indices, split, metrics, classes):
    seen, errors, valid_causes = set(), [], []
    confusion = {(a, b): 0 for a in classes for b in classes}
    for row in rows:
        index = int(row["sample_index"])
        if index in seen or index not in expected_indices or row["split"] != split:
            raise ValueError("Predictions contain repeated, missing or cross-split windows")
        seen.add(index)
        predicted, target = row["predicted_cause"], row["target_cause"]
        if target and target not in classes:
            raise ValueError("Unknown target cause label")
        if predicted not in CAUSES:
            raise ValueError("Predicted cause is outside the four supported classes")
        if target in CAUSES:
            valid_causes.append((target, predicted)); confusion[target, predicted] += 1
        error = abs(float(row["predicted_remain_len_windows"]) - float(row["target_remain_len_windows"]))
        if not math.isfinite(error):
            raise ValueError("Non-finite remaining-time prediction")
        errors.append(error)
    if seen != expected_indices or len(errors) != metrics["sample_count"]:
        raise ValueError("Prediction CSV does not cover each original split window once")
    close(metrics["remain"]["remain_len_mae"], math.fsum(errors) / len(errors), tolerance=2e-6)
    cause = metrics["cause"]
    if len(valid_causes) != cause["cause_n"]:
        raise ValueError("Supported cause denominator differs")
    accuracy = sum(a == b for a, b in valid_causes) / len(valid_causes) if valid_causes else 0.
    close(cause["cause_acc"], accuracy)
    recalls = []
    for label in CAUSES:
        support = [b for a, b in valid_causes if a == label]
        if support:
            recall = support.count(label) / len(support)
            close(cause[f"cause_recall_{label}"], recall); recalls.append(recall)
    close(cause["cause_macro_recall"], math.fsum(recalls) / len(recalls) if recalls else 0.)
    return confusion


def artifact_snapshot(directory, next_archive, expected):
    if any(Path(name).name != name for name in expected):
        raise ValueError("Artifact names must be direct children")
    try:
        if next_archive is not None and next_archive.exists():
            with zipfile.ZipFile(next_archive) as archive:
                manifest = json.loads(archive.read("archive_manifest.json"))
                if manifest != expected:
                    raise ValueError("Completed-stage archive manifest differs from its result record")
                contents = {name: archive.read(name) for name in expected}
            origin = str(next_archive)
        else:
            contents = {name: (directory / name).read_bytes() for name in expected}
            origin = str(directory)
    except (zipfile.BadZipFile, FileNotFoundError, KeyError) as error:
        raise SnapshotNotReady("Completed-stage archive is transitioning; re-read the same stage") from error
    if any(digest(contents[name]) != value for name, value in expected.items()):
        if next_archive is not None and origin != str(next_archive) and next_archive.exists():
            raise SnapshotNotReady("Stage artifacts moved during the snapshot; retry the same stage")
        raise ValueError("Artifact SHA differs from the completed-stage record")
    return contents, origin


def verify_stage(dataset, plan, model, cap, source_commit):
    import torch
    task = next(t for t in plan["tasks"] if (t["model"], t["max_start"]) == (model, cap))
    record_path = Path(task["record"])
    if not record_path.exists():
        raise SnapshotNotReady("The requested stage has not started")
    raw_record = record_path.read_bytes(); record = json.loads(raw_record)
    if record["status"] != "validation_completed":
        raise SnapshotNotReady("The requested stage has not completed")
    if record["source_commit"] != RUNTIME or record["test_evaluated"]:
        raise ValueError("Unexpected stage source or test evaluation")
    directory = Path(task["output_dir"])
    next_archive = directory / f"model_before_matched20260913_s{5 + cap}.zip" if cap < 15 else None
    files, origin = artifact_snapshot(directory, next_archive, record["artifact_sha256"])
    config, summary, metrics = (json.loads(files[name]) for name in ("config.json", "run_summary.json", "metrics.json"))
    if summary != record["summary"] or metrics != record["metrics"] or set(metrics) != {"train", "validation"}:
        raise ValueError("Result record differs from the saved run")
    if config["training"] != task["training"] or config["loss"] != task["loss"]:
        raise ValueError("Saved configuration differs from preregistration")
    meta, training = config["metadata"], config["training"]
    if training["seed"] != 42 or training["evaluate_test"] or not training["evaluate_train"]:
        raise ValueError("Unexpected seed or evaluated splits")
    if meta["git_commit"] != RUNTIME or meta["dataset_manifest_sha256"] != SOURCE_MANIFEST:
        raise ValueError("Wrong runtime or dataset identity")
    contract = meta["evaluation_contract"]
    expected_contract = json.loads((dataset / "baseline_matched_protocol_preflight20260913.json").read_text())["tasks"][str(cap)]["contract"]
    if contract != expected_contract or contract["version"] != VERSION or contract["reference_commit"] != REFERENCE:
        raise ValueError("Evaluation contract differs from preflight")
    if config["model"]["max_remain_windows"] != 20 or config["model"]["gru_hidden"] != 128:
        raise ValueError("Wrong output horizon or backbone width")
    if any(config["model"].get(k) != v for k, v in task["model_overrides"].items()):
        raise ValueError("Backbone configuration changed")
    history = list(csv.DictReader(io.StringIO(files["history.csv"].decode())))
    epochs = int(summary["epochs_trained"])
    if [int(row["epoch"]) for row in history] != list(range(1, epochs + 1)) or not 1 <= epochs <= training["max_epochs"]:
        raise ValueError("Incomplete training history")
    chosen, feasible = selected_epoch(history, training)
    epoch = int(chosen["epoch"])
    if epoch != summary["best_epoch"] or epoch != summary["checkpoint_epoch"] or feasible != summary["checkpoint_constraint_met"]:
        raise ValueError("Saved selection does not match an independent history reconstruction")
    parent = meta["warm_start_parent"]
    steps_per_epoch = math.ceil(23859 / training["batch_size"])
    for row in history:
        for key in ("train_total", "validation_total_loss", "learning_rate"):
            if not math.isfinite(float(row[key])):
                raise ValueError("Non-finite training history")
        steps = int(row["epoch"]) * steps_per_epoch
        if int(row["stage_optimizer_steps"]) != steps or int(row["cumulative_optimizer_steps"]) != parent["optimizer_steps"] + steps:
            raise ValueError("History update counts disagree with the registered sampling budget")
    budget = summary["training_budget"]
    for key, value in {"stage_epochs_trained": epochs, "stage_optimizer_steps": epochs * steps_per_epoch,
                       "parent_epochs_trained": parent["epochs_trained"], "parent_optimizer_steps": parent["optimizer_steps"],
                       "cumulative_epochs_trained": parent["epochs_trained"] + epochs,
                       "cumulative_optimizer_steps": parent["optimizer_steps"] + epochs * steps_per_epoch,
                       "cumulative_max_epochs": parent["max_epochs"] + training["max_epochs"]}.items():
        if budget[key] != value:
            raise ValueError(f"Incorrect cumulative budget: {key}")
    parent_path = Path(parent["archive"])
    if sha(parent_path) != parent["archive_sha256"]:
        raise ValueError("Parent archive changed")
    for name, expected_epoch in (("best.pt", epoch), ("last.pt", epochs)):
        checkpoint = torch.load(io.BytesIO(files[name]), map_location="cpu", weights_only=False)
        if (checkpoint["epoch"] != expected_epoch or checkpoint["metadata"]["git_commit"] != RUNTIME
                or checkpoint["model_kind"] != ("b4_gcn_gru" if model == "B4" else "b5_gat_gru")
                or json.loads(json.dumps(checkpoint["train_config"])) != training
                or json.loads(json.dumps(checkpoint["loss_config"])) != config["loss"]
                or checkpoint["model_config"] != config["model"]
                or checkpoint["metadata"]["evaluation_contract"] != contract
                or checkpoint["metadata"]["input_feature_contract"] != meta["input_feature_contract"]
                or checkpoint["metadata"]["warm_start_parent"] != parent):
            raise ValueError("Checkpoint header disagrees with saved stage identity")
        if checkpoint["metadata"]["stage_optimizer_steps"] != expected_epoch * steps_per_epoch:
            raise ValueError("Checkpoint update count disagrees with its epoch")
        if any(not bool(torch.isfinite(value).all()) for value in checkpoint["model_state_dict"].values()):
            raise ValueError("Non-finite checkpoint weights")
    split_manifest = json.loads((dataset / "split_manifest.json").read_text())
    classes = json.loads((dataset / "dataset_manifest.json").read_text())["cause_classes"]
    scores = {}
    for split, measured in metrics.items():
        if measured != json.loads(files[f"metrics_{split}.json"]) or measured["evaluation_contract"] != contract:
            raise ValueError("Per-split metrics or contract disagree")
        expected = plan["preflight_label_counts"][str(cap)][split]
        if measured["sample_count"] != expected["samples"]:
            raise ValueError("Wrong split sample count")
        scores[split] = station_counts(measured["station_report"], expected)
        close(scores[split]["threshold"], summary["event_report_threshold"])
        rows = list(csv.DictReader(io.StringIO(files[f"predictions_{split}.csv"].decode())))
        confusion = verify_prediction_rows(rows, set(split_manifest[split]["sample_indices"]), split, measured, classes)
        for row in csv.DictReader(io.StringIO(files[f"confusion_matrix_{split}.csv"].decode())):
            for predicted in classes:
                if int(row[f"predicted__{predicted}"]) != confusion[row["target_cause"], predicted]:
                    raise ValueError("Cause confusion matrix disagrees with prediction rows")
        phase_counts = [count(measured["remain"][f"remain_len_n_{p}"]) for p in ("early", "middle", "late")]
        if sum(phase_counts) != measured["sample_count"]:
            raise ValueError("Remaining-time phase supports do not partition the split")
        primary = measured["remain"].get("remain_len_mae_middle_weighted")
        if measured["remain"]["remain_len_mae_primary"] != primary:
            raise ValueError("Remaining-time primary field is not middle weighted")
    for suffix in ("precision", "recall", "f1"):
        close(metrics["validation"]["station_report"][f"will15_{suffix}"], chosen[f"validation_primary_{suffix}"])
        close(metrics["validation"]["station_report"][f"report_{suffix}"], chosen[f"validation_report_{suffix}"])
    if digest(record_path.read_bytes()) != digest(raw_record):
        raise SnapshotNotReady("Stage record changed during verification")
    return {"status": "completed_stage_verified", "model": model, "max_start": cap, "seed": 42,
            "runtime_commit": RUNTIME, "verification_source_commit": source_commit,
            "record_sha256": digest(raw_record), "artifact_origin": origin,
            "artifact_sha256": record["artifact_sha256"], "selected_epoch": epoch,
            "epochs_trained": epochs, "checkpoint_constraint_met": feasible, "training_budget": budget,
            "scores": scores, "test_evaluated": False, "model_forward": False, "model_training": False,
            "verification_scope": "Artifact/checkpoint identity; history-based selection; metric count identities; exhaustive split CSV support; four-cause confusion and unweighted remain MAE recomputed.",
            "limitations": "Station counts/labels and weighted phase errors use saved metrics, not a second inference. Source CSV event_will_any is legacy metadata, not the new task target. A single seed/curriculum is not an architecture-only causal comparison."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--model", choices=("B4", "B5"), required=True)
    parser.add_argument("--max_start", type=int, choices=(5, 10, 15), required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
    assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    dataset = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    output = dataset / f"baseline_matched_{args.model.lower()}_start{args.max_start}_verification20260913.json"
    if output.exists():
        raise FileExistsError("Read the existing verification instead of repeating it")
    plan = json.loads((dataset / "baseline_matched_protocol_training20260913_plan.json").read_text())
    assert plan["source_commit"] == RUNTIME and plan["source_manifest_sha256"] == sha(dataset / "dataset_manifest.json") == SOURCE_MANIFEST
    for name, expected in plan["protected_files_stat_before_training"].items():
        if not name.startswith("models/"):
            path = dataset / name
            assert [path.stat().st_size, path.stat().st_mtime_ns] == expected, name
    try:
        result = verify_stage(dataset, plan, args.model, args.max_start, args.source_commit)
    except SnapshotNotReady as error:
        print("MATCHED_STAGE_NOT_READY", str(error), flush=True)
        raise SystemExit(2)
    result["verification_source_sha256"] = digest(subprocess.check_output(["git", "show", args.source_commit + ":" + RELATIVE]))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print("MATCHED_STAGE_VERIFIED", args.model, args.max_start, sha(output), json.dumps(result["scores"]), flush=True)
