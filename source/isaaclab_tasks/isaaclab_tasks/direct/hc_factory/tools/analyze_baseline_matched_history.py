#!/usr/bin/env python3
"""Describe completed, verified validation trajectories without selecting a new model."""

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

from verify_baseline_matched_results import RUNTIME, selected_epoch, close, SnapshotNotReady


RELATIVE = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/analyze_baseline_matched_history.py"
FIELDS = ("train_total", "validation_total_loss", "validation_primary_f1",
          "validation_primary_precision", "validation_primary_recall",
          "validation_report_recall_upcoming", "validation_event_threshold",
          "validation_event_will_pr_auc")


def summarize_history(rows, training, chosen_epoch, support):
    if support <= 0 or [int(r["epoch"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Need a complete chronological history and positive upcoming support")
    chosen, feasible = selected_epoch(rows, training)
    if int(chosen["epoch"]) != chosen_epoch:
        raise ValueError("Saved selection differs from its history")
    values = []
    for row in rows:
        item = {"epoch": int(row["epoch"]), **{k: float(row[k]) for k in FIELDS}}
        if not all(math.isfinite(item[k]) for k in FIELDS):
            raise ValueError("Non-finite trajectory values")
        if any(not 0 <= item[k] <= 1 for k in FIELDS[2:]):
            raise ValueError("Invalid score, probability or threshold")
        hits = support * item["validation_report_recall_upcoming"]
        close(hits, round(hits))
        item["upcoming_strict_hits"] = round(hits)
        item["checkpoint_constraints_met"] = (
            item["validation_primary_precision"] >= training["checkpoint_min_report_precision"]
            and item["validation_primary_recall"] >= training["checkpoint_min_report_recall"])
        values.append(item)
    selected = values[chosen_epoch - 1]
    maximum = max(v["upcoming_strict_hits"] for v in values)
    return {"epochs_trained": len(values), "upcoming_support": support,
            "checkpoint_constraint_met": feasible, "first": values[0], "selected": selected,
            "last": values[-1], "retrospective_maximum_hits": maximum,
            "retrospective_maximum_recall": maximum / support,
            "retrospective_maximum_epochs": [v for v in values if v["upcoming_strict_hits"] == maximum],
            "extra_hits_over_selected_at_observed_thresholds": maximum - selected["upcoming_strict_hits"],
            "epochs_with_more_hits_than_selected": sum(v["upcoming_strict_hits"] > selected["upcoming_strict_hits"] for v in values),
            "trajectory": values}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def analyze(dataset, model, cap, source_commit):
    output = dataset / f"baseline_matched_{model.lower()}_start{cap}_history_audit20260913.json"
    if output.exists():
        raise FileExistsError("Reuse the completed audit")
    plan = json.loads((dataset / "baseline_matched_protocol_training20260913_plan.json").read_text())
    task = next(t for t in plan["tasks"] if (t["model"], t["max_start"]) == (model, cap))
    proof_path = dataset / f"baseline_matched_{model.lower()}_start{cap}_verification20260913.json"
    if not proof_path.exists():
        raise SnapshotNotReady("Wait for the independent completed-stage verification")
    raw_proof, raw_record = proof_path.read_bytes(), Path(task["record"]).read_bytes()
    proof, record = json.loads(raw_proof), json.loads(raw_record)
    if (proof["status"] != "completed_stage_verified" or proof["record_sha256"] != digest(raw_record)
            or record["source_commit"] != RUNTIME or record["test_evaluated"]
            or (record["model"], record["max_start"]) != (model, cap)):
        raise ValueError("Stage identity or completed verification changed")
    directory = Path(task["output_dir"])
    archive = directory / f"model_before_matched20260913_s{cap + 5}.zip"
    if cap < 15 and archive.exists():
        with zipfile.ZipFile(archive) as z:
            raw_history = z.read("history.csv")
        origin = str(archive)
    else:
        raw_history = (directory / "history.csv").read_bytes()
        origin = str(directory / "history.csv")
    if digest(raw_history) != record["artifact_sha256"]["history.csv"]:
        raise ValueError("Completed history changed")
    rows = list(csv.DictReader(io.StringIO(raw_history.decode())))
    group = proof["scores"]["validation"]["groups"]["upcoming"]
    summary = summarize_history(rows, task["training"], proof["selected_epoch"], group["true"])
    if (summary["selected"]["upcoming_strict_hits"] != group["strict_hits"]
            or summary["epochs_trained"] != proof["epochs_trained"]
            or summary["checkpoint_constraint_met"] != proof["checkpoint_constraint_met"]):
        raise ValueError("History differs from independently verified results")
    result = {"status": "completed_history_audited", "model": model, "max_start": cap,
              "runtime_commit": RUNTIME, "analysis_source_commit": source_commit,
              "analysis_source_sha256": digest(subprocess.check_output(["git", "show", source_commit + ":" + RELATIVE])),
              "history_sha256": digest(raw_history), "history_origin": origin,
              "stage_record_sha256": digest(raw_record), "stage_verification_sha256": digest(raw_proof),
              "summary": summary, "new_model_selected": False, "test_evaluated": False,
              "model_forward": False, "model_training": False,
              "limits": "Retrospective maximum covers only saved epoch reports at their original validation-selected thresholds. It is not a new selected checkpoint, a threshold search or an architecture bound. Train loss uses event-resampled windows; validation loss uses original validation windows. No train recall is inferred from train loss."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print("MATCHED_HISTORY_AUDITED", model, cap, digest(output.read_bytes()),
          json.dumps({k: v for k, v in summary.items() if k != "trajectory"}), flush=True)


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
    dataset = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    analyze(dataset, args.model, args.max_start, args.source_commit)
