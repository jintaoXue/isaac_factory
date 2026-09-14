"""Read a verified baseline archive for the registered 5 -> 10 -> 15 curriculum."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import torch

from . import protocol_20260913 as protocol


MEMBERS = ("best.pt", "config.json", "history.csv", "run_summary.json", "metrics_validation.json")
HORIZON_KEYS = {
    "heads.remain_time_embedding.weight",
    "heads.event_start_head.2.weight",
    "heads.event_start_head.2.bias",
}


def _json(value):
    return json.loads(json.dumps(value))


def validate_parent(contents, *, model_kind, model_config, config_class, initial_state,
                    seed, dataset_manifest_sha256, train_sample_count, max_start):
    """Validate identity, task ancestry, selected metrics, weights and cumulative budget."""
    checkpoint = torch.load(io.BytesIO(contents["best.pt"]), map_location="cpu", weights_only=False)
    config = json.loads(contents["config.json"])
    summary = json.loads(contents["run_summary.json"])
    validation = json.loads(contents["metrics_validation.json"])
    meta, training = checkpoint["metadata"], checkpoint["train_config"]
    if summary["status"] != "validation_completed" or training["evaluate_test"]:
        raise ValueError("Parent must be a completed validation-only baseline")
    if checkpoint["model_kind"] != model_kind or summary["model_kind"] != model_kind:
        raise ValueError("Parent model kind differs")
    if any(x != seed for x in (training["seed"], meta["seed"], summary["seed"])):
        raise ValueError("Parent seed differs")
    if any(x != dataset_manifest_sha256 for x in (
        meta["dataset_manifest_sha256"], config["metadata"]["dataset_manifest_sha256"],
        summary["dataset_manifest_sha256"],
    )):
        raise ValueError("Parent source dataset differs")
    if _json(training) != config["training"] or _json(checkpoint["loss_config"]) != config["loss"]:
        raise ValueError("Parent saved configurations disagree")
    parent_config = config_class(**checkpoint["model_config"]).to_dict()
    if parent_config != config_class(**config["model"]).to_dict():
        raise ValueError("Parent model configurations disagree")
    first = max_start == 5
    if first:
        if training.get("evaluation_protocol", "legacy") != "legacy" or "warm_start_parent" in meta:
            raise ValueError("First stage requires the registered legacy near parent")
        if parent_config["max_remain_windows"] != 15:
            raise ValueError("Legacy near parent must have 15 future grids")
    else:
        expected = {10: 5, 15: 10}.get(max_start)
        if (expected is None or training.get("evaluation_protocol") != protocol.VERSION
                or training.get("event_max_start_windows") != expected
                or meta["evaluation_contract"] != protocol.evaluation_contract(expected)):
            raise ValueError("Parent must be the immediately preceding matched task")
    comparable = {**parent_config, "max_remain_windows": 20}
    b3_matched = model_kind == "b3_lstm" and not first
    expected_precursor = None if b3_matched else "near"
    if comparable != model_config or model_config.get("event_precursor") != expected_precursor:
        raise ValueError("Only the legacy 15-to-20 output horizon may change")
    if not meta.get("git_commit") or meta["git_commit"] == "unknown":
        raise ValueError("Parent source commit is missing")

    epochs, selected = int(summary["epochs_trained"]), int(checkpoint["epoch"])
    if not (int(first) <= selected <= epochs <= int(training["max_epochs"])):
        raise ValueError("Parent epoch budget is invalid")
    if selected != summary["best_epoch"] or selected != summary["checkpoint_epoch"]:
        raise ValueError("Archive does not contain the selected parent")
    history = list(csv.DictReader(io.StringIO(contents["history.csv"].decode())))
    if [int(row["epoch"]) for row in history] != list(range(1, epochs + 1)):
        raise ValueError("Parent history is incomplete")
    if any(not math.isfinite(float(row[k])) for row in history
           for k in ("train_total", "validation_total_loss", "learning_rate")):
        raise ValueError("Parent history is non-finite")
    if selected == 0:
        initial_report = json.loads(contents["metrics_initial_validation.json"])["station_report"]
    for suffix in ("precision", "recall", "f1"):
        measured = float(validation["station_report"][f"report_{suffix}"])
        recorded = (float(history[selected - 1][f"validation_report_{suffix}"])
                    if selected else float(initial_report[f"report_{suffix}"]))
        if not all(math.isclose(float(v), measured, abs_tol=1e-6) for v in
                   (recorded, summary[f"best_validation_report_{suffix}"])):
            raise ValueError("Parent selected strict validation scores disagree")
        if not first and not math.isclose(float(summary[f"best_validation_primary_{suffix}"]),
                                         float(validation["station_report"][f"will15_{suffix}"]), abs_tol=1e-6):
            raise ValueError("Parent selected primary validation scores disagree")

    state = checkpoint["model_state_dict"]
    if set(state) != set(initial_state):
        raise ValueError("Parent state keys differ")
    skipped = {k for k in state if state[k].shape != initial_state[k].shape}
    if skipped != (HORIZON_KEYS if first else set()):
        raise ValueError("Unexpected shape changes outside the three horizon tensors")
    if any(not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError("Parent weights are non-finite")
    loaded = {k: initial_state[k] if k in skipped else state[k] for k in state}
    steps = epochs * math.ceil(train_sample_count / int(training["batch_size"]))
    budget = summary["training_budget"]
    if budget["stage_epochs_trained"] != epochs or budget["stage_optimizer_steps"] != steps:
        raise ValueError("Parent stage update accounting disagrees with history")
    parent = meta.get("warm_start_parent", {})
    if (budget["cumulative_epochs_trained"] != epochs + parent.get("epochs_trained", 0)
            or budget["cumulative_optimizer_steps"] != steps + parent.get("optimizer_steps", 0)
            or budget["cumulative_max_epochs"] != training["max_epochs"] + parent.get("max_epochs", 0)):
        raise ValueError("Parent cumulative budget is inconsistent")
    if not math.isfinite(budget["cumulative_elapsed_seconds"]) or budget["cumulative_elapsed_seconds"] < 0:
        raise ValueError("Parent elapsed time is invalid")
    provenance = {
        "mode": "weights_only_new_optimizer", "source_commit": meta["git_commit"],
        "dataset_manifest_sha256": dataset_manifest_sha256, "seed": seed,
        "selected_epoch": selected, "parent_task_max_start": 2 if first else expected,
        "epochs_trained": budget["cumulative_epochs_trained"],
        "optimizer_steps": budget["cumulative_optimizer_steps"],
        "max_epochs": budget["cumulative_max_epochs"],
        "elapsed_seconds": budget["cumulative_elapsed_seconds"],
        "copied_tensor_count": len(loaded) - len(skipped), "reinitialized_keys": sorted(skipped),
        "budget_scope": "all_executed_parent_search_epochs_not_only_selected_checkpoint_epoch",
        "artifact_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
    }
    return loaded, provenance


def load_matched_archive_parent(path: Path, **kwargs):
    path = path.resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        manifest = json.loads(archive.read("archive_manifest.json"))
        for name, expected in manifest.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Archived member hash mismatch: {name}")
        if not set(MEMBERS) <= manifest.keys():
            raise ValueError("Missing verified parent artifacts")
        contents = {name: archive.read(name) for name in MEMBERS}
        if "metrics_initial_validation.json" in manifest:
            contents["metrics_initial_validation.json"] = archive.read("metrics_initial_validation.json")
    state, provenance = validate_parent(contents, **kwargs)
    with path.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance.update(archive=str(path), archive_sha256=archive_sha)
    return state, provenance
