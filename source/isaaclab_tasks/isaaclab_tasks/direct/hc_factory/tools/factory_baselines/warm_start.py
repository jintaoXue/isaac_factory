"""Validate a completed first-stage run before a weights-only second stage."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import torch


def load_warm_start_parent(
    path: Path,
    *,
    model_kind: str,
    model_config: dict[str, Any],
    seed: int,
    dataset_manifest_sha256: str,
    train_sample_count: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Require a complete, same-seed, same-contract validation-only first stage."""
    path = path.resolve()
    files = {
        "checkpoint": path,
        "config": path.parent / "config.json",
        "history": path.parent / "history.csv",
        "summary": path.parent / "run_summary.json",
        "validation": path.parent / "metrics_validation.json",
    }
    contents = {name: file.read_bytes() for name, file in files.items()}
    checkpoint = torch.load(
        io.BytesIO(contents["checkpoint"]), map_location="cpu", weights_only=False
    )
    config = json.loads(contents["config"])
    summary = json.loads(contents["summary"])
    validation = json.loads(contents["validation"])
    metadata = checkpoint["metadata"]
    training = checkpoint["train_config"]
    if summary["status"] != "validation_completed" or training["evaluate_test"]:
        raise ValueError("Warm start requires a completed validation-only parent")
    if "warm_start_parent" in metadata:
        raise ValueError("Warm start requires a first-stage parent, not another warm stage")
    if checkpoint["model_kind"] != model_kind or summary["model_kind"] != model_kind:
        raise ValueError("Warm-start parent model kind does not match")
    if checkpoint["model_config"] != model_config or config["model"] != model_config:
        raise ValueError("Warm-start parent model configuration does not match")
    if any(value != seed for value in (training["seed"], metadata["seed"], summary["seed"])):
        raise ValueError("Warm-start parent seed does not match")
    if any(value != dataset_manifest_sha256 for value in (
        metadata["dataset_manifest_sha256"], config["metadata"]["dataset_manifest_sha256"],
    )):
        raise ValueError("Warm-start parent dataset manifest does not match")
    if json.loads(json.dumps(training)) != config["training"]:
        raise ValueError("Warm-start parent training configuration is inconsistent")
    if checkpoint["loss_config"] != config["loss"]:
        raise ValueError("Warm-start parent loss configuration is inconsistent")
    if not metadata["git_commit"] or metadata["git_commit"] == "unknown":
        raise ValueError("Warm-start parent has no recorded source commit")

    epochs = int(summary["epochs_trained"])
    selected_epoch = int(checkpoint["epoch"])
    if not 1 <= selected_epoch <= epochs <= int(training["max_epochs"]):
        raise ValueError("Warm-start parent epoch budget is inconsistent")
    if selected_epoch != summary["best_epoch"] or selected_epoch != summary["checkpoint_epoch"]:
        raise ValueError("Warm start requires the selected parent checkpoint")
    history = list(csv.DictReader(io.StringIO(contents["history"].decode("utf-8"))))
    if [int(row["epoch"]) for row in history] != list(range(1, epochs + 1)):
        raise ValueError("Warm-start parent history is incomplete")
    for row in history:
        for key in ("train_total", "validation_total_loss", "learning_rate"):
            if not math.isfinite(float(row[key])):
                raise ValueError("Warm-start parent history contains non-finite values")
    for suffix in ("precision", "recall", "f1"):
        measured = float(validation["station_report"][f"report_{suffix}"])
        recorded = float(history[selected_epoch - 1][f"validation_report_{suffix}"])
        best = float(summary[f"best_validation_report_{suffix}"])
        if not all(math.isfinite(value) and math.isclose(value, measured, abs_tol=1e-6)
                   for value in (recorded, best)):
            raise ValueError("Warm-start parent selected validation metrics are inconsistent")
    elapsed = float(summary["elapsed_seconds"])
    batch_size = int(training["batch_size"])
    if not math.isfinite(elapsed) or elapsed < 0 or batch_size <= 0 or train_sample_count <= 0:
        raise ValueError("Warm-start parent training budget is invalid")
    state = checkpoint["model_state_dict"]
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Warm-start parent weights contain non-finite values")
    provenance = {
        "mode": "weights_only_new_optimizer",
        "checkpoint": str(path),
        "artifact_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
        },
        "source_commit": metadata["git_commit"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "epochs_trained": epochs,
        "max_epochs": int(training["max_epochs"]),
        "batch_size": batch_size,
        "optimizer_steps": epochs * math.ceil(train_sample_count / batch_size),
        "elapsed_seconds": elapsed,
    }
    return state, provenance
