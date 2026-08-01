"""Training, checkpointing, and evaluation for the BSTAN baseline."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BstanTensorDataset
from .losses import BstanLossConfig, compute_multitask_loss
from .metrics import compute_metrics
from .model import BstanGatGru, BstanModelConfig


@dataclass
class BstanTrainConfig:
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_epochs", "patience"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _torch_load(
    path: Path, map_location: str | torch.device, weights_only: bool
) -> Any:
    options: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = weights_only
    return torch.load(path, **options)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "x": batch["x"],
        "adjacency": batch["adjacency"],
        "node_mask": batch["node_mask"],
        "global_features": batch["global_features"],
    }


def _load_dataset(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    payload = _torch_load(
        dataset_dir / "dataset.pt", map_location="cpu", weights_only=True
    )
    manifest = _read_json(manifest_path)
    required = {
        "x",
        "adjacency",
        "node_mask",
        "global_features",
        "y_occurrence",
        "split_indices",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"dataset.pt is missing keys: {sorted(missing)}")
    return payload, manifest


def _loaders(
    payload: dict[str, Any], config: BstanTrainConfig
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed)
    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "validation", "test"):
        indices = payload["split_indices"][split_name].tolist()
        loaders[split_name] = DataLoader(
            BstanTensorDataset(payload, indices),
            batch_size=config.batch_size,
            shuffle=split_name == "train",
            num_workers=config.num_workers,
            pin_memory=(
                torch.cuda.is_available()
                and (config.device == "auto" or config.device.startswith("cuda"))
            ),
            generator=generator if split_name == "train" else None,
        )
    return loaders


def _positive_weight(payload: dict[str, Any]) -> float:
    train_indices = payload["split_indices"]["train"]
    labels = payload["y_occurrence"][train_indices]
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    if positives <= 0:
        raise ValueError("Train split has no positive occurrence samples")
    return negatives / positives


def _run_train_epoch(
    model: BstanGatGru,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_config: BstanLossConfig,
    pos_weight: torch.Tensor,
    device: torch.device,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**_model_inputs(batch))
        loss, components = compute_multitask_loss(
            outputs, batch, loss_config, pos_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = int(batch["x"].shape[0])
        sample_count += batch_size
        values = {"total": loss, **components}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.item()) * batch_size
    return {name: value / sample_count for name, value in totals.items()}


def _evaluate_loader(
    model: BstanGatGru,
    loader: DataLoader,
    loss_config: BstanLossConfig,
    pos_weight: torch.Tensor,
    device: torch.device,
    class_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    totals: dict[str, float] = {}
    sample_count = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            outputs = model(**_model_inputs(batch))
            loss, components = compute_multitask_loss(
                outputs, batch, loss_config, pos_weight
            )
            batch_size = int(batch["x"].shape[0])
            sample_count += batch_size
            for name, value in {"total": loss, **components}.items():
                totals[name] = totals.get(name, 0.0) + float(value.item()) * batch_size

            values = {
                "sample_index": batch["sample_index"],
                "y_occurrence": batch["y_occurrence"],
                "y_node": batch["y_node"],
                "y_type": batch["y_type"],
                "y_time_to_start": batch["y_time_to_start"],
                "y_duration": batch["y_duration"],
                "y_severity": batch["y_severity"],
                "positive_mask": batch["positive_mask"],
                "duration_mask": batch["duration_mask"],
                "occurrence_probability": torch.sigmoid(outputs["occurrence_logit"]),
                "node_probabilities": torch.softmax(outputs["node_logits"], dim=-1),
                "type_predictions": outputs["type_logits"].argmax(dim=-1),
                "time_to_start": outputs["time_to_start"],
                "duration": outputs["duration"],
                "severity": outputs["severity"],
            }
            for name, value in values.items():
                collected.setdefault(name, []).append(value.detach().cpu().numpy())

    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    metrics, confusion = compute_metrics(arrays, class_count)
    metrics["loss"] = {name: value / sample_count for name, value in totals.items()}
    metrics["sample_count"] = sample_count
    return metrics, arrays, confusion


def save_checkpoint(
    path: Path,
    model: BstanGatGru,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_validation_pr_auc: float,
    model_config: BstanModelConfig,
    loss_config: BstanLossConfig,
    train_config: BstanTrainConfig,
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "epoch": epoch,
            "best_validation_pr_auc": best_validation_pr_auc,
            "model_config": model_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "train_config": asdict(train_config),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[BstanGatGru, dict[str, Any]]:
    # Checkpoints are generated by this trainer and include optimizer metadata.
    checkpoint = _torch_load(path, map_location=device, weights_only=False)
    model = BstanGatGru(BstanModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def _prediction_rows(
    arrays: dict[str, np.ndarray],
    sample_lookup: dict[int, dict[str, str]],
    node_ids: list[str],
    resource_types: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for position, raw_index in enumerate(arrays["sample_index"]):
        sample_index = int(raw_index)
        node_target = int(arrays["y_node"][position])
        node_prediction = int(arrays["node_probabilities"][position].argmax())
        type_target = int(arrays["y_type"][position])
        type_prediction = int(arrays["type_predictions"][position])
        rows.append(
            {
                **sample_lookup[sample_index],
                "occurrence_probability": float(
                    arrays["occurrence_probability"][position]
                ),
                "occurrence_prediction": int(
                    arrays["occurrence_probability"][position] >= 0.5
                ),
                "predicted_node_id": node_ids[node_prediction],
                "target_node_id_indexed": (
                    node_ids[node_target] if node_target >= 0 else ""
                ),
                "predicted_type": resource_types[type_prediction],
                "target_type_indexed": (
                    resource_types[type_target] if type_target >= 0 else ""
                ),
                "predicted_time_to_start_s": float(arrays["time_to_start"][position]),
                "target_time_to_start_s": float(arrays["y_time_to_start"][position]),
                "predicted_duration_s": float(arrays["duration"][position]),
                "target_duration_s": float(arrays["y_duration"][position]),
                "predicted_severity": float(arrays["severity"][position]),
                "target_severity": float(arrays["y_severity"][position]),
            }
        )
    return rows


def _write_evaluation_artifacts(
    output_dir: Path,
    split_name: str,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
    confusion: np.ndarray,
    dataset_dir: Path,
    manifest: dict[str, Any],
) -> None:
    _write_json(output_dir / f"metrics_{split_name}.json", metrics)
    sample_rows = _read_csv(dataset_dir / "model_sample_index.csv")
    sample_lookup = {int(row["sample_index"]): row for row in sample_rows}
    prediction_rows = _prediction_rows(
        arrays,
        sample_lookup,
        manifest["node_ids"],
        manifest["resource_types"],
    )
    prediction_fields = list(sample_rows[0]) + [
        "occurrence_probability",
        "occurrence_prediction",
        "predicted_node_id",
        "target_node_id_indexed",
        "predicted_type",
        "target_type_indexed",
        "predicted_time_to_start_s",
        "target_time_to_start_s",
        "predicted_duration_s",
        "target_duration_s",
        "predicted_severity",
        "target_severity",
    ]
    _write_csv(
        output_dir / f"predictions_{split_name}.csv",
        prediction_rows,
        prediction_fields,
    )
    confusion_rows = []
    for target_index, target_name in enumerate(manifest["resource_types"]):
        confusion_rows.append(
            {
                "target_type": target_name,
                **{
                    f"predicted__{name}": int(confusion[target_index, prediction_index])
                    for prediction_index, name in enumerate(manifest["resource_types"])
                },
            }
        )
    confusion_fields = ["target_type"] + [
        f"predicted__{name}" for name in manifest["resource_types"]
    ]
    _write_csv(
        output_dir / f"confusion_matrix_{split_name}.csv",
        confusion_rows,
        confusion_fields,
    )
    if split_name == "test":
        _write_csv(
            output_dir / "confusion_matrix.csv",
            confusion_rows,
            confusion_fields,
        )


def train_bstan_baseline(
    dataset_dir: Path,
    output_dir: Path,
    model_overrides: dict[str, Any] | None = None,
    train_config: BstanTrainConfig | None = None,
    loss_config: BstanLossConfig | None = None,
) -> dict[str, Any]:
    """Train a baseline, select by validation PR-AUC, and evaluate test data."""
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config = train_config or BstanTrainConfig()
    loss_config = loss_config or BstanLossConfig()
    _seed_everything(train_config.seed)
    device = _resolve_device(train_config.device)
    payload, manifest = _load_dataset(dataset_dir)
    if not math.isclose(
        loss_config.prediction_horizon,
        float(manifest["prediction_horizon_s"]),
    ):
        raise ValueError(
            "prediction_horizon must match dataset_manifest.json: "
            f"{loss_config.prediction_horizon} vs "
            f"{manifest['prediction_horizon_s']}"
        )
    model_values = {
        "input_dim": int(payload["x"].shape[-1]),
        "global_dim": int(payload["global_features"].shape[-1]),
        "num_nodes": int(payload["x"].shape[-2]),
        "num_types": len(manifest["resource_types"]),
        "prediction_horizon": loss_config.prediction_horizon,
        **(model_overrides or {}),
    }
    model_config = BstanModelConfig(**model_values)
    model = BstanGatGru(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loaders = _loaders(payload, train_config)
    pos_weight_value = _positive_weight(payload)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    manifest_path = dataset_dir / "dataset_manifest.json"
    metadata = {
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": _manifest_hash(manifest_path),
        "dataset_version": manifest["dataset_version"],
        "label_version": manifest["label_version"],
        "feature_names": manifest["feature_names"],
        "global_feature_names": manifest["global_feature_names"],
        "node_ids": manifest["node_ids"],
        "resource_types": manifest["resource_types"],
        "normalization": _read_json(dataset_dir / "normalization.json"),
        "git_commit": _git_commit(_repo_root(Path(__file__).resolve())),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "pos_weight": pos_weight_value,
    }
    config_payload = {
        "model": model_config.to_dict(),
        "loss": loss_config.to_dict(),
        "training": asdict(train_config),
        "metadata": metadata,
    }
    _write_json(output_dir / "config.json", config_payload)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started_at = time.time()
    for epoch in range(1, train_config.max_epochs + 1):
        train_losses = _run_train_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_config,
            pos_weight,
            device,
            train_config.gradient_clip_norm,
        )
        validation_metrics, _, _ = _evaluate_loader(
            model,
            loaders["validation"],
            loss_config,
            pos_weight,
            device,
            model_config.num_types,
        )
        validation_score = validation_metrics["occurrence"]["pr_auc"]
        score = float(validation_score) if validation_score is not None else -1.0
        row = {"epoch": epoch}
        row.update({f"train_{name}": value for name, value in train_losses.items()})
        row.update(
            {
                "validation_total_loss": validation_metrics["loss"]["total"],
                "validation_pr_auc": validation_score,
                "validation_roc_auc": validation_metrics["occurrence"]["roc_auc"],
                "validation_f1_at_0_5": validation_metrics["occurrence"]["f1_at_0_5"],
            }
        )
        history.append(row)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            max(best_score, score),
            model_config,
            loss_config,
            train_config,
            metadata,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_score,
                model_config,
                loss_config,
                train_config,
                metadata,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:03d} train_loss={train_losses['total']:.6f} "
            f"val_loss={validation_metrics['loss']['total']:.6f} "
            f"val_pr_auc={score:.6f}",
            flush=True,
        )
        if epochs_without_improvement >= train_config.patience:
            break

    _write_csv(output_dir / "history.csv", history, list(history[0]))
    best_model, checkpoint = load_checkpoint(output_dir / "best.pt", device)
    all_metrics: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        metrics, arrays, confusion = _evaluate_loader(
            best_model,
            loaders[split_name],
            loss_config,
            pos_weight,
            device,
            model_config.num_types,
        )
        all_metrics[split_name] = metrics
        _write_evaluation_artifacts(
            output_dir,
            split_name,
            metrics,
            arrays,
            confusion,
            dataset_dir,
            manifest,
        )
    _write_json(output_dir / "metrics.json", all_metrics)
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "best_validation_pr_auc": best_score,
        "test_pr_auc": all_metrics["test"]["occurrence"]["pr_auc"],
        "test_f1_at_0_5": all_metrics["test"]["occurrence"]["f1_at_0_5"],
        "elapsed_seconds": time.time() - started_at,
        "checkpoint_epoch": checkpoint["epoch"],
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def evaluate_checkpoint(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split_name: str = "test",
    device_name: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint against one dataset split."""
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split_name}")
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)
    payload, manifest = _load_dataset(dataset_dir)
    model, checkpoint = load_checkpoint(checkpoint_path.resolve(), device)
    expected_hash = checkpoint["metadata"]["dataset_manifest_sha256"]
    actual_hash = _manifest_hash(dataset_dir / "dataset_manifest.json")
    if expected_hash != actual_hash:
        raise ValueError("Checkpoint and dataset manifest hashes do not match")
    train_config = BstanTrainConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_name,
    )
    loaders = _loaders(payload, train_config)
    loss_config = BstanLossConfig.from_dict(checkpoint["loss_config"])
    pos_weight = torch.tensor(checkpoint["metadata"]["pos_weight"], device=device)
    metrics, arrays, confusion = _evaluate_loader(
        model,
        loaders[split_name],
        loss_config,
        pos_weight,
        device,
        model.config.num_types,
    )
    _write_evaluation_artifacts(
        output_dir,
        split_name,
        metrics,
        arrays,
        confusion,
        dataset_dir,
        manifest,
    )
    metrics_path = output_dir / "metrics.json"
    all_metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    all_metrics[split_name] = metrics
    _write_json(metrics_path, all_metrics)

    summary_path = output_dir / "run_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    summary.update(
        {
            "status": "evaluation_completed",
            "best_epoch": checkpoint["epoch"],
            "best_validation_pr_auc": checkpoint["best_validation_pr_auc"],
            "checkpoint": str(checkpoint_path.resolve()),
            "output_dir": str(output_dir),
        }
    )
    if split_name == "test":
        summary.update(
            {
                "test_pr_auc": metrics["occurrence"]["pr_auc"],
                "test_f1_at_0_5": metrics["occurrence"]["f1_at_0_5"],
            }
        )
    _write_json(summary_path, summary)
    return metrics
