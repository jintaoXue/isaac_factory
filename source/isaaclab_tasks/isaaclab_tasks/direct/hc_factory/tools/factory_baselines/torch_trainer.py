"""Training, checkpointing, and evaluation shared by B3-B5."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from factory_bn_shared.remain import (
    occupancy_event_metrics,
    rasterize_node_events,
    station_report_metrics,
)

from .dataset import FactoryBaselineTensorDataset, load_shared_dataset
from .torch_losses import MultiTaskLossConfig, compute_multitask_loss
from .metrics import (
    REPORT_THRESHOLD_SWEEP,
    _binary_metrics,
    compute_metrics,
    hot_grid_metrics,
    select_report_threshold,
    training_cause_majority,
)
from .b3_lstm import B3Lstm, B3ModelConfig
from .b4_gcn_gru import B4GcnGru, B4ModelConfig
from .b5_gat_gru import B5GatGru, B5ModelConfig


@dataclass
class TorchTrainConfig:
    training_profile: str = "baseline_fair_v1"
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 25
    min_epochs: int = 25
    learning_rate: float = 1.5e-4
    weight_decay: float = 5.0e-2
    lr_min: float = 1.0e-6
    lr_schedule: str = "cosine"
    gradient_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    hot_eval_threshold: float = 0.55
    event_report_threshold: float = 0.68
    report_threshold_sweep: tuple[float, ...] = REPORT_THRESHOLD_SWEEP
    checkpoint_min_report_precision: float = 0.80
    checkpoint_min_report_recall: float = 0.35

    def __post_init__(self) -> None:
        if not self.training_profile.strip():
            raise ValueError("training_profile must not be empty")
        for name in ("batch_size", "max_epochs", "patience", "min_epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.lr_schedule not in {"none", "cosine"}:
            raise ValueError("lr_schedule must be 'none' or 'cosine'")
        for name in (
            "hot_eval_threshold",
            "event_report_threshold",
            "checkpoint_min_report_precision",
            "checkpoint_min_report_recall",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.report_threshold_sweep:
            raise ValueError("report_threshold_sweep must not be empty")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.report_threshold_sweep):
            raise ValueError("report_threshold_sweep values must be in [0, 1]")


def _model_spec(model_kind: str) -> tuple[type[nn.Module], type, str, str]:
    specs = {
        "b3_lstm": (B3Lstm, B3ModelConfig, "B3", "LSTM"),
        "b4_gcn_gru": (B4GcnGru, B4ModelConfig, "B4", "GCN-GRU"),
        "b5_gat_gru": (B5GatGru, B5ModelConfig, "B5", "BSTAN-style GAT-GRU"),
    }
    try:
        return specs[model_kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model_kind {model_kind!r}; expected one of {sorted(specs)}"
        ) from exc


def _validation_checkpoint_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Lexicographic validation rank used after the precision gate.

    Report F1 remains primary. Hot F1 and lower validation loss only break ties,
    so an all-zero report curve cannot pin the fallback checkpoint to epoch 1.
    """
    return (
        float(metrics["station_report"]["report_f1"]),
        float(metrics["remain"]["hot_f1"]),
        -float(metrics["loss"]["total"]),
    )


def _rank_improved(
    candidate: tuple[float, ...],
    incumbent: tuple[float, ...],
    epsilon: float = 1.0e-6,
) -> bool:
    for candidate_value, incumbent_value in zip(candidate, incumbent):
        if candidate_value > incumbent_value + epsilon:
            return True
        if candidate_value < incumbent_value - epsilon:
            return False
    return False


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
        "target_node_mask": batch["target_node_mask"],
        "global_features": batch["global_features"],
        "jobs_remaining": batch["jobs_remaining"],
        "jobs_total": batch["jobs_total"],
    }


def _loaders(
    payload: dict[str, Any], config: TorchTrainConfig
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed)
    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "validation", "test"):
        indices = payload["split_indices"][split_name].tolist()
        loaders[split_name] = DataLoader(
            FactoryBaselineTensorDataset(payload, indices),
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


def _occupancy_type_masks(
    dataset_dir: Path,
    payload: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = _read_csv(dataset_dir / "node_catalog.csv")
    node_count = int(payload["x"].shape[-2])
    masks = {
        name: torch.zeros(node_count, dtype=torch.float32, device=device)
        for name in ("machine", "workbench", "gantry", "agv")
    }
    for row in rows:
        index = int(row["node_index"])
        resource_id = str(row["resource_id"]).lower()
        resource_type = str(row["resource_type"]).lower()
        if "workbench" in resource_id:
            name = "workbench"
        elif resource_type == "machine":
            name = "machine"
        elif resource_type == "gantry":
            name = "gantry"
        elif resource_type in {"transport_robot", "agv"}:
            name = "agv"
        else:
            continue
        masks[name][index] = 1.0
    tagged = torch.stack(list(masks.values())).sum(dim=0) > 0
    target = payload["target_node_mask"].any(dim=0).to(device=device)
    if bool((target & ~tagged).any()):
        missing = torch.nonzero(target & ~tagged, as_tuple=False).reshape(-1).tolist()
        raise ValueError(f"Occupancy target nodes lack type masks: {missing}")
    return masks


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_config: MultiTaskLossConfig,
    pos_weight: torch.Tensor,
    device: torch.device,
    gradient_clip_norm: float,
    occupancy_type_masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**_model_inputs(batch))
        loss, components = compute_multitask_loss(
            outputs,
            batch,
            loss_config,
            pos_weight,
            occupancy_type_masks=occupancy_type_masks,
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


def _occupancy_events(grid: np.ndarray, length: int) -> list[dict[str, int]]:
    """Collapse connected hot runs into A.1 node/start/duration events."""
    events = []
    event_id = 0
    usable = grid[: max(min(length, grid.shape[0]), 0)]
    for node_index in range(usable.shape[1]):
        offset = 0
        while offset < usable.shape[0]:
            if not usable[offset, node_index]:
                offset += 1
                continue
            end = offset + 1
            while end < usable.shape[0] and usable[end, node_index]:
                end += 1
            events.append(
                {
                    "event_id": event_id,
                    "node_index": node_index,
                    "start_offset_windows": offset,
                    "duration_windows": end - offset,
                }
            )
            event_id += 1
            offset = end
    return events


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_config: MultiTaskLossConfig,
    pos_weight: torch.Tensor,
    device: torch.device,
    cause_class_count: int,
    cause_classes: list[str] | tuple[str, ...] | None = None,
    cause_majority: int = -1,
    event_threshold: float = 0.68,
    report_threshold_sweep: tuple[float, ...] | list[float] = (),
    min_report_precision: float = 0.80,
    hot_threshold: float = 0.55,
    occupancy_type_masks: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    totals: dict[str, float] = {}
    sample_count = 0
    score_abs_sum = 0.0
    score_count = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            outputs = model(**_model_inputs(batch))
            loss, components = compute_multitask_loss(
                outputs,
                batch,
                loss_config,
                pos_weight,
                occupancy_type_masks=occupancy_type_masks,
            )
            batch_size = int(batch["x"].shape[0])
            sample_count += batch_size
            for name, value in {"total": loss, **components}.items():
                totals[name] = totals.get(name, 0.0) + float(value.item()) * batch_size

            values = {
                "sample_index": batch["sample_index"],
                "y_cause": batch["y_cause"],
                "target_remain_len": batch["target_remain_len"],
                "cause_predictions": outputs["cause_logits"].argmax(dim=-1),
                "remain_len": outputs["remain_len"],
                "y_hot_grid": batch["y_hot"],
                "remain_mask_grid": batch["remain_mask"],
                "occ_node_mask_grid": batch["occ_node_mask"],
                "hist_last_hot_grid": batch["hist_last_hot"],
                "hot_probability_grid": torch.sigmoid(outputs["remain_hot_logit"]),
                "event_will_target": batch["event_will"],
                "event_will_probability": torch.sigmoid(outputs["event_will_logit"]),
                "event_start_index": outputs["event_start_logit"].argmax(dim=-1),
                "event_duration_windows": outputs["event_duration"],
            }
            for name, value in values.items():
                collected.setdefault(name, []).append(value.detach().cpu().numpy())

            predicted_grid = (
                (torch.sigmoid(outputs["remain_hot_logit"]) >= hot_threshold)
                .detach()
                .cpu()
                .numpy()
            )
            target_grid = batch["y_hot"].bool().detach().cpu().numpy()
            predicted_lengths = outputs["remain_len"].round().long().detach().cpu()
            target_lengths = batch["target_remain_len"].long().detach().cpu()
            predicted_events = np.asarray(
                [
                    json.dumps(
                        _occupancy_events(predicted_grid[i], int(predicted_lengths[i])),
                        separators=(",", ":"),
                    )
                    for i in range(batch_size)
                ]
            )
            target_events = np.asarray(
                [
                    json.dumps(
                        _occupancy_events(target_grid[i], int(target_lengths[i])),
                        separators=(",", ":"),
                    )
                    for i in range(batch_size)
                ]
            )
            collected.setdefault("predicted_events_json", []).append(predicted_events)
            collected.setdefault("target_events_json", []).append(target_events)

            valid = (
                batch["remain_mask"].bool()[:, :, None]
                & batch["occ_node_mask"].bool()[:, None, :]
            )
            score_valid = valid[:, :, :, None].expand_as(outputs["remain_score"])
            score_abs_sum += float(
                torch.abs(outputs["remain_score"] - batch["y_score"])[score_valid]
                .sum()
                .item()
            )
            score_count += int(score_valid.sum().item())

    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    metrics, confusion = compute_metrics(
        arrays,
        cause_class_count,
        cause_classes=cause_classes,
        cause_majority=cause_majority,
    )
    type_masks = {
        name: mask.detach().cpu().numpy()
        for name, mask in (occupancy_type_masks or {}).items()
    }
    hot_metrics = hot_grid_metrics(
        arrays["y_hot_grid"],
        arrays["hot_probability_grid"],
        arrays["remain_mask_grid"],
        arrays["occ_node_mask_grid"],
        threshold=hot_threshold,
        type_masks=type_masks,
    )
    remain_len_mae = float(
        np.abs(arrays["remain_len"] - arrays["target_remain_len"]).mean()
    )
    metrics["remain"] = {
        **hot_metrics,
        "score_mae": score_abs_sum / max(score_count, 1),
        "remain_len_mae": remain_len_mae,
        "remain_len_mae_windows": remain_len_mae,
    }
    metrics.update(hot_metrics)
    metrics["remain_len_mae"] = remain_len_mae
    if report_threshold_sweep:
        report_metrics = select_report_threshold(
            arrays["y_hot_grid"],
            arrays["event_will_probability"],
            arrays["event_start_index"],
            arrays["event_duration_windows"],
            arrays["remain_mask_grid"],
            arrays["occ_node_mask_grid"],
            default_threshold=event_threshold,
            threshold_sweep=report_threshold_sweep,
            min_precision=min_report_precision,
            min_windows=8,
            start_tol_windows=3,
            hist_last_hot=arrays["hist_last_hot_grid"],
        )
    else:
        report_metrics = station_report_metrics(
            arrays["y_hot_grid"],
            arrays["event_will_probability"],
            arrays["event_start_index"],
            arrays["event_duration_windows"],
            arrays["remain_mask_grid"],
            arrays["occ_node_mask_grid"],
            threshold=event_threshold,
            min_windows=8,
            start_tol_windows=3,
            hist_last_hot=arrays["hist_last_hot_grid"],
            force_ongoing_will=False,
        )
        report_metrics["report_threshold_used"] = float(event_threshold)
    metrics["station_report"] = report_metrics
    metrics.update(report_metrics)
    selected_threshold = float(report_metrics["report_threshold_used"])
    event_valid = arrays["occ_node_mask_grid"] > 0.5
    event_labels = arrays["event_will_target"][event_valid]
    event_probabilities = arrays["event_will_probability"][event_valid]
    event_will_metrics = _binary_metrics(
        event_labels,
        event_probabilities,
        threshold=selected_threshold,
    )
    event_will_metrics["probability_quantiles"] = {
        str(quantile): float(np.quantile(event_probabilities, quantile))
        for quantile in (0.5, 0.9, 0.95, 0.99)
    }
    event_will_metrics["probability_max"] = float(event_probabilities.max())
    metrics["event_will"] = event_will_metrics
    event_occupancy = rasterize_node_events(
        arrays["event_will_probability"],
        arrays["event_start_index"],
        arrays["event_duration_windows"],
        arrays["y_hot_grid"].shape[1],
        threshold=selected_threshold,
        min_windows=8,
    )
    occupancy_metrics = occupancy_event_metrics(
        arrays["y_hot_grid"],
        event_occupancy,
        arrays["remain_mask_grid"],
        arrays["occ_node_mask_grid"],
        [str(index) for index in range(arrays["y_hot_grid"].shape[-1])],
        threshold=0.5,
        min_windows=8,
        iou_min=0.5,
        window_size_s=60.0,
    )
    metrics["occupancy_event"] = occupancy_metrics
    metrics.update(occupancy_metrics)
    metrics["loss"] = {name: value / sample_count for name, value in totals.items()}
    metrics["sample_count"] = sample_count
    return metrics, arrays, confusion


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_validation_report_f1: float,
    model_kind: str,
    model_config: Any,
    loss_config: MultiTaskLossConfig,
    train_config: TorchTrainConfig,
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_kind": model_kind,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "epoch": epoch,
            "best_validation_report_f1": best_validation_report_f1,
            "model_config": model_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "train_config": asdict(train_config),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    # Checkpoints are generated by this trainer and include optimizer metadata.
    checkpoint = _torch_load(path, map_location=device, weights_only=False)
    model_kind = checkpoint["model_kind"]
    model_class, config_class, _baseline_id, _model_name = _model_spec(model_kind)
    model = model_class(config_class.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def _prediction_rows(
    arrays: dict[str, np.ndarray],
    sample_lookup: dict[int, dict[str, str]],
    cause_classes: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for position, raw_index in enumerate(arrays["sample_index"]):
        sample_index = int(raw_index)
        cause_target = int(arrays["y_cause"][position])
        cause_prediction = int(arrays["cause_predictions"][position])
        rows.append(
            {
                **sample_lookup[sample_index],
                "predicted_cause": cause_classes[cause_prediction],
                "target_cause": (
                    cause_classes[cause_target] if cause_target >= 0 else ""
                ),
                "predicted_remain_len_windows": float(arrays["remain_len"][position]),
                "target_remain_len_windows": int(arrays["target_remain_len"][position]),
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
        manifest["cause_classes"],
    )
    prediction_fields = list(sample_rows[0]) + [
        "predicted_cause",
        "target_cause",
        "predicted_remain_len_windows",
        "target_remain_len_windows",
    ]
    _write_csv(
        output_dir / f"predictions_{split_name}.csv",
        prediction_rows,
        prediction_fields,
    )
    event_rows = []
    window_size_s = float(manifest["window_size_s"])
    for position, raw_index in enumerate(arrays["sample_index"]):
        sample = sample_lookup[int(raw_index)]
        first_future_start_s = float(sample["first_future_start_s"])
        for source, field in (
            ("prediction", "predicted_events_json"),
            ("target", "target_events_json"),
        ):
            for event in json.loads(str(arrays[field][position])):
                start_s = first_future_start_s + (
                    event["start_offset_windows"] * window_size_s
                )
                duration_s = event["duration_windows"] * window_size_s
                event_rows.append(
                    {
                        "sample_index": int(raw_index),
                        "split": split_name,
                        "source": source,
                        "event_id": event["event_id"],
                        "resource_id": manifest["node_ids"][event["node_index"]],
                        "start_s": start_s,
                        "end_s": start_s + duration_s,
                        "duration_s": duration_s,
                        "n_windows": event["duration_windows"],
                    }
                )
    _write_csv(
        output_dir / f"occupancy_events_{split_name}.csv",
        event_rows,
        [
            "sample_index",
            "split",
            "source",
            "event_id",
            "resource_id",
            "start_s",
            "end_s",
            "duration_s",
            "n_windows",
        ],
    )
    confusion_rows = []
    for target_index, target_name in enumerate(manifest["cause_classes"]):
        confusion_rows.append(
            {
                "target_cause": target_name,
                **{
                    f"predicted__{name}": int(confusion[target_index, prediction_index])
                    for prediction_index, name in enumerate(manifest["cause_classes"])
                },
            }
        )
    confusion_fields = ["target_cause"] + [
        f"predicted__{name}" for name in manifest["cause_classes"]
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


def train_torch_baseline(
    model_kind: str,
    dataset_dir: Path,
    output_dir: Path,
    model_overrides: dict[str, Any] | None = None,
    train_config: TorchTrainConfig | None = None,
    loss_config: MultiTaskLossConfig | None = None,
) -> dict[str, Any]:
    """Train one B3-B5 model and evaluate it with the shared protocol."""
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config = train_config or TorchTrainConfig()
    loss_config = loss_config or MultiTaskLossConfig()
    _seed_everything(train_config.seed)
    device = _resolve_device(train_config.device)
    payload, manifest = load_shared_dataset(dataset_dir)
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
        "prediction_horizon": loss_config.prediction_horizon,
        "max_remain_windows": int(manifest["max_remain_windows"]),
        "num_causes": len(manifest["cause_classes"]),
        **(model_overrides or {}),
    }
    model_class, config_class, baseline_id, model_name = _model_spec(model_kind)
    model_config = config_class(**model_values)
    model = model_class(model_config).to(device)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(train_config.max_epochs, 1),
            eta_min=train_config.lr_min,
        )
        if train_config.lr_schedule == "cosine"
        else None
    )
    loaders = _loaders(payload, train_config)
    occupancy_type_masks = _occupancy_type_masks(dataset_dir, payload, device)
    cause_classes = list(manifest["cause_classes"])
    train_indices = payload["split_indices"]["train"]
    cause_majority = training_cause_majority(
        payload["y_cause"][train_indices].numpy(),
        cause_classes,
    )
    pos_weight_value = 1.0
    pos_weight = torch.tensor(pos_weight_value, device=device)
    manifest_path = dataset_dir / "dataset_manifest.json"
    metadata = {
        "baseline_id": baseline_id,
        "model_name": model_name,
        "model_kind": model_kind,
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": _manifest_hash(manifest_path),
        "dataset_version": manifest["dataset_version"],
        "dataset_contract": manifest["dataset_contract"],
        "label_version": manifest["label_version"],
        "feature_names": manifest["feature_names"],
        "global_feature_names": manifest["global_feature_names"],
        "node_ids": manifest["node_ids"],
        "resource_types": manifest["resource_types"],
        "normalization": _read_json(dataset_dir / "normalization.json"),
        "git_commit": _git_commit(_repo_root(Path(__file__).resolve())),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "training_profile": train_config.training_profile,
        "trainable_parameter_count": trainable_parameter_count,
        "pos_weight": pos_weight_value,
        "cause_majority": cause_majority,
        "occupancy_type_masks": {
            name: torch.nonzero(mask > 0.5, as_tuple=False).reshape(-1).tolist()
            for name, mask in occupancy_type_masks.items()
        },
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
    best_precision = 0.0
    best_recall = 0.0
    best_hot_f1 = -1.0
    best_validation_loss = float("inf")
    empty_rank = (float("-inf"),) * 3
    best_rank = empty_rank
    fallback_score = -1.0
    fallback_epoch = 0
    fallback_precision = 0.0
    fallback_recall = 0.0
    fallback_hot_f1 = -1.0
    fallback_validation_loss = float("inf")
    fallback_rank = empty_rank
    checkpoint_constraint_met = False
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
            occupancy_type_masks,
        )
        validation_metrics, _, _ = _evaluate_loader(
            model,
            loaders["validation"],
            loss_config,
            pos_weight,
            device,
            model_config.num_causes,
            cause_classes=cause_classes,
            cause_majority=cause_majority,
            event_threshold=train_config.event_report_threshold,
            report_threshold_sweep=train_config.report_threshold_sweep,
            min_report_precision=train_config.checkpoint_min_report_precision,
            hot_threshold=train_config.hot_eval_threshold,
            occupancy_type_masks=occupancy_type_masks,
        )
        report = validation_metrics["station_report"]
        validation_score = float(report["report_f1"])
        validation_precision = float(report["report_precision"])
        validation_recall = float(report["report_recall"])
        selected_threshold = float(report["report_threshold_used"])
        validation_hot_f1 = float(validation_metrics["remain"]["hot_f1"])
        validation_loss = float(validation_metrics["loss"]["total"])
        validation_will_ap = validation_metrics["event_will"]["pr_auc"]
        candidate_rank = _validation_checkpoint_rank(validation_metrics)
        feasible = (
            validation_precision >= train_config.checkpoint_min_report_precision
            and validation_recall >= train_config.checkpoint_min_report_recall
        )
        fallback_improved = _rank_improved(candidate_rank, fallback_rank)
        feasible_improved = feasible and _rank_improved(candidate_rank, best_rank)
        row = {"epoch": epoch}
        row.update({f"train_{name}": value for name, value in train_losses.items()})
        row.update(
            {
                "validation_total_loss": validation_metrics["loss"]["total"],
                "validation_hot_f1": validation_metrics["remain"]["hot_f1"],
                "validation_report_f1": report["report_f1"],
                "validation_report_precision": report["report_precision"],
                "validation_report_recall": report["report_recall"],
                "validation_event_threshold": selected_threshold,
                "validation_hot_threshold": train_config.hot_eval_threshold,
                "validation_event_will_pr_auc": validation_metrics["event_will"][
                    "pr_auc"
                ],
                "validation_event_will_probability_q95": validation_metrics[
                    "event_will"
                ]["probability_quantiles"]["0.95"],
                "fallback_checkpoint_improved": int(fallback_improved),
                "feasible_checkpoint_improved": int(feasible_improved),
                "checkpoint_precision_constraint_met": int(
                    validation_precision >= train_config.checkpoint_min_report_precision
                ),
                "checkpoint_recall_constraint_met": int(
                    validation_recall >= train_config.checkpoint_min_report_recall
                ),
                "checkpoint_constraints_met": int(feasible),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        history.append(row)
        epoch_metadata = {
            **metadata,
            "event_report_threshold": selected_threshold,
            "hot_eval_threshold": train_config.hot_eval_threshold,
        }
        if fallback_improved:
            fallback_rank = candidate_rank
            fallback_score = validation_score
            fallback_epoch = epoch
            fallback_precision = validation_precision
            fallback_recall = validation_recall
            fallback_hot_f1 = validation_hot_f1
            fallback_validation_loss = validation_loss
            save_checkpoint(
                output_dir / "fallback_best.pt",
                model,
                optimizer,
                epoch,
                fallback_score,
                model_kind,
                model_config,
                loss_config,
                train_config,
                epoch_metadata,
            )
        if feasible_improved:
            checkpoint_constraint_met = True
            best_rank = candidate_rank
            best_score = validation_score
            best_epoch = epoch
            best_precision = validation_precision
            best_recall = validation_recall
            best_hot_f1 = validation_hot_f1
            best_validation_loss = validation_loss
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_score,
                model_kind,
                model_config,
                loss_config,
                train_config,
                epoch_metadata,
            )
        selection_improved = (
            feasible_improved if checkpoint_constraint_met else fallback_improved
        )
        if selection_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_score if checkpoint_constraint_met else fallback_score,
            model_kind,
            model_config,
            loss_config,
            train_config,
            epoch_metadata,
        )
        print(
            f"epoch={epoch:03d} train_loss={train_losses['total']:.6f} "
            f"val_loss={validation_metrics['loss']['total']:.6f} "
            f"val_report_f1={report['report_f1']:.6f} "
            f"val_report_precision={report['report_precision']:.6f} "
            f"val_hot_f1={validation_hot_f1:.6f} "
            f"will_ap={float(validation_will_ap or 0.0):.6f} "
            f"feasible={int(feasible)}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()
        if (
            epoch >= min(train_config.min_epochs, train_config.max_epochs)
            and epochs_without_improvement >= train_config.patience
        ):
            break

    _write_csv(output_dir / "history.csv", history, list(history[0]))
    if not checkpoint_constraint_met:
        shutil.copy2(output_dir / "fallback_best.pt", output_dir / "best.pt")
        best_score = fallback_score
        best_epoch = fallback_epoch
        best_precision = fallback_precision
        best_recall = fallback_recall
        best_hot_f1 = fallback_hot_f1
        best_validation_loss = fallback_validation_loss
    best_model, checkpoint = load_checkpoint(output_dir / "best.pt", device)
    event_threshold = float(
        checkpoint["metadata"].get(
            "event_report_threshold", train_config.event_report_threshold
        )
    )
    validation_metrics, validation_arrays, validation_confusion = _evaluate_loader(
        best_model,
        loaders["validation"],
        loss_config,
        pos_weight,
        device,
        model_config.num_causes,
        cause_classes=cause_classes,
        cause_majority=cause_majority,
        event_threshold=event_threshold,
        hot_threshold=train_config.hot_eval_threshold,
        occupancy_type_masks=occupancy_type_masks,
    )
    checkpoint["metadata"]["event_report_threshold"] = event_threshold
    checkpoint["metadata"]["hot_eval_threshold"] = train_config.hot_eval_threshold
    checkpoint["metadata"]["checkpoint_constraint_met"] = checkpoint_constraint_met
    torch.save(checkpoint, output_dir / "best.pt")
    config_payload["metadata"]["event_report_threshold"] = event_threshold
    config_payload["metadata"]["hot_eval_threshold"] = train_config.hot_eval_threshold
    config_payload["metadata"]["checkpoint_constraint_met"] = checkpoint_constraint_met
    _write_json(output_dir / "config.json", config_payload)
    test_metrics, test_arrays, test_confusion = _evaluate_loader(
        best_model,
        loaders["test"],
        loss_config,
        pos_weight,
        device,
        model_config.num_causes,
        cause_classes=cause_classes,
        cause_majority=cause_majority,
        event_threshold=event_threshold,
        hot_threshold=train_config.hot_eval_threshold,
        occupancy_type_masks=occupancy_type_masks,
    )
    evaluations = {
        "validation": (
            validation_metrics,
            validation_arrays,
            validation_confusion,
        ),
        "test": (test_metrics, test_arrays, test_confusion),
    }
    all_metrics: dict[str, Any] = {}
    for split_name, (metrics, arrays, confusion) in evaluations.items():
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
        "baseline_id": baseline_id,
        "model_name": model_name,
        "model_kind": model_kind,
        "training_profile": train_config.training_profile,
        "trainable_parameter_count": trainable_parameter_count,
        "dataset_contract": manifest["dataset_contract"],
        "dataset_version": manifest["dataset_version"],
        "label_version": manifest["label_version"],
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "best_validation_report_f1": best_score,
        "best_validation_report_precision": best_precision,
        "best_validation_report_recall": best_recall,
        "best_validation_hot_f1": best_hot_f1,
        "best_validation_total_loss": best_validation_loss,
        "checkpoint_precision_constraint": (
            train_config.checkpoint_min_report_precision
        ),
        "checkpoint_recall_constraint": (train_config.checkpoint_min_report_recall),
        "checkpoint_constraint_met": checkpoint_constraint_met,
        "checkpoint_selection": (
            "constrained_report_f1_hot_f1_val_loss"
            if checkpoint_constraint_met
            else "fallback_report_f1_hot_f1_val_loss"
        ),
        "event_report_threshold": event_threshold,
        "report_threshold_selected_on": "validation",
        "hot_eval_threshold": train_config.hot_eval_threshold,
        "test_hot_f1": all_metrics["test"]["remain"]["hot_f1"],
        "test_report_f1": all_metrics["test"]["station_report"]["report_f1"],
        "test_report_precision": all_metrics["test"]["station_report"][
            "report_precision"
        ],
        "test_report_recall": all_metrics["test"]["station_report"]["report_recall"],
        "elapsed_seconds": time.time() - started_at,
        "checkpoint_epoch": checkpoint["epoch"],
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def evaluate_torch_checkpoint(
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
    payload, manifest = load_shared_dataset(dataset_dir)
    model, checkpoint = load_checkpoint(checkpoint_path.resolve(), device)
    expected_hash = checkpoint["metadata"]["dataset_manifest_sha256"]
    actual_hash = _manifest_hash(dataset_dir / "dataset_manifest.json")
    if expected_hash != actual_hash:
        raise ValueError("Checkpoint and dataset manifest hashes do not match")
    train_config = TorchTrainConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_name,
    )
    loaders = _loaders(payload, train_config)
    loss_config = MultiTaskLossConfig.from_dict(checkpoint["loss_config"])
    pos_weight = torch.tensor(checkpoint["metadata"]["pos_weight"], device=device)
    event_threshold = float(checkpoint["metadata"]["event_report_threshold"])
    hot_threshold = float(checkpoint["metadata"].get("hot_eval_threshold", 0.55))
    occupancy_type_masks = _occupancy_type_masks(dataset_dir, payload, device)
    cause_classes = list(manifest["cause_classes"])
    cause_majority = int(checkpoint["metadata"].get("cause_majority", -1))
    metrics, arrays, confusion = _evaluate_loader(
        model,
        loaders[split_name],
        loss_config,
        pos_weight,
        device,
        model.config.num_causes,
        cause_classes=cause_classes,
        cause_majority=cause_majority,
        event_threshold=event_threshold,
        hot_threshold=hot_threshold,
        occupancy_type_masks=occupancy_type_masks,
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
            "baseline_id": checkpoint["metadata"]["baseline_id"],
            "model_name": checkpoint["metadata"]["model_name"],
            "model_kind": checkpoint["model_kind"],
            "best_epoch": checkpoint["epoch"],
            "best_validation_report_f1": checkpoint["best_validation_report_f1"],
            "checkpoint": str(checkpoint_path.resolve()),
            "output_dir": str(output_dir),
        }
    )
    if split_name == "test":
        summary.update(
            {
                "event_report_threshold": event_threshold,
                "test_report_f1": metrics["station_report"]["report_f1"],
                "test_hot_f1": metrics["remain"]["hot_f1"],
            }
        )
    _write_json(summary_path, summary)
    return metrics
