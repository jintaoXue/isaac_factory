"""B2 XGBoost baseline on the shared factory bottleneck tensor contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from factory_bn_shared.causes import cause_ignore_ids
from factory_bn_shared.remain import (
    node_event_targets,
    occupancy_event_metrics,
    station_report_metrics,
)

from .dataset import FactoryBaselineTensorDataset, load_shared_dataset
from .metrics import compute_metrics


@dataclass
class B2XGBoostConfig:
    """Training and bounded cell-sampling settings for B2."""

    seed: int = 42
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    n_jobs: int = 8
    near_remain_windows: int = 60
    negative_cell_ratio: float = 4.0
    empty_sample_negative_cells: int = 32
    prediction_cell_chunk_size: int = 65536

    def __post_init__(self) -> None:
        for name in (
            "n_estimators",
            "max_depth",
            "n_jobs",
            "near_remain_windows",
            "empty_sample_negative_cells",
            "prediction_cell_chunk_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.negative_cell_ratio <= 0:
            raise ValueError("negative_cell_ratio must be positive")


@dataclass
class _Head:
    kind: str
    model: Any | None = None
    constant: float | int | None = None
    classes: list[int] | None = None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_xgboost() -> tuple[Any, Any]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "B2 requires xgboost. Install it in the server environment with "
            "`python -m pip install xgboost`."
        ) from exc
    return XGBClassifier, XGBRegressor


def _base_features(payload: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    """Compact non-graph history representation shared by scalar B2 heads."""
    x = payload["x"][indices].float().numpy()
    node_mask = payload["node_mask"][indices].float().numpy()
    observation = payload["observation_mask"][indices].float().numpy()
    global_features = payload["global_features"][indices].float().numpy()
    jobs_remaining = payload["jobs_remaining"][indices].float().numpy()
    jobs_total = payload["jobs_total"][indices].float().numpy()

    last = x[:, -1]
    temporal_mean = x.mean(axis=1)
    temporal_max = x.max(axis=1)
    temporal_delta = x[:, -1] - x[:, 0]
    parts = [
        last.reshape(len(indices), -1),
        temporal_mean.reshape(len(indices), -1),
        temporal_max.reshape(len(indices), -1),
        temporal_delta.reshape(len(indices), -1),
        node_mask,
        observation.mean(axis=1),
    ]
    if global_features.shape[-1]:
        parts.extend(
            [
                global_features[:, -1],
                global_features.mean(axis=1),
                global_features[:, -1] - global_features[:, 0],
            ]
        )
    jobs_ratio = jobs_remaining / np.maximum(jobs_total, 1.0)
    parts.append(np.stack((jobs_remaining, jobs_total, jobs_ratio), axis=1))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _cell_features(
    payload: dict[str, Any],
    sample_indices: np.ndarray,
    offsets: np.ndarray,
    node_indices: np.ndarray,
) -> np.ndarray:
    """Represent one future node-window cell without using adjacency."""
    unique_samples, inverse = np.unique(sample_indices, return_inverse=True)
    x_unique = payload["x"][unique_samples].float().numpy()
    node_history = x_unique[inverse, :, node_indices, :]
    node_last = node_history[:, -1]
    node_mean = node_history.mean(axis=1)
    node_max = node_history.max(axis=1)
    node_delta = node_history[:, -1] - node_history[:, 0]
    graph_last_mean = x_unique[:, -1].mean(axis=1)[inverse]
    graph_last_max = x_unique[:, -1].max(axis=1)[inverse]
    jobs_remaining = payload["jobs_remaining"][sample_indices].float().numpy()
    jobs_total = payload["jobs_total"][sample_indices].float().numpy()
    jobs_ratio = jobs_remaining / np.maximum(jobs_total, 1.0)
    max_windows = max(int(payload["max_remain_windows"]), 1)
    num_nodes = max(int(payload["x"].shape[2]), 1)
    offset_norm = offsets.astype(np.float32) / max(max_windows - 1, 1)
    node_norm = node_indices.astype(np.float32) / max(num_nodes - 1, 1)
    phase = 2.0 * np.pi * offset_norm
    scalar = np.stack(
        (
            offset_norm,
            np.sin(phase),
            np.cos(phase),
            node_norm,
            jobs_remaining,
            jobs_total,
            jobs_ratio,
        ),
        axis=1,
    )
    return np.concatenate(
        (
            node_last,
            node_mean,
            node_max,
            node_delta,
            graph_last_mean,
            graph_last_max,
            scalar,
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def _sample_target(
    payload: dict[str, Any], sample_index: int
) -> tuple[np.ndarray, np.ndarray]:
    dataset = FactoryBaselineTensorDataset(payload, [sample_index])
    sample = dataset[0]
    length = min(int(sample["target_remain_len"]), int(payload["max_remain_windows"]))
    return (
        sample["y_score"][:length, :, 0].numpy(),
        sample["y_hot"][:length].numpy().astype(np.int8),
    )


def _training_cells(
    payload: dict[str, Any],
    indices: Iterable[int],
    config: B2XGBoostConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep every positive cell and deterministic sampled negatives."""
    rng = np.random.default_rng(config.seed)
    sample_parts: list[np.ndarray] = []
    offset_parts: list[np.ndarray] = []
    node_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    hot_parts: list[np.ndarray] = []
    for sample_index in indices:
        scores, hot = _sample_target(payload, int(sample_index))
        usable = min(len(hot), config.near_remain_windows)
        if usable <= 0:
            continue
        valid_nodes = np.flatnonzero(
            payload["occ_node_mask"][sample_index].numpy() > 0.5
        )
        offsets, node_indices = np.meshgrid(
            np.arange(usable, dtype=np.int64), valid_nodes, indexing="ij"
        )
        offsets = offsets.ravel()
        node_indices = node_indices.ravel()
        labels = hot[:usable, valid_nodes].reshape(-1)
        values = scores[:usable, valid_nodes].reshape(-1)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        negative_limit = max(
            int(math.ceil(len(positive) * config.negative_cell_ratio)),
            config.empty_sample_negative_cells,
        )
        if len(negative) > negative_limit:
            negative = rng.choice(negative, size=negative_limit, replace=False)
        selected = np.concatenate((positive, negative))
        rng.shuffle(selected)
        sample_parts.append(np.full(len(selected), sample_index, dtype=np.int64))
        offset_parts.append(offsets[selected])
        node_parts.append(node_indices[selected])
        score_parts.append(values[selected].astype(np.float32))
        hot_parts.append(labels[selected].astype(np.int8))
    if not sample_parts:
        raise ValueError("No valid B2 training cells were generated")
    sample_indices = np.concatenate(sample_parts)
    offsets = np.concatenate(offset_parts)
    node_indices = np.concatenate(node_parts)
    features = _cell_features(payload, sample_indices, offsets, node_indices)
    return features, np.concatenate(score_parts), np.concatenate(hot_parts)


def _fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    config: B2XGBoostConfig,
    *,
    scale_pos_weight: float | None = None,
) -> _Head:
    classes = np.unique(y.astype(np.int64))
    if len(classes) == 1:
        return _Head(
            kind="constant", constant=int(classes[0]), classes=classes.tolist()
        )
    XGBClassifier, _ = _require_xgboost()
    values: dict[str, Any] = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "reg_lambda": config.reg_lambda,
        "n_jobs": config.n_jobs,
        "tree_method": "hist",
        "random_state": config.seed,
    }
    if len(classes) == 2 and set(classes.tolist()) == {0, 1}:
        values.update(objective="binary:logistic", eval_metric="logloss")
        if scale_pos_weight is not None:
            values["scale_pos_weight"] = scale_pos_weight
        model = XGBClassifier(**values)
        model.fit(X, y.astype(np.int64))
        return _Head(kind="xgboost_binary", model=model, classes=[0, 1])
    encoded = np.searchsorted(classes, y.astype(np.int64))
    values.update(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(classes),
    )
    model = XGBClassifier(**values)
    model.fit(X, encoded)
    return _Head(kind="xgboost_multiclass", model=model, classes=classes.tolist())


def _fit_regressor(X: np.ndarray, y: np.ndarray, config: B2XGBoostConfig) -> _Head:
    if np.allclose(y, y[0]):
        return _Head(kind="constant", constant=float(y[0]))
    _, XGBRegressor = _require_xgboost()
    model = XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        n_jobs=config.n_jobs,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=config.seed,
    )
    model.fit(X, y.astype(np.float32))
    return _Head(kind="xgboost_regression", model=model)


def _predict_probability(head: _Head, X: np.ndarray) -> np.ndarray:
    if head.kind == "constant":
        return np.full(len(X), float(int(head.constant or 0) == 1), dtype=np.float32)
    return np.asarray(head.model.predict_proba(X)[:, 1], dtype=np.float32)


def _predict_regression(head: _Head, X: np.ndarray) -> np.ndarray:
    if head.kind == "constant":
        return np.full(len(X), float(head.constant or 0.0), dtype=np.float32)
    return np.asarray(head.model.predict(X), dtype=np.float32)


def _predict_multiclass(
    head: _Head, X: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((len(X), class_count), dtype=np.float32)
    classes = head.classes or []
    if head.kind == "constant":
        prediction = int(head.constant or 0)
        probabilities[:, prediction] = 1.0
        return np.full(len(X), prediction, dtype=np.int64), probabilities
    local_probabilities = np.asarray(head.model.predict_proba(X), dtype=np.float32)
    probabilities[:, np.asarray(classes, dtype=np.int64)] = local_probabilities
    predictions = np.asarray(classes, dtype=np.int64)[
        local_probabilities.argmax(axis=1)
    ]
    return predictions, probabilities


def _events(grid: np.ndarray, length: int) -> list[dict[str, int]]:
    result = []
    event_id = 0
    usable = grid[: max(min(length, len(grid)), 0)]
    for node_index in range(usable.shape[1]):
        start = 0
        while start < len(usable):
            if not usable[start, node_index]:
                start += 1
                continue
            end = start + 1
            while end < len(usable) and usable[end, node_index]:
                end += 1
            result.append(
                {
                    "event_id": event_id,
                    "node_index": node_index,
                    "start_offset_windows": start,
                    "duration_windows": end - start,
                }
            )
            event_id += 1
            start = end
    return result


def _predict_cells(
    payload: dict[str, Any],
    sample_index: int,
    hot_head: _Head,
    score_head: _Head,
    threshold: float,
    config: B2XGBoostConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_score, target_hot = _sample_target(payload, sample_index)
    length, node_count = target_hot.shape
    valid_nodes = np.flatnonzero(
        payload["occ_node_mask"][sample_index].numpy() > 0.5
    )
    probabilities = np.zeros((length, node_count), dtype=np.float32)
    predicted_score = np.zeros((length, node_count), dtype=np.float32)
    offsets, nodes = np.meshgrid(
        np.arange(length, dtype=np.int64), valid_nodes, indexing="ij"
    )
    offsets = offsets.ravel()
    nodes = nodes.ravel()
    for start in range(0, len(offsets), config.prediction_cell_chunk_size):
        stop = min(start + config.prediction_cell_chunk_size, len(offsets))
        sample_indices = np.full(stop - start, sample_index, dtype=np.int64)
        features = _cell_features(
            payload, sample_indices, offsets[start:stop], nodes[start:stop]
        )
        probabilities[offsets[start:stop], nodes[start:stop]] = _predict_probability(
            hot_head, features
        )
        predicted_score[offsets[start:stop], nodes[start:stop]] = _predict_regression(
            score_head, features
        )
    predicted_hot = probabilities >= threshold
    return predicted_hot, probabilities, predicted_score, target_score


def _save_head(output_dir: Path, name: str, head: _Head) -> dict[str, Any]:
    metadata = {
        "kind": head.kind,
        "constant": head.constant,
        "classes": head.classes,
        "path": None,
    }
    if head.model is not None:
        path = output_dir / f"{name}.json"
        head.model.save_model(str(path))
        metadata["path"] = path.name
    return metadata


def train_b2_xgboost(
    dataset_dir: Path,
    output_dir: Path,
    config: B2XGBoostConfig | None = None,
) -> dict[str, Any]:
    """Train B2 heads, select thresholds on validation, and evaluate test."""
    config = config or B2XGBoostConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    payload, manifest = load_shared_dataset(dataset_dir)
    split_indices = {
        name: payload["split_indices"][name].numpy().astype(np.int64)
        for name in ("train", "validation", "test")
    }
    train_indices = split_indices["train"]
    train_X = _base_features(payload, train_indices)
    cause_values = payload["y_cause"][train_indices].numpy()
    cause_valid = cause_values >= 0
    for cause_id in cause_ignore_ids():
        cause_valid &= cause_values != int(cause_id)
    cause_trained = bool(cause_valid.any())
    cause_head = (
        _fit_classifier(train_X[cause_valid], cause_values[cause_valid], config)
        if cause_trained
        else _Head(kind="constant", constant=0, classes=[0])
    )
    remain_len_head = _fit_regressor(
        train_X,
        payload["target_remain_len"][train_indices].numpy(),
        config,
    )
    cell_X, cell_score_y, cell_hot_y = _training_cells(
        payload, train_indices.tolist(), config
    )
    cell_positives = max(int(cell_hot_y.sum()), 1)
    hot_head = _fit_classifier(
        cell_X,
        cell_hot_y,
        config,
        scale_pos_weight=float(len(cell_hot_y) - cell_positives) / cell_positives,
    )
    score_head = _fit_regressor(cell_X, cell_score_y, config)
    heads = {
        "cause": cause_head,
        "remain_len": remain_len_head,
        "remain_hot": hot_head,
        "remain_score": score_head,
    }

    def scalar_arrays(split_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        indices = split_indices[split_name]
        X = _base_features(payload, indices)
        cause_predictions, _ = _predict_multiclass(
            cause_head, X, len(manifest["cause_classes"])
        )
        arrays = {
            "sample_index": indices,
            "y_cause": payload["y_cause"][indices].numpy(),
            "target_remain_len": payload["target_remain_len"][indices].numpy(),
            "cause_predictions": cause_predictions,
            "remain_len": np.maximum(_predict_regression(remain_len_head, X), 0.0),
        }
        return X, arrays

    _, validation_arrays = scalar_arrays("validation")
    occupancy_threshold = 0.65

    sample_rows = list(
        csv.DictReader(
            (dataset_dir / "model_sample_index.csv").open(newline="", encoding="utf-8")
        )
    )
    sample_lookup = {int(row["sample_index"]): row for row in sample_rows}
    all_metrics: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        _, arrays = (
            (None, validation_arrays)
            if split_name == "validation"
            else scalar_arrays(split_name)
        )
        metrics, confusion = compute_metrics(
            arrays, len(manifest["cause_classes"])
        )
        hot_tp = hot_fp = hot_fn = score_count = 0
        score_abs_sum = 0.0
        predicted_event_json = []
        target_event_json = []
        target_grids = []
        probability_grids = []
        remain_masks = []
        occupancy_masks = []
        history_hot = []
        event_will_probabilities = []
        event_start_indices = []
        event_durations = []
        for position, sample_index in enumerate(arrays["sample_index"]):
            predicted_hot, _probability, predicted_score, target_score = _predict_cells(
                payload,
                int(sample_index),
                hot_head,
                score_head,
                occupancy_threshold,
                config,
            )
            _, target_hot = _sample_target(payload, int(sample_index))
            valid_nodes = (
                payload["occ_node_mask"][sample_index].numpy() > 0.5
            )
            predicted_valid = predicted_hot[:, valid_nodes]
            target_valid = target_hot[:, valid_nodes].astype(bool)
            hot_tp += int(np.logical_and(predicted_valid, target_valid).sum())
            hot_fp += int(np.logical_and(predicted_valid, ~target_valid).sum())
            hot_fn += int(np.logical_and(~predicted_valid, target_valid).sum())
            score_abs_sum += float(
                np.abs(
                    predicted_score[:, valid_nodes] - target_score[:, valid_nodes]
                ).sum()
            )
            score_count += int(target_score[:, valid_nodes].size)
            sample = FactoryBaselineTensorDataset(payload, [int(sample_index)])[0]
            remain_mask = sample["remain_mask"].numpy()
            occ_mask = sample["occ_node_mask"].numpy()
            will, start, duration = node_event_targets(
                _probability,
                min_windows=8,
                remain_mask=remain_mask,
                occ_node_mask=occ_mask,
            )
            padded_target = np.zeros(
                (int(payload["max_remain_windows"]), target_hot.shape[1]),
                dtype=np.float32,
            )
            padded_probability = np.zeros_like(padded_target)
            padded_target[: len(target_hot)] = target_hot
            padded_probability[: len(_probability)] = _probability
            target_grids.append(padded_target)
            probability_grids.append(padded_probability)
            remain_masks.append(remain_mask)
            occupancy_masks.append(occ_mask)
            history_hot.append(sample["hist_last_hot"].numpy())
            event_will_probabilities.append(will)
            event_start_indices.append(start)
            event_durations.append(duration)
            predicted_length = int(round(float(arrays["remain_len"][position])))
            target_length = int(arrays["target_remain_len"][position])
            predicted_event_json.append(
                json.dumps(
                    _events(predicted_hot, predicted_length), separators=(",", ":")
                )
            )
            target_event_json.append(
                json.dumps(
                    _events(target_hot.astype(bool), target_length),
                    separators=(",", ":"),
                )
            )
        arrays["predicted_events_json"] = np.asarray(predicted_event_json)
        arrays["target_events_json"] = np.asarray(target_event_json)
        hot_precision = hot_tp / max(hot_tp + hot_fp, 1)
        hot_recall = hot_tp / max(hot_tp + hot_fn, 1)
        metrics["remain"] = {
            "hot_threshold": occupancy_threshold,
            "hot_precision": hot_precision,
            "hot_recall": hot_recall,
            "hot_f1": 2.0
            * hot_precision
            * hot_recall
            / max(hot_precision + hot_recall, 1.0e-12),
            "score_mae": score_abs_sum / max(score_count, 1),
            "remain_len_mae_windows": float(
                np.abs(arrays["remain_len"] - arrays["target_remain_len"]).mean()
            ),
            "valid_node_windows": score_count,
        }
        metrics["station_report"] = station_report_metrics(
            np.stack(target_grids),
            np.stack(event_will_probabilities),
            np.stack(event_start_indices),
            np.stack(event_durations),
            np.stack(remain_masks),
            np.stack(occupancy_masks),
            threshold=0.65,
            min_windows=8,
            start_tol_windows=3,
            hist_last_hot=np.stack(history_hot),
        )
        metrics["occupancy_event"] = occupancy_event_metrics(
            np.stack(target_grids),
            np.stack(probability_grids),
            np.stack(remain_masks),
            np.stack(occupancy_masks),
            manifest["node_ids"],
            threshold=0.65,
            min_windows=8,
            iou_min=0.5,
            window_size_s=float(manifest["window_size_s"]),
        )
        metrics["sample_count"] = len(arrays["sample_index"])
        all_metrics[split_name] = metrics
        _write_json(output_dir / f"metrics_{split_name}.json", metrics)
        prediction_rows = []
        event_rows = []
        for position, sample_index in enumerate(arrays["sample_index"]):
            lookup = sample_lookup[int(sample_index)]
            cause_prediction = int(arrays["cause_predictions"][position])
            prediction_rows.append(
                {
                    **lookup,
                    "predicted_cause": manifest["cause_classes"][cause_prediction],
                    "predicted_remain_len_windows": float(
                        arrays["remain_len"][position]
                    ),
                }
            )
            for source, field in (
                ("prediction", "predicted_events_json"),
                ("target", "target_events_json"),
            ):
                for event in json.loads(str(arrays[field][position])):
                    start_s = float(lookup["first_future_start_s"]) + (
                        event["start_offset_windows"] * float(manifest["window_size_s"])
                    )
                    duration_s = event["duration_windows"] * float(
                        manifest["window_size_s"]
                    )
                    event_rows.append(
                        {
                            "sample_index": int(sample_index),
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
            output_dir / f"predictions_{split_name}.csv",
            prediction_rows,
            list(sample_rows[0])
            + [
                "predicted_cause",
                "predicted_remain_len_windows",
            ],
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
                        f"predicted__{name}": int(
                            confusion[target_index, prediction_index]
                        )
                        for prediction_index, name in enumerate(
                            manifest["cause_classes"]
                        )
                    },
                }
            )
        _write_csv(
            output_dir / f"confusion_matrix_{split_name}.csv",
            confusion_rows,
            ["target_cause"]
            + [f"predicted__{name}" for name in manifest["cause_classes"]],
        )

    model_metadata = {
        name: _save_head(output_dir, name, head) for name, head in heads.items()
    }
    metadata = {
        "baseline_id": "B2",
        "model_name": "XGBoost",
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": _manifest_hash(
            dataset_dir / "dataset_manifest.json"
        ),
        "dataset_contract": manifest["dataset_contract"],
        "dataset_version": manifest["dataset_version"],
        "label_version": manifest["label_version"],
        "prediction_target_version": manifest["prediction_target_version"],
        "config": asdict(config),
        "models": model_metadata,
        "event_report_threshold": occupancy_threshold,
        "occupancy_threshold": occupancy_threshold,
        "training_cell_count": int(len(cell_hot_y)),
        "training_positive_cell_count": int(cell_hot_y.sum()),
        "cause_head_trained": cause_trained,
    }
    _write_json(output_dir / "config.json", metadata)
    _write_json(output_dir / "metrics.json", all_metrics)
    summary = {
        "status": "completed",
        "baseline_id": "B2",
        "model_name": "XGBoost",
        "dataset_contract": manifest["dataset_contract"],
        "dataset_version": manifest["dataset_version"],
        "label_version": manifest["label_version"],
        "validation_hot_f1": all_metrics["validation"]["remain"]["hot_f1"],
        "test_hot_f1": all_metrics["test"]["remain"]["hot_f1"],
        "validation_report_f1": all_metrics["validation"]["station_report"][
            "report_f1"
        ],
        "test_report_f1": all_metrics["test"]["station_report"]["report_f1"],
        "event_report_threshold": occupancy_threshold,
        "occupancy_threshold": occupancy_threshold,
        "elapsed_seconds": time.time() - started_at,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary
