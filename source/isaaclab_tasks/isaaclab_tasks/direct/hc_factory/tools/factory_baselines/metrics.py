"""Evaluation metrics shared by factory bottleneck baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from factory_bn_shared.causes import (
    CAUSE_REPORT_CLASSES,
    ROOT_CAUSE_CLASSES,
    cause_ignore_ids,
)
from factory_bn_shared.remain import station_report_metrics


REPORT_THRESHOLD_SWEEP: tuple[float, ...] = (
    0.55,
    0.60,
    0.62,
    0.65,
    0.68,
    0.70,
    0.72,
    0.75,
    0.78,
    0.80,
    0.82,
    0.85,
)
OCCUPANCY_EVAL_TYPES: tuple[str, ...] = (
    "machine",
    "gantry",
    "agv",
    "workbench",
)


def _binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.logical_and(predictions, positives).sum())
    false_positive = int(np.logical_and(predictions, negatives).sum())
    false_negative = int(np.logical_and(~predictions, positives).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)

    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count:
        order = np.argsort(-probabilities, kind="stable")
        sorted_scores = probabilities[order]
        sorted_positive = positives[order].astype(np.int64)
        threshold_ends = np.flatnonzero(
            np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
        )
        true_positives = np.cumsum(sorted_positive)[threshold_ends]
        predicted_positives = threshold_ends + 1
        recall_curve = true_positives / positive_count
        precision_curve = true_positives / predicted_positives
        average_precision = float(
            (np.diff(np.r_[0.0, recall_curve]) * precision_curve).sum()
        )
    else:
        average_precision = None

    if positive_count and negative_count:
        order = np.argsort(probabilities, kind="stable")
        sorted_scores = probabilities[order]
        sorted_positive = positives[order]
        group_starts = np.r_[
            0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
        ]
        group_ends = np.r_[group_starts[1:], len(sorted_scores)]
        average_ranks = (group_starts + 1 + group_ends) / 2.0
        ranks = np.repeat(average_ranks, group_ends - group_starts)
        positive_rank_sum = float(ranks[sorted_positive].sum())
        roc_auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
            positive_count * negative_count
        )
    else:
        roc_auc = None

    return {
        "pr_auc": average_precision,
        "roc_auc": roc_auc,
        "decision_threshold": threshold,
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "f1_at_threshold": f1,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _multiclass_metrics(
    labels: np.ndarray, predictions: np.ndarray, class_count: int
) -> tuple[dict[str, Any], np.ndarray]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1

    class_f1 = []
    for class_index in range(class_count):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        class_f1.append(2.0 * precision * recall / max(precision + recall, 1.0e-12))
    return {
        "accuracy": float((labels == predictions).mean()) if len(labels) else None,
        "macro_f1": float(np.mean(class_f1)) if class_f1 else None,
        "class_f1": class_f1,
        "sample_count": int(len(labels)),
    }, confusion


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select the highest validation threshold among equal best F1 values."""
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    scored = [
        (
            _binary_metrics(labels, probabilities, float(threshold))["f1_at_threshold"],
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def _main_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Match BNPDFormer train._average_precision, including stable tie order."""
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    positive_count = float(y.sum())
    if positive_count <= 0.0 or not y.size:
        return 0.0
    order = np.argsort(-p, kind="stable")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float((precision * ranked).sum() / positive_count)


def select_report_threshold(
    y_hot: np.ndarray,
    will_probability: np.ndarray,
    start_index: np.ndarray,
    duration_windows: np.ndarray,
    remain_mask: np.ndarray,
    occ_node_mask: np.ndarray,
    *,
    default_threshold: float,
    threshold_sweep: tuple[float, ...] | list[float] = REPORT_THRESHOLD_SWEEP,
    min_precision: float = 0.80,
    min_windows: int = 8,
    start_tol_windows: int = 3,
    hist_last_hot: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select report threshold on validation exactly as the main experiment."""
    thresholds = [float(default_threshold)]
    thresholds.extend(
        float(value)
        for value in threshold_sweep
        if not np.isclose(float(value), float(default_threshold))
    )
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        candidate = station_report_metrics(
            y_hot,
            will_probability,
            start_index,
            duration_windows,
            remain_mask,
            occ_node_mask,
            threshold=threshold,
            min_windows=min_windows,
            start_tol_windows=start_tol_windows,
            hist_last_hot=hist_last_hot,
            force_ongoing_will=False,
        )
        candidate["report_threshold_used"] = threshold
        candidates.append(candidate)
    return choose_report_metrics(candidates, min_precision=min_precision)


def choose_report_metrics(
    candidates: list[dict[str, Any]],
    *,
    min_precision: float = 0.80,
) -> dict[str, Any]:
    """Choose max report F1, preferring candidates that satisfy precision."""
    if not candidates:
        raise ValueError("report threshold candidates must not be empty")
    best = candidates[0]
    for candidate in candidates[1:]:
        candidate_ok = float(candidate["report_precision"]) + 1.0e-12 >= min_precision
        best_ok = float(best["report_precision"]) + 1.0e-12 >= min_precision
        candidate_f1 = float(candidate["report_f1"])
        best_f1 = float(best["report_f1"])
        if candidate_ok and (not best_ok or candidate_f1 > best_f1 + 1.0e-6):
            best = candidate
        elif not candidate_ok and not best_ok and candidate_f1 > best_f1 + 1.0e-6:
            best = candidate
    return best


def hot_grid_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    remain_mask: np.ndarray,
    occ_node_mask: np.ndarray,
    *,
    threshold: float = 0.55,
    type_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute canonical micro hot metrics on valid future node-windows."""
    y = np.asarray(labels, dtype=np.float32)
    p = np.asarray(probabilities, dtype=np.float32)
    remain = np.asarray(remain_mask) > 0.5
    occupancy = np.asarray(occ_node_mask) > 0.5
    if occupancy.ndim == 1:
        occupancy = np.broadcast_to(occupancy, (y.shape[0], occupancy.shape[0]))
    valid = remain[:, :, None] & occupancy[:, None, :]

    def score(mask: np.ndarray) -> dict[str, Any]:
        binary = _binary_metrics(
            y[mask].astype(np.int64),
            p[mask],
            threshold=threshold,
        )
        count = int(mask.sum())
        predicted = p[mask] >= threshold
        return {
            "precision": float(binary["precision_at_threshold"]),
            "recall": float(binary["recall_at_threshold"]),
            "f1": float(binary["f1_at_threshold"]),
            "ap": _main_average_precision(y[mask], p[mask]),
            "positive_rate": float(y[mask].mean()) if count else 0.0,
            "predicted_positive_rate": (float(predicted.mean()) if count else 0.0),
            "count": count,
        }

    overall = score(valid)
    result: dict[str, Any] = {
        "hot_threshold": float(threshold),
        "hot_precision": overall["precision"],
        "hot_recall": overall["recall"],
        "hot_f1": overall["f1"],
        "hot_ap": overall["ap"],
        "hot_pos_rate": overall["positive_rate"],
        "hot_pred_pos_rate": overall["predicted_positive_rate"],
        "valid_node_windows": overall["count"],
    }
    positive_type_f1: list[float] = []
    for name in OCCUPANCY_EVAL_TYPES:
        if type_masks is None or name not in type_masks:
            continue
        node_mask = np.asarray(type_masks[name]) > 0.5
        typed = valid & node_mask.reshape(1, 1, -1)
        type_score = score(typed)
        result[f"hot_precision_{name}"] = type_score["precision"]
        result[f"hot_recall_{name}"] = type_score["recall"]
        result[f"hot_f1_{name}"] = type_score["f1"]
        result[f"hot_pos_rate_{name}"] = type_score["positive_rate"]
        result[f"hot_pred_pos_rate_{name}"] = type_score["predicted_positive_rate"]
        if type_score["positive_rate"] > 0.0:
            positive_type_f1.append(type_score["f1"])
    if positive_type_f1:
        epsilon_values = [max(value, 1.0e-8) for value in positive_type_f1]
        result["hot_type_hmean"] = len(epsilon_values) / sum(
            1.0 / value for value in epsilon_values
        )
    else:
        result["hot_type_hmean"] = result["hot_f1"]
    return result


def training_cause_majority(
    labels: np.ndarray,
    cause_classes: list[str] | tuple[str, ...],
) -> int:
    """Return the valid A.3 majority class using training labels only."""
    y = np.asarray(labels, dtype=np.int64)
    valid = y >= 0
    for cause_id in cause_ignore_ids(cause_classes):
        valid &= y != int(cause_id)
    if not valid.any():
        return -1
    counts = np.bincount(y[valid], minlength=len(cause_classes))
    return int(np.argmax(counts))


def compute_metrics(
    arrays: dict[str, np.ndarray],
    cause_class_count: int,
    cause_classes: list[str] | tuple[str, ...] | None = None,
    cause_majority: int = -1,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute A.3 process-cause metrics; A.1 is scored as station events."""
    class_names = list(cause_classes or ROOT_CAUSE_CLASSES)
    if len(class_names) != cause_class_count:
        raise ValueError("cause_classes must match cause_class_count")
    metrics: dict[str, Any] = {}
    valid_cause = arrays["y_cause"] >= 0
    for cause_id in cause_ignore_ids(class_names):
        valid_cause &= arrays["y_cause"] != int(cause_id)
    if valid_cause.any():
        labels = arrays["y_cause"][valid_cause].astype(np.int64)
        predictions = arrays["cause_predictions"][valid_cause].astype(np.int64)
        cause_metrics, confusion = _multiclass_metrics(
            labels,
            predictions,
            cause_class_count,
        )
        cause_metrics["cause_acc"] = cause_metrics["accuracy"]
        cause_metrics["cause_n"] = cause_metrics["sample_count"]
        if cause_majority >= 0:
            cause_metrics["cause_majority_acc"] = float(
                (labels == int(cause_majority)).mean()
            )
        recalls = []
        for class_id, class_name in enumerate(class_names):
            if class_name not in CAUSE_REPORT_CLASSES:
                continue
            support = labels == class_id
            if not support.any():
                continue
            recall = float((predictions[support] == class_id).mean())
            cause_metrics[f"cause_recall_{class_name}"] = recall
            recalls.append(recall)
        cause_metrics["cause_macro_recall"] = (
            float(np.mean(recalls)) if recalls else 0.0
        )
        metrics["cause"] = cause_metrics
    else:
        confusion = np.zeros((cause_class_count, cause_class_count), dtype=np.int64)
        metrics["cause"] = {
            "accuracy": None,
            "macro_f1": None,
            "class_f1": [0.0] * cause_class_count,
            "sample_count": 0,
            "cause_acc": 0.0,
            "cause_n": 0,
            "cause_macro_recall": 0.0,
        }
        if cause_majority >= 0:
            metrics["cause"]["cause_majority_acc"] = 0.0
    for name, value in metrics["cause"].items():
        if name.startswith("cause_"):
            metrics[name] = value
    return metrics, confusion
