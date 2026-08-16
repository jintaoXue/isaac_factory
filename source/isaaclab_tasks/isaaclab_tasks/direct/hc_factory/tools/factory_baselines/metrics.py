"""Evaluation metrics shared by factory bottleneck baselines."""

from __future__ import annotations

from typing import Any

import numpy as np


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
        positive_scores = probabilities[positives][:, None]
        negative_scores = probabilities[negatives][None, :]
        roc_auc = float(
            (
                (positive_scores > negative_scores).sum()
                + 0.5 * (positive_scores == negative_scores).sum()
            )
            / (positive_count * negative_count)
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


def compute_metrics(
    arrays: dict[str, np.ndarray],
    cause_class_count: int,
    occurrence_threshold: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute shared will/mark/time-to-start and A.3 cause metrics."""
    occurrence = arrays["y_occurrence"].astype(np.int64)
    occurrence_probability = arrays["occurrence_probability"]
    metrics: dict[str, Any] = {
        "occurrence": _binary_metrics(
            occurrence, occurrence_probability, occurrence_threshold
        ),
        "no_event_baseline": _binary_metrics(
            occurrence, np.zeros_like(occurrence_probability), 0.5
        ),
    }

    positive = arrays["positive_mask"].astype(bool)
    positive_count = int(positive.sum())
    if positive_count:
        node_probabilities = arrays["node_probabilities"][positive]
        node_targets = arrays["y_node"][positive].astype(np.int64)
        ranking = np.argsort(-node_probabilities, axis=1, kind="stable")
        target_ranks = np.argmax(ranking == node_targets[:, None], axis=1) + 1
        metrics["node"] = {
            "top_1_accuracy": float((target_ranks <= 1).mean()),
            "top_3_accuracy": float((target_ranks <= 3).mean()),
            "mrr": float((1.0 / target_ranks).mean()),
            "sample_count": positive_count,
        }
        metrics["regression"] = {
            "time_to_start_mae_s": float(
                np.abs(
                    arrays["time_to_start"][positive]
                    - arrays["y_time_to_start"][positive]
                ).mean()
            ),
        }
    else:
        metrics["node"] = {
            "top_1_accuracy": None,
            "top_3_accuracy": None,
            "mrr": None,
            "sample_count": 0,
        }
        metrics["regression"] = {
            "time_to_start_mae_s": None,
        }
    valid_cause = arrays["y_cause"] >= 0
    if valid_cause.any():
        cause_metrics, confusion = _multiclass_metrics(
            arrays["y_cause"][valid_cause].astype(np.int64),
            arrays["cause_predictions"][valid_cause].astype(np.int64),
            cause_class_count,
        )
        labels = arrays["y_cause"][valid_cause].astype(np.int64)
        majority = int(np.bincount(labels, minlength=cause_class_count).argmax())
        cause_metrics["majority_accuracy"] = float((labels == majority).mean())
        cause_metrics["majority_class"] = majority
        metrics["cause"] = cause_metrics
    else:
        confusion = np.zeros((cause_class_count, cause_class_count), dtype=np.int64)
        metrics["cause"] = {
            "accuracy": None,
            "macro_f1": None,
            "class_f1": [0.0] * cause_class_count,
            "sample_count": 0,
            "majority_accuracy": None,
            "majority_class": None,
        }
    return metrics, confusion
