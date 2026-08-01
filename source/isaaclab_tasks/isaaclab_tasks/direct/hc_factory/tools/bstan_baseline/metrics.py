"""Evaluation metrics for the BSTAN multi-task prediction heads."""

from __future__ import annotations

from typing import Any

import numpy as np


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    predictions = probabilities >= 0.5
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
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": f1,
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


def compute_metrics(
    arrays: dict[str, np.ndarray], class_count: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute all Phase-D metrics and the bottleneck-type confusion matrix."""
    occurrence = arrays["y_occurrence"].astype(np.int64)
    occurrence_probability = arrays["occurrence_probability"]
    metrics: dict[str, Any] = {
        "occurrence": _binary_metrics(occurrence, occurrence_probability),
        "no_event_baseline": _binary_metrics(
            occurrence, np.zeros_like(occurrence_probability)
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
        type_targets = arrays["y_type"][positive].astype(np.int64)
        type_predictions = arrays["type_predictions"][positive].astype(np.int64)
        type_metrics, confusion = _multiclass_metrics(
            type_targets, type_predictions, class_count
        )
        metrics["type"] = type_metrics
        metrics["regression"] = {
            "time_to_start_mae_s": float(
                np.abs(
                    arrays["time_to_start"][positive]
                    - arrays["y_time_to_start"][positive]
                ).mean()
            ),
            "severity_mae": float(
                np.abs(
                    arrays["severity"][positive] - arrays["y_severity"][positive]
                ).mean()
            ),
        }
    else:
        confusion = np.zeros((class_count, class_count), dtype=np.int64)
        metrics["node"] = {
            "top_1_accuracy": None,
            "top_3_accuracy": None,
            "mrr": None,
            "sample_count": 0,
        }
        metrics["type"] = {
            "accuracy": None,
            "macro_f1": None,
            "class_f1": [0.0] * class_count,
            "sample_count": 0,
        }
        metrics["regression"] = {
            "time_to_start_mae_s": None,
            "severity_mae": None,
        }

    duration_mask = arrays["duration_mask"].astype(bool) & positive
    metrics["regression"]["duration_mae_s"] = (
        float(
            np.abs(
                arrays["duration"][duration_mask] - arrays["y_duration"][duration_mask]
            ).mean()
        )
        if duration_mask.any()
        else None
    )
    metrics["regression"]["duration_sample_count"] = int(duration_mask.sum())
    return metrics, confusion
