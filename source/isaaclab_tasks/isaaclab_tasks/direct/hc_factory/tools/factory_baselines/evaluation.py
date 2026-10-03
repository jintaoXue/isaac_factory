"""Versioned evaluation semantics, independent of each baseline's decoder."""

from __future__ import annotations

import numpy as np
from typing import Any

from factory_bn_shared.remain import rasterize_node_events


EVALUATION_CONTRACT = {
    "version": "factory_dense_i1_eval_v1",
    "reference_commit": "20c40e230aedee6aef2429d352413fbcf0fa571a",
    "reference_entrypoint": "factory_bn.train._epoch",
    "ongoing_min_windows": 1,
    "max_start_windows": 2,
    "start_tol_windows": 3,
    "history_label_policy": "full_episode_smoothed_ops_hot_at_last_history_window",
    "time_unit": "window",
    "mae_support": "who_true_positives",
    "remain_mae_support": "all_sample_anchors_to_jobs_done",
    "test_threshold_policy": "frozen_validation_threshold",
}


def event_rule_kwargs(min_windows: int, contract: dict | None = None) -> dict[str, int]:
    contract = EVALUATION_CONTRACT if contract is None else contract
    return {
        "min_windows": int(min_windows),
        "ongoing_min_windows": contract["ongoing_min_windows"],
        "max_start_windows": contract["max_start_windows"],
    }


def add_time_metric_metadata(
    metrics: dict, *, window_size_s: float, sample_count: int, contract: dict | None = None
) -> None:
    """Keep canonical window-valued keys and add explicit units and support.

    Upstream returns zero for unmatched-event MAE. The explicit-unit fields
    are null in that case, so an empty prediction cannot look perfectly timed.
    """
    if not np.isfinite(window_size_s) or window_size_s <= 0:
        raise ValueError("window_size_s must be finite and positive")
    metrics["evaluation_contract"] = dict(EVALUATION_CONTRACT if contract is None else contract)
    metrics["evaluation_contract"]["window_size_s"] = float(window_size_s)
    report = metrics["station_report"]
    for suffix in ("", "_ongoing", "_upcoming"):
        count = int(report[f"n_matched_who{suffix}"])
        report[f"time_mae_sample_count{suffix}"] = count
        for name in ("start_mae", "dur_mae"):
            key = name + suffix
            for unit, scale in (("seconds", window_size_s), ("minutes", window_size_s / 60)):
                report[f"{key}_{unit}"] = float(report[key]) * scale if count else None
    remain = metrics["remain"]
    remain["remain_len_mae_sample_count"] = int(sample_count)
    remain["remain_len_mae_seconds"] = remain["remain_len_mae"] * window_size_s
    remain["remain_len_mae_minutes"] = remain["remain_len_mae"] * window_size_s / 60
    # Unlike the main unsupervised loop's zero accumulator, this is measured.
    remain["score_mae_role"] = "baseline_auxiliary_not_main_unsupervised_metric"
    metrics.update(report)
    metrics.update({key: value for key, value in remain.items() if key.startswith("remain_len_mae")})


def _mae_rmse(error: np.ndarray) -> tuple[float, float]:
    error = np.asarray(error, dtype=np.float64)
    if error.size == 0:
        return float("nan"), float("nan")
    return float(np.abs(error).mean()), float(np.sqrt(np.square(error).mean()))


def _json_metric(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def add_bottleneck_forecast_metrics(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    *,
    threshold: float,
    window_size_s: float,
    min_windows: int,
) -> None:
    """Add the current dev_tyx bottleneck duration/state metrics.

    Durations and starts are represented in windows by the shared baseline
    decoder.  The explicit minute fields make the unit conversion visible;
    with the current 60-second windows the canonical values are numerically
    equal to window counts.
    """
    required = (
        "event_will_target", "event_will_probability", "event_start_index",
        "event_duration_windows", "y_hot_grid", "remain_mask_grid",
        "occ_node_mask_grid",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Bottleneck metrics require arrays: {missing}")
    y_will = np.asarray(arrays["event_will_target"]) > 0.5
    will = np.asarray(arrays["event_will_probability"])
    starts = np.asarray(arrays["event_start_index"])
    durations = np.asarray(arrays["event_duration_windows"])
    node_ok = np.asarray(arrays["occ_node_mask_grid"]) > 0.5
    remain = np.asarray(arrays["remain_mask_grid"]) > 0.5
    predicted = (will >= float(threshold)) & node_ok
    truth = y_will & node_ok
    pred_dur = np.where(predicted, durations, 0.0)
    # The baseline loader stores event duration targets separately only in the
    # event target tensor.  Callers can provide it explicitly; otherwise the
    # event-duration fields remain TP-only and union duration is rejected.
    if "event_duration_windows_target" not in arrays:
        raise ValueError("event_duration_windows_target is required for union duration metrics")
    true_dur = np.where(truth, np.asarray(arrays["event_duration_windows_target"]), 0.0)
    pred_start = np.where(predicted, starts, 0)
    true_start = np.where(truth, np.asarray(arrays["event_start_index_target"]), 0)
    tp = predicted & truth
    union = predicted | truth

    duration_error = pred_dur - true_dur
    start_error = pred_start - true_start
    dur_mae, dur_rmse = _mae_rmse(duration_error[union])
    start_mae, start_rmse = _mae_rmse(start_error[union])
    tp_dur_mae, tp_dur_rmse = _mae_rmse(
        (durations - np.asarray(arrays["event_duration_windows_target"]))[tp]
    )
    tp_start_mae, tp_start_rmse = _mae_rmse(
        (starts - np.asarray(arrays["event_start_index_target"]))[tp]
    )
    scale_min = float(window_size_s) / 60.0
    metrics.update(
        {
            "dur_mae_union": _json_metric(dur_mae),
            "dur_rmse_union": _json_metric(dur_rmse),
            "dur_mae_union_minutes": _json_metric(dur_mae * scale_min),
            "dur_rmse_union_minutes": _json_metric(dur_rmse * scale_min),
            "dur_mae_tp": _json_metric(tp_dur_mae),
            "dur_rmse_tp": _json_metric(tp_dur_rmse),
            "start_mae_union": _json_metric(start_mae),
            "start_rmse_union": _json_metric(start_rmse),
            "start_mae_tp": _json_metric(tp_start_mae),
            "start_rmse_tp": _json_metric(tp_start_rmse),
            "dur_mae_cells": _json_metric(_mae_rmse((pred_dur - true_dur)[node_ok])[0]),
            "dur_rmse_cells": _json_metric(_mae_rmse((pred_dur - true_dur)[node_ok])[1]),
            "duration_union_support": int(union.sum()),
            "duration_tp_support": int(tp.sum()),
        }
    )
    predicted_grid = rasterize_node_events(
        predicted.astype(np.float32), starts, durations,
        np.asarray(arrays["y_hot_grid"]).shape[1], threshold=0.5,
        min_windows=int(min_windows),
    ) > 0.5
    observed_grid = np.asarray(arrays["y_hot_grid"]) > 0.5
    event_true_grid = rasterize_node_events(
        truth.astype(np.float32), np.asarray(arrays["event_start_index_target"]),
        np.asarray(arrays["event_duration_windows_target"]),
        np.asarray(arrays["y_hot_grid"]).shape[1], threshold=0.5, min_windows=1,
    ) > 0.5
    cell = remain[:, :, None] & node_ok[:, None, :]
    for name, sl in (("1step", slice(0, 1)), ("horizon", slice(None))):
        valid_true = observed_grid[:, sl][cell[:, sl]]
        valid_pred = predicted_grid[:, sl][cell[:, sl]]
        tp_cells = float((valid_true & valid_pred).sum())
        pred_cells = float(valid_pred.sum())
        true_cells = float(valid_true.sum())
        precision = tp_cells / pred_cells if pred_cells else 0.0
        recall = tp_cells / true_cells if true_cells else 0.0
        metrics[f"state_precision_{name}"] = precision
        metrics[f"state_recall_{name}"] = recall
        metrics[f"state_f1_{name}"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        metrics[f"state_acc_{name}"] = _json_metric(float(
            (valid_true == valid_pred).mean()
        )) if valid_true.size else None
        event_true = event_true_grid[:, sl][cell[:, sl]]
        metrics[f"state_vs_event_acc_{name}"] = _json_metric(float(
            (event_true == valid_pred).mean()
        )) if event_true.size else None


def bottleneck_series(
    arrays: dict[str, np.ndarray],
    *,
    node_ids: list[str] | tuple[str, ...],
    node_types: list[str] | tuple[str, ...] | None,
    threshold: float,
    min_windows: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build per-device metrics and chart-ready arrays for one test split."""
    y_will = np.asarray(arrays["event_will_target"]) > 0.5
    will = np.asarray(arrays["event_will_probability"])
    starts = np.asarray(arrays["event_start_index"])
    durations = np.asarray(arrays["event_duration_windows"])
    target_starts = np.asarray(arrays["event_start_index_target"])
    target_durations = np.asarray(arrays["event_duration_windows_target"])
    node_ok = np.asarray(arrays["occ_node_mask_grid"]) > 0.5
    remain = np.asarray(arrays["remain_mask_grid"]) > 0.5
    predicted = (will >= float(threshold)) & node_ok
    truth = y_will & node_ok
    pred_dur = np.where(predicted, durations, 0.0)
    true_dur = np.where(truth, target_durations, 0.0)
    tp = predicted & truth
    union = predicted | truth
    pred_grid = rasterize_node_events(
        predicted.astype(np.float32), starts, durations,
        np.asarray(arrays["y_hot_grid"]).shape[1], threshold=0.5,
        min_windows=int(min_windows),
    ) > 0.5
    observed_grid = np.asarray(arrays["y_hot_grid"]) > 0.5
    valid = remain[:, :, None] & node_ok[:, None, :]
    per_device: list[dict[str, Any]] = []
    for index, device in enumerate(node_ids):
        ok = node_ok[:, index]
        tp_i = tp[:, index]
        pred_i = predicted[:, index]
        truth_i = truth[:, index]
        precision = float(tp_i.sum() / pred_i.sum()) if pred_i.any() else 0.0
        recall = float(tp_i.sum() / truth_i.sum()) if truth_i.any() else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        dur_union = union[:, index]
        dur_mae, dur_rmse = _mae_rmse((pred_dur[:, index] - true_dur[:, index])[dur_union])
        dur_tp_mae, dur_tp_rmse = _mae_rmse(
            (durations[:, index] - target_durations[:, index])[tp_i]
        )
        c1 = valid[:, 0, index] & ok
        state_acc = _json_metric(float((observed_grid[:, 0, index][c1] == pred_grid[:, 0, index][c1]).mean())) if c1.any() else None
        per_device.append(
            {
                "device": str(device),
                "type": str(node_types[index]) if node_types is not None else "",
                "n_true": int(truth_i.sum()),
                "n_pred": int(pred_i.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "dur_mae_union": _json_metric(dur_mae),
                "dur_rmse_union": _json_metric(dur_rmse),
                "dur_mae_tp": _json_metric(dur_tp_mae),
                "dur_rmse_tp": _json_metric(dur_tp_rmse),
                "state_acc_1step": state_acc,
            }
        )
    series = {
        "true_state": (observed_grid[:, 0, :] & node_ok).astype(np.bool_),
        "pred_state": (pred_grid[:, 0, :] & node_ok).astype(np.bool_),
        "valid_1step": valid[:, 0, :].astype(np.bool_),
        "true_dur": true_dur.astype(np.float32),
        "pred_dur": pred_dur.astype(np.float32),
        "true_will": truth.astype(np.bool_),
        "pred_will": predicted.astype(np.bool_),
        "will_prob": will.astype(np.float32),
        "sample_index": np.asarray(arrays["sample_index"], dtype=np.int64),
        "resource_ids": np.asarray([str(x) for x in node_ids]),
        "resource_types": np.asarray([str(x) for x in (node_types or [""] * len(node_ids))]),
    }
    return series, per_device
