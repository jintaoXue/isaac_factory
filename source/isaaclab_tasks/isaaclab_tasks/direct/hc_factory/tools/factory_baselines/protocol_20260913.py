"""Versioned Min8 start-5/10/15 evaluation, matching dev_tyx@7b2ab39.

The source 208-episode tensors remain immutable. This module changes task views
and measurement, not a model's encoder or decoder. Its 20-grid/Min8 censoring
is intentionally the reference behavior, including the unreachable starts >12.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from factory_bn_shared.remain import node_event_targets
from factory_bn_shared.remain import station_report_metrics as legacy_station_metrics

from .evaluation import EVALUATION_CONTRACT


VERSION = "factory_dense_i1_eval_tyx_7b2ab39_v1"
REFERENCE_COMMIT = "7b2ab393e838c1b54a78b2f125d41e0066d53ce0"
THRESHOLDS = (.45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .94)
CAUSE_CLASSES = (
    "transport_delay", "material_shortage", "starved_upstream", "queue_buildup",
)


def evaluation_contract(max_start_windows: int) -> dict[str, Any]:
    if max_start_windows not in (5, 10, 15):
        raise ValueError("The matched Min8 tasks are start<=5/10/15")
    return {
        **EVALUATION_CONTRACT,
        "version": VERSION,
        "reference_commit": REFERENCE_COMMIT,
        "max_start_windows": int(max_start_windows),
        "min_windows": 8,
        "occupancy_horizon_windows": 20,
        "hot_smoothing_order": "legacy",
        "report_primary": "will15",
        "will15_semantics": "who_f1_on_the_configured_start_limit_not_a_fixed_15_minute_target",
        "event_partition": "positive_and_last_history_hot_vs_cold",
        "checkpoint_tie_break": "first_primary_f1_improvement_above_1e-6",
        "max_observable_upcoming_start_index": min(int(max_start_windows), 12),
        "right_boundary_policy": "require_eight_observed_future_hot_grids",
        "cause_report_classes": list(CAUSE_CLASSES),
        "remain_progress_weight_floor": .25,
        "remain_progress_weight_power": 1.5,
        "remain_eval_primary_phase": "middle_weighted",
        "baseline_decoder_policy": "existing_baseline_predictions_no_main_prefix_or_occupancy_union",
    }


def protocol_view(payload: dict, manifest: dict, max_start_windows: int) -> tuple[dict, dict]:
    """Reuse inputs, normalization, full episode targets, and split membership."""
    if payload["evaluation_contract"] != EVALUATION_CONTRACT:
        raise ValueError("Expected the frozen legacy source dataset contract")
    if int(payload["event_min_windows"]) != 8 or float(payload["window_size_s"]) != 60:
        raise ValueError("The Min8 view requires the source minute/Min8 labels")
    if int(payload["x"].shape[1]) != 30:
        raise ValueError("The matched input history must contain 30 minute grids")
    if manifest.get("hot_smoothing_order", "legacy") != "legacy":
        raise ValueError("The matched three-tier configs use legacy hot smoothing")
    if not payload.get("remain_series"):
        raise ValueError("Full episode target series are required; do not pad old 15-grid labels")
    contract = evaluation_contract(max_start_windows)
    view = {**payload, "max_remain_windows": 20, "evaluation_contract": contract}
    view_manifest = {
        **manifest,
        "max_remain_windows": 20,
        "occupancy_horizon_windows": 20,
        "occupancy_horizon_s": 1200,
        "evaluation_contract": contract,
        "source_evaluation_contract": dict(EVALUATION_CONTRACT),
        "task_view_only": True,
    }
    return view, view_manifest


def station_metrics(
    y_hot: np.ndarray, will_probability: np.ndarray, start_index: np.ndarray,
    duration_windows: np.ndarray, remain_mask: np.ndarray, occ_node_mask: np.ndarray,
    *, hist_last_hot: np.ndarray, threshold: float, contract: dict,
) -> dict[str, Any]:
    """Score baseline decisions with the reference's target/metric semantics.

    Baselines keep their existing last-hot start=0 handling. The main model's
    learned prefix and occupancy-union decision paths are not transplanted.
    """
    if contract.get("version") != VERSION:
        raise ValueError("Expected an explicit 7b2ab39 evaluation contract")
    y = np.asarray(y_hot, dtype=np.float32)
    wp = np.asarray(will_probability, dtype=np.float32)
    sp = np.asarray(start_index, dtype=np.int64)
    dp = np.asarray(duration_windows, dtype=np.float32)
    if y.ndim != 3 or y.shape[1] != int(contract["occupancy_horizon_windows"]):
        raise ValueError("Matched three-tier evaluation requires 20-grid targets")
    shape = (y.shape[0], y.shape[2])
    if any(a.shape != shape for a in (wp, sp, dp)):
        raise ValueError("Station probabilities, starts and durations must match the target grid")
    last = np.broadcast_to(np.asarray(hist_last_hot, dtype=np.float32), shape) > .5
    valid = np.broadcast_to(np.asarray(occ_node_mask), shape) > .5
    rules = {
        "min_windows": int(contract["min_windows"]),
        "max_start_windows": int(contract["max_start_windows"]),
        "ongoing_min_windows": int(contract["ongoing_min_windows"]),
    }
    report = legacy_station_metrics(
        y, wp, sp, dp, remain_mask, occ_node_mask, hist_last_hot=last,
        threshold=threshold, start_tol_windows=int(contract["start_tol_windows"]),
        force_ongoing_will=False, **rules,
    )
    will, start, dur = node_event_targets(
        y, remain_mask=remain_mask, occ_node_mask=occ_node_mask,
        hist_last_hot=last, **rules,
    )
    positive = (will > .5) & valid
    predicted = (wp >= float(threshold)) & valid
    who_hit = positive & predicted
    predicted_start = np.where(last, 0, sp)
    error = np.abs(predicted_start - start)
    duration_error = np.abs(dp - dur)
    report_hit = who_hit & (error <= int(contract["start_tol_windows"]))
    for suffix in ("precision", "recall", "f1"):
        report[f"will15_{suffix}"] = report[f"who_{suffix}"]
    groups = {"ongoing": positive & last, "upcoming": positive & ~last}
    for name, mask in groups.items():
        count = int(mask.sum())
        matches = who_hit & mask
        report[f"n_true_{name}"] = float(count)
        report[f"n_matched_who_{name}"] = float(matches.sum())
        report[f"n_matched_report_{name}"] = float((report_hit & mask).sum())
        report[f"who_recall_{name}"] = float(matches.sum()) / count if count else 0.0
        report[f"report_recall_{name}"] = float((report_hit & mask).sum()) / count if count else 0.0
        report[f"start_mae_{name}"] = float(error[matches].mean()) if matches.any() else 0.0
        report[f"dur_mae_{name}"] = float(duration_error[matches].mean()) if matches.any() else 0.0
    report["exact_start_accuracy"] = float((error[who_hit] == 0).mean()) if who_hit.any() else 0.0
    up_hit = who_hit & groups["upcoming"]
    report["exact_start_accuracy_upcoming"] = float((error[up_hit] == 0).mean()) if up_hit.any() else 0.0
    true_bucket = np.select([groups["ongoing"], start <= 5, start <= 10], [0, 1, 2], default=3)
    pred_bucket = np.select([predicted_start == 0, predicted_start <= 5, predicted_start <= 10], [0, 1, 2], default=3)
    report["onset_bucket_accuracy"] = float((true_bucket[who_hit] == pred_bucket[who_hit]).mean()) if who_hit.any() else 0.0
    for suffix, start_mask in (
        ("start_le_5", start <= 5),
        ("start_6_10", (start >= 6) & (start <= 10)),
        ("start_gt_10", start > 10),
    ):
        mask = groups["upcoming"] & start_mask
        count = int(mask.sum())
        matches = who_hit & mask
        report[f"n_true_upcoming_{suffix}"] = float(count)
        report[f"report_recall_upcoming_{suffix}"] = float((report_hit & mask).sum()) / count if count else 0.0
        report[f"start_mae_upcoming_{suffix}"] = float(error[matches].mean()) if matches.any() else 0.0
    for tolerance in (1, 2, 3):
        tp = float((who_hit & (error <= tolerance)).sum())
        p = tp / float(predicted.sum()) if predicted.any() else 0.0
        r = tp / float(positive.sum()) if positive.any() else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        for suffix, value in (("precision", p), ("recall", r), ("f1", f)):
            report[f"report_{suffix}_at_{tolerance}"] = value
    report["report_threshold_used"] = float(threshold)
    return report


def remain_metrics(predicted: np.ndarray, target: np.ndarray, jobs_remaining: np.ndarray,
                   jobs_total: np.ndarray, contract: dict) -> dict[str, Any]:
    error = np.abs(np.asarray(predicted).reshape(-1) - np.asarray(target).reshape(-1))
    remaining, total = np.asarray(jobs_remaining).reshape(-1), np.asarray(jobs_total).reshape(-1)
    if not (error.shape == remaining.shape == total.shape) or not len(error):
        raise ValueError("Remaining-time predictions and progress must share nonempty sample support")
    progress = np.clip(1.0 - remaining / np.maximum(total, 1.0), 0.0, 1.0)
    floor = float(contract["remain_progress_weight_floor"])
    weight = floor + (1.0 - floor) * progress ** float(contract["remain_progress_weight_power"])
    result = {
        "remain_len_mae": float(error.mean()),
        "remain_len_mae_progress_weighted": float(np.sum(error * weight) / weight.sum()),
    }
    for name, mask in (("early", progress < 1/3),
                       ("middle", (progress >= 1/3) & (progress < 2/3)),
                       ("late", progress >= 2/3)):
        result[f"remain_len_n_{name}"] = int(mask.sum())
        if mask.any():
            result[f"remain_len_mae_{name}"] = float(error[mask].mean())
            result[f"remain_len_mae_{name}_weighted"] = float(np.sum(error[mask] * weight[mask]) / weight[mask].sum())
    key = "remain_len_mae_" + str(contract["remain_eval_primary_phase"])
    result["remain_len_mae_primary"] = result.get(key)
    return result
