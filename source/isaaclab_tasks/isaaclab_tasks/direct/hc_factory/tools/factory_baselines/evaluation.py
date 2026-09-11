"""Versioned evaluation semantics, independent of each baseline's decoder."""

from __future__ import annotations

import numpy as np


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


def event_rule_kwargs(min_windows: int) -> dict[str, int]:
    return {
        "min_windows": int(min_windows),
        "ongoing_min_windows": EVALUATION_CONTRACT["ongoing_min_windows"],
        "max_start_windows": EVALUATION_CONTRACT["max_start_windows"],
    }


def add_time_metric_metadata(
    metrics: dict, *, window_size_s: float, sample_count: int
) -> None:
    """Keep canonical window-valued keys and add explicit units and support.

    Upstream returns zero for unmatched-event MAE. The explicit-unit fields
    are null in that case, so an empty prediction cannot look perfectly timed.
    """
    if not np.isfinite(window_size_s) or window_size_s <= 0:
        raise ValueError("window_size_s must be finite and positive")
    metrics["evaluation_contract"] = dict(EVALUATION_CONTRACT)
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
