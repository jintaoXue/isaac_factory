"""Shared dense-i1 event boundaries and unambiguous MAE support."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))

from factory_bn_shared.remain import node_event_targets, station_report_metrics
from factory_baselines.evaluation import add_time_metric_metadata, event_rule_kwargs


def test_ongoing_tail_and_upcoming_start_boundary():
    hot = np.zeros((15, 5), dtype=np.float32)
    hot[:1, 0] = 1
    hot[:7, 1] = 1
    hot[2:10, 2] = 1
    hot[3:11, 3] = 1
    hot[1:8, 4] = 1
    will, start, dur = node_event_targets(
        hot, hist_last_hot=np.array([1, 0, 0, 0, 0]),
        **event_rule_kwargs(8),
    )
    np.testing.assert_array_equal(will, [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(start, [0, 0, 2, 0, 0])
    np.testing.assert_array_equal(dur, [1, 0, 8, 0, 0])


def test_masked_nodes_and_episode_tail_are_not_invented_events():
    hot = np.ones((15, 2), dtype=np.float32)
    remain = np.r_[np.ones(2), np.zeros(13)]
    will, _, dur = node_event_targets(
        hot, remain_mask=remain, occ_node_mask=np.array([1, 0]),
        hist_last_hot=np.ones(2), **event_rule_kwargs(8),
    )
    np.testing.assert_array_equal(will, [1, 0])
    np.testing.assert_array_equal(dur, [2, 0])


def test_mae_is_on_who_matches_not_just_report_matches():
    hot = np.zeros((1, 15, 2), dtype=np.float32)
    hot[:, 1:9, :] = 1
    report = station_report_metrics(
        hot, np.ones((1, 2)), np.array([[2, 6]]), np.array([[6., 4.]]),
        np.ones((1, 15)), np.ones((1, 2)),
        hist_last_hot=np.zeros((1, 2)), threshold=.7,
        start_tol_windows=3, **event_rule_kwargs(8),
    )
    assert report["report_recall"] == .5
    assert report["who_recall"] == 1
    assert report["start_mae"] == 3
    assert report["dur_mae"] == 3
    metrics = {"station_report": report, "remain": {"remain_len_mae": 2.}}
    add_time_metric_metadata(metrics, window_size_s=60, sample_count=1)
    assert metrics["start_mae_minutes"] == 3
    assert metrics["start_mae_seconds"] == 180
    assert metrics["time_mae_sample_count"] == 2
    assert metrics["time_mae_sample_count_ongoing"] == 0
    assert metrics["start_mae_ongoing_minutes"] is None
    assert metrics["remain_len_mae_minutes"] == 2


def test_no_prediction_has_no_valid_mae():
    hot = np.ones((1, 15, 1), dtype=np.float32)
    report = station_report_metrics(
        hot, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
        np.ones((1, 15)), np.ones((1, 1)),
        hist_last_hot=np.ones((1, 1)), **event_rule_kwargs(8),
    )
    metrics = {"station_report": report, "remain": {"remain_len_mae": 4.}}
    add_time_metric_metadata(metrics, window_size_s=30, sample_count=1)
    assert metrics["start_mae"] == 0  # Canonical key, not evidence of accuracy.
    assert metrics["start_mae_minutes"] is None
    assert metrics["time_mae_sample_count"] == 0
    assert metrics["remain_len_mae_minutes"] == 2


def test_causal_decoder_sensitivity_keeps_ground_truth_fixed():
    hot = np.zeros((1, 15, 1), dtype=np.float32)
    hot[:, :2] = 1
    truth_history = np.ones((1, 1))
    reports = []
    for observed_history in (truth_history, np.zeros((1, 1))):
        reports.append(station_report_metrics(
            hot, np.ones((1, 1)), np.full((1, 1), 7), np.full((1, 1), 2),
            np.ones((1, 15)), np.ones((1, 1)),
            hist_last_hot=observed_history, target_hist_last_hot=truth_history,
            **event_rule_kwargs(8),
        ))
    assert [report["n_true_who"] for report in reports] == [1, 1]
    assert [report["report_f1"] for report in reports] == [1, 0]
