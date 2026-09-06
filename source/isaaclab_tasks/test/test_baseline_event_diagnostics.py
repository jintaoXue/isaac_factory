"""Event diagnostics must partition misses without changing canonical metrics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from diagnose_baseline_events import summarize_events


class TestEventDiagnostics(unittest.TestCase):
    def test_probability_and_timing_misses_are_disjoint(self):
        hot = np.zeros((1, 15, 4), dtype=np.float32)
        hot[0, :8, 0] = 1
        hot[0, 5:13, 1:3] = 1
        arrays = {
            "y_hot": hot,
            "remain_mask": np.ones((1, 15)),
            "occ_node_mask": np.ones((1, 4)),
            "hist_last_hot": np.array([[1, 0, 0, 0]]),
            "event_will": np.array([[1, 1, 1, 0]]),
            "event_start": np.array([[0, 5, 5, -1]]),
            "will_probability": np.array([[.9, .2, .8, .8]]),
            "predicted_start": np.array([[9, 5, 0, 0]]),
            "predicted_duration": np.full((1, 4), 8),
        }
        result = summarize_events(arrays, [.5])
        row = result["thresholds"][0]
        self.assertEqual(row["upcoming_probability_misses"], 1)
        self.assertEqual(row["upcoming_timing_misses"], 1)
        self.assertEqual(row["false_positive_stations"], 1)
        self.assertEqual(row["report_false_alarm_count"], 2)
        self.assertEqual(row["report_false_alarm_breakdown"]["true_event_wrong_start"], 1)
        self.assertEqual(row["report_recall_ongoing"], 1)
        self.assertEqual(row["report_recall_upcoming"], 0)
        self.assertAlmostEqual(row["report_f1"], 1 / 3)
        self.assertEqual(result["groups"]["ongoing"]["start_within_tolerance_rate"], 1)

    def test_false_alarm_partition_handles_short_horizon_and_short_hot_runs(self):
        hot = np.zeros((2, 15, 3), dtype=np.float32)
        hot[0, :3, 0] = 1
        hot[0, :8, 1] = 1
        hot[1, :4, :] = 1
        remain = np.ones((2, 15))
        remain[1, 5:] = 0
        arrays = {
            "y_hot": hot, "remain_mask": remain,
            "occ_node_mask": np.array([[1, 1, 0], [1, 1, 1]]),
            "hist_last_hot": np.array([[1, 0, 0], [1, 0, 0]]),
            "event_will": np.array([[0, 1, 0], [0, 0, 0]]),
            "event_start": np.zeros((2, 3)),
            "will_probability": np.full((2, 3), .9),
            "predicted_start": np.zeros((2, 3)),
            "predicted_duration": np.full((2, 3), 8),
        }
        row = summarize_events(arrays, [.5])["thresholds"][0]
        parts = row["report_false_alarm_breakdown"]
        self.assertEqual(row["n_pred_who"], 5)
        self.assertAlmostEqual(row["report_precision"], .2)
        self.assertEqual(row["report_false_alarm_count"], 4)
        self.assertEqual(parts["short_observed_horizon_historically_hot"], 1)
        self.assertEqual(parts["short_observed_horizon_historically_cold"], 2)
        self.assertEqual(parts["hot_without_qualifying_event_historically_hot"], 1)
        self.assertEqual(sum(parts.values()), 4)
        self.assertEqual(sum(node["false_alarms"] for node in row["per_node"]), 4)
        self.assertEqual(row["predicted_duration_q25_q50_q75"], [8, 8, 8])

    def test_future_hot_outside_observation_does_not_explain_false_alarm(self):
        hot = np.zeros((1, 15, 1), dtype=np.float32)
        hot[0, 10:, 0] = 1
        remain = np.ones((1, 15))
        remain[:, 10:] = 0
        arrays = {
            "y_hot": hot, "remain_mask": remain, "occ_node_mask": np.ones((1, 1)),
            "hist_last_hot": np.zeros((1, 1)), "event_will": np.zeros((1, 1)),
            "event_start": np.zeros((1, 1)), "will_probability": np.full((1, 1), .9),
            "predicted_start": np.zeros((1, 1)), "predicted_duration": np.full((1, 1), 8),
        }
        row = summarize_events(arrays, [.5])["thresholds"][0]
        self.assertEqual(row["report_false_alarm_breakdown"]["no_future_hot_historically_cold"], 1)
        empty = summarize_events(arrays, [.95])["thresholds"][0]
        self.assertEqual(empty["report_false_alarm_count"], 0)
        self.assertIsNone(empty["predicted_duration_q25_q50_q75"])


if __name__ == "__main__":
    unittest.main()
