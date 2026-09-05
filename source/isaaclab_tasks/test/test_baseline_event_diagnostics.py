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
        self.assertEqual(row["report_recall_ongoing"], 1)
        self.assertEqual(row["report_recall_upcoming"], 0)
        self.assertAlmostEqual(row["report_f1"], 1 / 3)
        self.assertEqual(result["groups"]["ongoing"]["start_within_tolerance_rate"], 1)


if __name__ == "__main__":
    unittest.main()
