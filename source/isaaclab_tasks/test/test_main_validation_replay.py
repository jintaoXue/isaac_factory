"""Strict cohort binding and scoring isolation for read-only main replay."""

import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from reevaluate_main_validation import compare_reports, load_standalone_metrics, validation_names


class TestMainValidationReplay(unittest.TestCase):
    def test_only_audited_validation_groups_are_selected(self):
        rows = [{"split": "validation", "group_id": "g2"}]
        self.assertEqual(validation_names(rows, {"train_ep": "g1", "val_ep": "g2"}), {"val_ep"})
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            validation_names(rows, {"a": "g2", "b": "g2"})
        with self.assertRaisesRegex(ValueError, "missing"):
            validation_names(rows, {"a": "g1"})
        with self.assertRaisesRegex(ValueError, "must not score"):
            validation_names([{"split": "test", "group_id": "g2"}], {"ep": "g2"})

    def test_reproduction_requires_both_ranking_and_error_metrics(self):
        actual = {"report_precision": .8, "dur_mae": 2.1}
        expected = {"report_precision": .8 + 1e-8, "dur_mae": 1.5}
        self.assertEqual(set(compare_reports(actual, expected, list(actual))), {"dur_mae"})

    def test_standalone_metric_import_does_not_replace_main_module(self):
        before = {name: value for name, value in sys.modules.items() if name.startswith("factory_bn")}
        metric = load_standalone_metrics(TOOLS.parent / "PDFormer/factory_bn/remain.py")
        for name, value in before.items():
            self.assertIs(sys.modules[name], value)
        hot = np.ones((1, 15, 1), dtype=np.float32)
        common = metric.station_report_metrics(hot, np.array([[.63]]), np.array([[0]]),
                 np.array([[10.]]), np.ones((1, 15)), np.ones((1, 1)), threshold=.65,
                 hist_last_hot=np.ones((1, 1)), force_ongoing_will=False)
        original = metric.station_report_metrics(hot, np.array([[1.]]), np.array([[0]]),
                   np.array([[10.]]), np.ones((1, 15)), np.ones((1, 1)), threshold=.65,
                   hist_last_hot=np.ones((1, 1)), force_ongoing_will=True)
        self.assertEqual(common["report_recall"], 0)
        self.assertEqual(original["report_recall"], 1)

    def test_full_episode_smoothing_is_not_a_causal_history_flag(self):
        metric = load_standalone_metrics(TOOLS.parent / "PDFormer/factory_bn/remain.py")
        same_history = np.ones((3, 1), dtype=np.float32)
        long_future = np.ones((10, 1), dtype=np.float32)
        ended_future = np.concatenate((same_history, np.zeros((7, 1), dtype=np.float32)))
        self.assertEqual(metric.smooth_occupancy_runs(long_future, min_windows=8)[2, 0], 1)
        self.assertEqual(metric.smooth_occupancy_runs(ended_future, min_windows=8)[2, 0], 0)
        self.assertEqual(metric.smooth_occupancy_runs(same_history, min_windows=8)[2, 0], 0)


if __name__ == "__main__":
    unittest.main()
