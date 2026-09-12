"""Retrospective support counts must not inflate distinct onsets or use masked nodes."""

from pathlib import Path
import sys
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from diagnose_dense_event_support import episode_support


class TestDenseEventSupport(unittest.TestCase):
    def test_two_upcoming_windows_are_one_onset_and_first_future_start_is_separate(self):
        x = np.zeros((60, 2, 27), dtype=np.float32)
        x[32:, 0, 18] = 1
        hot = np.zeros((60, 2), dtype=np.float32)
        hot[32:45] = 1
        scores = np.zeros((60, 2, 1), dtype=np.float32)
        result = episode_support(x, scores, hot, 60, [30, 31, 32, 33], np.array([1, 0]), ["node0", "ignored"])
        self.assertEqual(result["counts"]["upcoming_targets"], 2)
        self.assertEqual(result["counts"]["ongoing_targets"], 2)
        self.assertEqual(result["counts"]["positive_start_zero_hist_cold"], 1)
        self.assertEqual(result["unique_upcoming_onsets"], 1)
        self.assertEqual(result["upcoming_onsets"][0]["anchor_support"], 2)
        self.assertEqual(result["upcoming_onsets"][0]["onset_position"], 32)
        self.assertEqual(result["counts"]["new_local_disturbance_by_onset"], 2)
        self.assertEqual(result["counts"]["local_disturbance_observed_at_anchor"], 0)
        self.assertEqual(result["upcoming_targets_by_node"], {"node0": 2})

    def test_inadequate_future_duration_cannot_become_an_upcoming_target(self):
        x = np.zeros((40, 1, 27), dtype=np.float32)
        hot = np.zeros((40, 1), dtype=np.float32)
        hot[32:36] = 1
        result = episode_support(x, np.zeros((40, 1, 1)), hot, 36, [30, 31], np.ones(1), ["node0"])
        self.assertEqual(result["counts"]["upcoming_targets"], 0)
        self.assertEqual(result["unique_upcoming_onsets"], 0)
        self.assertEqual(result["counts"]["negative_targets"], 2)


if __name__ == "__main__":
    unittest.main()
