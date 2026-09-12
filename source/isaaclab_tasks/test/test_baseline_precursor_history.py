"""History-budget audit must distinguish far inputs without any future access."""

import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from audit_baseline_precursor_history import history_summary, load_reference


class TestPrecursorHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference, cls.provenance = load_reference(TOOLS.parents[5])

    def test_far_change_affects_only_last_five_summary_channels(self):
        features = np.zeros((80, 2, 27), dtype=np.float32)
        changed = features.copy()
        changed[10:40, :, [0, 6, 14, 15]] = [5, 60, 60, 1]
        actual = history_summary(self.reference, changed, 70, include_far=True)
        np.testing.assert_array_equal(actual[:, :18], 0)
        np.testing.assert_array_equal(actual[:, 18:], 1)
        np.testing.assert_array_equal(history_summary(self.reference, changed, 70, include_far=False), 0)

    def test_future_and_past_older_than_sixty_windows_do_not_change_summary(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(100, 2, 27)).astype(np.float32)
        expected = history_summary(self.reference, features, 70, include_far=True)
        features[:10] = 10000
        features[70:] = -10000
        np.testing.assert_array_equal(expected, history_summary(self.reference, features, 70, include_far=True))

    def test_episode_start_has_zero_far_and_partial_history_is_not_future_padded(self):
        features = np.zeros((80, 1, 27), dtype=np.float32)
        features[:10, :, 0] = 5
        np.testing.assert_array_equal(history_summary(self.reference, features, 30, include_far=True)[:, 18:], 0)
        actual = history_summary(self.reference, features, 40, include_far=True)
        np.testing.assert_array_equal(actual[:, 18:], [[1, 0, 0, 0, 1]])
        for t in (29, 81):
            with self.assertRaises(ValueError):
                history_summary(self.reference, features, t, include_far=True)


if __name__ == "__main__":
    unittest.main()
