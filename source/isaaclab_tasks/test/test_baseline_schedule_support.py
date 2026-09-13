"""Test time alignment and distinct-event support without reading test episodes."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from diagnose_upcoming_schedule_support import match_future_runtime, summarize_support_cells


class TestScheduleSupport(unittest.TestCase):
    def test_actual_target_and_half_open_time_boundaries(self):
        onset = dict(resource_id="machine_ws1", onset_window_index=32)
        anchor = dict(start_index=2)
        runtime = [dict(event_id="at_begin", target="machine_ws1", start=1800., type="machine_down"),
                   dict(event_id="at_end", target="machine_ws1", start=1980., type="machine_down"),
                   dict(event_id="other", target="machine_ws0", start=1900., type="machine_down")]
        result = match_future_runtime(onset, anchor, runtime)
        self.assertEqual(result["matched_runtime_event_ids"], ["at_begin"])
        self.assertEqual((result["first_future_time_s"], result["onset_window_end_s"]), (1800, 1980))
        self.assertTrue(result["no_prior_recorded_runtime_start"])
        runtime.append(dict(event_id="prior_other", target="human0", start=1799., type="human_absence"))
        result = match_future_runtime(onset, anchor, runtime)
        self.assertFalse(result["no_prior_recorded_runtime_start"])
        self.assertEqual(result["matched_runtime_event_ids"], ["at_begin"])

    def test_overlapping_anchors_have_different_future_boundaries(self):
        onset = dict(resource_id="robot_1", onset_window_index=32)
        runtime = [dict(event_id="one", target="robot_1", start=1830., type="transport_delay")]
        self.assertTrue(match_future_runtime(onset, dict(start_index=2), runtime)["matched_future_local_runtime_start"])
        result = match_future_runtime(onset, dict(start_index=1), runtime)
        self.assertFalse(result["matched_future_local_runtime_start"])
        self.assertFalse(result["no_prior_recorded_runtime_start"])
        for invalid in (0, 3, -1):
            with self.assertRaises(ValueError): match_future_runtime(onset, dict(start_index=invalid), runtime)

    def test_support_counts_onsets_once_and_windows_separately(self):
        def ep(split, scenario, node, windows):
            return dict(split=split, scenario_id=scenario,
                        upcoming_onsets=[dict(resource_id=node, anchors=[{}] * windows)])
        episodes = [ep("train", "A", "m0", 2), ep("train", "A", "m0", 1),
                    ep("train", "B", "m1", 2), ep("validation", "A", "m0", 2),
                    ep("validation", "A", "m1", 2), ep("validation", "C", "m2", 1)]
        result = summarize_support_cells(episodes)
        unseen = {name: value["coverage"][0] for name, value in result.items()}
        self.assertEqual(unseen["node"]["validation_window_targets"], 1)
        self.assertEqual(unseen["scenario"]["validation_window_targets"], 1)
        self.assertEqual(unseen["node_scenario"]["validation_window_targets"], 3)
        self.assertEqual(unseen["node_scenario"]["validation_unique_onsets"], 2)
        row = next(row for row in result["node_scenario"]["cells"] if row["cell"] == ["m0", "A"])
        self.assertEqual(row["train_unique_onsets"], 2)
        self.assertEqual(sum(row["validation_unique_onsets"] for row in result["node"]["cells"]), 3)
        self.assertEqual(summarize_support_cells([])["node"]["coverage"][0]["validation_unique_onsets"], 0)


if __name__ == "__main__":
    unittest.main()
