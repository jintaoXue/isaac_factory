"""Test retrospective joins, count units, and immutable predictions in memory."""

import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from diagnose_baseline_events import summarize_schedule_strata


def fixture():
    arrays = dict(sample_index=np.array([19, 7]), occ_node_mask=np.ones((2, 2)),
        event_will=np.array([[1, 0], [1, 1]]), event_start=np.array([[1, 0], [0, 2]]),
        hist_last_hot=np.array([[1, 0], [1, 0]]), predicted_start=np.array([[12, 0], [0, 6]]),
        will_probability=np.array([[.5, .2], [.9, .7]]))
    samples = [dict(sample_index=7, split="validation", group_id="v2", scenario_id="S", first_future_start_s=1200),
               dict(sample_index=19, split="validation", group_id="v1", scenario_id="S", first_future_start_s=600),
               dict(sample_index=999, split="test")]
    nodes = ["m0", "m1"]
    plans = [dict(group_id=g, split="validation", schedule_mode="resample_per_episode") for g in ("v1", "v2")]
    targets = [dict(split="validation", group_id="v1", resource_id="m0", onset_window_index=11,
                   start_index=1, first_future_time_s=600, matched_future_local_runtime_start=True, no_prior_recorded_runtime_start=True),
               dict(split="validation", group_id="v2", resource_id="m1", onset_window_index=22,
                   start_index=2, first_future_time_s=1200, matched_future_local_runtime_start=False, no_prior_recorded_runtime_start=False)]
    audit = dict(episode_plan_reconstruction=plans, upcoming_targets=targets)
    support = dict(episodes=[dict(group_id="tr", split="train", scenario_id="S", upcoming_onsets=[dict(resource_id="m0")]),
        dict(group_id="v1", split="validation", scenario_id="S", upcoming_onsets=[dict(resource_id="m0")]),
        dict(group_id="v2", split="validation", scenario_id="S", upcoming_onsets=[dict(resource_id="m1")])])
    historical = dict(episodes=[dict(group_id="v1", match_class="both"), dict(group_id="v2", match_class="historical_only")])
    return arrays, samples, nodes, audit, support, historical


class TestScheduleStrata(unittest.TestCase):
    def test_shuffled_sample_identity_threshold_and_timing(self):
        result = summarize_schedule_strata(*fixture(), "validation", .5)
        summary = result["summary"]
        self.assertEqual([summary[k] for k in ("window_targets", "unique_onsets", "report_hits", "probability_misses", "timing_misses")], [2, 2, 1, 0, 1])
        first, second = result["upcoming_targets"]
        self.assertEqual((first["sample_index"], first["decoded_start"], first["train_joint_unique_onset_support"]), (19, 0, 1))
        self.assertEqual(second["train_joint_unique_onset_support"], 0)  # Validation positives cannot support themselves.
        self.assertEqual(result["compatibility_upcoming_vs_negative"]["historical_only"]["negative_count"], 0)  # Exclude ongoing.

    def test_identity_corruption_is_rejected(self):
        for mutation in ("sample_duplicate", "target_duplicate", "future_offset", "node_order"):
            args = list(fixture())
            if mutation == "sample_duplicate": args[0]["sample_index"][1] = 19
            if mutation == "target_duplicate": args[3]["upcoming_targets"].append(copy.deepcopy(args[3]["upcoming_targets"][0]))
            if mutation == "future_offset": args[1][0]["first_future_start_s"] += 60
            if mutation == "node_order": args[2].reverse()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_schedule_strata(*args, "validation", .5)

    def test_diagnostic_labels_do_not_mutate_predictions_and_test_is_forbidden(self):
        args = fixture(); before = copy.deepcopy(args[0])
        first = summarize_schedule_strata(*args, "validation", .5)
        args[3]["upcoming_targets"][0]["matched_future_local_runtime_start"] = False
        args[5]["episodes"][0]["match_class"] = "current_only"
        second = summarize_schedule_strata(*args, "validation", .5)
        self.assertEqual(first["summary"], second["summary"])
        for key in before: np.testing.assert_array_equal(before[key], args[0][key])
        self.assertNotEqual(first["groups"], second["groups"])
        with self.assertRaises(ValueError): summarize_schedule_strata(*args, "test", .5)


if __name__ == "__main__":
    unittest.main()
