"""Exact input audit must not create conflicts by omitting actual model inputs."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from audit_baseline_input_identity import (
    audit_dataset, equal_observations, input_fingerprint, observed_inputs, parse_args,
)


def sample(index=0, episode=0, will=(1., 1.), start=(2., 0.)):
    return dict(
        x=torch.zeros(3, 2, 4), adjacency=torch.eye(2), node_mask=torch.ones(2),
        target_node_mask=torch.ones(2), global_features=torch.zeros(3, 1),
        jobs_remaining=torch.tensor(5.), jobs_total=torch.tensor(10.),
        hist_last_hot=torch.tensor([0., 1.]), occ_node_mask=torch.ones(2),
        event_will=torch.tensor(will), event_start=torch.tensor(start),
        sample_index=torch.tensor(index), sample_group_id=torch.tensor(episode),
        y_hot=torch.zeros(15, 2), remain_mask=torch.ones(15),
    )


class TestInputIdentity(unittest.TestCase):
    def test_event_view_omits_only_other_head_fields(self):
        first, second = sample(1, 1), sample(2, 2, will=(0., 1.), start=(0., 0.))
        view = "event_context_disabled"
        excluded = {"target_node_mask", "global_features", "jobs_remaining", "jobs_total"}
        self.assertEqual(set(observed_inputs(first)) - set(observed_inputs(first, view)), excluded)
        for key in excluded:
            second[key] += 1
        self.assertNotEqual(input_fingerprint(first), input_fingerprint(second))
        self.assertEqual(input_fingerprint(first, view), input_fingerprint(second, view))
        self.assertTrue(equal_observations(first, second, view))
        mapping = {1: "train", 2: "validation"}
        full = audit_dataset([first, second], mapping)
        event = audit_dataset([first, second], mapping, view)
        self.assertEqual(full["train_validation_union"]["input_groups"], 2)
        self.assertEqual(event["train_validation_union"]["input_groups"], 1)
        self.assertEqual(event["train_validation_union"]["upcoming_targets_with_negative_counterexample"], 1)
        first["event_precursor"] = torch.zeros(2, 23)
        expected = input_fingerprint(first, view)
        for key in observed_inputs(first, view):
            changed = copy.deepcopy(first)
            changed[key].view(-1)[0] += 1
            self.assertNotEqual(input_fingerprint(changed, view), expected, key)
        with self.assertRaisesRegex(ValueError, "Unknown observation view"):
            audit_dataset([], {}, "unsupported")

    def test_actual_b4_b5_event_outputs_ignore_excluded_fields_without_context(self):
        from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
        from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
        from factory_baselines.torch_trainer import _model_inputs
        for model_class, config_class, spatial in (
            (B4GcnGru, B4ModelConfig, dict(gcn_hidden=4)),
            (B5GatGru, B5ModelConfig, dict(gat_hidden=4, gat_heads=2)),
        ):
            for mode in ("none", "near"):
                with self.subTest(model=model_class.__name__, precursor=mode):
                    model = model_class(config_class(input_dim=4, global_dim=1, num_nodes=2,
                        gru_hidden=8, dropout=0, event_context=False, event_onset_aux=True,
                        event_precursor=mode, temporal_readout="last_mean", **spatial)).eval()
                    first = sample()
                    first["x"] = torch.randn_like(first["x"])
                    if mode != "none":
                        first["event_precursor"] = torch.randn(2, 23)
                    second = copy.deepcopy(first)
                    second["target_node_mask"].zero_()
                    second["global_features"] += 100
                    second["jobs_remaining"] += 50
                    second["jobs_total"] += 200
                    with torch.no_grad():
                        a, b = [model(**{k: v.unsqueeze(0) for k, v in _model_inputs(s).items()})
                                for s in (first, second)]
                    for key in ("event_will_logit", "event_onset_logit", "event_start_logit", "event_duration"):
                        self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_fingerprint_uses_every_actual_input_and_no_future_or_sample_identity(self):
        original = sample()
        original["event_precursor"] = torch.zeros(2, 23)
        expected = input_fingerprint(original)
        for key in observed_inputs(original):
            changed = copy.deepcopy(original)
            changed[key].view(-1)[0] += 1
            self.assertNotEqual(input_fingerprint(changed), expected, key)
        for key in ("event_will", "event_start", "sample_index", "sample_group_id",
                    "y_hot", "remain_mask", "occ_node_mask"):
            changed = copy.deepcopy(original)
            changed[key].view(-1)[0] += 1
            self.assertEqual(input_fingerprint(changed), expected, key)
        same = copy.deepcopy(original)
        same["x"].fill_(-0.)
        self.assertEqual(input_fingerprint(same), expected)
        self.assertTrue(equal_observations(original, same))
        same["x"] = same["x"].double()
        self.assertNotEqual(input_fingerprint(same), expected)
        self.assertFalse(equal_observations(original, same))
        original["x"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            input_fingerprint(original)

    def test_conflicts_count_only_valid_nodes_and_separate_within_from_cross_split(self):
        samples = [sample(10, 1), sample(11, 2, will=(0., 1.), start=(0., 0.)),
                   sample(12, 3), sample(13, 4, will=(0., 0.), start=(0., 0.)), sample(14, 5)]
        samples[3]["x"][0, 0, 0] = .01
        samples[4]["occ_node_mask"][0] = 0
        original = copy.deepcopy(samples)
        report = audit_dataset(samples, {10: "train", 11: "train", 12: "validation",
                                         13: "validation", 14: "train"})
        train, val = report["splits"]["train"], report["splits"]["validation"]
        self.assertEqual([train[k] for k in ("samples", "input_groups", "repeated_input_groups",
                                             "ongoing_targets", "upcoming_targets", "negative_targets")],
                         [3, 1, 1, 3, 1, 1])
        self.assertEqual(train["upcoming_targets_with_negative_counterexample"], 1)
        self.assertEqual(val["upcoming_targets_with_negative_counterexample"], 0)
        self.assertEqual(report["train_validation_union"]["upcoming_targets_with_negative_counterexample"], 2)
        self.assertEqual(report["train_validation_union"]["samples_in_repeated_input_groups"], 4)
        self.assertEqual(report["cross_split"]["input_groups"], 1)
        self.assertEqual(report["cross_split"]["validation_upcoming_with_train_negative_counterexample"], 1)
        self.assertEqual(report["cross_split"]["validation_negative_with_train_upcoming_counterexample"], 0)
        for before, after in zip(original, samples):
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_precursor_can_split_core_duplicates_without_merging_distinct_core_inputs(self):
        first = sample(1, 1)
        second = sample(2, 2, will=(0., 1.), start=(0., 0.))
        mapping = {1: "train", 2: "train"}
        core = audit_dataset([first, second], mapping)
        self.assertEqual(core["splits"]["train"]["input_groups"], 1)
        first["event_precursor"] = torch.zeros(2, 23)
        second["event_precursor"] = torch.zeros(2, 23)
        second["event_precursor"][0, -1] = .1
        full = audit_dataset([first, second], mapping)
        self.assertEqual(full["splits"]["train"]["input_groups"], 2)
        self.assertEqual(full["splits"]["train"]["upcoming_targets_with_negative_counterexample"], 0)
        second["x"][0, 0, 0] = 1
        second["event_precursor"].zero_()
        self.assertNotEqual(input_fingerprint(first), input_fingerprint(second))

    def test_hash_matches_require_actual_tensor_equality(self):
        first, second = sample(1, 1), sample(2, 2)
        second["jobs_remaining"] += 1
        with patch("audit_baseline_input_identity.input_fingerprint", return_value="forced_collision"):
            with self.assertRaisesRegex(ValueError, "Hash collision"):
                audit_dataset([first, second], {1: "train", 2: "train"})

    def test_rejects_test_and_incomplete_or_duplicate_sample_coverage(self):
        class Unreadable:
            def __len__(self):
                raise AssertionError("Must reject test before reading samples")
        with self.assertRaisesRegex(ValueError, "Only train and validation"):
            audit_dataset(Unreadable(), {0: "test"})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            audit_dataset([sample()], {0: "train", 1: "validation"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_dataset([sample(), sample()], {0: "train"})
        with self.assertRaisesRegex(ValueError, "schema"):
            changed = sample(1)
            changed["global_features"] = torch.zeros(3, 2)
            audit_dataset([sample(), changed], {0: "train", 1: "train"})
        empty = audit_dataset([], {})
        self.assertEqual(empty["train_validation_union"]["samples"], 0)
        self.assertEqual(empty["cross_split"]["input_groups"], 0)
        with self.assertRaises(SystemExit):
            parse_args(["--dataset_dir", "data", "--output", "result.json", "--split", "test"])


if __name__ == "__main__":
    unittest.main()
