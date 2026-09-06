"""Hard-negative supervision must not redefine labels, inputs or decoding."""

from dataclasses import replace
import sys
import unittest
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from factory_baselines.torch_losses import MultiTaskLossConfig, _short_hot_negative_mask, compute_multitask_loss
from factory_baselines.torch_heads import FactoryPredictionHeads
from run_staged_baseline import hard_negative_configuration


class TestShortHotNegatives(unittest.TestCase):
    def test_only_observed_eligible_event_negatives_are_weighted(self):
        hot = torch.zeros(2, 15, 5)
        hot[0, :3, 0:3] = 1
        hot[0, 12:, 3] = 1
        hot[1, :3] = 1
        remain = torch.ones(2, 15)
        remain[0, 10:] = 0
        remain[1, 5:] = 0
        will = torch.zeros(2, 5)
        will[0, 1] = 1
        eligible = torch.ones(2, 5, dtype=torch.bool)
        eligible[0, 2] = False
        batch = {"y_hot": hot, "remain_mask": remain, "event_will": will, "occ_node_mask": eligible}
        expected = torch.zeros(2, 5, dtype=torch.bool)
        expected[0, 0] = True
        self.assertTrue(torch.equal(_short_hot_negative_mask(batch), expected))

    def test_configuration_changes_only_multiplier_between_arms(self):
        parent = {"model": {"temporal_readout": "last_mean", "node_embedding": 0, "event_context": False},
                  "training": {"max_epochs": 60, "evaluate_test": False, "learning_rate": .0003},
                  "loss": MultiTaskLossConfig().to_dict()}
        control = hard_negative_configuration(parent, "weight1_control", "same", "cpu", 42)
        for arm, value in (("weight2", 2.), ("weight4", 4.)):
            changed = hard_negative_configuration(parent, arm, "same", "cpu", 42)
            self.assertEqual(changed["loss"].pop("event_short_hot_fp_multiplier"), value)
            expected = {key: dict(part) for key, part in control.items()}
            expected["loss"].pop("event_short_hot_fp_multiplier")
            self.assertEqual(changed, expected)
        self.assertEqual(control["training"]["max_epochs"], 60)
        self.assertEqual(parent["loss"]["event_short_hot_fp_multiplier"], 1)
        for value in (0., .5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MultiTaskLossConfig(event_short_hot_fp_multiplier=value)

    def test_default_exact_and_hard_negative_gradient_strengthens(self):
        torch.manual_seed(8)
        heads = FactoryPredictionHeads(8, 0, 3, 180, 15, 10)
        mask = torch.ones(1, 3, dtype=torch.bool)
        outputs = heads(torch.randn(1, 3, 8), mask, mask, torch.empty(1, 4, 0),
                        torch.ones(1), torch.ones(1))
        hot = torch.zeros(1, 15, 3)
        hot[:, :3, 0] = 1
        hot[:, :8, 2] = 1
        batch = {"remain_mask": torch.ones(1, 15), "occ_node_mask": mask,
                 "y_score": torch.zeros(1, 15, 3, 1), "y_hot": hot,
                 "target_remain_len": torch.ones(1), "y_cause": torch.tensor([-1]),
                 "event_will": torch.tensor([[0., 0., 1.]]), "event_start": torch.tensor([[-1, -1, 0]]),
                 "event_duration": torch.tensor([[0., 0., 8.]]), "hist_last_hot": torch.ones(1, 3)}
        original = {key: value.clone() for key, value in batch.items()}
        config = MultiTaskLossConfig(type_balanced_occupancy=False)
        _, base = compute_multitask_loss(outputs, batch, config)
        _, explicit = compute_multitask_loss(outputs, batch, replace(config, event_short_hot_fp_multiplier=1.))
        self.assertTrue(torch.equal(base["event_will"], explicit["event_will"]))
        base_grad = torch.autograd.grad(base["event_will"], outputs["event_will_logit"], retain_graph=True)[0]
        _, hard = compute_multitask_loss(outputs, batch, replace(config, event_short_hot_fp_multiplier=4.))
        hard_grad = torch.autograd.grad(hard["event_will"], outputs["event_will_logit"], retain_graph=True)[0]
        self.assertGreater(float(hard_grad[0, 0]), float(base_grad[0, 0]))
        self.assertGreater(float(hard_grad[0, 0] / hard_grad[0, 1]),
                           float(base_grad[0, 0] / base_grad[0, 1]))
        for key in batch:
            self.assertTrue(torch.equal(original[key], batch[key]), key)


if __name__ == "__main__":
    unittest.main()
