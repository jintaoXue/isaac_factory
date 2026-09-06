"""Historical-flag audit must isolate future dependence and supervision changes."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from audit_baseline_history_causality import ongoing_group, prefix_flag
from reevaluate_main_validation import load_standalone_metrics
from factory_baselines.torch_heads import FactoryPredictionHeads
from factory_baselines.torch_losses import MultiTaskLossConfig, compute_multitask_loss


class TestHistoryCausality(unittest.TestCase):
    def test_prefix_never_uses_later_operational_states(self):
        reference = load_standalone_metrics(TOOLS.parent / "PDFormer/factory_bn/remain.py")
        features = np.zeros((12, 1, 27), dtype=np.float32)
        features[:, :, 21] = 1
        features[:, :, 0] = 4
        features[:, :, 6] = 45
        altered = features.copy()
        altered[3:, :, :21] = 0
        self.assertEqual(prefix_flag(reference, features, 3, 60, 8, 1)[0], 0)
        self.assertTrue(np.array_equal(prefix_flag(reference, features, 3, 60, 8, 1),
                                      prefix_flag(reference, altered, 3, 60, 8, 1)))
        self.assertEqual(prefix_flag(reference, features, 9, 60, 8, 1)[0], 1)
        with self.assertRaises(ValueError):
            prefix_flag(reference, features, 0, 60, 8, 1)

    def test_group_audit_excludes_negative_and_masked_cells(self):
        will, start = np.array([1, 1, 0, 1]), np.array([0, 4, -1, 4])
        valid = np.array([1, 1, 1, 0])
        old = ongoing_group(will, start, np.ones(4), valid)
        new = ongoing_group(will, start, np.zeros(4), valid)
        self.assertEqual((old != new).tolist(), [False, True, False, False])

    def test_unchanged_groups_leave_current_loss_and_gradients_unchanged(self):
        torch.manual_seed(19)
        heads = FactoryPredictionHeads(8, 0, 2, 180, 15, 10)
        hidden = torch.randn(1, 2, 8, requires_grad=True)
        node = torch.ones(1, 2, dtype=torch.bool)
        outputs = heads(hidden, node, node, torch.empty(1, 4, 0), torch.ones(1), torch.ones(1))
        hot = torch.zeros(1, 15, 2)
        hot[:, :8, 0] = 1
        batch = {"remain_mask": torch.ones(1, 15), "occ_node_mask": node,
                 "y_score": torch.zeros(1, 15, 2, 1), "y_hot": hot,
                 "target_remain_len": torch.tensor([15.]), "y_cause": torch.tensor([-1]),
                 "event_will": torch.tensor([[1., 0.]]), "event_start": torch.tensor([[0, -1]]),
                 "event_duration": torch.tensor([[8., 0.]]), "hist_last_hot": torch.ones(1, 2)}
        first, _ = compute_multitask_loss(outputs, batch, MultiTaskLossConfig())
        old_gradient = torch.autograd.grad(first, hidden, retain_graph=True)[0]
        batch["hist_last_hot"] = torch.zeros(1, 2)
        second, _ = compute_multitask_loss(outputs, batch, MultiTaskLossConfig())
        new_gradient = torch.autograd.grad(second, hidden)[0]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(old_gradient, new_gradient))


if __name__ == "__main__":
    unittest.main()
