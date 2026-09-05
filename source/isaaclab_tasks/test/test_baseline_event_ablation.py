"""Tests for shared event context and focal-loss ablations."""

from dataclasses import replace
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from factory_baselines.torch_heads import FactoryPredictionHeads
from factory_baselines.torch_losses import MultiTaskLossConfig, _event_binary_loss
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_trainer import TorchTrainConfig, save_checkpoint, load_checkpoint


class TestEventAblation(unittest.TestCase):
    def test_focal_zero_is_bce_and_easy_negatives_are_downweighted(self):
        logits = torch.tensor([-10., 0., 10., -10.], requires_grad=True)
        target = torch.tensor([0., 0., 0., 1.])
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        self.assertTrue(torch.equal(_event_binary_loss(logits, target, 0), bce))
        focal = _event_binary_loss(logits, target, 2)
        self.assertLess(float(focal[0].detach()), float(bce[0].detach()) * .01)
        self.assertAlmostEqual(float(focal[1].detach()), float(bce[1].detach()) * .25)
        focal.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        for gamma in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MultiTaskLossConfig(event_focal_gamma=gamma)

    def test_context_uses_other_valid_nodes_but_not_padding(self):
        torch.manual_seed(31)
        heads = FactoryPredictionHeads(8, 0, 3, 180, 15, 10, event_context=True).eval()
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        def predict(value):
            return heads(value, mask, mask, torch.empty(2, 4, 0),
                         torch.ones(2), torch.full((2,), 10.))["event_will_logit"]
        original = predict(hidden)
        changed = hidden.detach().clone()
        changed[:, 2] = 1000
        self.assertTrue(torch.equal(original, predict(changed)))
        original[:, 0].sum().backward()
        self.assertGreater(float(hidden.grad[:, 1].abs().sum()), 0)
        self.assertEqual(float(hidden.grad[:, 2].abs().sum()), 0)

    def test_context_model_checkpoint_round_trip(self):
        configs = (
            ("b4_gcn_gru", B4GcnGru, B4ModelConfig(6, 0, 3, gcn_hidden=8, gru_hidden=8)),
            ("b5_gat_gru", B5GatGru, B5ModelConfig(6, 0, 3, gat_hidden=8, gat_heads=2, gru_hidden=8)),
        )
        for kind, model_class, config in configs:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                model = model_class(replace(config, event_context=True, dropout=0)).eval()
                inputs = dict(x=torch.randn(2, 4, 3, 6), adjacency=torch.ones(2, 3, 3),
                              node_mask=torch.ones(2, 3, dtype=torch.bool),
                              target_node_mask=torch.ones(2, 3, dtype=torch.bool),
                              global_features=torch.empty(2, 4, 0),
                              jobs_remaining=torch.ones(2), jobs_total=torch.ones(2))
                expected = model(**inputs)["event_will_logit"]
                path = Path(temp) / "best.pt"
                save_checkpoint(path, model, None, 1, .1, kind, model.config,
                                MultiTaskLossConfig(event_focal_gamma=2), TorchTrainConfig(), {})
                loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
                loaded.eval()
                self.assertTrue(torch.equal(expected, loaded(**inputs)["event_will_logit"]))
                self.assertEqual(checkpoint["loss_config"]["event_focal_gamma"], 2)


if __name__ == "__main__":
    unittest.main()
