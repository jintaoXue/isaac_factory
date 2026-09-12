"""Verify that post-GRU graph exchange is a trainable, controlled ablation."""

from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_trainer import load_checkpoint
from train_dense_baseline_control import dense_configuration


def candidates():
    return (
        ("b4_gcn_gru", B4GcnGru, B4ModelConfig(6, 2, 4, gcn_hidden=8, gru_hidden=8,
            event_precursor="near", temporal_readout="last_mean")),
        ("b5_gat_gru", B5GatGru, B5ModelConfig(6, 2, 4, gat_hidden=8, gat_heads=2, gru_hidden=8,
            event_precursor="near", temporal_readout="last_mean")),
    )


def inputs():
    return dict(x=torch.randn(2, 5, 4, 6), adjacency=torch.ones(2, 4, 4),
        node_mask=torch.tensor([[1., 1., 1., 0.]] * 2),
        target_node_mask=torch.tensor([[1., 1., 1., 0.]] * 2),
        global_features=torch.randn(2, 5, 2), jobs_remaining=torch.full((2,), 5.),
        jobs_total=torch.full((2,), 10.), event_precursor=torch.randn(2, 4, 23))


class TestHistoryGraphRefinement(unittest.TestCase):
    def test_common_initial_weights_rng_and_predictions_match_parent(self):
        batch = inputs()
        for kind, cls, config in candidates():
            for seed in (42, 43):
                with self.subTest(model=kind, seed=seed):
                    torch.manual_seed(seed)
                    parent = cls(config)
                    expected_rng = torch.get_rng_state().clone()
                    torch.manual_seed(seed)
                    candidate = cls(replace(config, history_graph_refine=True))
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                    for key, value in parent.state_dict().items():
                        self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)
                    for train in (False, True):
                        parent.train(train); candidate.train(train)
                        torch.manual_seed(902)
                        expected = parent(**batch)
                        after = torch.get_rng_state().clone()
                        torch.manual_seed(902)
                        actual = candidate(**batch)
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        for key in expected:
                            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_graph_exchange_uses_neighbor_histories_and_respects_edges_and_padding(self):
        for kind, cls, config in candidates():
            with self.subTest(model=kind):
                torch.manual_seed(25)
                block = cls(replace(config, history_graph_refine=True)).history_graph.eval()
                with torch.no_grad():
                    block.output.weight.normal_(std=.1)
                history = torch.randn(1, 4, 8, requires_grad=True)
                mask = torch.tensor([[1., 1., 1., 0.]])
                adjacency = torch.eye(4)[None]
                adjacency[0, 0, 1] = adjacency[0, 1, 0] = 1
                adjacency[0, 0, 3] = adjacency[0, 3, 0] = 1
                out = block(history, adjacency, mask)
                out[0, 0, 0].backward()
                self.assertGreater(history.grad[0, 1].abs().sum().item(), 0)
                self.assertEqual(history.grad[0, 2:].abs().sum().item(), 0)
                changed = history.detach().clone(); changed[0, 2:] += 1000
                torch.testing.assert_close(block(changed, adjacency, mask)[0, 0], out[0, 0], rtol=0, atol=0)
                self.assertEqual(out[0, 3].abs().sum().item(), 0)
                permutation = torch.tensor([2, 0, 3, 1])
                swapped = block(history[:, permutation], adjacency[:, permutation][:, :, permutation], mask[:, permutation])
                torch.testing.assert_close(swapped, out[:, permutation], rtol=1e-5, atol=1e-6)

    def test_zero_initial_projection_learns_then_trains_the_graph_layer(self):
        for kind, cls, config in candidates():
            with self.subTest(model=kind):
                torch.manual_seed(61)
                model = cls(replace(config, dropout=0, history_graph_refine=True)).train()
                batch = inputs(); target = torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.]])
                optimizer = torch.optim.SGD(model.parameters(), lr=.05)
                graph_weight = (model.history_graph.graph_layer.linear.weight if kind.startswith("b4")
                                else model.history_graph.graph_layer.projection.weight)
                for step in range(2):
                    optimizer.zero_grad()
                    logit = model(**batch)["event_will_logit"]
                    F.binary_cross_entropy_with_logits(logit[:, :3], target[:, :3]).backward()
                    self.assertGreater(model.history_graph.output.weight.grad.abs().sum().item(), 0)
                    self.assertTrue(torch.isfinite(graph_weight.grad).all())
                    if step == 0:
                        self.assertEqual(graph_weight.grad.abs().sum().item(), 0)
                    else:
                        self.assertGreater(graph_weight.grad.abs().sum().item(), 0)
                    optimizer.step()

    def test_trained_candidate_checkpoint_round_trip_and_parent_remains_loadable(self):
        for kind, cls, config in candidates():
            for enabled in (False, True):
                with self.subTest(model=kind, enabled=enabled):
                    model = cls(replace(config, dropout=0, history_graph_refine=enabled)).eval()
                    if enabled:
                        with torch.no_grad():
                            model.history_graph.output.weight.normal_(std=.05)
                    batch = inputs(); expected = model(**batch)
                    saved = io.BytesIO()
                    config_dict = model.config.to_dict()
                    if not enabled:
                        del config_dict["history_graph_refine"]
                    torch.save(dict(model_kind=kind, model_config=config_dict,
                                    model_state_dict=model.state_dict()), saved)
                    saved.seek(0); restored, _ = load_checkpoint(saved, torch.device("cpu")); restored.eval()
                    self.assertEqual(restored.config.history_graph_refine, enabled)
                    actual = restored(**batch)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_dense_variant_preserves_inputs_losses_training_and_reports_added_capacity(self):
        for name, cls, config_cls, extra in (("B4", B4GcnGru, B4ModelConfig, 16576),
                                           ("B5", B5GatGru, B5ModelConfig, 16704)):
            for seed in (42, 43):
                with self.subTest(model=name, seed=seed):
                    train, arch, loss = dense_configuration(name, "near_precursor", seed, "cpu")
                    candidate_train, candidate_arch, candidate_loss = dense_configuration(name, "history_graph_refine", seed, "cpu")
                    candidate_train.training_profile = train.training_profile
                    self.assertEqual(candidate_train, train)
                    self.assertEqual(candidate_loss, loss)
                    self.assertTrue(candidate_arch.pop("history_graph_refine"))
                    self.assertEqual(candidate_arch, arch)
                    parent_config = config_cls(input_dim=27, global_dim=2, num_nodes=4, **arch)
                    parent = cls(parent_config)
                    candidate = cls(replace(parent_config, history_graph_refine=True))
                    self.assertEqual(sum(p.numel() for p in candidate.parameters())-
                                     sum(p.numel() for p in parent.parameters()), extra)
                    self.assertFalse(candidate_train.evaluate_test)


if __name__ == "__main__":
    unittest.main()
