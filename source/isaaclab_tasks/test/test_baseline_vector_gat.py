"""Test query-dependent graph ranking without changing the baseline's capacity."""

from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig, DenseGraphAttention
from factory_baselines.torch_trainer import load_checkpoint
from train_dense_baseline_control import dense_configuration
from preflight_baseline_vector_gat import restore_training_config


def model_inputs():
    return dict(x=torch.randn(2, 5, 4, 6), adjacency=torch.ones(2, 4, 4),
                node_mask=torch.tensor([[1., 1., 1., 0.]] * 2),
                target_node_mask=torch.tensor([[1., 1., 1., 0.]] * 2),
                global_features=torch.empty(2, 5, 0),
                jobs_remaining=torch.full((2,), 5.), jobs_total=torch.full((2,), 10.),
                event_precursor=torch.randn(2, 4, 23))


class TestVectorGat(unittest.TestCase):
    def test_query_can_reverse_shared_neighbor_priority_only_with_vector_score(self):
        # Two queries see the very same two neighbors. Neighbor features make
        # the two attention weights directly recoverable from the layer output.
        x = torch.tensor([[[-2., 0.], [0., -3.], [1., 0.], [0., 2.]]])
        graph = torch.eye(4)[None]
        graph[0, :2] = torch.tensor([0., 0., 1., 1.])
        mask = torch.ones(1, 4)
        observed = {}
        for mode in ("scalar_additive", "vector_additive"):
            layer = DenseGraphAttention(2, 2, 1, True, 0, mode).eval()
            with torch.no_grad():
                layer.projection.weight.copy_(torch.eye(2))
                layer.attention_source.fill_(1)
                layer.attention_target.fill_(1)
                layer.bias.zero_()
            out = layer(x, graph, mask)[0, :2]
            weights = torch.stack((out[:, 0], out[:, 1] / 2), dim=-1)
            torch.testing.assert_close(weights.sum(-1), torch.ones(2))
            observed[mode] = weights.argmax(-1).tolist()
        self.assertEqual(observed["scalar_additive"], [1, 1])
        self.assertEqual(observed["vector_additive"], [1, 0])

    def test_default_scalar_path_matches_previous_formula_exactly(self):
        torch.manual_seed(3)
        layer = DenseGraphAttention(6, 4, 2, False, 0).eval()
        x, graph, mask = torch.randn(2, 4, 6), torch.ones(2, 4, 4), torch.ones(2, 4)
        projected = layer.projection(x).view(2, 4, 2, 4)
        source = (projected * layer.attention_source).sum(-1)
        target = (projected * layer.attention_target).sum(-1)
        attention = F.leaky_relu(source[:, :, None] + target[:, None, :], .2).softmax(2)
        previous = torch.einsum("bijh,bjhd->bihd", attention, projected).mean(2) + layer.bias
        torch.testing.assert_close(layer(x, graph, mask), previous, rtol=0, atol=0)

    def test_same_parameters_initial_values_rng_and_dropout_draws(self):
        config = B5ModelConfig(6, 0, 4, gat_hidden=8, gat_heads=2, gru_hidden=8,
                               temporal_readout="last_mean", event_precursor="near")
        batch = model_inputs()
        for seed in (42, 43):
            with self.subTest(seed=seed):
                torch.manual_seed(seed); parent = B5GatGru(config)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                candidate = B5GatGru(replace(config, gat_score_mode="vector_additive"))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(list(parent.state_dict()), list(candidate.state_dict()))
                for key, value in parent.state_dict().items():
                    torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
                self.assertEqual(sum(p.numel() for p in parent.parameters()),
                                 sum(p.numel() for p in candidate.parameters()))
                torch.manual_seed(75); parent_out = parent(**batch)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(75); candidate_out = candidate(**batch)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                # Unlike a zero-gated residual, this changes the actual scoring
                # rule immediately. Identical initial predictions are not claimed.
                self.assertFalse(torch.equal(parent_out["node_hidden"], candidate_out["node_hidden"]))

    def test_vector_edges_padding_and_permutation_equivariance(self):
        torch.manual_seed(22)
        layer = DenseGraphAttention(6, 4, 2, True, 0, "vector_additive").eval()
        x = torch.randn(1, 4, 6, requires_grad=True)
        graph = torch.eye(4)[None]; graph[0, 0, 1] = 1; graph[0, 0, 3] = 1
        mask = torch.tensor([[1., 1., 1., 0.]])
        out = layer(x, graph, mask)
        out[0, 0].square().sum().backward()
        self.assertGreater(x.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(x.grad[0, 2:].abs().sum().item(), 0)
        changed = x.detach().clone(); changed[0, 2:] += 1000
        torch.testing.assert_close(layer(changed, graph, mask)[0, 0], out[0, 0], rtol=0, atol=0)
        self.assertEqual(out[0, 3].abs().sum().item(), 0)
        permutation = torch.tensor([2, 0, 3, 1])
        swapped = layer(x[:, permutation], graph[:, permutation][:, :, permutation], mask[:, permutation])
        torch.testing.assert_close(swapped, out[:, permutation], rtol=1e-5, atol=1e-6)

    def test_event_loss_updates_both_dynamic_layers_and_gru(self):
        torch.manual_seed(44)
        model = B5GatGru(B5ModelConfig(6, 0, 4, gat_hidden=8, gat_heads=2, gru_hidden=8,
            dropout=0, temporal_readout="last_mean", event_precursor="near",
            gat_score_mode="vector_additive"))
        batch = model_inputs(); out = model(**batch)
        target = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
        F.binary_cross_entropy_with_logits(out["event_will_logit"][:, :3], target).backward()
        for name, param in model.named_parameters():
            if name.startswith(("gat1.", "gat2.", "gru.")):
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(param.grad.abs().sum().item(), 0, name)

    def test_checkpoint_roundtrip_and_old_config_default(self):
        batch = model_inputs()
        for mode in ("scalar_additive", "vector_additive"):
            with self.subTest(mode=mode):
                model = B5GatGru(B5ModelConfig(6, 0, 4, gat_hidden=8, gat_heads=2,
                    gru_hidden=8, temporal_readout="last_mean", event_precursor="near",
                    gat_score_mode=mode)).eval()
                config = model.config.to_dict()
                if mode == "scalar_additive":
                    del config["gat_score_mode"]
                buffer = io.BytesIO()
                torch.save(dict(model_kind="b5_gat_gru", model_config=config,
                                model_state_dict=model.state_dict()), buffer)
                buffer.seek(0); restored, _ = load_checkpoint(buffer, torch.device("cpu")); restored.eval()
                self.assertEqual(restored.config.gat_score_mode, mode)
                expected, actual = model(**batch), restored(**batch)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_invalid_score_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "GAT score"):
            B5ModelConfig(6, 0, 4, gat_score_mode="typo")
        with self.assertRaisesRegex(ValueError, "GAT score"):
            DenseGraphAttention(6, 4, 2, True, 0, "typo")

    def test_registered_candidate_changes_only_b5_attention_formula(self):
        for seed in (42, 43):
            with self.subTest(seed=seed):
                train, arch, loss = dense_configuration("B5", "near_precursor", seed, "cpu")
                new_train, new_arch, new_loss = dense_configuration("B5", "vector_gat", seed, "cpu")
                new_train.training_profile = train.training_profile
                self.assertEqual(new_train, train)
                self.assertEqual(new_loss, loss)
                self.assertEqual(new_arch.pop("gat_score_mode"), "vector_additive")
                self.assertEqual(new_arch, arch)
                self.assertFalse(new_train.evaluate_test)
        with self.assertRaisesRegex(ValueError, "B5-only"):
            dense_configuration("B4", "vector_gat", 42, "cpu")

    def test_preflight_restores_json_tuple_without_ignoring_changed_values(self):
        original = dense_configuration("B5", "near_precursor", 42, "cuda:0")[0]
        saved = json.loads(json.dumps(asdict(original)))
        self.assertIsInstance(saved["report_threshold_sweep"], list)
        self.assertEqual(restore_training_config(saved), original)
        saved["report_threshold_sweep"][0] += .01
        self.assertNotEqual(restore_training_config(saved), original)
        saved = json.loads(json.dumps(asdict(original)))
        saved["batch_size"] += 1
        self.assertNotEqual(restore_training_config(saved), original)


if __name__ == "__main__":
    unittest.main()
