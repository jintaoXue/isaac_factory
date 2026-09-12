"""Check the event F-beta ablation against source and actual gradient paths."""

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_losses import MultiTaskLossConfig, _event_soft_fbeta, compute_multitask_loss
from train_dense_baseline_control import dense_configuration
import test_b5_gat_gru as fixture_module


class TestEventFbeta(unittest.TestCase):
    def test_value_and_gradients_match_the_pinned_main_source_block(self):
        path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer/factory_bn/model.py"
        source = subprocess.check_output(["git", "show", "20c40e230aedee6aef2429d352413fbcf0fa571a:" + path],
                                         cwd=TOOLS.parents[5], text=True)
        blocks = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Attribute)
                  and n.test.left.attr == "w_event_f1"]
        self.assertEqual(len(blocks), 1)
        reference = compile(ast.Module(body=blocks, type_ignores=[]), path, "exec")
        generator = torch.Generator().manual_seed(8)
        for trial in range(16):
            logits = torch.randn(3, 7, generator=generator, dtype=torch.float64).requires_grad_()
            labels = torch.randint(0, 2, (3, 7), generator=generator).double()
            weights = torch.randint(0, 5, (3, 7), generator=generator).double()
            weights[0, 0] = 1
            up = labels.bool() & (weights > 0) & (torch.rand(3, 7, generator=generator) > .5)
            if trial % 4 == 0:
                up.zero_()
            env = dict(torch=torch, self=SimpleNamespace(w_event_f1=1., event_f1_beta=1.5),
                       logit=logits, y_will=labels, w=weights, upcoming=up,
                       ongoing=labels.bool() & ~up, loss_will=logits.sum() * 0)
            exec(reference, env)
            expected = env["loss_will"]
            actual = _event_soft_fbeta(logits, labels, up, weights, 1.5)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
            b = torch.autograd.grad(expected, logits)[0]
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)

    def test_gradient_directions_upcoming_emphasis_and_beta_tradeoff(self):
        logits = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[1., 1., 0., 0.]], dtype=torch.float64)
        up = torch.tensor([[False, True, False, False]])
        loss = _event_soft_fbeta(logits, y, up, torch.ones_like(logits), 1.5)
        grad = torch.autograd.grad(loss, logits)[0]
        self.assertTrue((grad[0, :2] < 0).all())
        self.assertTrue((grad[0, 2:] > 0).all())
        self.assertGreater(abs(grad[0, 1].item()), abs(grad[0, 0].item()))
        ratios = []
        for beta in (1., 1.5):
            x = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
            value = _event_soft_fbeta(x, torch.tensor([[1., 0.]]), torch.zeros_like(x).bool(), torch.ones_like(x), beta)
            g = torch.autograd.grad(value, x)[0]
            ratios.append(abs((g[0, 0] / g[0, 1]).item()))
        self.assertAlmostEqual(ratios[0], 3., places=4)
        self.assertAlmostEqual(ratios[1], 5.5, places=4)

    def test_masked_nodes_absent_positive_and_empty_support(self):
        x = torch.tensor([[0., 0., 500.]], requires_grad=True)
        y = torch.tensor([[1., 0., 1.]])
        up = torch.tensor([[True, False, False]])
        weights = torch.tensor([[4., 2., 0.]], requires_grad=True)
        loss = _event_soft_fbeta(x, y, up, weights, 1.5)
        loss.backward()
        self.assertEqual(x.grad[0, 2].item(), 0)
        self.assertIsNone(weights.grad)
        changed = x.detach().clone(); changed[0, 2] = -500
        torch.testing.assert_close(_event_soft_fbeta(changed, y, up, weights, 1.5), loss, rtol=0, atol=0)
        for valid, expected in ((True, 1.), (False, 0.)):
            values = torch.tensor([[-1000., 1000.]], requires_grad=True)
            value = _event_soft_fbeta(values, torch.zeros_like(values), torch.zeros_like(values).bool(),
                                      torch.full_like(values, float(valid)), 1.5)
            self.assertEqual(value.item(), expected)
            value.backward()
            self.assertTrue((values.grad == 0).all())

    def test_only_the_registered_loss_term_changes_and_gradients_reach_event_encoder(self):
        fixture = fixture_module.TestB5GatGru(); fixture.setUp(); batch = fixture._batch()
        batch["event_will"][:, 0] = 1; batch["event_start"][:, 0] = 1; batch["event_duration"][:, 0] = 8
        inputs = fixture._inputs(batch); inputs["event_precursor"] = torch.randn(4, 5, 23)
        for cls, cfg, spatial in ((B4GcnGru, B4ModelConfig, "gcn_hidden"),
                                  (B5GatGru, B5ModelConfig, "gat_hidden")):
            with self.subTest(model=cls.__name__):
                model = cls(cfg(6, 2, 5, **{spatial: 8}, gru_hidden=8, dropout=0,
                                event_precursor="near", temporal_readout="last_mean")).eval()
                output = model(**inputs); before = {k: v.clone() for k, v in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                config = MultiTaskLossConfig()
                old, original_parts = compute_multitask_loss(output, batch, config)
                total, parts = compute_multitask_loss(output, batch, replace(config, event_fbeta_weight=.8))
                self.assertNotIn("event_fbeta", original_parts)
                for k, v in original_parts.items():
                    torch.testing.assert_close(parts[k], v, rtol=0, atol=0)
                torch.testing.assert_close(total-old, config.lambda_event_will * .8 * parts["event_fbeta"])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
                parts["event_fbeta"].backward()
                self.assertGreater(model.gru.weight_ih_l0.grad.abs().sum().item(), 0)
                self.assertGreater(model.heads.event_will_head[-1].weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.heads.precursor_projection[-1].weight.grad.abs().sum().item(), 0)
                self.assertIsNone(model.heads.event_start_head[-1].weight.grad)
                self.assertIsNone(model.heads.remain_hot_head[-1].weight.grad)

    def test_dense_configuration_changes_only_fbeta_weight_and_no_architecture(self):
        for model in ("B4", "B5"):
            for seed in (42, 43):
                with self.subTest(model=model, seed=seed):
                    train, arch, loss = dense_configuration(model, "near_precursor", seed, "cpu")
                    other_train, other_arch, other_loss = dense_configuration(model, "event_fbeta", seed, "cpu")
                    other_train.training_profile = train.training_profile
                    self.assertEqual(other_train, train); self.assertEqual(other_arch, arch)
                    self.assertEqual(replace(other_loss, event_fbeta_weight=0), loss)
                    self.assertEqual(other_loss.event_fbeta_weight, .8)
                    self.assertEqual(other_loss.event_fbeta_beta, 1.5)
                    self.assertEqual(other_train.event_oversample_factor, 1.)
                    self.assertFalse(other_train.evaluate_test)
                    self.assertEqual(MultiTaskLossConfig.from_dict(other_loss.to_dict()), other_loss)

    def test_invalid_configuration_and_target_grids_are_rejected(self):
        for v in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError): MultiTaskLossConfig(event_fbeta_weight=v)
        for v in (0., .09, float("nan"), float("inf")):
            with self.assertRaises(ValueError): MultiTaskLossConfig(event_fbeta_beta=v)
        x = torch.zeros(1, 2); up = x.bool(); w = torch.ones_like(x)
        for y in (torch.ones(2, 2), torch.full_like(x, .5), torch.full_like(x, float("nan"))):
            with self.assertRaises(ValueError): _event_soft_fbeta(x, y, up, w, 1.5)
        for bad in (torch.full_like(x, -1), torch.full_like(x, float("nan"))):
            with self.assertRaises(ValueError): _event_soft_fbeta(x, x, up, bad, 1.5)
        with self.assertRaises(ValueError): _event_soft_fbeta(x, x, torch.ones_like(up), w, 1.5)


if __name__ == "__main__":
    unittest.main()
