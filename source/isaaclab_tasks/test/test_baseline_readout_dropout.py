"""Controlled training-only temporal-readout regularization and old-source parity."""

from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
import types

import torch
from torch.nn import functional as F


TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from train_dense_baseline_control import dense_configuration


def cases():
    return (("b4_gcn_gru", B4GcnGru, B4ModelConfig(6, 2, 4, gcn_hidden=8, gru_hidden=8,
                event_precursor="near", temporal_readout="last_mean")),
            ("b5_gat_gru", B5GatGru, B5ModelConfig(6, 2, 4, gat_hidden=8, gat_heads=2, gru_hidden=8,
                event_precursor="near", temporal_readout="last_mean")))


def inputs():
    torch.manual_seed(1001)
    return dict(x=torch.randn(4, 30, 4, 6), adjacency=torch.ones(4, 4, 4),
        node_mask=torch.tensor([[1., 1., 1., 0.]] * 4), target_node_mask=torch.tensor([[1., 1., 1., 0.]] * 4),
        global_features=torch.randn(4, 30, 2), jobs_remaining=torch.full((4,), 5.), jobs_total=torch.full((4,), 10.),
        event_precursor=torch.randn(4, 4, 23))


def prior_module(name):
    path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/factory_baselines/" + name + ".py"
    source = subprocess.check_output(["git", "show", "979c680fb4d1919ae760cdfb0038f69fb7cc6708:" + path])
    module = types.ModuleType("factory_baselines._prior_" + name); module.__package__ = "factory_baselines"
    sys.modules[module.__name__] = module
    exec(compile(source, "prior_" + name, "exec"), module.__dict__)
    return module


def test_default_zero_preserves_prior_source_outputs_rng_and_checkpoint_schema():
    batch = inputs()
    for name, cls, config in cases():
        old = prior_module(name); old_cls = getattr(old, cls.__name__); old_config_cls = getattr(old, type(config).__name__)
        values = config.to_dict(); del values["readout_dropout"]
        assert type(config).from_dict(values).readout_dropout == 0
        torch.manual_seed(42); parent = old_cls(old_config_cls(**values)); initial_rng = torch.get_rng_state().clone()
        torch.manual_seed(42); candidate = cls(config)
        assert torch.equal(initial_rng, torch.get_rng_state())
        assert parent.state_dict().keys() == candidate.state_dict().keys()
        candidate.load_state_dict(parent.state_dict(), strict=True)
        for train in (False, True):
            parent.train(train); candidate.train(train)
            torch.manual_seed(912); expected = parent(**batch); rng = torch.get_rng_state().clone()
            torch.manual_seed(912); actual = candidate(**batch)
            assert torch.equal(rng, torch.get_rng_state())
            assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_positive_dropout_has_no_parameters_or_eval_difference():
    batch = inputs()
    for _, cls, config in cases():
        for seed in (42, 43):
            torch.manual_seed(seed); parent = cls(config); rng = torch.get_rng_state().clone()
            torch.manual_seed(seed); candidate = cls(replace(config, readout_dropout=.2))
            assert torch.equal(rng, torch.get_rng_state())
            assert parent.state_dict().keys() == candidate.state_dict().keys()
            assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in parent.state_dict().items())
            assert sum(p.numel() for p in parent.parameters()) == sum(p.numel() for p in candidate.parameters())
            parent.eval(); candidate.eval()
            with torch.no_grad():
                expected, actual = parent(**batch), candidate(**batch)
            assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_training_mask_is_shared_and_backbone_receives_gradients():
    batch = inputs()
    for name, cls, config in cases():
        torch.manual_seed(42); model = cls(replace(config, readout_dropout=.2)).train()
        captured = []
        def observe(_module, args, output): captured.append((args[0], output))
        hook = model.readout_dropout.register_forward_hook(observe)
        result = model(**batch); hook.remove(); before, after = captured[0]
        valid = batch["node_mask"].bool()[:, :, None].expand_as(before)
        kept, dropped = valid & (after != 0), valid & (after == 0) & (before != 0)
        assert kept.any() and dropped.any()
        torch.testing.assert_close(after[kept], before[kept] / .8, rtol=1e-6, atol=1e-6)
        assert torch.equal(result["node_hidden"], after * batch["node_mask"][:, :, None])
        loss = F.binary_cross_entropy_with_logits(result["event_will_logit"][:, :3], torch.zeros(4, 3))
        loss.backward()
        names = ("gru.weight_ih_l0", "heads.event_will_head.0.weight",
                 "gcn1.linear.weight" if name.startswith("b4") else "gat1.projection.weight")
        parameters = dict(model.named_parameters())
        for key in names:
            assert parameters[key].grad is not None and torch.isfinite(parameters[key].grad).all()
            assert parameters[key].grad.abs().sum() > 0


def test_configuration_changes_only_training_readout_dropout_and_rejects_invalid_p():
    for model in ("B4", "B5"):
        for seed in (42, 43):
            parent, arch, loss = dense_configuration(model, "near_precursor", seed, "cpu")
            candidate, new_arch, new_loss = dense_configuration(model, "readout_dropout", seed, "cpu")
            first, second = asdict(parent), asdict(candidate); first.pop("training_profile"); second.pop("training_profile")
            assert first == second and loss == new_loss and new_arch == {**arch, "readout_dropout": .2}
            assert not candidate.evaluate_test and candidate.max_epochs == 60
    for _, _, config in cases():
        for value in (-1., 1., float("nan"), float("inf")):
            try: replace(config, readout_dropout=value)
            except ValueError: pass
            else: raise AssertionError("Invalid probability accepted")
