"""Controlled onset supervision: gradients, inference isolation and serialization."""

from dataclasses import replace
import io
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_heads import FactoryPredictionHeads
from factory_baselines.torch_losses import MultiTaskLossConfig, _onset_auxiliary_loss, compute_multitask_loss
from factory_baselines.torch_trainer import load_checkpoint
import test_b5_gat_gru as fixture_module


def test_onset_loss_excludes_ongoing_and_masks_and_does_not_depend_on_history_hot():
    logits = torch.tensor([[0., .2, -.4, 1., 2., 3.]], requires_grad=True)
    batch = {"event_will": torch.tensor([[1., 1., 0., 0., 1., 1.]]),
             "event_start": torch.tensor([[0, 2, -1, -1, 1, 0]]),
             "occ_node_mask": torch.tensor([[1, 1, 1, 1, 0, 1]]),
             "hist_last_hot": torch.tensor([[1., 1., 0., 0., 0., 0.]])}
    loss = _onset_auxiliary_loss(logits, batch)
    expected = .5 * (F.softplus(-logits[0, 1]) + F.softplus(logits[0, 2:4]).mean())
    torch.testing.assert_close(loss, expected)
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad[0, 1] < 0 and (grad[0, 2:4] > 0).all()
    assert torch.equal(grad[0, [0, 4, 5]], torch.zeros(3))
    batch["hist_last_hot"] = 1 - batch["hist_last_hot"]
    torch.testing.assert_close(_onset_auxiliary_loss(logits, batch), loss, rtol=0, atol=0)
    # Duplicating identical negatives must not dilute the positive class gradient.
    doubled = {k: torch.cat([v, v[:, 2:4]], dim=1) for k, v in batch.items()}
    more = torch.cat([logits, logits[:, 2:4]], dim=1)
    torch.testing.assert_close(_onset_auxiliary_loss(more, doubled), loss)
    batch["event_start"][0, 1] = -1
    with pytest.raises(ValueError, match="valid start"):
        _onset_auxiliary_loss(logits, batch)


@pytest.mark.parametrize("case", ["negative", "upcoming", "ongoing", "masked"])
def test_absent_class_contributes_zero_without_rescaling_or_nan(case):
    logits = torch.zeros(1, 3, requires_grad=True)
    batch = {"event_will": torch.full((1, 3), 0. if case == "negative" else 1.),
             "event_start": torch.full((1, 3), 1 if case == "upcoming" else 0),
             "occ_node_mask": torch.full((1, 3), case != "masked")}
    loss = _onset_auxiliary_loss(logits, batch)
    loss.backward()
    if case in {"negative", "upcoming"}:
        torch.testing.assert_close(loss, .5 * torch.log(torch.tensor(2.)))
        assert (logits.grad > 0).all() if case == "negative" else (logits.grad < 0).all()
    else:
        assert loss == 0 and (logits.grad == 0).all()


@pytest.mark.parametrize("kind", ["b4_gcn_gru", "b5_gat_gru"])
def test_same_initialization_outputs_shared_gradients_and_checkpoint_roundtrip(kind):
    fixture = fixture_module.TestB5GatGru()
    fixture.setUp()
    batch = fixture._batch()
    batch["event_will"][:, 0] = 1
    batch["event_start"][:, 0] = 1
    batch["event_duration"][:, 0] = 8
    inputs = fixture._inputs(batch)
    inputs["event_precursor"] = torch.randn(4, 5, 23)
    cls, config_cls, spatial = (B4GcnGru, B4ModelConfig, "gcn_hidden") if kind.startswith("b4") else (B5GatGru, B5ModelConfig, "gat_hidden")
    values = dict(input_dim=6, global_dim=2, num_nodes=5, gru_hidden=8,
                  dropout=0., temporal_readout="last_mean", event_precursor="near", **{spatial: 8})
    torch.manual_seed(30)
    control = cls(config_cls(**values)).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(30)
    candidate = cls(config_cls(**values, event_onset_aux=True)).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in control.state_dict().items():
        assert torch.equal(v, candidate.state_dict()[k]), k
    original, output = control(**inputs), candidate(**inputs)
    for k, v in original.items():
        torch.testing.assert_close(output[k], v, rtol=0, atol=0)
    torch.testing.assert_close(output["event_onset_logit"], output["event_will_logit"], rtol=0, atol=0)
    config = MultiTaskLossConfig(lambda_event_onset_aux=1.)
    base, base_parts = compute_multitask_loss(original, batch, replace(config, lambda_event_onset_aux=0.))
    total, parts = compute_multitask_loss(output, batch, config)
    torch.testing.assert_close(total - base, parts["event_onset_aux"])
    for k, v in base_parts.items():
        torch.testing.assert_close(parts[k], v, rtol=0, atol=0)
    parts["event_onset_aux"].backward()
    assert candidate.gru.weight_ih_l0.grad.abs().sum() > 0
    assert candidate.heads.event_onset_head[-1].weight.grad.abs().sum() > 0
    assert candidate.heads.event_will_head[-1].weight.grad is None
    assert candidate.heads.precursor_projection[-1].weight.grad.abs().sum() > 0
    # An auxiliary-only parameter update cannot alter official predictions.
    with torch.no_grad():
        candidate.heads.event_onset_head[-1].bias.add_(4.)
    changed = candidate(**inputs)
    for k in original:
        torch.testing.assert_close(changed[k], original[k], rtol=0, atol=0)
    assert not torch.equal(changed["event_onset_logit"], output["event_onset_logit"])
    buffer = io.BytesIO()
    torch.save(dict(model_kind=kind, model_config=candidate.config.to_dict(),
                    model_state_dict=candidate.state_dict()), buffer)
    buffer.seek(0)
    loaded, _ = load_checkpoint(buffer, torch.device("cpu"))
    loaded.eval()
    for k, v in changed.items():
        torch.testing.assert_close(loaded(**inputs)[k], v, rtol=0, atol=0)
    with pytest.raises(ValueError, match="independent head"):
        compute_multitask_loss(original, batch, config)


def test_invalid_auxiliary_configuration_rejected():
    for coefficient in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MultiTaskLossConfig(lambda_event_onset_aux=coefficient)
    with pytest.raises(ValueError, match="binary"):
        FactoryPredictionHeads(8, 0, 3, 180, 15, 10, event_head="three_class", event_onset_aux=True)
