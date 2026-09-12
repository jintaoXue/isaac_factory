"""Three-class event supervision preserves the public binary report contract."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))

from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_heads import FactoryPredictionHeads
from factory_baselines.torch_losses import MultiTaskLossConfig, compute_multitask_loss
from factory_baselines import torch_trainer as trainer
import test_b5_gat_gru as model_fixture
from test_baseline_warm_start import stage_parents


@pytest.mark.parametrize("kind", ("B4", "B5"))
def test_event_classes_marginalize_and_supervise_only_valid_targets(kind):
    fixture = model_fixture.TestB5GatGru()
    fixture.setUp()
    batch = fixture._batch()
    batch["event_will"][:, 1:3] = 1
    batch["event_start"][:, 2] = 2
    batch["event_duration"][:, 1:3] = 8
    batch["hist_last_hot"][:, 2] = 1
    original = {key: value.clone() for key, value in batch.items()}
    if kind == "B4":
        model = B4GcnGru(B4ModelConfig(input_dim=6, global_dim=2, num_nodes=5,
                                       gcn_hidden=8, gru_hidden=8, dropout=0, event_head="three_class"))
    else:
        model = B5GatGru(replace(fixture.config, event_head="three_class"))
    model.eval()
    output = model(**fixture._inputs(batch))
    logits = output["event_kind_logits"]
    assert logits.shape == (4, 5, 3)
    torch.testing.assert_close(output["event_will_logit"].sigmoid(), logits.softmax(-1)[..., 1:].sum(-1))
    _, components = compute_multitask_loss(output, batch, MultiTaskLossConfig())
    targets = torch.zeros(4, 5, dtype=torch.long)
    targets[:, 1] = 1
    targets[:, 2] = 2
    weights = torch.tensor([[2., 3., 4., 0., 0.]]).expand(4, -1)
    expected = (F.cross_entropy(logits.movedim(-1, 1), targets, reduction="none") * weights).sum() / weights.sum()
    torch.testing.assert_close(components["event_will"], expected)
    gradient = torch.autograd.grad(components["event_will"], logits, retain_graph=True)[0]
    assert (gradient[:, 0, 0] < 0).all()
    assert (gradient[:, 1, 1] < 0).all()
    assert (gradient[:, 2, 2] < 0).all()
    assert (gradient[:, 3:] == 0).all()
    components["event_will"].backward()
    assert model.input_projection.weight.grad.abs().sum() > 0
    changed = {**batch, "hist_last_hot": 1 - batch["hist_last_hot"]}
    _, changed_components = compute_multitask_loss(output, changed, MultiTaskLossConfig())
    torch.testing.assert_close(components["event_will"], changed_components["event_will"], rtol=0, atol=0)
    changed["event_will"] = 1 - batch["event_will"]
    changed["event_start"] = torch.full_like(batch["event_start"], 14)
    with torch.no_grad():
        repeated = model(**fixture._inputs(changed))
    torch.testing.assert_close(output["event_will_logit"], repeated["event_will_logit"], rtol=0, atol=0)
    for key in batch:
        assert torch.equal(batch[key], original[key])


def test_default_binary_head_is_unchanged_and_three_class_prior_is_matched():
    torch.manual_seed(47)
    original = FactoryPredictionHeads(8, 0, 3, 180, 15, 10)
    torch.manual_seed(47)
    explicit = FactoryPredictionHeads(8, 0, 3, 180, 15, 10, event_head="binary")
    binary_rng = torch.get_rng_state().clone()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(47)
    ternary = FactoryPredictionHeads(8, 0, 3, 180, 15, 10, event_head="three_class")
    assert torch.equal(torch.get_rng_state(), binary_rng)
    for key, value in original.state_dict().items():
        if not key.startswith("event_will_head.2."):
            torch.testing.assert_close(value, ternary.state_dict()[key], rtol=0, atol=0)
    bias = ternary.event_will_head[-1].bias
    torch.testing.assert_close(bias.softmax(-1)[1:].sum(), original.event_will_head[-1].bias.sigmoid().squeeze())
    with pytest.raises(ValueError, match="event_head"):
        FactoryPredictionHeads(8, 0, 3, 180, 15, 10, event_head="unknown")
    for cls in (B4ModelConfig, B5ModelConfig):
        with pytest.raises(ValueError, match="event_head"):
            cls(input_dim=6, global_dim=0, num_nodes=3, event_head="unknown")


def test_three_class_no_valid_events_has_zero_event_gradient():
    fixture = model_fixture.TestB5GatGru()
    fixture.setUp()
    batch = fixture._batch()
    batch["occ_node_mask"].zero_()
    model = B5GatGru(replace(fixture.config, event_head="three_class"))
    output = model(**fixture._inputs(batch))
    _, components = compute_multitask_loss(output, batch, MultiTaskLossConfig(event_focal_gamma=2))
    assert components["event_will"] == 0
    gradient = torch.autograd.grad(components["event_will"], output["event_kind_logits"])[0]
    assert torch.isfinite(gradient).all() and (gradient == 0).all()


def test_three_class_head_can_fit_all_three_distinguishable_training_classes():
    torch.manual_seed(9)
    heads = FactoryPredictionHeads(8, 0, 3, 900, 15, 10, event_head="three_class")
    hidden = torch.eye(3, 8)
    target = torch.arange(3)
    weights = torch.tensor([2., 3., 4.])
    optimizer = torch.optim.Adam(heads.event_will_head.parameters(), lr=.05)
    initial = F.cross_entropy(heads.event_will_head(hidden), target, weight=weights).detach()
    for _ in range(50):
        optimizer.zero_grad()
        loss = F.cross_entropy(heads.event_will_head(hidden), target, weight=weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = heads.event_will_head(hidden)
        assert torch.equal(logits.argmax(-1), target)
        assert F.cross_entropy(logits, target, weight=weights) < initial * .05


@pytest.mark.parametrize("kind", ("b4_gcn_gru", "b5_gat_gru"))
def test_three_class_real_training_and_checkpoint_loading_do_not_evaluate_test(stage_parents, tmp_path, kind):
    _, dataset, _, models = stage_parents
    output = tmp_path / kind
    summary = trainer.train_torch_baseline(
        kind, dataset, output, model_overrides={**models[kind], "event_head": "three_class"},
        train_config=trainer.TorchTrainConfig(evaluate_test=False, batch_size=8, max_epochs=1,
                                               min_epochs=1, patience=1, device="cpu"),
    )
    assert summary["status"] == "validation_completed"
    loaded, checkpoint = trainer.load_checkpoint(output / "best.pt", torch.device("cpu"))
    assert checkpoint["model_config"]["event_head"] == loaded.config.event_head == "three_class"
    assert loaded.heads.event_will_head[-1].out_features == 3
    metrics = json.loads((output / "metrics_validation.json").read_text())
    assert 0 <= metrics["station_report"]["report_f1"] <= 1
    assert "time_mae_sample_count" in metrics["station_report"]
    assert not (output / "metrics_test.json").exists()
