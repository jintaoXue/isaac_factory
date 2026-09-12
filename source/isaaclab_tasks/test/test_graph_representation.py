"""B4/B5 representation controls preserve masking and checkpoint semantics."""

from dataclasses import asdict, replace
import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_heads import TemporalAttentionPool
from factory_baselines.torch_losses import MultiTaskLossConfig
from factory_baselines.torch_trainer import TorchTrainConfig, load_checkpoint, save_checkpoint

MODELS = [
    ("b4_gcn_gru", B4GcnGru, B4ModelConfig, {"gcn_hidden": 8}),
    ("b5_gat_gru", B5GatGru, B5ModelConfig, {"gat_hidden": 8, "gat_heads": 2}),
]


def inputs():
    mask = torch.tensor([[True, True, False], [True, True, False]])
    return dict(
        x=torch.randn(2, 4, 3, 6), adjacency=torch.ones(2, 3, 3),
        node_mask=mask, target_node_mask=mask, global_features=torch.empty(2, 4, 0),
        jobs_remaining=torch.ones(2), jobs_total=torch.full((2,), 10.),
    )


@pytest.mark.parametrize("identity,readout", [(0, "last"), (4, "last"), (0, "last_mean"), (4, "last_mean")])
@pytest.mark.parametrize("kind,model_class,config_class,spatial", MODELS)
def test_masking_gradients_and_round_trip(tmp_path, identity, readout, kind, model_class, config_class, spatial):
    torch.manual_seed(17)
    config = config_class(6, 0, 3, **spatial, gru_hidden=8, dropout=0,
                          node_embedding=identity, temporal_readout=readout)
    model = model_class(config).eval()
    batch = inputs()
    output = model(**batch)
    changed = {**batch, "x": batch["x"].clone()}
    changed["x"][:, :, 2] = 1000
    assert torch.equal(output["event_will_logit"], model(**changed)["event_will_logit"])
    assert torch.count_nonzero(output["node_hidden"][:, 2]) == 0
    output["event_will_logit"][:, :2].sum().backward()
    if identity:
        gradient = model.node_identity[0].weight.grad
        assert float(gradient[:2].abs().sum()) > 0
        assert torch.count_nonzero(gradient[2]) == 0
    if readout == "last_mean":
        assert float(model.history_readout[0].weight.grad.abs().sum()) > 0
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, None, 1, .1, kind, config,
                    MultiTaskLossConfig(), TorchTrainConfig(), {})
    loaded, _ = load_checkpoint(path, torch.device("cpu"))
    loaded.eval()
    assert torch.equal(output["event_will_logit"], loaded(**batch)["event_will_logit"])


@pytest.mark.parametrize("kind,model_class,config_class,spatial", MODELS)
def test_last_mean_receives_both_history_and_last_state(kind, model_class, config_class, spatial):
    model = model_class(config_class(6, 0, 3, **spatial, gru_hidden=8,
                                     temporal_readout="last_mean", dropout=0)).eval()
    seen = {}
    gru_hook = model.gru.register_forward_hook(lambda _, args, result: seen.update(gru=result[0]))
    read_hook = model.history_readout.register_forward_pre_hook(lambda _, args: seen.update(read=args[0]))
    model(**inputs())
    gru_hook.remove()
    read_hook.remove()
    assert torch.equal(seen["read"][:, :8], seen["gru"][:, -1])
    assert torch.equal(seen["read"][:, 8:], seen["gru"].mean(1))


@pytest.mark.parametrize("kind,model_class,config_class,spatial", MODELS)
def test_reject_invalid_representation(kind, model_class, config_class, spatial):
    with pytest.raises(ValueError, match="node_embedding"):
        config_class(6, 0, 3, node_embedding=-1)
    with pytest.raises(ValueError, match="temporal_readout"):
        config_class(6, 0, 3, temporal_readout="future")


@pytest.mark.parametrize("steps", [1, 4, 30])
def test_attention_pool_starts_at_exact_mean_and_can_select_history(steps):
    pool = TemporalAttentionPool(2)
    history = torch.stack((torch.linspace(-2, 2, steps), torch.zeros(steps)), dim=-1)[None]
    assert torch.equal(pool(history), history.mean(1))
    if steps == 1:
        return
    with torch.no_grad():
        pool.query[0] = 20
    focused = pool(history)
    assert focused[0, 0] > 1.9
    weights = (history @ pool.query / (2 ** .5)).softmax(1)
    torch.testing.assert_close(focused, (weights[:, :, None] * history).sum(1))
    with torch.no_grad():
        pool.query.neg_()
    assert pool(history)[0, 0] < -1.9


@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("kind,model_class,config_class,spatial", MODELS)
def test_attention_control_initialization_learning_masking_and_checkpoint(seed, kind, model_class, config_class, spatial):
    config = config_class(6, 0, 3, **spatial, gru_hidden=128, dropout=.2,
                          temporal_readout="last_mean", event_precursor="near")
    torch.manual_seed(seed)
    control = model_class(config)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = model_class(replace(config, temporal_readout="last_attention"))
    assert torch.equal(rng, torch.get_rng_state())
    original_state, candidate_state = control.state_dict(), candidate.state_dict()
    assert set(candidate_state) - set(original_state) == {"history_attention.query"}
    for k, v in original_state.items():
        assert torch.equal(v, candidate_state[k]), k
    assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in control.parameters()) == 128

    batch = inputs()
    batch["x"] = torch.randn(2, 30, 3, 6)
    batch["event_precursor"] = torch.randn(2, 3, 23)
    rng = torch.get_rng_state().clone()
    original = control(**batch)
    after_forward_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    output = candidate(**batch)
    assert torch.equal(after_forward_rng, torch.get_rng_state())
    for k, v in original.items():
        torch.testing.assert_close(output[k], v, rtol=0, atol=0)

    candidate.eval()
    batch["x"].requires_grad_()
    output = candidate(**batch)
    output["event_will_logit"][:, :2].sum().backward()
    query_grad = candidate.history_attention.query.grad
    assert torch.isfinite(query_grad).all() and query_grad.abs().sum() > 0
    assert candidate.gru.weight_ih_l0.grad.abs().sum() > 0
    assert torch.count_nonzero(batch["x"].grad[:, :, 2]) == 0
    with torch.no_grad():
        candidate.history_attention.query.copy_(torch.linspace(-1, 1, 128))
    changed = candidate(**batch)
    assert not torch.equal(changed["node_hidden"], output["node_hidden"])
    masked_batch = {**batch, "x": batch["x"].detach().clone()}
    masked_batch["x"][:, :, 2] = 1000
    masked_batch["event_precursor"] = batch["event_precursor"].clone()
    masked_batch["event_precursor"][:, 2] = -1000
    assert torch.equal(changed["event_will_logit"], candidate(**masked_batch)["event_will_logit"])
    assert torch.count_nonzero(changed["node_hidden"][:, 2]) == 0
    buffer = io.BytesIO()
    torch.save(dict(model_kind=kind, model_config=candidate.config.to_dict(),
                    model_state_dict=candidate.state_dict()), buffer)
    buffer.seek(0)
    loaded, _ = load_checkpoint(buffer, torch.device("cpu"))
    loaded.eval()
    restored = loaded(**batch)
    for k, v in changed.items():
        torch.testing.assert_close(restored[k], v, rtol=0, atol=0)


@pytest.mark.parametrize("model", ["B4", "B5"])
def test_attention_variant_changes_only_temporal_pool_and_profile(model):
    from train_dense_baseline_control import dense_configuration
    parent_train, parent_model, parent_loss = dense_configuration(model, "near_precursor", 42, "cpu")
    train, architecture, loss = dense_configuration(model, "temporal_attention", 42, "cpu")
    assert architecture == {**parent_model, "temporal_readout": "last_attention"}
    assert asdict(loss) == asdict(parent_loss)
    assert asdict(train) == {**asdict(parent_train), "training_profile": "dense_temporal_attention_v2"}
