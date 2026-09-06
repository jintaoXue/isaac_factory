"""B4/B5 representation controls preserve masking and checkpoint semantics."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
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
