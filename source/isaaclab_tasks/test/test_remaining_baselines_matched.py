"""No-directory, in-memory checks for B2/B3 matched training and safe launching."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from factory_baselines import protocol_20260913 as protocol
from factory_baselines import b2_xgboost as b2
from factory_baselines.b3_lstm import B3Lstm, B3ModelConfig
from factory_baselines.evaluation import EVALUATION_CONTRACT
from factory_baselines.dataset import FactoryBaselineTensorDataset
from factory_baselines.torch_losses import compute_multitask_loss
from factory_baselines.torch_trainer import _model_inputs
from torch.utils.data import default_collate
from factory_baselines.matched_warm_start import validate_parent
from factory_bn_shared.causes import ROOT_CAUSE_CLASSES
from run_remaining_baselines_matched import b2_config, b3_config, TASKS
from launch_remaining_baselines_matched import check_terminal, validate_completion

def payload():
    n, nodes = 3, 2
    hot = torch.zeros(20, nodes)
    hot[:10, 0] = 1
    hot[7:17, 1] = 1
    return {
        "x": torch.randn(n, 30, nodes, 27), "adjacency": torch.ones(n, nodes, nodes, dtype=torch.bool),
        "node_mask": torch.ones(n, nodes, dtype=torch.bool),
        "target_node_mask": torch.ones(n, nodes, dtype=torch.bool),
        "occ_node_mask": torch.ones(n, nodes),
        "observation_mask": torch.ones(n, 30, nodes, dtype=torch.bool),
        "hist_last_hot": torch.tensor([[0., 0.], [1., 0.], [0., 0.]]),
        "global_features": torch.empty(n, 30, 0),
        "jobs_remaining": torch.full((n,), 2.), "jobs_total": torch.full((n,), 4.),
        "target_start_position": torch.zeros(n, dtype=torch.int64),
        "target_remain_len": torch.full((n,), 20, dtype=torch.int64),
        "sample_group_id": torch.arange(n), "y_cause": torch.zeros(n, dtype=torch.int64),
        "max_remain_windows": 15, "event_min_windows": 8, "window_size_s": 60.,
        "evaluation_contract": dict(EVALUATION_CONTRACT),
        "split_indices": {"train": torch.tensor([0, 1]), "validation": torch.tensor([2])},
        "remain_series": {str(i): {"score": torch.zeros(20, nodes, 1), "hot": hot.clone()} for i in range(n)},
    }

def manifest():
    return {"cause_classes": list(ROOT_CAUSE_CLASSES), "node_ids": ["machine_a", "machine_b"],
            "window_size_s": 60., "input_windows": 30, "hot_smoothing_order": "legacy",
            "dataset_contract": "synthetic", "dataset_version": "synthetic", "label_version": "synthetic",
            "prediction_target_version": "synthetic", "evaluation_contract": dict(EVALUATION_CONTRACT)}

@pytest.mark.parametrize("cap,upcoming", [(5, 1), (10, 3), (15, 3)])
def test_b2_cold_start_zero_uses_upcoming_and_cap_targets(cap, upcoming):
    view, _ = protocol.protocol_view(payload(), manifest(), cap)
    rows = b2._event_training_data(view, [0, 1])
    mask = (rows["will"] > 0) & ~rows["ongoing"]
    assert int(mask.sum()) == upcoming
    assert 0 in rows["start"][mask]
    assert int(rows["ongoing"].sum()) == 1

@pytest.mark.parametrize("cap", [5, 10, 15])
def test_registered_recipes_preserve_models_and_freeze_development_protocol(cap):
    tree = b2_config(cap)
    train, overrides, loss = b3_config(cap, "cpu", ROOT_CAUSE_CLASSES)
    assert not tree.evaluate_test and tree.evaluate_train and tree.n_estimators == 500
    assert tree.event_will_scale_pos_weight == 12
    assert tree.report_threshold_sweep == train.report_threshold_sweep == protocol.THRESHOLDS
    assert train.event_max_start_windows == cap and train.max_epochs == 100 and train.patience == 40
    assert not train.evaluate_test and train.evaluate_train
    assert overrides["lstm_hidden"] == 128 and "event_precursor" not in overrides
    assert loss.event_will_upcoming_pos_weight == {5: 9., 10: 10., 15: 11.}[cap]
    assert set(TASKS) == {(m, s) for m in ("B2", "B3") for s in (5, 10, 15)}

def test_matched_b2_refuses_test_scope():
    with pytest.raises(ValueError, match="must not evaluate test"):
        b2.B2XGBoostConfig(evaluation_protocol=protocol.VERSION, event_max_start_windows=5)

@pytest.mark.parametrize("cap", [5, 10, 15])
def test_b3_matched_horizon_and_loss_backward(cap):
    view, meta = protocol.protocol_view(payload(), manifest(), cap)
    dataset = FactoryBaselineTensorDataset(view, [0, 1])
    batch = default_collate([dataset[0], dataset[1]])
    _, overrides, loss = b3_config(cap, "cpu", meta["cause_classes"])
    model = B3Lstm(B3ModelConfig(input_dim=27, global_dim=0, num_nodes=2,
        num_causes=len(meta["cause_classes"]), max_remain_windows=20, **overrides))
    outputs = model(**_model_inputs(batch, model))
    value, _ = compute_multitask_loss(outputs, batch, loss)
    value.backward()
    assert outputs["remain_hot_logit"].shape == (2, 20, 2)
    assert torch.isfinite(value)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

@pytest.mark.parametrize("raw", ["0", "0 0", "1 1", "1", "2 0"])
def test_launcher_never_replaces_live_or_failed_pane(raw):
    with pytest.raises(ValueError):
        check_terminal(raw)
    check_terminal("1 0")

@pytest.mark.parametrize("damage", ["status", "order", "test", "hash", "state"])
def test_launcher_requires_exact_completed_b45_evidence(damage):
    final = {"status": "six_matched_tasks_independently_verified", "test_evaluated": False,
             "results": [{"model": m, "max_start": s} for s in (5,10,15) for m in ("B4","B5")]}
    raw = json.dumps(final).encode()
    state = {"status": "completed", "final_sha256": hashlib.sha256(raw).hexdigest()}
    validate_completion(final, state, raw)
    if damage == "status": final["status"] = "running"
    if damage == "order": final["results"].reverse()
    if damage == "test": final["test_evaluated"] = True
    if damage == "hash": state["final_sha256"] = "changed"
    if damage == "state": state["status"] = "waiting"
    with pytest.raises(ValueError):
        validate_completion(final, state, raw)

@pytest.mark.parametrize("cap", [10, 15])
def test_b3_copies_own_preceding_matched_parent_without_near_projection(cap):
    from test_baseline_matched_curriculum import parent_fixture
    contents, kwargs, _ = parent_fixture(cap=cap)
    cfg = B3ModelConfig(input_dim=2, global_dim=0, num_nodes=2, max_remain_windows=20)
    old = B3Lstm(cfg).state_dict()
    ckpt = torch.load(io.BytesIO(contents["best.pt"]), weights_only=False)
    config = json.loads(contents["config.json"])
    summary = json.loads(contents["run_summary.json"])
    ckpt.update(model_kind="b3_lstm", model_config=cfg.to_dict(), model_state_dict=old)
    config["model"] = cfg.to_dict()
    summary["model_kind"] = "b3_lstm"
    buffer = io.BytesIO(); torch.save(ckpt, buffer)
    contents.update({"best.pt": buffer.getvalue(), "config.json": json.dumps(config).encode(),
                     "run_summary.json": json.dumps(summary).encode()})
    kwargs.update(model_kind="b3_lstm", model_config=cfg.to_dict(), config_class=B3ModelConfig,
                  initial_state=B3Lstm(cfg).state_dict())
    state, metadata = validate_parent(contents, **kwargs)
    assert not metadata["reinitialized_keys"]
    assert all(torch.equal(old[k], state[k]) for k in old)
    kwargs["max_start"] = 5
    with pytest.raises(ValueError, match="legacy near parent"):
        validate_parent(contents, **kwargs)

def test_b2_full_matched_exports_use_new_targets_four_causes_and_frozen_train_threshold(monkeypatch):
    data, meta = payload(), manifest()
    supported = [meta["cause_classes"].index(n) for n in protocol.CAUSE_CLASSES]
    data["y_cause"] = torch.tensor([supported[0], supported[1], supported[0]])
    outputs = {}
    rows = "sample_index,split,target_cause,target_remain_len_windows,first_future_start_s,group_id,run_id,env_id,episode_id,anchor_time_s\n"
    rows += "\n".join(f"{i},{'train' if i < 2 else 'validation'},{meta['cause_classes'][int(data['y_cause'][i])]},20,1800,g{i},r0,0,{i},1740" for i in range(3))
    class MemoryPath:
        def __init__(self, name): self.name = str(name)
        def resolve(self): return self
        def is_dir(self): return True
        def mkdir(self, **kwargs): pass
        def __truediv__(self, child): return MemoryPath(self.name + "/" + child)
        def __str__(self): return self.name
        def open(self, *args, **kwargs):
            assert self.name.endswith("model_sample_index.csv")
            return io.StringIO(rows)
    monkeypatch.setattr(b2, "Path", MemoryPath)
    monkeypatch.setattr(b2, "load_shared_dataset", lambda _: (data, meta))
    monkeypatch.setattr(b2, "_manifest_hash", lambda _: "synthetic")
    monkeypatch.setattr(b2, "_occupancy_type_masks", lambda *_: {})
    def fit(X, y, cfg, **kwargs):
        assert len(X) == len(y) and len(y)
        return b2._Head(kind="constant", constant=int(y[0]), classes=[int(y[0])])
    monkeypatch.setattr(b2, "_fit_classifier", fit)
    monkeypatch.setattr(b2, "_fit_regressor", lambda X,y,cfg: b2._Head(kind="constant", constant=float(np.mean(y))))
    monkeypatch.setattr(b2, "_save_head", lambda d,n,h: {"kind":h.kind,"path":None,"constant":h.constant,"classes":h.classes})
    monkeypatch.setattr(b2, "_write_json", lambda p,x: outputs.update({str(p).split("/")[-1]:x}))
    monkeypatch.setattr(b2, "_write_csv", lambda p,x,f: outputs.update({str(p).split("/")[-1]:x}))
    real_choose = b2.choose_report_metrics
    calls = []
    def choose(candidates, **kwargs):
        calls.append((len(candidates),kwargs["primary"]))
        return real_choose(candidates, **kwargs)
    monkeypatch.setattr(b2, "choose_report_metrics", choose)
    summary = b2.train_b2_xgboost("dataset", "output", b2_config(10))
    assert calls == [(len(protocol.THRESHOLDS), "will15"), (1, "will15")]
    assert summary["primary_metric"] == "will15_f1"
    assert set(outputs["metrics.json"]) == {"train", "validation"}
    assert {int(r["sample_index"]) for r in outputs["predictions_train.csv"]} == {0,1}
    assert {int(r["sample_index"]) for r in outputs["predictions_validation.csv"]} == {2}
    train, val = (outputs["metrics.json"][s] for s in ("train","validation"))
    assert train["station_report"]["n_true_upcoming"] == 3
    assert val["station_report"]["n_true_upcoming"] == 2
    assert train["station_report"]["report_threshold_used"] == val["station_report"]["report_threshold_used"]
    assert all(v["evaluation_contract"]["max_start_windows"] == 10 for v in (train,val))
    assert all("remain_len_mae_middle_weighted" in v["remain"] for v in (train,val))
    assert not any("test" in key for key in outputs)
