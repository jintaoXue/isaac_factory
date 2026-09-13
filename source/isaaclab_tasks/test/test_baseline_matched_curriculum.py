"""In-memory checks for archive ancestry, horizon transfer and training registration."""

import io
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))

from factory_baselines import protocol_20260913 as protocol
from factory_baselines.matched_warm_start import HORIZON_KEYS, validate_parent
from factory_baselines.torch_trainer import _model_spec, _event_sampling_weights
from train_baseline_matched_curriculum import configuration


def parent_fixture(model="B4", cap=5, selected=1):
    kind = "b4_gcn_gru" if model == "B4" else "b5_gat_gru"
    cls, cfg_cls, _, _ = _model_spec(kind)
    _, overrides, _ = configuration(model, cap, "cpu")
    cfg = cfg_cls(input_dim=27, global_dim=0, num_nodes=38, num_causes=10,
                  max_remain_windows=20, **overrides)
    old_cfg = cfg_cls(**{**cfg.to_dict(), "max_remain_windows": 15 if cap == 5 else 20})
    torch.manual_seed(42)
    initial = cls(cfg).state_dict()
    torch.manual_seed(711)
    old = cls(old_cfg).state_dict()
    training = {"seed": 42, "evaluate_test": False, "max_epochs": 10, "batch_size": 4}
    meta = {"seed": 42, "dataset_manifest_sha256": "source", "git_commit": "parent_source"}
    budget = {"stage_epochs_trained": 2, "stage_optimizer_steps": 4,
              "cumulative_epochs_trained": 2, "cumulative_optimizer_steps": 4,
              "cumulative_max_epochs": 10, "cumulative_elapsed_seconds": 5.}
    if cap != 5:
        prev = {10: 5, 15: 10}[cap]
        training.update(evaluation_protocol=protocol.VERSION, event_max_start_windows=prev)
        meta.update(evaluation_contract=protocol.evaluation_contract(prev),
                    warm_start_parent={"epochs_trained": 7, "optimizer_steps": 14, "max_epochs": 20})
        budget.update(cumulative_epochs_trained=9, cumulative_optimizer_steps=18, cumulative_max_epochs=30)
    summary = {"status": "validation_completed", "model_kind": kind, "seed": 42,
               "dataset_manifest_sha256": "source", "epochs_trained": 2, "best_epoch": selected,
               "checkpoint_epoch": selected, "training_budget": budget}
    report = {}
    for suffix in ("precision", "recall", "f1"):
        report[f"report_{suffix}"] = .5
        report[f"will15_{suffix}"] = .8
        summary[f"best_validation_report_{suffix}"] = .5
        summary[f"best_validation_primary_{suffix}"] = .8
    checkpoint = {"model_kind": kind, "model_config": old_cfg.to_dict(), "train_config": training,
                  "loss_config": {}, "metadata": meta, "model_state_dict": old, "epoch": selected}
    config = {"model": old_cfg.to_dict(), "training": training, "loss": {}, "metadata": meta}
    checkpoint_bytes = io.BytesIO(); torch.save(checkpoint, checkpoint_bytes)
    history = "epoch,train_total,validation_total_loss,learning_rate,validation_report_precision,validation_report_recall,validation_report_f1\n1,1,1,.001,.5,.5,.5\n2,1,1,.001,.5,.5,.5\n"
    contents = {"best.pt": checkpoint_bytes.getvalue(), "history.csv": history.encode(),
                "config.json": json.dumps(config).encode(), "run_summary.json": json.dumps(summary).encode(),
                "metrics_validation.json": json.dumps({"station_report": report}).encode(),
                "metrics_initial_validation.json": json.dumps({"station_report": report}).encode()}
    kwargs = dict(model_kind=kind, model_config=cfg.to_dict(), config_class=cfg_cls,
                  initial_state=initial, seed=42, dataset_manifest_sha256="source",
                  train_sample_count=8, max_start=cap)
    return contents, kwargs, old


@pytest.mark.parametrize("model", ["B4", "B5"])
def test_legacy_transfer_only_reinitializes_three_horizon_tensors(model):
    contents, kwargs, old = parent_fixture(model)
    state, info = validate_parent(contents, **kwargs)
    assert set(info["reinitialized_keys"]) == HORIZON_KEYS
    for name, value in state.items():
        assert torch.equal(value, kwargs["initial_state"][name] if name in HORIZON_KEYS else old[name])
    assert info["epochs_trained"] == 2 and info["optimizer_steps"] == 4


@pytest.mark.parametrize("cap,selected", [(10, 1), (15, 1), (10, 0)])
def test_curriculum_copies_all_tensors_and_counts_all_ancestral_epochs(cap, selected):
    contents, kwargs, old = parent_fixture(cap=cap, selected=selected)
    state, info = validate_parent(contents, **kwargs)
    assert not info["reinitialized_keys"]
    assert all(torch.equal(state[k], old[k]) for k in state)
    assert (info["epochs_trained"], info["optimizer_steps"], info["max_epochs"]) == (9, 18, 30)


@pytest.mark.parametrize("damage", ["seed", "dataset", "test", "backbone", "nan", "budget", "incomplete", "ancestry"])
def test_reject_invalid_parent_before_loading(damage):
    contents, kwargs, _ = parent_fixture(cap=10)
    checkpoint = torch.load(io.BytesIO(contents["best.pt"]), weights_only=False)
    if damage == "seed":
        kwargs["seed"] = 43
    elif damage == "dataset":
        kwargs["dataset_manifest_sha256"] = "different"
    elif damage == "test":
        checkpoint["train_config"]["evaluate_test"] = True
    elif damage == "backbone":
        key = next(k for k in checkpoint["model_state_dict"] if not k.startswith("heads."))
        checkpoint["model_state_dict"][key] = torch.zeros(1)
    elif damage == "nan":
        checkpoint["model_state_dict"]["heads.event_start_head.2.bias"][0] = float("nan")
    elif damage == "budget":
        summary = json.loads(contents["run_summary.json"])
        summary["training_budget"]["cumulative_optimizer_steps"] = 4
        contents["run_summary.json"] = json.dumps(summary).encode()
    elif damage == "incomplete":
        contents["history.csv"] = contents["history.csv"].splitlines()[0] + b"\n"
    elif damage == "ancestry":
        kwargs["max_start"] = 15
    buffer = io.BytesIO(); torch.save(checkpoint, buffer); contents["best.pt"] = buffer.getvalue()
    with pytest.raises(ValueError):
        validate_parent(contents, **kwargs)


def test_frozen_six_job_configuration_and_cold_start_zero_sampling():
    for model in ("B4", "B5"):
        for cap, weight in ((5, 9.), (10, 10.), (15, 11.)):
            training, overrides, loss = configuration(model, cap, "cpu")
            assert not training.evaluate_test and training.evaluate_train
            assert (training.max_epochs, training.patience, training.seed) == (100, 40, 42)
            assert training.report_threshold_sweep == protocol.THRESHOLDS
            assert loss.event_will_upcoming_pos_weight == weight
            assert overrides["gru_hidden"] == 128 and overrides["event_precursor"] == "near"
            assert not overrides.get("event_onset_joint", False)
    samples = [{"event_will": torch.tensor([will]), "occ_node_mask": torch.ones(1),
                "event_start": torch.tensor([start]), "hist_last_hot": torch.zeros(1)}
               for will, start in ((1., 0), (1., 12), (0., -1))]
    assert _event_sampling_weights(samples, training).tolist() == [4., 4., 1.]
