"""Joint onset reporting: reference gradients, real heads, trainer and old weights."""

import ast
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_heads import combine_onset_event_logits
from factory_baselines.onset_history import observed_history_hot, attach_onset_history
from factory_bn_shared.remain import ops_hot_mask
from factory_baselines.torch_losses import MultiTaskLossConfig, compute_multitask_loss
from factory_baselines.torch_trainer import (
    _evaluate_loader, _model_inputs, _run_train_epoch, load_checkpoint,
)
import test_b5_gat_gru as fixture_module
from train_dense_baseline_control import dense_configuration


def test_joint_scores_and_bce_gradients_match_pinned_main_method_including_ties():
    path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer/factory_bn/model.py"
    source = subprocess.check_output(
        ["git", "show", "20c40e230aedee6aef2429d352413fbcf0fa571a:" + path],
        cwd=TOOLS.parents[5], text=True,
    )
    methods = [n for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.FunctionDef) and n.name == "_combine_will_logit"]
    assert len(methods) == 1
    env = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), path, "exec"), env)
    reference = env["_combine_will_logit"]
    generator = torch.Generator().manual_seed(19)
    for _ in range(12):
        cont = torch.randn(3, 6, generator=generator, dtype=torch.float64).requires_grad_()
        onset = torch.randn(3, 6, generator=generator, dtype=torch.float64)
        onset[:, 0] = cont.detach()[:, 0]  # Equal maxima have a defined split gradient.
        onset.requires_grad_()
        last = torch.tensor([[0., .5, 1., 0., 1., 0.]]).expand_as(cont)
        actual = combine_onset_event_logits(cont, onset, last)
        expected = reference(SimpleNamespace(split_will_heads=True, event_cold_will_max=True),
                             cont, onset, {"hist_last_hot": last})
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        labels = torch.randint(0, 2, cont.shape, generator=generator).double()
        a = torch.autograd.grad(F.binary_cross_entropy_with_logits(actual, labels),
                                (cont, onset), retain_graph=True)
        b = torch.autograd.grad(F.binary_cross_entropy_with_logits(expected, labels), (cont, onset))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def _fixture(kind):
    f = fixture_module.TestB5GatGru(); f.setUp(); batch = f._batch()
    batch["event_precursor"] = torch.randn(4, 5, 23)
    batch["event_history_hot"] = batch["hist_last_hot"].clone()
    batch["event_history_hot"][:, 0] = 1
    batch["event_will"][:, 2] = 1
    batch["event_start"][:, 2] = 1
    batch["event_duration"][:, 2] = 8
    batch["y_hot"][:, 1:9, 2] = 1
    batch["sample_index"] = torch.arange(4)
    batch["target_remain_len"][:] = 15
    cls, cfg, spatial = ((B4GcnGru, B4ModelConfig, "gcn_hidden") if kind.startswith("b4")
                         else (B5GatGru, B5ModelConfig, "gat_hidden"))
    values = dict(input_dim=6, global_dim=2, num_nodes=5, gru_hidden=8, dropout=0.,
                  temporal_readout="last_mean", event_precursor="near", event_onset_aux=True,
                  **{spatial: 8})
    torch.manual_seed(30); parent = cls(cfg(**values))
    rng = torch.get_rng_state().clone()
    torch.manual_seed(30); candidate = cls(cfg(**values, event_onset_joint=True))
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in parent.parameters()) == sum(p.numel() for p in candidate.parameters())
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in parent.state_dict().items())
    return parent, candidate, batch


@pytest.mark.parametrize("kind", ["b4_gcn_gru", "b5_gat_gru"])
def test_real_heads_initial_equality_then_joint_event_loss_reaches_the_correct_branch(kind):
    parent, model, batch = _fixture(kind)
    loss_config = MultiTaskLossConfig(lambda_event_onset_aux=1.)
    for training in (True, False):
        parent.train(training); model.train(training)
        a, b = parent(**_model_inputs(batch, parent)), model(**_model_inputs(batch, model))
        for k, v in a.items():
            torch.testing.assert_close(b[k], v, rtol=0, atol=0)
        assert set(b) - set(a) == {"event_will_continue_logit"}
        torch.testing.assert_close(compute_multitask_loss(a, batch, loss_config)[0],
                                   compute_multitask_loss(b, batch, loss_config)[0], rtol=0, atol=0)
    # Same changed onset parameters in both models: only the reporting connection differs.
    with torch.no_grad():
        parent.heads.event_onset_head[-1].bias.add_(4)
        model.heads.event_onset_head[-1].bias.add_(4)
    a, b = parent(**_model_inputs(batch, parent)), model(**_model_inputs(batch, model))
    for k, v in a.items():
        if k != "event_will_logit":
            torch.testing.assert_close(b[k], v, rtol=0, atol=0)
    torch.testing.assert_close(b["event_will_logit"][:, 0], a["event_will_logit"][:, 0])
    torch.testing.assert_close(b["event_will_logit"][:, 1:], b["event_onset_logit"][:, 1:])
    _, old_parts = compute_multitask_loss(a, batch, loss_config)
    _, parts = compute_multitask_loss(b, batch, loss_config)
    for k, v in old_parts.items():
        if k != "event_will":
            torch.testing.assert_close(parts[k], v, rtol=0, atol=0)
    b["event_onset_logit"].retain_grad(); b["event_will_continue_logit"].retain_grad()
    parts["event_will"].backward()
    grad = b["event_onset_logit"].grad
    assert (grad[:, 1] > 0).all()  # Cold negative: the main loss penalizes onset false alarms.
    assert (grad[:, 2] < 0).all()  # Cold upcoming: the main loss rewards the onset branch.
    assert (grad[:, [0, 3, 4]] == 0).all()  # Hot gate and target masks remain effective.
    assert (b["event_will_continue_logit"].grad[:, 1:] == 0).all()
    assert model.gru.weight_ih_l0.grad.abs().sum() > 0
    assert model.heads.precursor_projection[-1].weight.grad.abs().sum() > 0
    assert model.heads.event_onset_head[-1].weight.grad.abs().sum() > 0
    assert model.heads.event_start_head[-1].weight.grad is None
    assert model.heads.remain_hot_head[-1].weight.grad is None
    # The true future may change loss/evaluation, but cannot change model inputs or predictions.
    changed = {**batch, "event_will": 1 - batch["event_will"], "y_hot": 1 - batch["y_hot"],
               "hist_last_hot": 1 - batch["hist_last_hot"]}
    for k, v in b.items():
        torch.testing.assert_close(model(**_model_inputs(changed, model))[k], v, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["b4_gcn_gru", "b5_gat_gru"])
def test_actual_training_and_evaluation_use_the_same_joint_score_and_checkpoint_roundtrip(kind):
    parent, model, batch = _fixture(kind)
    with torch.no_grad(): model.heads.event_onset_head[-1].bias.add_(4)

    class Samples(Dataset):
        payload = {"event_min_windows": 8, "window_size_s": 60.}
        def __len__(self): return len(batch["x"])
        def __getitem__(self, i): return {k: v[i] for k, v in batch.items()}

    loader = DataLoader(Samples(), batch_size=2, shuffle=False)
    observed = []
    def hook(m, args, kwargs, output):
        assert "event_history_hot" in kwargs and "event_will" not in kwargs and "y_hot" not in kwargs
        expected = combine_onset_event_logits(output["event_will_continue_logit"],
                                               output["event_onset_logit"], kwargs["event_history_hot"])
        torch.testing.assert_close(output["event_will_logit"], expected, rtol=0, atol=0)
        observed.append((m.training, torch.is_grad_enabled(), expected.detach().sigmoid().numpy()))
    handle = model.register_forward_hook(hook, with_kwargs=True)
    config = MultiTaskLossConfig(lambda_event_onset_aux=1.)
    before = model.heads.event_onset_head[-1].weight.detach().clone()
    result = _run_train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=.001),
                              config, torch.ones(5), torch.device("cpu"), 1., {})
    assert np.isfinite(result["total"])
    assert not torch.equal(before, model.heads.event_onset_head[-1].weight)
    assert all(training and grad for training, grad, _ in observed)
    observed.clear()
    _, arrays, _ = _evaluate_loader(model, loader, config, torch.ones(5), torch.device("cpu"), 10)
    assert all(not training and not grad for training, grad, _ in observed)
    np.testing.assert_array_equal(arrays["event_will_probability"], np.concatenate([a[2] for a in observed]))
    handle.remove()
    buffer = io.BytesIO()
    torch.save(dict(model_kind=kind, model_config=model.config.to_dict(),
                    model_state_dict=model.state_dict()), buffer); buffer.seek(0)
    loaded, _ = load_checkpoint(buffer, torch.device("cpu")); loaded.eval()
    for k, v in model(**_model_inputs(batch, model)).items():
        torch.testing.assert_close(loaded(**_model_inputs(batch, loaded))[k], v, rtol=0, atol=0)
    # Old checkpoints have no joint key and keep the auxiliary-only behavior.
    old_config = parent.config.to_dict(); old_config.pop("event_onset_joint")
    buffer = io.BytesIO()
    torch.save(dict(model_kind=kind, model_config=old_config, model_state_dict=parent.state_dict()), buffer)
    buffer.seek(0); old, _ = load_checkpoint(buffer, torch.device("cpu")); old.eval(); parent.eval()
    assert "event_history_hot" not in _model_inputs(batch, old)
    for k, v in parent(**_model_inputs(batch, parent)).items():
        torch.testing.assert_close(old(**_model_inputs(batch, old))[k], v, rtol=0, atol=0)


def test_missing_history_invalid_gates_and_incompatible_configurations_are_rejected():
    x = torch.zeros(2, 3)
    for last in (None, torch.zeros(3), torch.full_like(x, float("nan")),
                 torch.full_like(x, -1), torch.full_like(x, 2)):
        with pytest.raises(ValueError): combine_onset_event_logits(x, x, last)
    with pytest.raises(ValueError): combine_onset_event_logits(x, torch.zeros(2, 4), x)
    for cls in (B4ModelConfig, B5ModelConfig):
        with pytest.raises(ValueError): cls(6, 2, 5, event_onset_joint=True)
        with pytest.raises(ValueError): cls(6, 2, 5, event_onset_aux=True, event_onset_joint=True,
                                           event_head="three_class")
    parent, model, batch = _fixture("b4_gcn_gru")
    assert set(_model_inputs(batch, model)) - set(_model_inputs(batch, parent)) == {"event_history_hot"}
    del batch["event_history_hot"]
    with pytest.raises(KeyError): _model_inputs(batch, model)
    with pytest.raises(ValueError): model(**_model_inputs(batch))


def test_observed_gate_rejects_full_episode_lookahead_and_uses_frozen_normalization():
    raw = np.zeros((45, 2, 27), dtype=np.float32)
    raw[:, :, 21] = 1
    raw[27:40, 0, 0] = 4; raw[27:40, 0, 6] = 45
    raw[20:40, 1, 0] = 4; raw[20:40, 1, 6] = 45
    mean = np.linspace(-.5, .5, 21, dtype=np.float32)
    std = np.full(21, 2., dtype=np.float32)
    x = raw[:30].copy(); x[..., :21] = (x[..., :21] - mean) / std
    original = x.copy()
    actual = observed_history_hot(x, mean, std)
    # The same current operational state can be a short observed run or a long one.
    np.testing.assert_array_equal(actual, [0., 1.])
    np.testing.assert_array_equal(ops_hot_mask(raw, min_hot_windows=8, gap_windows=1)[29], [1., 1.])
    np.testing.assert_array_equal(actual, ops_hot_mask(raw[:30], min_hot_windows=8, gap_windows=1)[-1])
    raw[30:] = 0  # Changing the future never enters this builder's input.
    np.testing.assert_array_equal(observed_history_hot(x, mean, std), actual)
    np.testing.assert_array_equal(x, original)
    for bad in (x[:29], x[:, :, :26], np.full_like(x, np.nan)):
        with pytest.raises(ValueError): observed_history_hot(bad, mean, std)


def test_attachment_uses_only_selected_existing_x_and_preserves_legacy_labels_and_masks():
    x = torch.zeros(3, 30, 2, 27); x[:, :, :, 21] = 1
    x[:, :, :, 0] = 4; x[:, :, :, 6] = 45
    x[2] = float("nan")  # Test is not inspected when only train/validation are requested.
    payload = dict(x=x, node_mask=torch.tensor([[1, 0], [1, 1], [1, 1]]),
                   split_indices={"train": [0], "validation": [1], "test": [2]},
                   hist_last_hot=torch.zeros(3, 2), event_will=torch.zeros(3, 2))
    manifest = dict(input_windows=30, window_size_s=60)
    norm = json.dumps(dict(feature_mean=[0.] * 21, feature_std=[1.] * 21))
    with patch("factory_baselines.onset_history.file_hash", return_value="verified-test-hash"), \
         patch.object(Path, "read_text", return_value=norm):
        output, contract = attach_onset_history(payload, manifest, Path("."), True, ("train", "validation"))
        assert all(output[k] is v for k, v in payload.items())
        torch.testing.assert_close(output["event_history_hot"], torch.tensor([[1., 0.], [1., 1.], [0., 0.]]))
        assert output["event_history_hot_valid"].tolist() == [True, True, False]
        assert contract["extra_history_windows"] == 0
        assert not contract["legacy_hist_last_hot_used_as_model_input"]
        changed = {**payload, "hist_last_hot": torch.ones(3, 2), "event_will": torch.ones(3, 2)}
        repeated, repeated_contract = attach_onset_history(changed, manifest, Path("."), True,
                                                            ("train", "validation"), contract)
        torch.testing.assert_close(output["event_history_hot"], repeated["event_history_hot"])
        assert repeated_contract == contract
        with pytest.raises(ValueError, match="contract differs"):
            attach_onset_history(payload, manifest, Path("."), True, ("train",), {})
        for splits in (("test",), ("train", "train")):
            with pytest.raises(ValueError): attach_onset_history(payload, manifest, Path("."), True, splits)
    # Legacy paths do not inspect files, input values, or add any field.
    old, contract = attach_onset_history(payload, {}, Path("unused"), False, ())
    assert old is payload and contract is None


def test_registered_four_configurations_change_only_joint_reporting_from_onset_parent():
    for model in ("B4", "B5"):
        for seed in (42, 43):
            train, arch, loss = dense_configuration(model, "onset_aux", seed, "cpu")
            new_train, new_arch, new_loss = dense_configuration(model, "joint_onset", seed, "cpu")
            assert replace(new_train, training_profile=train.training_profile) == train
            assert new_arch == {**arch, "event_onset_joint": True}
            assert new_loss == loss and new_loss.event_fbeta_weight == 0
            assert new_loss.lambda_event_onset_aux == 1
            assert new_arch["event_precursor"] == "near"
            assert new_arch["temporal_readout"] == "last_mean"
            assert not new_train.evaluate_test and new_train.event_oversample_factor == 1
            assert new_train.max_epochs == 60


def test_control_builds_causal_history_before_archiving_and_records_actual_contract():
    import train_dense_baseline_control as control
    repo = Path("/home/sci/work/BSTAN_isaac_factory")
    dataset, output = repo / "existing_dataset", repo / "existing_model"
    norm_contract = {"mode": "near"}
    history_contract = {"field": "event_history_hot", "extra_history_windows": 0}
    order, records = [], []
    def git(args, **kwargs):
        if "--show-toplevel" in args: return str(repo)
        if "--show-current" in args: return "dev_xwt"
        if "HEAD" in args: return "pinned-test-commit"
        raise AssertionError(args)
    def history(payload, manifest, directory, enabled, splits):
        assert enabled and splits == ("train", "validation")
        order.append("history"); return payload, history_contract
    def archive(*args): order.append("archive"); return output / "verified.zip"
    with patch.object(control.subprocess, "check_output", side_effect=git), \
         patch.object(Path, "resolve", lambda self: self), \
         patch.object(Path, "is_dir", return_value=True), \
         patch.object(Path, "exists", lambda self: self.name == "best.pt"), \
         patch.object(Path, "write_text", lambda self, text: records.append(json.loads(text))), \
         patch.object(control, "load_shared_dataset", return_value=({}, {"shared_bundle_alignment": {"status": "passed"}})), \
         patch("factory_baselines.precursor.attach_precursor", return_value=({}, norm_contract)), \
         patch("factory_baselines.onset_history.attach_onset_history", side_effect=history) as builder, \
         patch.object(control, "archive_files", side_effect=archive) as archiver, \
         patch.object(control, "file_hash", return_value="manifest-test-hash"), \
         patch.object(control, "train_torch_baseline", return_value={"status": "validation_completed"}):
        control.run_control("B4", dataset, output, "joint_test", 42, "cpu", "joint_onset")
        assert order == ["history", "archive"]
        record = records[-1]
        assert record["parent_control"] == "onset_aux"
        assert record["onset_history_contract"] == history_contract
        assert record["input_feature_contract"] == norm_contract
        assert record["onset_auxiliary"]["used_for_report_decision"]
        assert not record["joint_onset_reporting"]["legacy_hist_last_hot_added_to_model_input"]
        assert not record["test_evaluated"]
        # A failed causal-input build cannot archive or replace old weights.
        builder.side_effect = ValueError("invalid observed history")
        archiver.reset_mock(); records.clear()
        with pytest.raises(ValueError, match="invalid observed history"):
            control.run_control("B5", dataset, output, "joint_test", 43, "cpu", "joint_onset")
        archiver.assert_not_called(); assert not records
