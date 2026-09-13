"""Parity with the pinned main implementation and immutable source views."""

import ast
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))

from factory_baselines import protocol_20260913 as protocol
from factory_baselines.evaluation import EVALUATION_CONTRACT
from factory_baselines.metrics import choose_report_metrics, compute_metrics, training_cause_majority
from factory_baselines.torch_trainer import _selection_metrics, _validation_checkpoint_rank
from factory_baselines.torch_trainer import _evaluate_loader, _model_inputs, _model_spec
from factory_baselines.torch_losses import MultiTaskLossConfig, compute_multitask_loss
from factory_bn_shared.causes import ROOT_CAUSE_CLASSES


@lru_cache(maxsize=1)
def reference():
    path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer/factory_bn/remain.py"
    source = subprocess.check_output(
        ["git", "show", protocol.REFERENCE_COMMIT + ":" + path], cwd=ROOT, text=True,
    )
    names = {"node_event_targets", "_prf", "_align_hist_last", "apply_ongoing_will_force",
             "_empty_report_metrics", "station_report_metrics"}
    tree = ast.parse(source)
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == names
    namespace = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("max_start", [5, 10, 15])
def test_station_fields_match_pinned_main(max_start):
    rng = np.random.default_rng(711)
    hot = np.zeros((8, 20, 9), dtype=np.float32)
    for batch in range(8):
        for node in range(9):
            start = int(rng.integers(0, 17))
            hot[batch, start:start + int(rng.integers(1, 21)), node] = 1
    # Cold at cutoff, hot in the first future grid must be upcoming.
    hot[0, :, 0] = 1
    last = rng.integers(0, 2, (8, 9)).astype(np.float32)
    last[0, 0] = 0
    remain = np.ones((8, 20), dtype=np.float32)
    remain[-1, 4:] = 0
    occ = rng.integers(0, 2, (8, 9)).astype(np.float32)
    occ[0, 0] = 1
    probability = rng.random((8, 9), dtype=np.float32)
    probability[0, 0] = 1
    start = rng.integers(0, 20, (8, 9))
    duration = rng.uniform(0, 20, (8, 9)).astype(np.float32)
    args = (hot, probability, start, duration, remain, occ)
    actual = protocol.station_metrics(*args, hist_last_hot=last, threshold=.6,
                                      contract=protocol.evaluation_contract(max_start))
    expected = reference()["station_report_metrics"](
        *args, hist_last_hot=last, threshold=.6, min_windows=8,
        ongoing_min_windows=1, max_start_windows=max_start, start_tol_windows=3,
        force_ongoing_will=False,
    )
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, abs=1e-7), key
    for tolerance in (1, 2, 3):
        strict = reference()["station_report_metrics"](
            *args, hist_last_hot=last, threshold=.6, min_windows=8,
            ongoing_min_windows=1, max_start_windows=max_start,
            start_tol_windows=tolerance, force_ongoing_will=False,
        )
        for key in ("report_precision", "report_recall", "report_f1"):
            assert actual[f"{key}_at_{tolerance}"] == pytest.approx(strict[key], abs=1e-7)


def test_target_view_keeps_original_inputs_splits_and_legacy_contract():
    source = {
        "x": torch.zeros(2, 30, 3, 27), "event_min_windows": 8,
        "window_size_s": 60, "max_remain_windows": 15,
        "evaluation_contract": dict(EVALUATION_CONTRACT),
        "remain_series": {"0": {"hot": torch.ones(80, 3)}},
        "split_indices": {"train": torch.tensor([0]), "validation": torch.tensor([1])},
    }
    manifest = {"max_remain_windows": 15, "evaluation_contract": dict(EVALUATION_CONTRACT)}
    views = [protocol.protocol_view(source, manifest, n) for n in (5, 10, 15)]
    assert source["max_remain_windows"] == 15
    assert source["evaluation_contract"] == EVALUATION_CONTRACT
    assert manifest["max_remain_windows"] == 15
    for view, view_manifest in views:
        for name in ("x", "remain_series", "split_indices"):
            assert view[name] is source[name]
        assert view["max_remain_windows"] == 20
        assert view_manifest["task_view_only"]
    assert [v[0]["evaluation_contract"]["max_start_windows"] for v in views] == [5, 10, 15]


def test_start15_keeps_reference_censoring_and_exposes_it():
    hot = np.zeros((20, 4), dtype=np.float32)
    for node, start in enumerate((12, 13, 14, 15)):
        hot[start:start+8, node] = 1
    will, _, _ = reference()["node_event_targets"](
        hot, min_windows=8, max_start_windows=15, hist_last_hot=np.zeros(4),
        ongoing_min_windows=1,
    )
    np.testing.assert_array_equal(will, [1, 0, 0, 0])
    assert protocol.evaluation_contract(15)["max_observable_upcoming_start_index"] == 12


def test_will_selection_does_not_select_on_strict_start_score_or_hot_tiebreak():
    a = dict(report_precision=.9, report_f1=.8, will15_precision=.8, will15_f1=.7)
    b = dict(report_precision=.6, report_f1=.6, will15_precision=.9, will15_f1=.85)
    assert choose_report_metrics([a, b]) is a
    assert choose_report_metrics([a, b], primary="will15") is b
    tied = {**b, "report_f1": .9}
    assert choose_report_metrics([b, tied], primary="will15") is b
    metrics = {"station_report": {**b, "will15_recall": .81},
               "evaluation_contract": protocol.evaluation_contract(5)}
    assert _selection_metrics(metrics) == (.85, .9, .81)
    assert _validation_checkpoint_rank(metrics) == (.85,)


def test_four_class_support_excludes_sparse_classes_without_renumbering():
    classes = ["queue_buildup", "blocked_downstream", "transport_delay", "high_utilization",
               "material_shortage", "starved_upstream"]
    labels = np.array([1, 1, 1, 3, 0, 2, 4, 5])
    predicted = np.array([0, 0, 0, 0, 0, 2, 4, 5])
    metrics, _ = compute_metrics({"y_cause": labels, "cause_predictions": predicted},
                                 len(classes), classes, report_classes=protocol.CAUSE_CLASSES)
    assert metrics["cause_n"] == 4
    assert metrics["cause_macro_recall"] == 1
    assert training_cause_majority(labels, classes, report_classes=protocol.CAUSE_CLASSES) == 0


def test_remain_global_and_middle_weighted_have_distinct_support():
    contract = protocol.evaluation_contract(5)
    actual = protocol.remain_metrics(np.array([10., 4., 8., 2.]), np.zeros(4),
                                     np.array([10., 6., 4., 1.]), np.full(4, 10.), contract)
    progress = torch.tensor([0., .4, .6, .9], dtype=torch.float64)
    weight = .25 + .75 * progress.pow(1.5)
    expected = float((torch.tensor([4., 8.]) * weight[1:3]).sum() / weight[1:3].sum())
    assert actual["remain_len_mae"] == 6
    assert actual["remain_len_n_middle"] == 2
    assert actual["remain_len_mae_primary"] == pytest.approx(expected)
    assert actual["remain_len_mae_primary"] != actual["remain_len_mae"]


@pytest.mark.parametrize("model_kind", ["b4_gcn_gru", "b5_gat_gru"])
@pytest.mark.parametrize("max_start", [5, 10, 15])
def test_real_baseline_loss_and_evaluation_path(model_kind, max_start):
    contract = protocol.evaluation_contract(max_start)
    hot = np.zeros((4, 20, 3), dtype=np.float32)
    hot[:, :1, 0] = 1; hot[:, :8, 1] = 1; hot[:, 12:, 2] = 1
    last = np.tile([1., 0., 0.], (4, 1)).astype(np.float32)
    will, start, dur = reference()["node_event_targets"](
        hot, min_windows=8, max_start_windows=max_start, hist_last_hot=last,
        ongoing_min_windows=1,
    )
    active = [i for i, name in enumerate(ROOT_CAUSE_CLASSES) if name in protocol.CAUSE_CLASSES]
    ignored = tuple(i for i in range(len(ROOT_CAUSE_CLASSES)) if i not in active)
    batch = {
        "x": torch.zeros(4, 30, 3, 27), "adjacency": torch.ones(4, 3, 3),
        "node_mask": torch.ones(4, 3, dtype=torch.bool),
        "target_node_mask": torch.ones(4, 3, dtype=torch.bool),
        "occ_node_mask": torch.ones(4, 3), "hist_last_hot": torch.from_numpy(last),
        "global_features": torch.empty(4, 30, 0),
        "jobs_remaining": torch.tensor([9., 6., 4., 1.]), "jobs_total": torch.full((4,), 10.),
        "target_remain_len": torch.full((4,), 20), "y_hot": torch.from_numpy(hot),
        "y_score": torch.zeros(4, 20, 3, 1), "remain_mask": torch.ones(4, 20),
        "event_will": torch.from_numpy(will), "event_start": torch.from_numpy(start),
        "event_duration": torch.from_numpy(dur), "sample_index": torch.arange(4),
        "y_cause": torch.tensor([ignored[0], *active[:3]]),
    }
    cls, config_cls, _, _ = _model_spec(model_kind)
    model = cls(config_cls(input_dim=27, global_dim=0, num_nodes=3,
                           max_remain_windows=20, num_causes=len(ROOT_CAUSE_CLASSES)))
    loss = MultiTaskLossConfig(near_remain_windows=20, event_partition="history",
                              cause_ignored_ids=ignored, remain_progress_weight_floor=.25,
                              remain_progress_weight_power=1.5)
    outputs = model(**_model_inputs(batch, model))
    total, components = compute_multitask_loss(outputs, batch, loss)
    cause_grad = torch.autograd.grad(components["cause"], outputs["cause_logits"], retain_graph=True)[0]
    assert torch.count_nonzero(cause_grad[0]) == 0
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    class OneBatch:
        dataset = SimpleNamespace(payload={"evaluation_contract": contract, "event_min_windows": 8,
                                           "window_size_s": 60})
        def __iter__(self):
            return iter([batch])
    metrics, arrays, _ = _evaluate_loader(
        model, OneBatch(), loss, torch.tensor(1.), torch.device("cpu"), len(ROOT_CAUSE_CLASSES),
        cause_classes=ROOT_CAUSE_CLASSES, report_threshold_sweep=protocol.THRESHOLDS,
    )
    assert metrics["cause_n"] == 3
    assert set(arrays["cause_predictions"].tolist()) <= set(active)
    assert metrics["will15_f1"] == metrics["who_f1"]
    assert metrics["n_true_upcoming"] == (8 if max_start == 15 else 4)
    assert metrics["remain_len_n_middle"] == 2
    assert metrics["evaluation_contract"]["max_start_windows"] == max_start
