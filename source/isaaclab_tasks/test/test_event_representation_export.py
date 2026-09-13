"""Check that observation preserves real model outputs and row identities."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("representation_export", TOOLS / "export_frozen_event_representations.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig


def inputs():
    torch.manual_seed(174)
    return dict(x=torch.randn(2, 30, 4, 27), adjacency=torch.ones(2, 4, 4),
        node_mask=torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]]), target_node_mask=torch.ones(2, 4),
        global_features=torch.zeros(2, 30, 0), jobs_remaining=torch.ones(2), jobs_total=torch.ones(2) * 3,
        event_precursor=torch.randn(2, 4, 23), event_history_hot=torch.zeros(2, 4))


def check_model(model):
    model.eval(); batch = inputs()
    with torch.no_grad():
        before = torch.random.get_rng_state().clone(); expected = model(**batch)
        observed, hidden = module.observe_event_input(model, batch)
    assert torch.equal(before, torch.random.get_rng_state())
    assert expected.keys() == observed.keys()
    assert all(torch.equal(expected[k], observed[k]) for k in expected)
    assert not model.heads.event_will_head._forward_pre_hooks
    assert hidden.shape == observed["node_hidden"].shape


def test_b4_joint_observer_leaves_every_prediction_and_rng_unchanged():
    check_model(B4GcnGru(B4ModelConfig(27, 0, 4, gcn_hidden=8, gru_hidden=8, event_context=True,
        event_precursor="near", event_onset_aux=True, event_onset_joint=True)))


def test_b5_vector_observer_leaves_every_prediction_and_rng_unchanged():
    check_model(B5GatGru(B5ModelConfig(27, 0, 4, gat_hidden=8, gat_heads=2, gru_hidden=8,
        event_context=True, event_precursor="near", gat_score_mode="vector_additive")))


def test_rows_preserve_sample_node_identity_and_labels_do_not_enter_features():
    batch = inputs(); batch.update(sample_index=torch.tensor([19, 4]),
        occ_node_mask=batch["node_mask"], event_will=torch.tensor([[0., 1., 1., 1.], [1., 0., 1., 0.]]),
        event_start=torch.tensor([[0, 0, 2, 1], [1, 0, 1, 0]]), hist_last_hot=torch.zeros(2, 4))
    result = dict(node_hidden=torch.arange(24).view(2, 4, 3).float(), event_will_logit=torch.zeros(2, 4), event_start_logit=torch.ones(2, 4, 15))
    event = result["node_hidden"] + 1
    first = module.pack_representation_rows(batch, result, event)
    assert list(zip(first["sample_index"], first["node_index"])) == [(19, 0), (19, 1), (19, 2), (4, 0), (4, 1)]
    assert first["label_kind"].tolist() == [0, 1, 2, 2, 0]
    assert first["raw_summary"].shape == (5, 84)
    changed = {**batch, "event_will": 1 - batch["event_will"], "event_start": batch["event_start"] + 3}
    second = module.pack_representation_rows(changed, result, event)
    for key in ("raw_summary", "backbone", "event_input", "probability"):
        assert np.array_equal(first[key], second[key])


def test_canonical_verification_includes_structured_counts_and_AP():
    report = dict(thresholds=[dict(threshold=.6, n_true_upcoming=2, report_false_alarm_breakdown={"short": 3})],
        groups={k: dict(count=v) for k, v in zip(("negative", "ongoing", "upcoming"), (10, 3, 2))},
        ranking=dict(upcoming_vs_negative=dict(tie_aware_average_precision=.25)))
    module.verify_old_report(report, report, .6)
    changed = {**report, "ranking": dict(upcoming_vs_negative=dict(tie_aware_average_precision=.3))}
    try: module.verify_old_report(changed, report, .6)
    except AssertionError: pass
    else: raise AssertionError("Changed AP was accepted")
