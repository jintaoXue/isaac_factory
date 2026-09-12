"""Event diagnostics must partition misses without changing canonical metrics."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from diagnose_baseline_events import (
    attach_node_catalog, load_diagnostic_checkpoint, parse_args, summarize_events,
    predicted_hot_run_score, summarize_prediction_heads,
    cold_onset_report_probability, summarize_onset_reports,
    independent_onset_report_mask, summarize_onset_frontier, _OnsetPrefixTree,
    predict_with_temporal_observation, summarize_temporal_attention,
    predict_with_history_graph_observation, summarize_history_graph,
    observe_joint_onset, summarize_joint_onset,
)
from factory_baselines.artifacts import archive_files
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig


class TestEventDiagnostics(unittest.TestCase):
    def test_joint_observation_uses_existing_real_outputs_without_mutation_or_forward(self):
        from test_baseline_joint_onset import _fixture
        from factory_baselines.torch_trainer import _model_inputs
        for kind in ("b4_gcn_gru", "b5_gat_gru"):
            with self.subTest(kind=kind):
                _, model, batch = _fixture(kind); model.eval()
                with torch.no_grad():
                    model.heads.event_onset_head[-1].bias.add_(2.)
                    result = model(**_model_inputs(batch, model))
                saved = {k: v.clone() for k, v in result.items()}
                state = {k: v.clone() for k, v in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                with patch.object(model, "forward", side_effect=AssertionError("No extra forward")):
                    observed = observe_joint_onset(result, batch["event_history_hot"])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))
                self.assertTrue(all(torch.equal(saved[k], v) for k, v in result.items()))
                self.assertTrue(torch.equal(observed["joint_onset_selected"], batch["event_history_hot"] <= .5))
                self.assertFalse(observed["joint_cold_logit_tie"].any())
                with self.assertRaisesRegex(ValueError, "registered joint rule"):
                    observe_joint_onset({**result, "event_will_logit": result["event_will_logit"] + 1}, batch["event_history_hot"])
        # Distinct saturated logits must not be misclassified as an exact tie.
        cont = torch.tensor([[100., 0., -1.]]); onset = torch.tensor([[101., 0., 0.]])
        gate = torch.tensor([[0., 0., 1.]])
        combined = torch.where(gate > .5, cont, torch.maximum(cont, onset))
        result = dict(event_will_continue_logit=cont, event_onset_logit=onset, event_will_logit=combined)
        observed = observe_joint_onset(result, gate)
        self.assertEqual(observed["joint_onset_selected"].tolist(), [[True, False, False]])
        self.assertEqual(observed["joint_cold_logit_tie"].tolist(), [[False, True, False]])
        for invalid in (gate[:, :2], torch.full_like(gate, .5), torch.full_like(gate, float("nan"))):
            with self.assertRaises(ValueError): observe_joint_onset(result, invalid)

    def _joint_arrays(self):
        arrays = self._three_class_arrays(); arrays.pop("event_kind_probability")
        arrays.update(will_probability=np.array([[.9, .8, .7, np.nan]]),
                      event_continue_probability=np.array([[.9, .1, .1, np.nan]]),
                      event_onset_probability=np.array([[.99, .8, .7, np.nan]]),
                      event_history_hot=np.array([[1., 0., 0., np.nan]]),
                      joint_onset_selected=np.array([[0., 1., 1., np.nan]]),
                      joint_cold_logit_tie=np.array([[0., 0., 0., np.nan]]))
        return arrays

    def test_joint_summary_separates_added_hits_false_alarms_and_preserves_canonical_report(self):
        arrays = self._joint_arrays(); saved = {k: v.copy() for k, v in arrays.items()}
        canonical = summarize_events(arrays, [.55]); report = summarize_joint_onset(arrays, .55)
        self.assertEqual(report["groups"]["upcoming"]["extra_hits"], 1)
        self.assertEqual(report["groups"]["negative"]["extra_false_alarms"], 1)
        self.assertEqual(report["groups"]["ongoing"]["onset_selected"], 0)
        self.assertEqual(report["groups"]["ongoing"]["extra_reports"], 0)
        joint, cont = (report["reports_at_saved_threshold"][k] for k in ("joint", "continue_only"))
        self.assertEqual(joint, canonical["thresholds"][0])
        self.assertEqual(joint["report_recall_upcoming"], 1)
        self.assertEqual(cont["report_recall_upcoming"], 0)
        self.assertAlmostEqual(joint["report_precision"], 2 / 3)
        self.assertEqual(cont["report_precision"], 1)
        self.assertIn("participates", canonical["onset_auxiliary_diagnostics"]["scope"])
        old = {k: v for k, v in arrays.items() if k != "event_continue_probability"}
        self.assertIn("excluded", summarize_events(old, [.55])["onset_auxiliary_diagnostics"]["scope"])
        self.assertEqual(canonical, summarize_events(arrays, [.55]))
        for k, v in saved.items(): np.testing.assert_array_equal(arrays[k], v)
        arrays["predicted_start"][0, 1] = 8
        missed = summarize_joint_onset(arrays, .55)["groups"]["upcoming"]
        self.assertEqual((missed["extra_hits"], missed["extra_false_alarms"]), (0, 1))

    def test_joint_summary_rejects_invalid_inputs_and_defines_empty_support(self):
        arrays = self._joint_arrays()
        for key in ("event_continue_probability", "event_history_hot", "joint_onset_selected"):
            bad = {k: v.copy() for k, v in arrays.items()}; bad[key][0, 1] = np.nan
            with self.assertRaises(ValueError): summarize_joint_onset(bad, .55)
        bad = {**arrays, "will_probability": arrays["event_continue_probability"]}
        with self.assertRaisesRegex(ValueError, "registered rule"): summarize_joint_onset(bad, .55)
        for threshold in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError): summarize_joint_onset(arrays, threshold)
        arrays["occ_node_mask"][:] = 0
        empty = summarize_joint_onset(arrays, .55)
        self.assertTrue(all(row["count"] == 0 for row in empty["groups"].values()))
        self.assertIsNone(empty["ranking"]["joint"]["upcoming_vs_negative_ap"])

    def test_history_graph_observer_preserves_outputs_state_rng_and_one_forward(self):
        import test_b5_gat_gru as fixture_module
        fixture = fixture_module.TestB5GatGru(); fixture.setUp()
        inputs = fixture._inputs(fixture._batch())
        for cls, cfg, spatial in ((B4GcnGru, B4ModelConfig, "gcn_hidden"),
                                  (B5GatGru, B5ModelConfig, "gat_hidden")):
            with self.subTest(model=cls.__name__):
                model = cls(cfg(6, 2, 5, **{spatial: 8}, gru_hidden=8,
                                temporal_readout="last_mean", history_graph_refine=True)).eval()
                _, zero = predict_with_history_graph_observation(model, inputs)
                self.assertTrue((zero["history_graph_shift_l2"] == 0).all())
                with torch.no_grad():
                    model.history_graph.output.weight.normal_(std=.1)
                state = {k: v.clone() for k, v in model.state_dict().items()}
                expected = model(**inputs); rng = torch.get_rng_state().clone()
                calls = []; handle = model.gru.register_forward_hook(lambda *args: calls.append(1))
                actual, observed = predict_with_history_graph_observation(model, inputs); handle.remove()
                self.assertEqual(len(calls), 1)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key, value in expected.items():
                    torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
                self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in state.items()))
                for value in observed.values():
                    self.assertEqual(value.shape, inputs["node_mask"].shape)
                    self.assertTrue(torch.isfinite(value).all())
                    self.assertTrue((value[~inputs["node_mask"].bool()] == 0).all())
                self.assertGreater(observed["history_graph_shift_l2"].max().item(), 0)
                self.assertEqual(len(model.history_graph._forward_hooks), 0)
                with patch.object(model, "forward", side_effect=RuntimeError("forward failed")):
                    with self.assertRaisesRegex(RuntimeError, "forward failed"):
                        predict_with_history_graph_observation(model, inputs)
                self.assertEqual(len(model.history_graph._forward_hooks), 0)
                with self.assertRaisesRegex(ValueError, "eval"):
                    predict_with_history_graph_observation(model.train(), inputs)

    def test_history_graph_summary_masks_labels_without_changing_canonical_report(self):
        arrays = self._three_class_arrays(); expected = summarize_events(arrays, [.55])
        arrays.update(history_graph_base_l2=np.array([[1., 2., 0., np.nan]]),
                      history_graph_refined_l2=np.array([[1., 3., 0., np.nan]]),
                      history_graph_shift_l2=np.array([[0., 1., 0., np.nan]]))
        saved = {k: v.copy() for k, v in arrays.items()}
        report = summarize_history_graph(arrays, .5)
        self.assertEqual(report["groups"]["upcoming"]["count"], 1)
        self.assertEqual(report["groups"]["upcoming"]["shift_over_base_l2_q10_q50_q90"], [.5] * 3)
        self.assertEqual(report["groups"]["negative"]["base_norm_below_floor_count"], 1)
        self.assertEqual(report["groups"]["negative"]["shift_over_base_l2_q10_q50_q90"], [0.] * 3)
        self.assertEqual(expected, summarize_events(arrays, [.55]))
        for key, value in saved.items():
            np.testing.assert_array_equal(arrays[key], value)
        arrays["occ_node_mask"][:] = 0
        self.assertTrue(all(row["count"] == 0 for row in summarize_history_graph(arrays, 0)["groups"].values()))

    def test_history_graph_summary_rejects_invalid_values_and_cli_options(self):
        arrays = self._three_class_arrays()
        arrays.update({"history_graph_"+key+"_l2": np.ones((1, 4)) for key in ("base", "refined", "shift")})
        for bad in (np.full((1, 4), -1.), np.full((1, 4), np.nan), np.ones((2, 4))):
            with self.assertRaises(ValueError):
                summarize_history_graph({**arrays, "history_graph_shift_l2": bad}, 0)
        for bad in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summarize_history_graph(arrays, bad)
        with self.assertRaisesRegex(ValueError, "refinement"):
            predict_with_history_graph_observation(torch.nn.Identity().eval(), {})
        common = ["--dataset_dir", "data", "--checkpoint", "best.pt", "--output", "result.json"]
        self.assertFalse(parse_args(common).inspect_history_graph)
        self.assertTrue(parse_args([*common, "--inspect_history_graph"]).inspect_history_graph)
        with self.assertRaises(SystemExit):
            parse_args([*common, "--inspect_history_graph", "--inspect_temporal_attention"])

    def test_onset_frontier_matches_exhaustive_two_threshold_enumeration(self):
        rng = np.random.default_rng(18)
        for trial in range(80):
            count = 1 + trial % 19
            kind = rng.integers(0, 3, size=(1, count))
            start = np.where(kind == 1, 0, 2)
            hot = np.zeros((1, 15, count), dtype=np.float32)
            for node in range(count):
                if kind[0, node]:
                    first = start[0, node]
                    hot[0, first:first+8, node] = 1
            scores = np.array([0., .2, .55, .7, .9, 1.], dtype=np.float32)
            arrays = dict(
                y_hot=hot, remain_mask=np.ones((1, 15)),
                occ_node_mask=(rng.random((1, count)) > .15).astype(float),
                hist_last_hot=rng.choice([0., .5, .51, 1.], size=(1, count)),
                event_will=(kind > 0).astype(float), event_start=start,
                will_probability=rng.choice(scores, size=(1, count)),
                event_onset_probability=rng.choice(scores, size=(1, count)),
                predicted_start=rng.choice([0, 2, 5, 14], size=(1, count)),
                predicted_duration=np.full((1, count), 8.),
            )
            saved = {key: value.copy() for key, value in arrays.items()}
            threshold = [.55, .7, 0., 1.][trial % 4]
            actual = summarize_onset_frontier(arrays, threshold)
            valid = arrays["occ_node_mask"] > .5
            positive = (kind > 0) & valid
            target_up = (kind == 2) & valid
            decoded = np.where(arrays["hist_last_hot"] > .5, 0, arrays["predicted_start"])
            hit = positive & (np.abs(decoded - start) <= 3)
            expected = dict.fromkeys(actual["empirical_maxima"])
            event_levels = [None, *np.unique(arrays["will_probability"][valid]).tolist()]
            onset_levels = [None, *np.unique(arrays["event_onset_probability"][valid]).tolist()]
            for fixed, event_thresholds in ((False, event_levels), (True, [threshold])):
                for event_threshold in event_thresholds:
                    for onset_threshold in onset_levels:
                        mask = np.zeros(valid.shape, dtype=bool)
                        if event_threshold is not None:
                            mask |= arrays["will_probability"] >= event_threshold
                        if onset_threshold is not None:
                            mask |= (arrays["hist_last_hot"] <= .5) & (arrays["event_onset_probability"] >= onset_threshold)
                        mask &= valid
                        n, h, u = int(mask.sum()), int((mask & hit).sum()), int((mask & hit & target_up).sum())
                        if n == 0 or 5*h < 4*n:
                            continue
                        keys = ["saved_event_threshold_P80"] if fixed else ["all_thresholds_P80"]
                        if not fixed and 10*h >= 7*int(positive.sum()):
                            keys.append("all_thresholds_P80_R70")
                        for key in keys:
                            expected[key] = max(expected[key] if expected[key] is not None else -1, u)
            for key, point in actual["empirical_maxima"].items():
                self.assertEqual(None if point is None else point["upcoming_hits"], expected[key], (trial, key))
            for key, value in saved.items():
                np.testing.assert_array_equal(arrays[key], value)

    def test_onset_prefix_tree_handles_disconnected_feasible_prefixes_and_removals(self):
        # Slack +1,+1,-4,+1,+1,+1 has disconnected feasible lengths 0,1,2,5,6.
        counts = np.ones(6, dtype=int)
        hits = np.array([1, 1, 0, 1, 1, 1])
        up = np.array([0, 1, 0, 0, 1, 0])
        tree = _OnsetPrefixTree(counts, hits, up)
        for required in range(-5, 5):
            prefix_slack = np.r_[0, np.cumsum(5*hits-4*counts)]
            feasible = np.flatnonzero(prefix_slack >= required)
            found = tree.rightmost(required)
            if not len(feasible):
                self.assertIsNone(found)
            else:
                end = int(feasible[-1])
                self.assertEqual(found, (end, int(counts[:end].sum()), int(hits[:end].sum()), int(up[:end].sum())))
        tree.remove(2, False, False)
        self.assertEqual(tree.rightmost(0), (6, 5, 5, 2))
        self.assertEqual(_OnsetPrefixTree(np.array([]), np.array([]), np.array([])).rightmost(0), (0, 0, 0, 0))

    def test_independent_onset_mask_has_no_target_inputs_and_respects_hot_gate(self):
        event = np.array([[.2, .2, .9, .2]])
        onset = np.array([[.9, .9, .1, .9]])
        history = np.array([[1., .5, 0., .51]])
        np.testing.assert_array_equal(
            independent_onset_report_mask(event, onset, history, .8, .9),
            [[False, True, True, False]],
        )
        self.assertFalse(independent_onset_report_mask(event, onset, history, None, None).any())
        for bad in (np.nan, -.1, 1.1):
            with self.assertRaises(ValueError):
                independent_onset_report_mask(event, onset, history, .5, bad)

    def test_onset_frontier_excludes_invalid_nodes_and_preserves_original_reports(self):
        arrays = self._three_class_arrays()
        arrays.pop("event_kind_probability")
        arrays["event_onset_probability"] = np.array([[.2, .9, .1, np.nan]])
        original = summarize_events(arrays, [.55])
        report = summarize_onset_frontier(arrays, .55)
        self.assertEqual(report["valid_targets"], 3)
        self.assertEqual(report["empirical_maxima"]["all_thresholds_P80"]["upcoming_hits"], 1)
        self.assertEqual(original, summarize_events(arrays, [.55]))
        arrays["occ_node_mask"][:] = 0
        self.assertTrue(all(v is None for v in summarize_onset_frontier(arrays, .55)["empirical_maxima"].values()))
        arrays["occ_node_mask"][:] = 1
        with self.assertRaises(ValueError):
            summarize_onset_frontier(arrays, .55)
        common = ["--dataset_dir", "data", "--checkpoint", "best.pt", "--output", "result.json"]
        self.assertFalse(parse_args(common).inspect_onset_frontier)
        self.assertTrue(parse_args([*common, "--inspect_onset_frontier"]).inspect_onset_frontier)

    def test_temporal_observer_preserves_single_forward_outputs_state_and_rng(self):
        import test_b5_gat_gru as fixture_module
        fixture = fixture_module.TestB5GatGru()
        fixture.setUp()
        inputs = fixture._inputs(fixture._batch())
        for cls, cfg, spatial in ((B4GcnGru, B4ModelConfig, "gcn_hidden"),
                                  (B5GatGru, B5ModelConfig, "gat_hidden")):
            with self.subTest(model=cls.__name__):
                model = cls(cfg(6, 2, 5, **{spatial: 8}, gru_hidden=8,
                                temporal_readout="last_attention")).eval()
                _, observed = predict_with_temporal_observation(model, inputs)
                self.assertTrue((observed["temporal_pool_shift_l2"] == 0).all())
                torch.testing.assert_close(observed["temporal_attention_weights"],
                                           torch.full_like(observed["temporal_attention_weights"], 1 / inputs["x"].shape[1]))
                with torch.no_grad():
                    model.history_attention.query.copy_(torch.linspace(-3, 3, 8))
                expected = model(**inputs)
                state = {key: value.clone() for key, value in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                calls = []
                handle = model.gru.register_forward_hook(lambda *args: calls.append(1))
                actual, observed = predict_with_temporal_observation(model, inputs)
                handle.remove()
                self.assertEqual(len(calls), 1)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key, value in expected.items():
                    torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
                for key, value in state.items():
                    self.assertTrue(torch.equal(model.state_dict()[key], value))
                weights = observed["temporal_attention_weights"]
                self.assertEqual(tuple(weights.shape), (*inputs["node_mask"].shape, inputs["x"].shape[1]))
                torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[:, :, 0]))
                self.assertGreater(float(observed["temporal_pool_shift_l2"].max()), 0)
                self.assertEqual(len(model.history_attention._forward_hooks), 0)
                with patch.object(model, "forward", side_effect=RuntimeError("forward failed")):
                    with self.assertRaisesRegex(RuntimeError, "forward failed"):
                        predict_with_temporal_observation(model, inputs)
                self.assertEqual(len(model.history_attention._forward_hooks), 0)
                with self.assertRaisesRegex(ValueError, "eval"):
                    predict_with_temporal_observation(model.train(), inputs)

    def test_temporal_summary_strata_masking_and_report_isolation(self):
        arrays = self._three_class_arrays()
        original = summarize_events(arrays, [.55])
        arrays.update(temporal_attention_weights=np.array([[
            [.25, .25, .25, .25], [0., 0., 0., 1.], [1., 0., 0., 0.], [np.nan] * 4]]),
            temporal_pool_shift_l2=np.array([[0., 1., .5, np.nan]]),
            temporal_pool_mean_l2=np.array([[1., 2., 1., np.nan]]))
        saved = {key: value.copy() for key, value in arrays.items()}
        report = summarize_temporal_attention(arrays, 1.)
        upcoming = report["groups"]["upcoming"]
        self.assertEqual(upcoming["count"], 1)
        self.assertEqual(upcoming["mean_weights_oldest_to_newest"], [0., 0., 0., 1.])
        self.assertEqual(upcoming["total_variation_from_uniform_q10_q50_q90"], [.75] * 3)
        self.assertEqual(upcoming["pool_shift_over_mean_l2_q10_q50_q90"], [.5] * 3)
        self.assertEqual(report["groups"]["ongoing"]["normalized_entropy_q10_q50_q90"], [1.] * 3)
        self.assertEqual(original, summarize_events(arrays, [.55]))
        for key, value in saved.items():
            np.testing.assert_array_equal(arrays[key], value)
        arrays["occ_node_mask"][:] = 0
        empty = summarize_temporal_attention(arrays, 1.)
        for row in empty["groups"].values():
            self.assertEqual(row["count"], 0)
            self.assertIsNone(row["mean_weights_oldest_to_newest"])

    def test_temporal_summary_rejects_invalid_observations_and_defines_one_step(self):
        arrays = self._three_class_arrays()
        arrays.update(temporal_attention_weights=np.ones((1, 4, 1)),
                      temporal_pool_shift_l2=np.zeros((1, 4)),
                      temporal_pool_mean_l2=np.zeros((1, 4)))
        report = summarize_temporal_attention(arrays, 0.)
        self.assertEqual(report["groups"]["upcoming"]["normalized_entropy_q10_q50_q90"], [0.] * 3)
        self.assertEqual(report["groups"]["upcoming"]["pool_shift_over_mean_l2_q10_q50_q90"], [0.] * 3)
        for bad in (np.zeros((1, 4, 1)), np.full((1, 4, 1), np.nan),
                    np.ones((1, 4)), np.ones((2, 4, 1))):
            with self.assertRaises(ValueError):
                summarize_temporal_attention({**arrays, "temporal_attention_weights": bad}, 0.)
        for bad in (np.full((1, 4), -1.), np.full((1, 4), np.nan), np.ones((1, 3))):
            with self.assertRaises(ValueError):
                summarize_temporal_attention({**arrays, "temporal_pool_shift_l2": bad}, 0.)
        with self.assertRaises(ValueError):
            summarize_temporal_attention(arrays, float("nan"))
        common = ["--dataset_dir", "data", "--checkpoint", "best.pt", "--output", "result.json"]
        self.assertFalse(parse_args(common).inspect_temporal_attention)
        self.assertTrue(parse_args([*common, "--inspect_temporal_attention"]).inspect_temporal_attention)

    def test_onset_report_policy_uses_only_predictions_and_history_without_mutation(self):
        event = np.array([[.1, .2, .8, .3]])
        onset = np.array([[.9, .7, .2, .9]])
        history = np.array([[1., 0., .5, .51]])
        originals = [x.copy() for x in (event, onset, history)]
        actual = cold_onset_report_probability(event, onset, history)
        np.testing.assert_allclose(actual, [[.1, .7, .8, .3]])
        for actual_input, saved in zip((event, onset, history), originals):
            np.testing.assert_array_equal(actual_input, saved)
        for bad in (onset[:, :1], onset[0], np.full_like(onset, np.nan), onset * 2):
            with self.assertRaises(ValueError):
                cold_onset_report_probability(event, bad, history)

    def test_onset_report_probe_counts_recovered_events_false_alarms_and_timing_errors(self):
        arrays = self._three_class_arrays()
        arrays.pop("event_kind_probability")
        arrays["will_probability"][0, 1] = .1
        arrays["event_onset_probability"] = np.array([[.99, .8, .7, .99]])
        saved = {k: v.copy() for k, v in arrays.items()}
        original = summarize_events(arrays, [.55])
        comparison = summarize_onset_reports(arrays, [.8, .55, .8], .55)
        self.assertEqual(comparison["thresholds"], [.55, .8])
        baseline = comparison["policies"]["event_only"]["thresholds"][0]
        candidate = comparison["policies"]["cold_onset_max"]["thresholds"][0]
        self.assertEqual(baseline, original["thresholds"][0])
        self.assertEqual(baseline["report_recall_upcoming"], 0)
        self.assertEqual(candidate["report_recall_upcoming"], 1)
        self.assertEqual(candidate["false_positive_stations"], 1)
        self.assertEqual(candidate["n_pred_who"], 3)  # invalid fourth node is excluded
        self.assertAlmostEqual(candidate["report_precision"], 2 / 3)
        self.assertEqual(original, summarize_events(arrays, [.55]))
        for key, value in saved.items():
            np.testing.assert_array_equal(arrays[key], value)
        arrays["predicted_start"][0, 1] = 12
        mistimed = summarize_onset_reports(arrays, [.55], .55)["policies"]["cold_onset_max"]["thresholds"][0]
        self.assertEqual(mistimed["report_recall_upcoming"], 0)
        self.assertEqual(mistimed["upcoming_timing_misses"], 1)
        self.assertEqual(mistimed["report_false_alarm_count"], 2)
        arrays["occ_node_mask"][:] = 0
        empty = summarize_onset_reports(arrays, [.55], .55)["policies"]["cold_onset_max"]
        self.assertEqual(empty["thresholds"][0]["n_pred_who"], 0)
        self.assertIsNone(empty["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"])

    def test_onset_probe_requires_the_registered_head_and_valid_thresholds(self):
        arrays = self._three_class_arrays()
        with self.assertRaisesRegex(ValueError, "independent onset"):
            summarize_onset_reports(arrays, [.55], .55)
        arrays["event_onset_probability"] = np.ones((1, 4))
        with self.assertRaisesRegex(ValueError, "binary"):
            summarize_onset_reports(arrays, [.55], .55)
        arrays.pop("event_kind_probability")
        for thresholds in ([np.nan], [2.]):
            with self.assertRaisesRegex(ValueError, "threshold"):
                summarize_onset_reports(arrays, thresholds, .55)
        common = ["--dataset_dir", "data", "--checkpoint", "best.pt", "--output", "result.json"]
        self.assertFalse(parse_args(common).compare_onset_report)
        self.assertTrue(parse_args([*common, "--compare_onset_report"]).compare_onset_report)

    def test_onset_auxiliary_ranking_cannot_change_reports(self):
        arrays = self._three_class_arrays()
        arrays.pop("event_kind_probability")
        expected = summarize_events(arrays, [.55])
        arrays["event_onset_probability"] = np.array([[.99, .8, .1, np.nan]])
        actual = summarize_events(arrays, [.55])
        aux = actual.pop("onset_auxiliary_diagnostics")
        self.assertEqual(expected, actual)
        self.assertEqual(aux["upcoming_count"], 1)
        self.assertEqual(aux["negative_count"], 1)
        self.assertEqual(aux["upcoming_vs_negative_ap"], 1)
        for bad in (np.ones((2, 4)), np.full((1, 4), np.nan), np.full((1, 4), 2.)):
            arrays["event_onset_probability"] = bad
            with self.assertRaises(ValueError):
                summarize_events(arrays, [.55])
        arrays["occ_node_mask"][:] = 0
        empty = summarize_events(arrays, [.55])["onset_auxiliary_diagnostics"]
        self.assertIsNone(empty["upcoming_vs_negative_ap"])

    def test_predicted_hot_score_requires_an_early_continuous_run(self):
        p = np.full((2, 15, 3), .1)
        p[0, 2:10, 0] = .8  # latest eligible onset
        p[0, 3:11, 1] = .9  # eight hot minutes, but onset too late
        p[0, :4, 2] = .9
        p[0, 5:9, 2] = .9  # eight hot minutes with a gap
        p[1, :8, :] = .7
        np.testing.assert_allclose(predicted_hot_run_score(p), [[.8, .1, .1], [.7, .7, .7]])
        p[:, 10:] = 1.0
        np.testing.assert_allclose(predicted_hot_run_score(p), [[.8, .1, .1], [.7, .7, .7]])
        for bad in (p[:, :9], p[:, 0], p * 2, p * np.nan):
            with self.assertRaises(ValueError):
                predicted_hot_run_score(bad)

    def test_head_comparison_preserves_reports_and_excludes_ongoing_and_masked_nodes(self):
        arrays = self._three_class_arrays()
        arrays["will_probability"][0, 1] = .1
        arrays.pop("event_kind_probability")
        p = np.full((1, 15, 4), .1)
        p[0, 1:9, 1] = .8
        p[0, :8, 2] = .6
        p[0, :8, 3] = .99
        arrays["predicted_hot_probability"] = p
        saved = {k: v.copy() for k, v in arrays.items()}
        canonical = summarize_events(arrays, [.55])
        result = summarize_prediction_heads(arrays, .55)
        self.assertEqual(result["heads"]["event_head"]["upcoming_vs_negative_ap"], .5)
        self.assertEqual(result["heads"]["predicted_hot_run"]["upcoming_vs_negative_ap"], 1.)
        self.assertEqual(result["heads"]["event_head"]["negative_count"], 1)
        self.assertEqual(result["overlap"]["upcoming"], dict(both=0, event_only=0, hot_only=1, neither=0))
        self.assertEqual(result["overlap"]["negative"]["hot_only"], 1)
        self.assertEqual(canonical, summarize_events(arrays, [.55]))
        for k, v in saved.items():
            np.testing.assert_array_equal(arrays[k], v)
        arrays["occ_node_mask"][:] = 0
        empty = summarize_prediction_heads(arrays, .55)
        self.assertIsNone(empty["heads"]["predicted_hot_run"]["upcoming_vs_negative_ap"])
        self.assertEqual(sum(empty["overlap"]["upcoming"].values()), 0)

    def _three_class_arrays(self):
        hot = np.zeros((1, 15, 4), dtype=np.float32)
        hot[0, :8, 0] = 1
        hot[0, 1:9, 1] = 1
        return {
            "y_hot": hot, "remain_mask": np.ones((1, 15)),
            "occ_node_mask": np.array([[1, 1, 1, 0]]),
            "hist_last_hot": np.array([[1, 0, 0, 0]]),
            "event_will": np.array([[1, 1, 0, 0]]),
            "event_start": np.array([[0, 1, 0, 0]]),
            "will_probability": np.array([[.95, .95, .1, .5]]),
            "predicted_start": np.zeros((1, 4)),
            "predicted_duration": np.full((1, 4), 8),
            "event_kind_probability": np.array([[
                [.05, .9, .05], [.05, .85, .1], [.9, .05, .05], [np.nan, np.nan, np.nan],
            ]]),
        }

    def test_three_class_diagnostic_does_not_change_binary_event_report(self):
        arrays = self._three_class_arrays()
        original = {key: value.copy() for key, value in arrays.items()}
        result = summarize_events(arrays, [.55])
        kinds = result.pop("event_kind_diagnostics")
        binary = summarize_events({k: v for k, v in arrays.items() if k != "event_kind_probability"}, [.55])
        self.assertEqual(result, binary)
        self.assertEqual(result["thresholds"][0]["report_recall_upcoming"], 1)
        self.assertEqual(kinds["sample_count"], 3)
        self.assertEqual(kinds["confusion_rows_true_columns_argmax"], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(kinds["per_true_class"]["upcoming"]["argmax_recall"], 0)
        self.assertEqual(kinds["per_true_class"]["upcoming"]["mean_predicted_probabilities"], [.05, .85, .1])
        self.assertEqual(kinds["upcoming_class_vs_negative"]["tie_aware_average_precision"], 1)
        for key, value in original.items():
            np.testing.assert_array_equal(arrays[key], value)

    def test_three_class_diagnostic_rejects_invalid_or_mismatched_probabilities(self):
        for change in ("shape", "distribution", "marginal"):
            with self.subTest(change=change):
                arrays = self._three_class_arrays()
                if change == "shape":
                    arrays["event_kind_probability"] = arrays["event_kind_probability"][..., :2]
                elif change == "distribution":
                    arrays["event_kind_probability"][0, 0, 0] = .5
                else:
                    arrays["will_probability"][0, 0] = .1
                with self.assertRaises(ValueError):
                    summarize_events(arrays, [.55])

    def test_three_class_diagnostic_empty_and_missing_classes_are_explicit(self):
        arrays = self._three_class_arrays()
        arrays["occ_node_mask"][:] = 0
        kinds = summarize_events(arrays, [.55])["event_kind_diagnostics"]
        self.assertEqual(kinds["sample_count"], 0)
        self.assertIsNone(kinds["argmax_accuracy"])
        self.assertEqual(kinds["confusion_rows_true_columns_argmax"], [[0, 0, 0]] * 3)
        for row in kinds["per_true_class"].values():
            self.assertEqual(row["count"], 0)
            self.assertIsNone(row["argmax_recall"])
            self.assertIsNone(row["mean_predicted_probabilities"])
        self.assertIsNone(kinds["upcoming_class_vs_negative"]["tie_aware_average_precision"])
        arrays["occ_node_mask"][0, 2] = 1
        kinds = summarize_events(arrays, [.55])["event_kind_diagnostics"]
        self.assertEqual(kinds["sample_count"], 1)
        self.assertEqual(kinds["per_true_class"]["none"]["argmax_recall"], 1)
        self.assertIsNone(kinds["per_true_class"]["upcoming"]["argmax_recall"])

    def test_archived_checkpoint_matches_direct_load_without_extraction(self):
        config = B4ModelConfig(input_dim=3, global_dim=0, num_nodes=2, gcn_hidden=2, gru_hidden=4)
        model = B4GcnGru(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "best.pt"
            torch.save({"model_kind": "b4_gcn_gru", "model_config": config.to_dict(),
                        "model_state_dict": model.state_dict(), "epoch": 7}, path)
            expected, _, direct = load_diagnostic_checkpoint(path, torch.device("cpu"), None)
            archive = archive_files(root, ["best.pt"], "saved.zip")
            before = archive.read_bytes()
            actual, checkpoint, source = load_diagnostic_checkpoint(archive, torch.device("cpu"), "best.pt")
            self.assertEqual(checkpoint["epoch"], 7)
            self.assertEqual(source["checkpoint_member_sha256"], direct["checkpoint_file_sha256"])
            self.assertEqual(source["checkpoint_file_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(set(root.iterdir()), {archive})
            self.assertEqual(archive.read_bytes(), before)
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)

    def test_archive_rejects_bad_checksum_before_deserializing(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "x") as stream:
                stream.writestr("archive_manifest.json", json.dumps({"best.pt": "incorrect"}))
                stream.writestr("best.pt", b"not a checkpoint")
            with patch("diagnose_baseline_events.load_checkpoint") as loader:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    load_diagnostic_checkpoint(archive, torch.device("cpu"), "best.pt")
                with self.assertRaisesRegex(ValueError, "direct-child"):
                    load_diagnostic_checkpoint(archive, torch.device("cpu"), "../best.pt")
                with self.assertRaisesRegex(ValueError, "Expected one"):
                    load_diagnostic_checkpoint(archive, torch.device("cpu"), "missing.pt")
                loader.assert_not_called()

    def test_restarted_upcoming_is_audited_without_relabeling(self):
        hot = np.zeros((1, 15, 3), dtype=np.float32)
        hot[0, 1:9, 0] = 1
        hot[0, :8, 1] = 1
        hot[0, 1:9, 2] = 1
        arrays = {
            "y_hot": hot, "remain_mask": np.ones((1, 15)),
            "occ_node_mask": np.array([[1, 1, 0]]),
            "hist_last_hot": np.array([[1, 0, 1]]),
            "event_will": np.array([[1, 1, 1]]),
            "event_start": np.array([[1, 0, 1]]),
            "will_probability": np.ones((1, 3)), "predicted_start": np.zeros((1, 3)),
            "predicted_duration": np.full((1, 3), 8),
        }
        report = summarize_events(arrays, [.65])
        audit = report["training_partition_audit"]
        self.assertEqual(audit["upcoming_with_hot_history"], 1)
        self.assertEqual(audit["upcoming_with_hot_history_windows"], 1)
        self.assertEqual(audit["ongoing_with_cold_history"], 1)
        self.assertEqual(report["groups"]["upcoming"]["count"], 1)
        self.assertEqual(report["thresholds"][0]["report_f1"], 1)

    def test_diagnostic_accepts_train_and_validation_but_not_test(self):
        common = ["--dataset_dir", "data", "--checkpoint", "best.pt", "--output", "result.json"]
        self.assertEqual(parse_args(common).split, "validation")
        for split in ("train", "validation"):
            self.assertEqual(parse_args([*common, "--split", split]).split, split)
        with self.assertRaises(SystemExit):
            parse_args([*common, "--split", "test"])

    def test_catalog_types_are_not_indexed_by_node_position(self):
        manifest = {"node_ids": ["a", "b", "c"], "resource_types": ["machine", "human"]}
        report = {"thresholds": [{"per_node": [{"node_index": 2}]}]}
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "node_catalog.csv"
            catalog.write_text(
                "node_index,resource_id,resource_type,resource_type_index\n"
                "2,c,machine,0\n0,a,human,1\n1,b,machine,0\n", encoding="utf-8"
            )
            attach_node_catalog(report, catalog, manifest)
            self.assertEqual(report["thresholds"][0]["per_node"][0]["resource_type"], "machine")
            self.assertEqual(report["thresholds"][0]["per_node"][0]["resource_id"], "c")
            manifest["node_ids"][2] = "different"
            with self.assertRaisesRegex(ValueError, "identities/types"):
                attach_node_catalog(report, catalog, manifest)

    def test_probability_and_timing_misses_are_disjoint(self):
        hot = np.zeros((1, 15, 4), dtype=np.float32)
        hot[0, :8, 0] = 1
        hot[0, 2:10, 1:3] = 1
        arrays = {
            "y_hot": hot,
            "remain_mask": np.ones((1, 15)),
            "occ_node_mask": np.ones((1, 4)),
            "hist_last_hot": np.array([[1, 0, 0, 0]]),
            "event_will": np.array([[1, 1, 1, 0]]),
            "event_start": np.array([[0, 2, 2, -1]]),
            "will_probability": np.array([[.9, .2, .8, .8]]),
            "predicted_start": np.array([[9, 2, 6, 0]]),
            "predicted_duration": np.full((1, 4), 8),
        }
        result = summarize_events(arrays, [.5])
        row = result["thresholds"][0]
        self.assertEqual(row["upcoming_probability_misses"], 1)
        self.assertEqual(row["upcoming_timing_misses"], 1)
        self.assertEqual(row["false_positive_stations"], 1)
        self.assertEqual(row["report_false_alarm_count"], 2)
        self.assertEqual(row["report_false_alarm_breakdown"]["true_event_wrong_start"], 1)
        self.assertEqual(row["report_recall_ongoing"], 1)
        self.assertEqual(row["report_recall_upcoming"], 0)
        self.assertAlmostEqual(row["report_f1"], 1 / 3)
        self.assertEqual(result["groups"]["ongoing"]["start_within_tolerance_rate"], 1)
        ranking = result["ranking"]
        self.assertEqual(ranking["all_events"]["sample_count"], 4)
        self.assertEqual(ranking["upcoming_vs_negative"]["positive_count"], 2)
        self.assertEqual(ranking["upcoming_vs_negative"]["negative_count"], 1)
        self.assertEqual(ranking["ongoing_vs_negative"]["positive_count"], 1)
        self.assertAlmostEqual(ranking["upcoming_vs_negative"]["tie_aware_average_precision"], 7 / 12)
        self.assertAlmostEqual(ranking["ongoing_vs_negative"]["tie_aware_average_precision"], 1)

    def test_false_alarm_partition_handles_short_horizon_and_short_hot_runs(self):
        hot = np.zeros((2, 15, 3), dtype=np.float32)
        hot[0, 1:4, 0] = 1
        hot[0, :8, 1] = 1
        hot[1, :4, :] = 1
        hot[1, 0, 0] = 0
        remain = np.ones((2, 15))
        remain[1, 5:] = 0
        arrays = {
            "y_hot": hot, "remain_mask": remain,
            "occ_node_mask": np.array([[1, 1, 0], [1, 1, 1]]),
            "hist_last_hot": np.array([[1, 0, 0], [1, 0, 0]]),
            "event_will": np.array([[0, 1, 0], [0, 0, 0]]),
            "event_start": np.zeros((2, 3)),
            "will_probability": np.full((2, 3), .9),
            "predicted_start": np.zeros((2, 3)),
            "predicted_duration": np.full((2, 3), 8),
        }
        row = summarize_events(arrays, [.5])["thresholds"][0]
        parts = row["report_false_alarm_breakdown"]
        self.assertEqual(row["n_pred_who"], 5)
        self.assertAlmostEqual(row["report_precision"], .2)
        self.assertEqual(row["report_false_alarm_count"], 4)
        self.assertEqual(parts["short_observed_horizon_historically_hot"], 1)
        self.assertEqual(parts["short_observed_horizon_historically_cold"], 2)
        self.assertEqual(parts["hot_without_qualifying_event_historically_hot"], 1)
        self.assertEqual(sum(parts.values()), 4)
        self.assertEqual(sum(node["false_alarms"] for node in row["per_node"]), 4)
        self.assertEqual(row["predicted_duration_q25_q50_q75"], [8, 8, 8])
        ranking = summarize_events(arrays, [.5])["ranking"]["events_vs_short_hot_negative"]
        self.assertEqual(ranking["sample_count"], 2)
        self.assertEqual(ranking["negative_count"], 1)
        self.assertEqual(ranking["positive_rate"], .5)
        self.assertEqual(ranking["tie_aware_average_precision"], .5)
        self.assertEqual(ranking["ap_over_prevalence"], 1)
        self.assertEqual(ranking["roc_auc"], .5)

    def test_future_hot_outside_observation_does_not_explain_false_alarm(self):
        hot = np.zeros((1, 15, 1), dtype=np.float32)
        hot[0, 10:, 0] = 1
        remain = np.ones((1, 15))
        remain[:, 10:] = 0
        arrays = {
            "y_hot": hot, "remain_mask": remain, "occ_node_mask": np.ones((1, 1)),
            "hist_last_hot": np.zeros((1, 1)), "event_will": np.zeros((1, 1)),
            "event_start": np.zeros((1, 1)), "will_probability": np.full((1, 1), .9),
            "predicted_start": np.zeros((1, 1)), "predicted_duration": np.full((1, 1), 8),
        }
        row = summarize_events(arrays, [.5])["thresholds"][0]
        self.assertEqual(row["report_false_alarm_breakdown"]["no_future_hot_historically_cold"], 1)
        empty = summarize_events(arrays, [.95])["thresholds"][0]
        self.assertEqual(empty["report_false_alarm_count"], 0)
        self.assertIsNone(empty["predicted_duration_q25_q50_q75"])
        result = summarize_events(arrays, [.95])
        self.assertIsNone(result["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"])
        self.assertIsNone(result["ranking"]["events_vs_short_hot_negative"]["positive_rate"])

    def test_invalid_nodes_are_excluded_from_ranking(self):
        arrays = {
            "y_hot": np.zeros((1, 15, 1)), "remain_mask": np.ones((1, 15)),
            "occ_node_mask": np.zeros((1, 1)), "hist_last_hot": np.zeros((1, 1)),
            "event_will": np.ones((1, 1)), "event_start": np.zeros((1, 1)),
            "will_probability": np.ones((1, 1)), "predicted_start": np.zeros((1, 1)),
            "predicted_duration": np.ones((1, 1)),
        }
        for row in summarize_events(arrays, [.5])["ranking"].values():
            self.assertEqual(row["sample_count"], 0)
            self.assertIsNone(row["tie_aware_average_precision"])
            self.assertIsNone(row["roc_auc"])


if __name__ == "__main__":
    unittest.main()
