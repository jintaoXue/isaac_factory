"""Read-only GAT observations must distinguish ordering from edge-mask changes."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from diagnose_baseline_events import (shared_neighbor_top_reversals,
    predict_with_gat_ranking_observation, summarize_gat_ranking)
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig, DenseGraphAttention
from test_baseline_vector_gat import model_inputs


class TestGatRanking(unittest.TestCase):
    def test_actual_layer_static_zero_and_dynamic_reversal(self):
        x = torch.tensor([[[-2., 0.], [0., -3.], [1., 0.], [0., 2.]]])
        graph = torch.eye(4)[None].bool()
        graph[0, :2] = torch.tensor([False, False, True, True])
        for mode in ("scalar_additive", "vector_additive"):
            layer = DenseGraphAttention(2, 2, 1, True, 0, mode).eval()
            with torch.no_grad():
                layer.projection.weight.copy_(torch.eye(2))
                layer.attention_source.fill_(1); layer.attention_target.fill_(1)
            captured = []
            hook = layer.dropout.register_forward_hook(lambda module, args, out: captured.append(args[0]))
            layer(x, graph, torch.ones(1, 4)); hook.remove()
            stats = shared_neighbor_top_reversals(captured[0], graph, 1, 1)
            self.assertEqual(stats["pair_eligible_by_head"].tolist(), [[1]])
            reverse = int(mode == "vector_additive")
            self.assertEqual(stats["pair_reversed_by_head"].tolist(), [[reverse]])
            self.assertEqual(stats["query_reversed"].tolist(), [[reverse, reverse, 0, 0]])

    def test_common_edges_ties_and_tolerance_do_not_create_reversals(self):
        # Different overall top neighbors caused solely by disjoint edges are
        # not dynamic ranking; these two queries have only one common neighbor.
        edges = torch.tensor([[[1, 1, 0], [0, 1, 1], [0, 0, 0]]]).bool()
        weights = torch.tensor([[[[.7], [.3], [0]], [[0], [.3], [.7]], [[1/3], [1/3], [1/3]]]])
        stats = shared_neighbor_top_reversals(weights, edges, 1, 1)
        self.assertEqual(stats["pair_eligible_by_head"].sum(), 0)
        edges[:] = True
        for eps in (0., 1e-8):
            weights[:] = 1/3
            weights[0, 0, 0] += eps; weights[0, 0, 1] -= eps
            weights[0, 1, 0] -= eps; weights[0, 1, 1] += eps
            self.assertEqual(shared_neighbor_top_reversals(weights, edges, 1, 1)["pair_reversed_by_head"].sum(), 0)
        with self.assertRaises(ValueError): shared_neighbor_top_reversals(weights, edges, 1, 2)
        weights[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): shared_neighbor_top_reversals(weights, edges, 1, 1)

    def test_original_forward_rng_state_and_hooks_preserved(self):
        torch.manual_seed(84)
        for mode in ("scalar_additive", "vector_additive"):
            model = B5GatGru(B5ModelConfig(6, 0, 4, gat_hidden=8, gat_heads=2,
                gru_hidden=8, event_precursor="near", gat_score_mode=mode)).eval()
            inputs = model_inputs()
            with torch.no_grad(): expected = model(**inputs)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            with patch.object(model, "forward", wraps=model.forward) as forward:
                result, observed = predict_with_gat_ranking_observation(model, inputs)
                self.assertEqual(forward.call_count, 1)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key in expected: torch.testing.assert_close(result[key], expected[key], atol=0, rtol=0)
            for key in state: torch.testing.assert_close(model.state_dict()[key], state[key], atol=0, rtol=0)
            for name in ("gat1", "gat2"):
                self.assertFalse(getattr(model, name)._forward_hooks)
                self.assertFalse(getattr(model, name).dropout._forward_hooks)
                self.assertEqual(observed[name + "_query_eligible"][:, -1].sum(), 0)
                if mode == "scalar_additive": self.assertEqual(observed[name + "_query_reversed"].sum(), 0)
            with patch.object(model.gat2, "forward", side_effect=RuntimeError("fixture")):
                with self.assertRaisesRegex(RuntimeError, "fixture"):
                    predict_with_gat_ranking_observation(model, inputs)
            self.assertFalse(model.gat1._forward_hooks); self.assertFalse(model.gat2.dropout._forward_hooks)

    def test_summary_counts_groups_permutation_and_empty_support(self):
        torch.manual_seed(63)
        attention = torch.rand(6, 5, 5, 2).softmax(2)
        edges = torch.ones(6, 5, 5, dtype=torch.bool)
        stats = shared_neighbor_top_reversals(attention, edges, 2, 3)
        permutation = torch.tensor([4, 1, 3, 0, 2])
        permuted = shared_neighbor_top_reversals(attention[:, permutation][:, :, permutation], edges, 2, 3)
        for key in stats:
            expected = stats[key][:, permutation] if key.startswith("query") else stats[key]
            torch.testing.assert_close(permuted[key], expected)
        arrays = dict(occ_node_mask=np.ones((2, 5)), event_will=np.zeros((2, 5)), event_start=np.zeros((2, 5)))
        arrays["event_will"][0, 0:2] = 1; arrays["event_start"][0, 0] = 1
        arrays.update({name + "_" + key: value.numpy() for name in ("gat1", "gat2") for key, value in stats.items()})
        report = summarize_gat_ranking(arrays, "vector_additive")
        groups = report["layers"]["gat1"]["groups"]
        self.assertEqual([groups[k]["count"] for k in ("upcoming", "ongoing", "negative")], [1, 1, 8])
        self.assertEqual(sum(r["eligible_query_pair_time_head_incidence"] for r in groups.values()), 2 * stats["pair_eligible_by_head"].sum())
        arrays["occ_node_mask"][:] = 0
        self.assertIsNone(summarize_gat_ranking(arrays, "vector_additive")["layers"]["gat1"]["groups"]["upcoming"]["reversal_fraction"])
        arrays["gat1_query_reversed"][0, 0] += 1
        with self.assertRaises(ValueError): summarize_gat_ranking(arrays, "vector_additive")


if __name__ == "__main__":
    unittest.main()
