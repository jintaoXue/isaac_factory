"""CPU tests for B2 tabular feature construction and constant heads."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


TOOLS_DIR = (
    Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
)
sys.path.insert(0, str(TOOLS_DIR))

from factory_baselines.b2_xgboost import (  # noqa: E402
    B2XGBoostConfig,
    _Head,
    _base_features,
    _cell_features,
    _predict_multiclass,
    _predict_probability,
    _predict_regression,
)


class TestB2XGBoost(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.payload = {
            "x": torch.randn(3, 4, 5, 6),
            "node_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                ],
                dtype=torch.bool,
            ),
            "occ_node_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                ],
                dtype=torch.float32,
            ),
            "observation_mask": torch.ones(3, 4, 5, dtype=torch.bool),
            "global_features": torch.empty(3, 4, 0),
            "jobs_remaining": torch.tensor([3.0, 2.0, 1.0]),
            "jobs_total": torch.tensor([3.0, 3.0, 3.0]),
            "max_remain_windows": 8,
        }

    def test_base_features_have_stable_shape_and_finite_values(self) -> None:
        features = _base_features(self.payload, np.array([0, 2]))
        self.assertEqual(features.shape, (2, 133))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_cell_features_select_each_requested_node_history(self) -> None:
        samples = np.array([0, 0, 1, 2])
        offsets = np.array([0, 1, 2, 3])
        nodes = np.array([0, 2, 1, 4])
        features = _cell_features(self.payload, samples, offsets, nodes)
        self.assertEqual(features.shape, (4, 43))
        np.testing.assert_allclose(
            features[:, :6],
            self.payload["x"][samples, -1, nodes].numpy(),
        )
        self.assertTrue(np.isfinite(features).all())

    def test_constant_heads_do_not_require_xgboost(self) -> None:
        X = np.zeros((4, 3), dtype=np.float32)
        probability = _predict_probability(
            _Head(kind="constant", constant=1, classes=[1]), X
        )
        regression = _predict_regression(_Head(kind="constant", constant=2.5), X)
        prediction, multiclass = _predict_multiclass(
            _Head(kind="constant", constant=2, classes=[2]), X, class_count=4
        )
        np.testing.assert_array_equal(probability, np.ones(4))
        np.testing.assert_allclose(regression, np.full(4, 2.5))
        np.testing.assert_array_equal(prediction, np.full(4, 2))
        np.testing.assert_array_equal(multiclass[:, 2], np.ones(4))

    def test_invalid_sampling_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative_cell_ratio"):
            B2XGBoostConfig(negative_cell_ratio=0.0)

    def test_main_experiment_thresholds_are_the_defaults(self) -> None:
        config = B2XGBoostConfig()
        self.assertEqual(config.training_profile, "baseline_fair_v1")
        self.assertEqual(config.n_estimators, 500)
        self.assertEqual(config.max_depth, 5)
        self.assertEqual(config.min_child_weight, 3.0)
        self.assertEqual(config.reg_lambda, 5.0)
        self.assertEqual(config.hot_eval_threshold, 0.55)
        self.assertEqual(config.event_report_threshold, 0.68)
        self.assertEqual(config.report_threshold_min_precision, 0.80)
        self.assertEqual(config.checkpoint_min_report_recall, 0.35)
        with self.assertRaisesRegex(ValueError, "hot_eval_threshold"):
            B2XGBoostConfig(hot_eval_threshold=1.01)
        with self.assertRaisesRegex(ValueError, "event_report_threshold"):
            B2XGBoostConfig(event_report_threshold=-0.01)
        with self.assertRaisesRegex(ValueError, "training_profile"):
            B2XGBoostConfig(training_profile="")


if __name__ == "__main__":
    unittest.main()
