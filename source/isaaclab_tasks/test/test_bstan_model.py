"""CPU unit tests for the BSTAN Phase-D model and training primitives."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


TOOLS_DIR = (
    Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
)
sys.path.insert(0, str(TOOLS_DIR))

from bstan_baseline.losses import (  # noqa: E402
    BstanLossConfig,
    compute_multitask_loss,
)
from bstan_baseline.model import BstanGatGru, BstanModelConfig  # noqa: E402
from bstan_baseline.metrics import compute_metrics, select_f1_threshold  # noqa: E402
from bstan_baseline.trainer import (  # noqa: E402
    BstanTrainConfig,
    load_checkpoint,
    save_checkpoint,
)


class TestBstanModel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = BstanModelConfig(
            input_dim=6,
            global_dim=2,
            num_nodes=5,
            gat_hidden=8,
            gat_heads=2,
            gru_hidden=8,
            dropout=0.0,
        )

    def _batch(self, positive: bool = True) -> dict[str, torch.Tensor]:
        batch_size, time_steps, node_count = 4, 4, self.config.num_nodes
        node_mask = torch.ones(batch_size, node_count, dtype=torch.bool)
        node_mask[:, -1] = False
        target_node_mask = node_mask.clone()
        target_node_mask[:, -2] = False
        adjacency = torch.ones(batch_size, node_count, node_count, dtype=torch.bool)
        adjacency[:, -1, :] = False
        adjacency[:, :, -1] = False
        occurrence = torch.ones(batch_size) if positive else torch.zeros(batch_size)
        return {
            "x": torch.randn(batch_size, time_steps, node_count, self.config.input_dim),
            "adjacency": adjacency,
            "node_mask": node_mask,
            "target_node_mask": target_node_mask,
            "global_features": torch.randn(
                batch_size, time_steps, self.config.global_dim
            ),
            "jobs_remaining": torch.full((batch_size,), 2.0),
            "jobs_total": torch.full((batch_size,), 2.0),
            "y_occurrence": occurrence,
            "y_node": torch.zeros(batch_size, dtype=torch.int64),
            "y_time_to_start": torch.full((batch_size,), 30.0),
            "positive_mask": occurrence.bool(),
            "y_cause": torch.zeros(batch_size, dtype=torch.int64),
            "target_remain_len": torch.full((batch_size,), 4, dtype=torch.int64),
            "remain_mask": torch.ones(batch_size, 512),
            "y_score": torch.rand(batch_size, 512, node_count, 1),
            "y_hot": torch.zeros(batch_size, 512, node_count),
        }

    @staticmethod
    def _inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: batch[name]
            for name in (
                "x",
                "adjacency",
                "node_mask",
                "target_node_mask",
                "global_features",
                "jobs_remaining",
                "jobs_total",
            )
        }

    def test_forward_shapes_and_masked_node_logits(self) -> None:
        model = BstanGatGru(self.config)
        batch = self._batch()
        outputs = model(**self._inputs(batch))
        self.assertEqual(outputs["occurrence_logit"].shape, (4,))
        self.assertEqual(outputs["node_logits"].shape, (4, 5))
        self.assertEqual(outputs["node_hidden"].shape, (4, 5, 8))
        self.assertTrue((outputs["node_logits"][:, -1] < -1.0e8).all())
        self.assertTrue((outputs["node_logits"][:, -2] < -1.0e8).all())
        probabilities = torch.softmax(outputs["node_logits"], dim=-1)
        self.assertTrue(torch.equal(probabilities[:, -1], torch.zeros(4)))
        self.assertTrue(torch.equal(probabilities[:, -2], torch.zeros(4)))
        self.assertEqual(outputs["remain_score"].shape, (4, 512, 5, 1))
        self.assertEqual(outputs["remain_hot_logit"].shape, (4, 512, 5))
        self.assertEqual(outputs["cause_logits"].shape, (4, 10))

    def test_forward_accepts_shared_contract_without_global_features(self) -> None:
        config = BstanModelConfig(
            input_dim=6,
            global_dim=0,
            num_nodes=5,
            gat_hidden=8,
            gat_heads=2,
            gru_hidden=8,
            dropout=0.0,
        )
        model = BstanGatGru(config)
        batch = self._batch()
        batch["global_features"] = torch.empty(4, 4, 0)
        outputs = model(**self._inputs(batch))
        self.assertEqual(outputs["occurrence_logit"].shape, (4,))

    def test_loss_without_positive_samples_is_finite(self) -> None:
        model = BstanGatGru(self.config)
        batch = self._batch(positive=False)
        outputs = model(**self._inputs(batch))
        loss, components = compute_multitask_loss(
            outputs, batch, BstanLossConfig(), pos_weight=torch.tensor(1.0)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(components["node"]), 0.0)
        self.assertTrue(torch.isfinite(components["remain_hot"]))
        loss.backward()

    def test_two_epoch_synthetic_overfit_smoke(self) -> None:
        model = BstanGatGru(self.config)
        batch = self._batch(positive=True)
        config = BstanLossConfig(
            lambda_node=0.0,
            lambda_time_to_start=0.0,
            lambda_remain_score=0.0,
            lambda_remain_hot=0.0,
            lambda_remain_len=0.0,
            lambda_cause=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        with torch.no_grad():
            initial, _ = compute_multitask_loss(
                model(**self._inputs(batch)), batch, config
            )
        for _epoch in range(2):
            for _step in range(8):
                optimizer.zero_grad()
                loss, _ = compute_multitask_loss(
                    model(**self._inputs(batch)), batch, config
                )
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            final, _ = compute_multitask_loss(
                model(**self._inputs(batch)), batch, config
            )
        self.assertLess(float(final), float(initial))

    def test_checkpoint_round_trip_preserves_predictions(self) -> None:
        model = BstanGatGru(self.config).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        batch = self._batch()
        with torch.no_grad():
            expected = model(**self._inputs(batch))["occurrence_logit"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=2,
                best_validation_hot_f1=0.5,
                model_config=self.config,
                loss_config=BstanLossConfig(),
                train_config=BstanTrainConfig(),
                metadata={"dataset_manifest_sha256": "test"},
            )
            loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
            loaded.eval()
            with torch.no_grad():
                actual = loaded(**self._inputs(batch))["occurrence_logit"]
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertTrue(torch.equal(expected, actual))

    def test_no_event_baseline_pr_auc_equals_prevalence(self) -> None:
        import numpy as np

        arrays = {
            "y_occurrence": np.array([0, 0, 1, 0, 1]),
            "occurrence_probability": np.zeros(5),
            "positive_mask": np.array([False, False, True, False, True]),
            "node_probabilities": np.array([[0.6, 0.3, 0.1]] * 5, dtype=np.float64),
            "y_node": np.array([-1, -1, 0, -1, 1]),
            "time_to_start": np.array([0, 0, 20, 0, 40]),
            "y_time_to_start": np.array([0, 0, 30, 0, 30]),
            "y_cause": np.array([-1, -1, 1, -1, 1]),
            "cause_predictions": np.array([0, 0, 1, 0, 1]),
        }
        metrics, confusion = compute_metrics(arrays, cause_class_count=2)
        self.assertAlmostEqual(metrics["no_event_baseline"]["pr_auc"], 0.4)
        self.assertEqual(confusion.sum(), 2)

    def test_occurrence_threshold_is_selected_on_validation_predictions(self) -> None:
        import numpy as np

        threshold = select_f1_threshold(
            np.array([0, 1, 0, 1]),
            np.array([0.20, 0.45, 0.40, 0.80]),
        )

        self.assertEqual(threshold, 0.45)


if __name__ == "__main__":
    unittest.main()
