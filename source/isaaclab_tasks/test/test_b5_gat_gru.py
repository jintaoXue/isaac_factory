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

from factory_baselines.torch_losses import (  # noqa: E402
    MultiTaskLossConfig,
    compute_multitask_loss,
)
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig  # noqa: E402
from factory_baselines.metrics import (  # noqa: E402
    REPORT_THRESHOLD_SWEEP,
    _binary_metrics,
    compute_metrics,
    hot_grid_metrics,
    select_f1_threshold,
    select_report_threshold,
    training_cause_majority,
)
from factory_baselines.torch_trainer import (  # noqa: E402
    TorchTrainConfig,
    load_checkpoint,
    save_checkpoint,
)


class TestB5GatGru(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = B5ModelConfig(
            input_dim=6,
            global_dim=2,
            num_nodes=5,
            gat_hidden=8,
            gat_heads=2,
            gru_hidden=8,
            dropout=0.0,
        )

    def _batch(self, positive: bool = True) -> dict[str, torch.Tensor]:
        del positive
        batch_size, time_steps, node_count = 4, 4, self.config.num_nodes
        node_mask = torch.ones(batch_size, node_count, dtype=torch.bool)
        node_mask[:, -1] = False
        target_node_mask = node_mask.clone()
        target_node_mask[:, -2] = False
        adjacency = torch.ones(batch_size, node_count, node_count, dtype=torch.bool)
        adjacency[:, -1, :] = False
        adjacency[:, :, -1] = False
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
            "y_cause": torch.zeros(batch_size, dtype=torch.int64),
            "target_remain_len": torch.full((batch_size,), 4, dtype=torch.int64),
            "remain_mask": torch.ones(batch_size, 15),
            "y_score": torch.rand(batch_size, 15, node_count, 1),
            "y_hot": torch.zeros(batch_size, 15, node_count),
            "occ_node_mask": target_node_mask.float(),
            "hist_last_hot": torch.zeros(batch_size, node_count),
            "event_will": torch.zeros(batch_size, node_count),
            "event_start": torch.zeros(batch_size, node_count, dtype=torch.int64),
            "event_duration": torch.zeros(batch_size, node_count),
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
        model = B5GatGru(self.config)
        batch = self._batch()
        outputs = model(**self._inputs(batch))
        self.assertEqual(outputs["node_hidden"].shape, (4, 5, 8))
        self.assertEqual(outputs["remain_score"].shape, (4, 15, 5, 1))
        self.assertEqual(outputs["remain_hot_logit"].shape, (4, 15, 5))
        self.assertEqual(outputs["event_will_logit"].shape, (4, 5))
        self.assertEqual(outputs["event_start_logit"].shape, (4, 5, 15))
        self.assertEqual(outputs["cause_logits"].shape, (4, 10))

    def test_forward_accepts_shared_contract_without_global_features(self) -> None:
        config = B5ModelConfig(
            input_dim=6,
            global_dim=0,
            num_nodes=5,
            gat_hidden=8,
            gat_heads=2,
            gru_hidden=8,
            dropout=0.0,
        )
        model = B5GatGru(config)
        batch = self._batch()
        batch["global_features"] = torch.empty(4, 4, 0)
        outputs = model(**self._inputs(batch))
        self.assertEqual(outputs["event_will_logit"].shape, (4, 5))

    def test_loss_without_positive_samples_is_finite(self) -> None:
        model = B5GatGru(self.config)
        batch = self._batch(positive=False)
        outputs = model(**self._inputs(batch))
        loss, components = compute_multitask_loss(
            outputs, batch, MultiTaskLossConfig(), pos_weight=torch.tensor(1.0)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(components["event_will"]))
        self.assertTrue(torch.isfinite(components["remain_hot"]))
        self.assertTrue(torch.isfinite(components["remain_dice"]))
        self.assertTrue(torch.isfinite(components["remain_iou"]))
        loss.backward()
        self.assertIsNotNone(model.input_projection.weight.grad)

    def test_occupancy_loss_excludes_non_target_nodes(self) -> None:
        batch = self._batch()
        outputs = B5GatGru(self.config)(**self._inputs(batch))
        _, original = compute_multitask_loss(outputs, batch, MultiTaskLossConfig())
        changed = dict(outputs)
        changed["remain_hot_logit"] = outputs["remain_hot_logit"].clone()
        changed["remain_hot_logit"][:, :, -2:] = 100.0
        _, modified = compute_multitask_loss(changed, batch, MultiTaskLossConfig())

        for name in ("remain_hot", "remain_dice", "remain_iou"):
            self.assertTrue(torch.equal(original[name], modified[name]))

    def test_event_start_and_duration_match_main_experiment_losses(self) -> None:
        batch = self._batch()
        batch["event_will"][0, 0] = 1.0
        batch["event_start"][0, 0] = 5
        batch["event_duration"][0, 0] = 9.0
        outputs = B5GatGru(self.config)(**self._inputs(batch))
        start_logits = torch.zeros_like(
            outputs["event_start_logit"], requires_grad=True
        )
        outputs["event_start_logit"] = start_logits
        outputs["event_duration"] = torch.ones_like(outputs["event_duration"])
        config = MultiTaskLossConfig(
            lambda_remain_hot=0.0,
            lambda_remain_dice=0.0,
            lambda_remain_iou=0.0,
            lambda_remain_len=0.0,
            lambda_cause=0.0,
            lambda_event_will=0.0,
            lambda_event_start=1.0,
            lambda_event_duration=0.0,
        )

        loss, components = compute_multitask_loss(outputs, batch, config)
        loss.backward()

        self.assertLess(float(start_logits.grad[0, 0, 4]), 0.0)
        expected_duration = torch.nn.functional.smooth_l1_loss(
            torch.log1p(torch.tensor(1.0)), torch.log1p(torch.tensor(9.0))
        )
        self.assertTrue(torch.allclose(components["event_duration"], expected_duration))

    def test_two_epoch_synthetic_overfit_smoke(self) -> None:
        model = B5GatGru(self.config)
        batch = self._batch(positive=True)
        config = MultiTaskLossConfig(
            lambda_remain_score=0.0,
            lambda_remain_hot=1.0,
            lambda_remain_len=0.0,
            lambda_cause=0.0,
            lambda_event_will=0.0,
            lambda_event_start=0.0,
            lambda_event_duration=0.0,
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
        model = B5GatGru(self.config).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        batch = self._batch()
        with torch.no_grad():
            expected = model(**self._inputs(batch))["event_will_logit"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=2,
                best_validation_report_f1=0.5,
                model_kind="b5_gat_gru",
                model_config=self.config,
                loss_config=MultiTaskLossConfig(),
                train_config=TorchTrainConfig(),
                metadata={"dataset_manifest_sha256": "test"},
            )
            loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
            loaded.eval()
            with torch.no_grad():
                actual = loaded(**self._inputs(batch))["event_will_logit"]
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertTrue(torch.equal(expected, actual))

    def test_ignored_cause_classes_are_excluded(self) -> None:
        import numpy as np

        arrays = {
            "y_cause": np.array([-1, 0, 4, 4]),
            "cause_predictions": np.array([0, 0, 4, 3]),
        }
        metrics, confusion = compute_metrics(arrays, cause_class_count=10)
        self.assertEqual(confusion.sum(), 2)
        self.assertEqual(metrics["cause"]["sample_count"], 2)

    def test_cause_protocol_uses_train_majority_and_six_report_classes(self) -> None:
        import numpy as np

        classes = [
            "machine_failure",
            "human_unavailable",
            "transport_delay",
            "material_shortage",
            "blocked_downstream",
            "starved_upstream",
            "unavailable",
            "high_utilization",
            "queue_buildup",
            "score_threshold",
        ]
        train_majority = training_cause_majority(np.array([0, 2, 2, 3, 9]), classes)
        arrays = {
            "y_cause": np.array([2, 2, 3, 0]),
            "cause_predictions": np.array([2, 3, 3, 0]),
        }
        metrics, confusion = compute_metrics(
            arrays,
            cause_class_count=len(classes),
            cause_classes=classes,
            cause_majority=train_majority,
        )

        self.assertEqual(train_majority, 2)
        self.assertEqual(confusion.sum(), 3)
        self.assertAlmostEqual(metrics["cause_macro_recall"], 0.75)
        self.assertAlmostEqual(metrics["cause_majority_acc"], 2 / 3)

    def test_hot_grid_metrics_use_valid_cells_and_resource_types(self) -> None:
        import numpy as np

        labels = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
        probabilities = np.array([[[0.9, 0.8], [0.1, 0.9]]], dtype=np.float32)
        metrics = hot_grid_metrics(
            labels,
            probabilities,
            remain_mask=np.array([[1, 1]]),
            occ_node_mask=np.array([[1, 1]]),
            type_masks={
                "machine": np.array([1, 0]),
                "gantry": np.array([0, 1]),
            },
        )

        self.assertAlmostEqual(metrics["hot_precision"], 2 / 3)
        self.assertEqual(metrics["hot_recall"], 1.0)
        self.assertEqual(metrics["hot_f1_machine"], 1.0)
        self.assertAlmostEqual(metrics["hot_f1_gantry"], 2 / 3)

    def test_report_threshold_is_selected_on_validation_with_precision_gate(
        self,
    ) -> None:
        import numpy as np

        y_hot = np.zeros((1, 15, 2), dtype=np.float32)
        y_hot[0, :8, 0] = 1.0
        metrics = select_report_threshold(
            y_hot,
            will_probability=np.array([[0.75, 0.65]], dtype=np.float32),
            start_index=np.zeros((1, 2), dtype=np.int64),
            duration_windows=np.full((1, 2), 8.0, dtype=np.float32),
            remain_mask=np.ones((1, 15), dtype=np.float32),
            occ_node_mask=np.ones((1, 2), dtype=np.float32),
            default_threshold=0.55,
            threshold_sweep=[0.55, 0.68, 0.80],
            min_precision=0.80,
        )

        self.assertEqual(metrics["report_threshold_used"], 0.68)
        self.assertEqual(metrics["report_precision"], 1.0)
        self.assertEqual(metrics["report_recall"], 1.0)

    def test_latest_main_evaluation_defaults(self) -> None:
        config = TorchTrainConfig()

        self.assertEqual(config.event_report_threshold, 0.68)
        self.assertEqual(config.report_threshold_sweep, REPORT_THRESHOLD_SWEEP)
        self.assertEqual(config.checkpoint_min_report_precision, 0.80)
        self.assertEqual(config.checkpoint_min_report_recall, 0.35)

    def test_occurrence_threshold_is_selected_on_validation_predictions(self) -> None:
        import numpy as np

        threshold = select_f1_threshold(
            np.array([0, 1, 0, 1]),
            np.array([0.20, 0.45, 0.40, 0.80]),
        )

        self.assertEqual(threshold, 0.45)

    def test_binary_roc_auc_uses_average_ranks_for_ties(self) -> None:
        import numpy as np

        metrics = _binary_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.5, 0.5, 0.9]),
        )

        self.assertEqual(metrics["roc_auc"], 0.875)


if __name__ == "__main__":
    unittest.main()
