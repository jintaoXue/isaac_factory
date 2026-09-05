"""CPU tests for B3 LSTM and B4 GCN-GRU baselines."""

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

from factory_baselines.b3_lstm import B3Lstm, B3ModelConfig  # noqa: E402
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig  # noqa: E402
from factory_baselines.torch_losses import MultiTaskLossConfig  # noqa: E402
from factory_baselines.torch_trainer import (  # noqa: E402
    TorchTrainConfig,
    load_checkpoint,
    save_checkpoint,
)


class TestB3B4Models(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        batch_size, time_steps, node_count, input_dim = 3, 4, 5, 6
        node_mask = torch.ones(batch_size, node_count, dtype=torch.bool)
        node_mask[:, -1] = False
        target_mask = node_mask.clone()
        target_mask[:, -2] = False
        self.batch = {
            "x": torch.randn(batch_size, time_steps, node_count, input_dim),
            "adjacency": torch.ones(
                batch_size, node_count, node_count, dtype=torch.bool
            ),
            "node_mask": node_mask,
            "target_node_mask": target_mask,
            "global_features": torch.empty(batch_size, time_steps, 0),
            "jobs_remaining": torch.full((batch_size,), 2.0),
            "jobs_total": torch.full((batch_size,), 3.0),
        }
        self.configs = {
            "b3_lstm": B3ModelConfig(
                input_dim=input_dim,
                global_dim=0,
                num_nodes=node_count,
                lstm_hidden=8,
                node_hidden=8,
                node_embedding=4,
                dropout=0.0,
                max_remain_windows=8,
                num_causes=3,
            ),
            "b4_gcn_gru": B4ModelConfig(
                input_dim=input_dim,
                global_dim=0,
                num_nodes=node_count,
                gcn_hidden=8,
                gru_hidden=8,
                dropout=0.0,
                max_remain_windows=8,
                num_causes=3,
            ),
        }

    def test_forward_contract_and_masks(self) -> None:
        for model in (
            B3Lstm(self.configs["b3_lstm"]),
            B4GcnGru(self.configs["b4_gcn_gru"]),
        ):
            with self.subTest(model=type(model).__name__):
                outputs = model(**self.batch)
                self.assertEqual(outputs["remain_score"].shape, (3, 8, 5, 1))
                self.assertEqual(outputs["remain_hot_logit"].shape, (3, 8, 5))
                self.assertEqual(outputs["cause_logits"].shape, (3, 3))
                self.assertEqual(outputs["event_will_logit"].shape, (3, 5))
                self.assertEqual(outputs["event_start_logit"].shape, (3, 5, 8))
                self.assertEqual(outputs["event_duration"].shape, (3, 5))

    def test_b3_ignores_adjacency(self) -> None:
        model = B3Lstm(self.configs["b3_lstm"]).eval()
        identity = torch.eye(5, dtype=torch.bool)[None].expand(3, -1, -1)
        with torch.no_grad():
            full = model(**self.batch)["node_hidden"]
            changed = model(**{**self.batch, "adjacency": identity})["node_hidden"]
        self.assertTrue(torch.equal(full, changed))

    def test_b4_uses_adjacency(self) -> None:
        model = B4GcnGru(self.configs["b4_gcn_gru"]).eval()
        identity = torch.eye(5, dtype=torch.bool)[None].expand(3, -1, -1)
        with torch.no_grad():
            full = model(**self.batch)["node_hidden"]
            changed = model(**{**self.batch, "adjacency": identity})["node_hidden"]
        self.assertFalse(torch.allclose(full, changed))

    def test_b4_residual_preserves_station_identity_on_dense_graph(self) -> None:
        model = B4GcnGru(self.configs["b4_gcn_gru"]).eval()
        with torch.no_grad():
            hidden = model(**self.batch)["node_hidden"]

        self.assertFalse(torch.allclose(hidden[:, 0], hidden[:, 1]))

    def test_training_profile_must_be_named(self) -> None:
        self.assertEqual(TorchTrainConfig().training_profile, "baseline_fair_v2")
        with self.assertRaisesRegex(ValueError, "training_profile"):
            TorchTrainConfig(training_profile="")

    def test_checkpoint_registry_round_trip(self) -> None:
        cases = (
            ("b3_lstm", B3Lstm(self.configs["b3_lstm"])),
            ("b4_gcn_gru", B4GcnGru(self.configs["b4_gcn_gru"])),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for model_kind, model in cases:
                with self.subTest(model_kind=model_kind):
                    path = Path(temp_dir) / f"{model_kind}.pt"
                    model.eval()
                    with torch.no_grad():
                        expected = model(**self.batch)["event_will_logit"]
                    save_checkpoint(
                        path=path,
                        model=model,
                        optimizer=None,
                        epoch=1,
                        best_validation_report_f1=0.2,
                        model_kind=model_kind,
                        model_config=model.config,
                        loss_config=MultiTaskLossConfig(),
                        train_config=TorchTrainConfig(),
                        metadata={"dataset_manifest_sha256": "test"},
                    )
                    loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
                    loaded.eval()
                    with torch.no_grad():
                        actual = loaded(**self.batch)["event_will_logit"]
                    self.assertEqual(checkpoint["model_kind"], model_kind)
                    self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
