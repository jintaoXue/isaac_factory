"""Pure PyTorch tests for the BSTAN Phase-C dataset builder."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader


TOOLS_DIR = (
    Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
)
sys.path.insert(0, str(TOOLS_DIR))

from bstan_baseline.dataset import BstanTensorDataset, build_bstan_dataset  # noqa: E402
from bstan_baseline.schema import CONTINUOUS_FEATURES, GLOBAL_FEATURES  # noqa: E402
from bstan_baseline.losses import BstanLossConfig  # noqa: E402
from bstan_baseline.trainer import BstanTrainConfig, train_bstan_baseline  # noqa: E402


class TestBstanDataset(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _make_run(self, root: Path) -> Path:
        run_dir = root / "run_seed42"
        for episode_id in range(6):
            raw_dir = run_dir / f"episode_{episode_id:02d}" / "env_00"
            derived_dir = (
                run_dir
                / "shared_bn_agg_v1"
                / f"episode_{episode_id:02d}"
                / "env_00"
            )
            config = {
                "run_id": "run_seed42",
                "env_id": 0,
                "episode_id": episode_id,
                "collector_version": "v0.3",
                "process_time_config": json.dumps(
                    {
                        "Product": {
                            "cut": {
                                "machine": "machine_a",
                                "required_materials": {"pipe": 1},
                            }
                        }
                    }
                ),
                "buffer_capacity_config": json.dumps(
                    {
                        "BlackStorage_00": {
                            "capacity": 2,
                            "supporting_materials": ["product_00_pipe"],
                        }
                    }
                ),
            }
            self._write_csv(raw_dir / "episode_config.csv", list(config), [config])
            (derived_dir / "shared_metadata.json").parent.mkdir(
                parents=True, exist_ok=True
            )
            (derived_dir / "shared_metadata.json").write_text(
                json.dumps(
                    {
                        "derived_contract_version": "tyx_bn_agg_v1",
                        "label_version": "tyx_bn_agg_event_v1",
                        "raw_contract_version": "tyx_raw_v0.3",
                        "scenario_id": f"scenario_{episode_id % 2}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            feature_rows = []
            label_rows = []
            for window_index in range(20):
                for node_id, resource_type in (
                    ("num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0", "machine"),
                    ("storage_BlackStorage_00", "buffer"),
                ):
                    row = {
                        "run_id": "run_seed42",
                        "env_id": 0,
                        "episode_id": episode_id,
                        "window_index": window_index,
                        "window_start_s": window_index * 60,
                        "window_end_s": (window_index + 1) * 60,
                        "window_size_s": 60,
                        "resource_id": node_id,
                        "resource_type": resource_type,
                    }
                    row.update(
                        {
                            feature_name: float(episode_id + window_index + 1)
                            for feature_name in CONTINUOUS_FEATURES
                        }
                    )
                    row.update(
                        {
                            feature_name: float(episode_id + window_index)
                            for feature_name in GLOBAL_FEATURES
                        }
                    )
                    feature_rows.append(row)

                positive = episode_id != 4 and window_index == 13
                label_rows.append(
                    {
                        "run_id": "run_seed42",
                        "env_id": 0,
                        "episode_id": episode_id,
                        "window_index": window_index,
                        "window_size_s": 60,
                        "window_start_s": window_index * 60,
                        "label_horizon_ready": 1,
                        "will_bottleneck": int(positive),
                        "future_bottleneck_object_id": (
                            "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0"
                            if positive else ""
                        ),
                        "future_bottleneck_type": "machine" if positive else "",
                        "time_to_start": 30 if positive else "",
                        "duration": 60 if positive else "",
                        "horizon_s": 180,
                    }
                )
            self._write_csv(
                derived_dir / "window_feature_table.csv",
                list(feature_rows[0]),
                feature_rows,
            )
            self._write_csv(
                derived_dir / "bottleneck_label.csv",
                list(label_rows[0]),
                label_rows,
            )
            job_rows = [
                {
                    "job_id": job_id,
                    "complete_s": 1140 + job_id,
                    "completed": 1,
                }
                for job_id in range(2)
            ]
            self._write_csv(
                derived_dir / "job_kpi.csv", list(job_rows[0]), job_rows
            )
        return run_dir

    def test_builds_fixed_graph_sequences_and_group_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self._make_run(root)
            out_dir = root / "dataset"
            result = build_bstan_dataset(
                run_dirs=[run_dir],
                out_dir=out_dir,
                window_size=60,
                stride=60,
                input_windows=12,
                horizon=180,
                seed=42,
            )
            payload = result["payload"]
            manifest = result["manifest"]

            self.assertEqual(payload["x"].shape[:3], (48, 12, 2))
            self.assertEqual(payload["x"].shape[-1], len(CONTINUOUS_FEATURES) + 2)
            self.assertEqual(payload["global_features"].shape, (48, 12, 0))
            self.assertEqual(payload["adjacency"].shape, (48, 2, 2))
            self.assertEqual(payload["target_node_mask"].shape, (48, 2))
            self.assertEqual(payload["observation_mask"].shape, (48, 12, 2))
            machine_index = manifest["node_ids"].index(
                "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0"
            )
            buffer_index = manifest["node_ids"].index("storage_BlackStorage_00")
            self.assertTrue(payload["target_node_mask"][:, machine_index].all())
            self.assertFalse(payload["target_node_mask"][:, buffer_index].any())
            self.assertTrue(torch.isfinite(payload["x"]).all())
            self.assertEqual(manifest["dataset_version"], "bstan_tyxbn_dataset_v2")
            self.assertEqual(
                manifest["prediction_target_version"], "factory_a1a3_remain_v1"
            )
            self.assertEqual(manifest["dataset_contract"], "tyx_bn_agg_v1")
            self.assertEqual(manifest["label_version"], "tyx_bn_agg_event_v1")
            self.assertEqual(manifest["target_node_category"], "process")
            self.assertEqual(manifest["positive_samples"], 5)
            for row in result["sample_rows"]:
                history = json.loads(row["input_window_indices"])
                self.assertEqual(len(history), 12)
                self.assertLess(max(history), row["anchor_window_index"])
            self.assertEqual(
                manifest["episode_counts"], {"train": 4, "validation": 1, "test": 1}
            )
            self.assertIn("buffer_affinity", manifest["edge_types"])

            split = result["split"]
            group_sets = {
                name: set(values["group_ids"]) for name, values in split.items()
            }
            self.assertFalse(group_sets["train"] & group_sets["validation"])
            self.assertFalse(group_sets["train"] & group_sets["test"])
            self.assertFalse(group_sets["validation"] & group_sets["test"])

            train_indices = payload["split_indices"]["train"]
            train_queue = payload["x"][train_indices, ..., 0]
            self.assertAlmostEqual(float(train_queue.mean()), 0.0, places=5)

            loader = DataLoader(
                BstanTensorDataset(payload, train_indices.tolist()), batch_size=3
            )
            batch = next(iter(loader))
            self.assertEqual(batch["x"].shape[1:], payload["x"].shape[1:])
            self.assertEqual(batch["adjacency"].dtype, torch.bool)
            self.assertEqual(batch["y_score"].shape[1:], (512, 2, 1))
            self.assertEqual(batch["y_hot"].shape[1:], (512, 2))
            self.assertEqual(batch["remain_mask"].shape[1:], (512,))

            expected_files = {
                "dataset.pt",
                "node_catalog.csv",
                "graph_edge_table.csv",
                "model_sample_index.csv",
                "split_manifest.json",
                "normalization.json",
                "dataset_manifest.json",
            }
            self.assertEqual(expected_files, {path.name for path in out_dir.iterdir()})
            loaded = torch.load(out_dir / "dataset.pt", map_location="cpu")
            self.assertTrue(torch.equal(payload["x"], loaded["x"]))

            repeated = build_bstan_dataset(
                run_dirs=[run_dir],
                out_dir=root / "dataset_repeated",
                window_size=60,
                stride=60,
                input_windows=12,
                horizon=180,
                seed=42,
            )
            self.assertEqual(result["split"], repeated["split"])
            self.assertTrue(torch.equal(payload["x"], repeated["payload"]["x"]))

            model_dir = root / "model"
            summary = train_bstan_baseline(
                dataset_dir=out_dir,
                output_dir=model_dir,
                model_overrides={
                    "gat_hidden": 8,
                    "gat_heads": 2,
                    "gru_hidden": 8,
                    "dropout": 0.0,
                },
                train_config=BstanTrainConfig(
                    batch_size=8,
                    max_epochs=1,
                    patience=1,
                    device="cpu",
                ),
                loss_config=BstanLossConfig(),
            )
            self.assertEqual(summary["status"], "completed")
            self.assertIn("test_hot_f1", summary)
            self.assertTrue((model_dir / "occupancy_events_test.csv").is_file())


if __name__ == "__main__":
    unittest.main()
