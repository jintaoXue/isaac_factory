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
from diagnose_baseline_events import attach_node_catalog, load_diagnostic_checkpoint, parse_args, summarize_events
from factory_baselines.artifacts import archive_files
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig


class TestEventDiagnostics(unittest.TestCase):
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
