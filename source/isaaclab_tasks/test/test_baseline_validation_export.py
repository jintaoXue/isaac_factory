"""Validation export ignores test files and incomplete runs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from export_baseline_validation import export_validation


class TestValidationExport(unittest.TestCase):
    def test_export_only_completed_validation_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "tuning/search/candidate_context/seed42"
            run.mkdir(parents=True)
            (run / "run_summary.json").write_text(json.dumps({
                "status": "validation_completed", "baseline_id": "B4", "seed": 42,
            }))
            (run / "metrics_validation.json").write_text(json.dumps({"report_f1": .3}))
            (run / "metrics_test.json").write_text("must never be parsed")
            (run / "metrics.json").write_text("must never be parsed")
            (run / "config.json").write_text(json.dumps({
                "model": {}, "loss": {}, "training": {"evaluate_test": False},
                "metadata": {"git_commit": "abc", "dataset_manifest_sha256": "xyz"},
            }))
            unfinished = root / "tuning/search/candidate_focal/seed42"
            unfinished.mkdir(parents=True)
            (unfinished / "config.json").write_text("in progress")
            result = export_validation(root)
            self.assertEqual(result["completed_runs"], 1)
            self.assertEqual(result["runs"][0]["validation"], {"report_f1": .3})
            self.assertFalse(result["test_metrics_opened"])
            self.assertEqual(len(result["runs"][0]["metrics_sha256"]), 64)

    def test_xgboost_native_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "tuning/b2/candidate_event/seed42"
            run.mkdir(parents=True)
            (run / "run_summary.json").write_text(json.dumps({
                "status": "validation_completed", "baseline_id": "B2", "seed": 42,
            }))
            (run / "metrics_validation.json").write_text(json.dumps({"report_f1": .2}))
            (run / "config.json").write_text(json.dumps({
                "config": {"evaluate_test": False, "max_depth": 5},
                "dataset_manifest_sha256": "xyz",
            }))
            record = export_validation(root)["runs"][0]
            self.assertEqual(record["configuration"]["max_depth"], 5)
            self.assertEqual(record["provenance"]["dataset_manifest_sha256"], "xyz")


if __name__ == "__main__":
    unittest.main()
