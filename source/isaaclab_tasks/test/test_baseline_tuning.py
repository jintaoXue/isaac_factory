"""Tests for validation-only baseline candidate selection."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = (
    Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
)
sys.path.insert(0, str(TOOLS_DIR))

from select_baseline_tuning import candidate_rank, summarize_candidate  # noqa: E402


class TestBaselineTuning(unittest.TestCase):
    @staticmethod
    def _write_run(
        candidate_dir: Path,
        seed: int,
        report_f1: float,
        upcoming_recall: float,
        include_test: bool = False,
    ) -> None:
        seed_dir = candidate_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        summary = {
            "status": "validation_completed",
            "seed": seed,
            "training_profile": candidate_dir.name,
            "checkpoint_constraint_met": False,
            "elapsed_seconds": 1.0,
        }
        validation = {
            "station_report": {
                "report_precision": 0.5,
                "report_recall": 0.3,
                "report_f1": report_f1,
                "who_f1": report_f1 + 0.01,
                "report_recall_ongoing": 0.6,
                "report_recall_upcoming": upcoming_recall,
            },
            "remain": {"hot_f1": 0.4, "hot_ap": 0.35},
            "event_will": {"pr_auc": 0.3},
        }
        metrics = {"validation": validation}
        if include_test:
            metrics["test"] = validation
        (seed_dir / "run_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (seed_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def test_ranking_uses_robust_f1_before_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stable = root / "candidate_stable"
            unstable = root / "candidate_unstable"
            self._write_run(stable, 42, 0.30, 0.05)
            self._write_run(stable, 43, 0.30, 0.05)
            self._write_run(unstable, 42, 0.10, 0.20)
            self._write_run(unstable, 43, 0.55, 0.20)

            stable_summary = summarize_candidate(stable, {42, 43})
            unstable_summary = summarize_candidate(unstable, {42, 43})

            self.assertGreater(
                candidate_rank(stable_summary), candidate_rank(unstable_summary)
            )
            self.assertAlmostEqual(stable_summary["report_f1_robust"], 0.30)
            self.assertAlmostEqual(unstable_summary["report_f1_robust"], 0.10)

    def test_rejects_missing_seed_and_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            incomplete = root / "candidate_incomplete"
            self._write_run(incomplete, 42, 0.3, 0.1)
            with self.assertRaisesRegex(ValueError, "expected seeds"):
                summarize_candidate(incomplete, {42, 43})

            leaked = root / "candidate_leaked"
            self._write_run(leaked, 42, 0.3, 0.1, include_test=True)
            with self.assertRaisesRegex(ValueError, "contains test metrics"):
                summarize_candidate(leaked, {42})


if __name__ == "__main__":
    unittest.main()
