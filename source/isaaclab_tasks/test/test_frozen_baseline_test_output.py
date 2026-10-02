"""Run the real test-report CLI flow with inference stubbed; no ML dependencies.

Run directly with python3 to avoid IsaacLab package initialization by pytest.
"""
from contextlib import nullcontext, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = (Path(__file__).resolve().parents[1] /
          "isaaclab_tasks/direct/hc_factory/tools/evaluate_frozen_baseline_test.py")


def load_evaluator():
    package = types.ModuleType("factory_baselines")
    package.__path__ = []
    modules = {"factory_baselines": package}
    for name in ("protocol_20260913", "torch_trainer", "b2_xgboost"):
        module = types.ModuleType("factory_baselines." + name)
        setattr(package, name, module)
        modules[module.__name__] = module
    package.b2_xgboost.B2XGBoostConfig = object
    spec = importlib.util.spec_from_file_location("frozen_test_output_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class FrozenTestOutputTests(unittest.TestCase):
    def run_report(self, empty_upcoming=False):
        evaluator = load_evaluator()
        # Other fields are inert values; distinct recalls detect wrong mappings.
        station = dict.fromkeys(evaluator.COMPACT_STATION_FIELDS, 1.0)
        station.update(report_threshold_used=0.8, will15_precision=0.8,
                       will15_recall=0.6, will15_f1=0.6857142857142857,
                       who_recall_upcoming=0.4, report_recall_upcoming=0.2,
                       who_recall_ongoing=0.9, report_recall_ongoing=0.7)
        remain = dict.fromkeys(evaluator.COMPACT_REMAIN_FIELDS, 2.5)
        cause = dict(cause_acc=0.8, cause_macro_recall=0.6, cause_n=10)
        if empty_upcoming:
            station.update(n_matched_who_upcoming=0, n_matched_report_upcoming=0,
                           who_recall_upcoming=0, report_recall_upcoming=0,
                           time_mae_sample_count_upcoming=0,
                           start_mae_upcoming=0, dur_mae_upcoming=0,
                           start_mae_upcoming_minutes=None,
                           dur_mae_upcoming_minutes=None)
        metrics = dict(station_report=station, remain=remain, cause=cause)
        original = copy.deepcopy(metrics)
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "BSTAN_isaac_factory"
            dataset = repo / "dataset"
            dataset.mkdir(parents=True)
            (repo / ".git").mkdir()
            (dataset / "dataset_manifest.json").write_text("{}")
            report = repo / "report.json"
            tasks = {(model, cap): {"model": model,
                                   "output_dir": str(dataset / "models" / model)}
                     for cap in (5, 10, 15) for model in ("B2", "B3", "B4", "B5")}
            def git_output(args, **kwargs):
                return "dev_xwt\n" if "--show-current" in args else "fixture-commit\n"
            output = io.StringIO()
            with (patch.object(sys, "argv", [str(SCRIPT), "--dataset_dir", str(dataset),
                                              "--report", str(report)]),
                  patch("subprocess.check_output", side_effect=git_output),
                  patch.object(evaluator, "_tasks", return_value=tasks),
                  patch.object(evaluator, "_stage_artifacts", side_effect=lambda task, cap:
                               nullcontext((Path(task["output_dir"]), "fixture-archive", 0.8))),
                  patch.object(evaluator, "_evaluate_b2", return_value=metrics) as b2,
                  patch.object(evaluator, "_evaluate_torch", return_value=metrics) as torch,
                  redirect_stdout(output)):
                evaluator.main()
            self.assertEqual(b2.call_count, 3)
            self.assertEqual(torch.call_count, 9)
            data = json.loads(report.read_text())
            lines = output.getvalue().splitlines()
            self.assertTrue(lines[-1].startswith("TEST_ALL_COMPLETE "))
        self.assertEqual(metrics, original)
        self.assertEqual(len(data["tasks"]), 12)
        self.assertFalse(data["training_performed"])
        self.assertFalse(data["threshold_or_checkpoint_selection"])
        stage_lines = [line for line in lines if line.startswith("TEST_STAGE_COMPLETE ")]
        self.assertEqual(len(stage_lines), 12)
        for line, task in zip(stage_lines, data["tasks"]):
            _, model, cap, payload = line.split(" ", 3)
            logged = json.loads(payload)
            self.assertEqual((model, int(cap)), (task["model"], task["max_start"]))
            self.assertEqual(logged["test_who_recall_upcoming"], station["who_recall_upcoming"])
            self.assertEqual(logged["test_report_recall_upcoming"], station["report_recall_upcoming"])
            self.assertEqual(logged["test_report_recall_ongoing"], 0.7)
            self.assertEqual(logged["test_ongoing_who_recall"], 0.9)
            self.assertEqual(task["test"], original)
            self.assertIsNone(task["test_cause_majority_acc"])
            self.assertEqual(task["test_start_mae_upcoming_minutes"],
                             station["start_mae_upcoming_minutes"])

    def test_all_twelve_tasks_log_and_write_report(self):
        self.run_report()

    def test_empty_matches_preserve_null_mae_and_optional_cause(self):
        self.run_report(empty_upcoming=True)


if __name__ == "__main__":
    unittest.main()
