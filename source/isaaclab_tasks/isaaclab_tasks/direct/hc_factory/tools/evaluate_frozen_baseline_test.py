#!/usr/bin/env python3
"""Evaluate the frozen B2--B5 matched checkpoints on test without refitting.

The v1 matched experiments selected checkpoints and thresholds on validation.
This entry point only loads those saved artifacts and scores the untouched test
split.  Temporary evaluation files are written under the system temp directory;
the only persistent output is the single JSON report requested by ``--report``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

from factory_baselines import protocol_20260913 as matched_protocol
from factory_baselines import torch_trainer
from factory_baselines.b2_xgboost import B2XGBoostConfig
from factory_baselines import b2_xgboost


MODELS = ("B2", "B3", "B4", "B5")
CAPS = (5, 10, 15)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@contextlib.contextmanager
def _stage_artifacts(task: dict[str, Any], cap: int) -> Iterator[tuple[Path, str, float]]:
    """Materialize the frozen artifacts for one stage without changing the repo.

    The matched runners keep the latest stage in ``output_dir`` and move the
    previous stages into the registered ZIP archive.  Test evaluation must use
    the stage-specific snapshot, otherwise every cap reads the final cap=15
    checkpoint and threshold.
    """
    output_dir = Path(task["output_dir"]).resolve()
    record_path = Path(task["record"]).resolve()
    record = _read(record_path)
    if record.get("status") != "validation_completed" or record.get("test_evaluated"):
        raise ValueError(f"Stage is not a frozen validation result: {record_path}")
    expected = record.get("artifact_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"Missing artifact hashes in stage record: {record_path}")
    summary = record.get("summary", {})
    threshold = float(summary["event_report_threshold"])
    registered_archive = Path(task["archive"]).resolve()
    # The archive registered on a task preserves the directory *before* that
    # stage started.  The completed stage itself is archived immediately before
    # the next stage starts, so use the next-stage archive (the same convention
    # as verify_baseline_matched_results.py).
    archive_path = registered_archive
    if cap < 15:
        suffix = f"_s{cap}.zip"
        if not registered_archive.name.endswith(suffix):
            raise ValueError(f"Unexpected registered archive name: {registered_archive}")
        archive_path = registered_archive.with_name(
            registered_archive.name[: -len(suffix)] + f"_s{cap + 5}.zip"
        )

    with tempfile.TemporaryDirectory(prefix=f"baseline_stage_{task['model'].lower()}_{cap}_") as temp:
        materialized = Path(temp)
        if cap < 15:
            if not archive_path.is_file():
                raise ValueError(f"Missing frozen stage archive: {archive_path}")
            with zipfile.ZipFile(archive_path) as archive:
                manifest = json.loads(archive.read("archive_manifest.json"))
                if manifest != expected:
                    raise ValueError(f"Archive manifest differs from stage record: {archive_path}")
                names = set(archive.namelist())
                for name, digest in expected.items():
                    if Path(name).name != name or name not in names:
                        raise ValueError(f"Missing or unsafe archive artifact: {name}")
                    value = archive.read(name)
                    if _sha_bytes(value) != digest:
                        raise ValueError(f"Archive artifact hash mismatch: {archive_path}:{name}")
                    (materialized / name).write_bytes(value)
            origin = str(archive_path)
        else:
            for name, digest in expected.items():
                if Path(name).name != name:
                    raise ValueError(f"Unsafe stage artifact name: {name}")
                source = output_dir / name
                if not source.is_file() or _sha(source) != digest:
                    raise ValueError(f"Current stage artifact hash mismatch: {source}")
                (materialized / name).write_bytes(source.read_bytes())
            origin = str(output_dir)
        yield materialized, origin, threshold


def _tasks(dataset_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    found: dict[tuple[str, int], dict[str, Any]] = {}
    plan_names = (
        "baseline_matched_protocol_training20260913_plan.json",
        "remaining_matched20260914_plan.json",
    )
    for plan_name in plan_names:
        plan_path = dataset_dir / plan_name
        if not plan_path.is_file():
            raise ValueError(f"Missing completed matched plan: {plan_path}")
        plan = _read(plan_path)
        for task in plan.get("tasks", []):
            identity = (str(task["model"]), int(task["max_start"]))
            if identity in {(model, cap) for cap in CAPS for model in MODELS}:
                if identity in found:
                    raise ValueError(f"Duplicate registered task: {identity}")
                found[identity] = task
    expected = {(model, cap) for cap in CAPS for model in MODELS}
    if set(found) != expected:
        raise ValueError(f"Expected twelve registered tasks, found {sorted(found)}")
    return found


def _load_b2_head(model_dir: Path, metadata: dict[str, Any]) -> b2_xgboost._Head:
    kind = metadata["kind"]
    if kind == "constant":
        return b2_xgboost._Head(
            kind=kind,
            constant=metadata.get("constant"),
            classes=metadata.get("classes"),
        )
    path = model_dir / str(metadata["path"])
    XGBClassifier, XGBRegressor = b2_xgboost._require_xgboost()
    if kind in {"xgboost_binary", "xgboost_multiclass"}:
        model = XGBClassifier()
    elif kind == "xgboost_regression":
        model = XGBRegressor()
    else:
        raise ValueError(f"Unknown saved B2 head kind: {kind}")
    model.load_model(str(path))
    return b2_xgboost._Head(
        kind=kind,
        model=model,
        constant=metadata.get("constant"),
        classes=metadata.get("classes"),
    )


def _evaluate_b2(
    dataset_dir: Path,
    model_dir: Path,
    cap: int,
    frozen_threshold: float,
) -> dict[str, Any]:
    """Reuse the B2 evaluator while replacing every fit with a saved head."""
    saved_config = _read(model_dir / "config.json")
    config_values = dict(saved_config["config"])
    if int(config_values.get("event_max_start_windows", cap)) != cap:
        raise ValueError(f"B2 artifact config cap does not match task cap: {model_dir}")
    config_values["evaluation_protocol"] = matched_protocol.VERSION
    config_values["event_max_start_windows"] = cap
    config_values["evaluate_test"] = False
    # The threshold was selected on validation and is frozen for test.  Restrict
    # the validation candidate list to this value so the test pass performs no
    # new threshold selection.
    config_values["event_report_threshold"] = frozen_threshold
    config_values["report_threshold_sweep"] = (frozen_threshold,)
    config = B2XGBoostConfig(**config_values)
    config.evaluate_test = True

    models = saved_config["models"]
    queue_classifier = [
        _load_b2_head(model_dir, models["cause"]),
        _load_b2_head(model_dir, models["remain_hot"]),
        _load_b2_head(model_dir, models["event_will"]),
        _load_b2_head(model_dir, models["event_start"]),
    ]
    queue_regressor = [
        _load_b2_head(model_dir, models["remain_len"]),
        _load_b2_head(model_dir, models["remain_score"]),
        _load_b2_head(model_dir, models["event_duration"]),
    ]
    original_classifier = b2_xgboost._fit_classifier
    original_regressor = b2_xgboost._fit_regressor

    def use_saved_classifier(*args: Any, **kwargs: Any) -> b2_xgboost._Head:
        if not queue_classifier:
            raise AssertionError("B2 requested more classifier fits than saved heads")
        return queue_classifier.pop(0)

    def use_saved_regressor(*args: Any, **kwargs: Any) -> b2_xgboost._Head:
        if not queue_regressor:
            raise AssertionError("B2 requested more regressor fits than saved heads")
        return queue_regressor.pop(0)

    b2_xgboost._fit_classifier = use_saved_classifier
    b2_xgboost._fit_regressor = use_saved_regressor
    try:
        with tempfile.TemporaryDirectory(prefix="baseline_b2_test_") as temp:
            b2_xgboost.train_b2_xgboost(dataset_dir, Path(temp), config)
            metrics = _read(Path(temp) / "metrics.json")["test"]
    finally:
        b2_xgboost._fit_classifier = original_classifier
        b2_xgboost._fit_regressor = original_regressor
    if queue_classifier or queue_regressor:
        raise AssertionError("Saved B2 heads were not all consumed")
    return metrics


def _evaluate_torch(
    dataset_dir: Path,
    model_dir: Path,
    cap: int,
    frozen_threshold: float,
) -> dict[str, Any]:
    checkpoint = model_dir / "best.pt"
    checkpoint_data = __import__("torch").load(
        checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_cap = int(checkpoint_data["train_config"]["event_max_start_windows"])
    if checkpoint_cap != cap:
        raise ValueError(
            f"Checkpoint cap does not match task cap for {checkpoint}: "
            f"checkpoint={checkpoint_cap}, task={cap}"
        )
    checkpoint_threshold = float(checkpoint_data["metadata"]["event_report_threshold"])
    if not math.isclose(checkpoint_threshold, frozen_threshold, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError(
            f"Frozen threshold mismatch for {checkpoint}: "
            f"checkpoint={checkpoint_threshold}, record={frozen_threshold}"
        )
    with tempfile.TemporaryDirectory(prefix="baseline_torch_test_") as temp:
        return torch_trainer.evaluate_torch_checkpoint(
            dataset_dir=dataset_dir,
            checkpoint_path=checkpoint,
            output_dir=Path(temp),
            split_name="test",
            device_name="auto",
            batch_size=32,
            num_workers=0,
        )


def _compact(
    model: str,
    cap: int,
    model_dir: Path,
    artifact_origin: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    report = metrics["station_report"]
    result = {
        "model": model,
        "max_start": cap,
        "model_dir": str(model_dir),
        "artifact_origin": artifact_origin,
        "test": metrics,
        "selected_threshold": report["report_threshold_used"],
        "test_will15_precision": report["will15_precision"],
        "test_will15_recall": report["will15_recall"],
        "test_will15_f1": report["will15_f1"],
        "test_upcoming_strict_hits": report["n_matched_report_upcoming"],
        "test_upcoming_who_hits": report["n_matched_who_upcoming"],
        "test_upcoming_support": report["n_true_upcoming"],
        "test_ongoing_who_recall": report["who_recall_ongoing"],
        "test_remain_MAE_middle_weighted": metrics["remain"]["remain_len_mae_middle_weighted"],
        "test_cause_accuracy": metrics["cause"]["cause_acc"],
        "test_cause_macro_recall": metrics["cause"]["cause_macro_recall"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    report_path = args.report.resolve()
    if not dataset_dir.is_dir() or not (dataset_dir / "dataset_manifest.json").is_file():
        raise ValueError(f"Invalid matched dataset directory: {dataset_dir}")
    repo_root = dataset_dir
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    if repo_root.name != "BSTAN_isaac_factory":
        raise ValueError(f"Unexpected repository root: {repo_root}")
    if __import__("subprocess").check_output(
        ["git", "branch", "--show-current"], cwd=repo_root, text=True
    ).strip() != "dev_xwt":
        raise ValueError("Test evaluation must run on BSTAN_isaac_factory/dev_xwt")

    tasks = _tasks(dataset_dir)
    results: list[dict[str, Any]] = []
    for cap in CAPS:
        for model in MODELS:
            task = tasks[(model, cap)]
            model_dir = Path(task["output_dir"]).resolve()
            if not model_dir.is_relative_to((dataset_dir / "models").resolve()):
                raise ValueError(f"Model output is outside authorized models directory: {model_dir}")
            with _stage_artifacts(task, cap) as (stage_dir, artifact_origin, frozen_threshold):
                metrics = (
                    _evaluate_b2(dataset_dir, stage_dir, cap, frozen_threshold)
                    if model == "B2"
                    else _evaluate_torch(dataset_dir, stage_dir, cap, frozen_threshold)
                )
            result = _compact(model, cap, model_dir, artifact_origin, metrics)
            results.append(result)
            print(
                f"TEST_STAGE_COMPLETE {model} {cap} "
                + json.dumps({k: result[k] for k in (
                    "selected_threshold", "test_will15_precision", "test_will15_recall",
                    "test_will15_f1", "test_upcoming_strict_hits", "test_upcoming_support",
                )}, ensure_ascii=False, separators=(",", ":")),
                flush=True,
            )

    report = {
        "status": "frozen_baseline_test_evaluation_completed",
        "test_evaluated": True,
        "training_performed": False,
        "threshold_or_checkpoint_selection": False,
        "dataset_manifest_sha256": _sha(dataset_dir / "dataset_manifest.json"),
        "source_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "evaluation_scope": "Load v1 validation-selected checkpoints and thresholds; score test only; no refit, no new selection.",
        "tasks": results,
    }
    if not report_path.parent.is_dir():
        raise ValueError(f"Report parent must already exist: {report_path.parent}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"TEST_ALL_COMPLETE {report_path}", flush=True)


if __name__ == "__main__":
    main()
