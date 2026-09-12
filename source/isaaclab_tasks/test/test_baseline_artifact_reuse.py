from pathlib import Path
from dataclasses import asdict
import json
import sys
import zipfile

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))

from factory_baselines.artifacts import archive_files
from rebuild_dense_factory_benchmark import extend_splits, identity, verify_export_alignment
from factory_baselines.dataset import _discover_groups, build_factory_baseline_dataset, load_shared_dataset
from bn_agg.pipeline import process_env_dir
from factory_bn.export_dataset import export_runs
import test_bottleneck_data_quality as raw_fixture
import test_factory_baseline_dataset as dataset_fixture
import train_dense_baseline_control as control


def test_prepared_dataset_matches_export_and_preserves_explicit_split(tmp_path):
    run = dataset_fixture.TestFactoryBaselineDataset()._make_run(tmp_path)
    groups = _discover_groups([run], tmp_path / "derived")
    prepared = [(run.name + f"__episode_{g['episode_id']:02d}", g["feature_rows"], {}, [], g["job_kpi_rows"])
                for g in groups]
    out = tmp_path / "output"
    export_runs([run], out, write_atomic=False, prepared_episodes=prepared)
    frozen = {"train": [g["group_id"] for g in groups[:4]],
              "validation": [groups[4]["group_id"]], "test": [groups[5]["group_id"]]}
    result = build_factory_baseline_dataset(
        [run], out, tmp_path / "derived", out, input_windows=12,
        episode_groups=groups, frozen_split_groups=frozen,
    )
    for split, ids in frozen.items():
        assert set(result["split"][split]["group_ids"]) == set(ids)
    assert verify_export_alignment(result, out)["samples"] == 48
    path = out / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["validation"] = "pending_shared_bundle_alignment"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unexpected validation"):
        load_shared_dataset(out)
    result["payload"]["jobs_remaining"][0] += 1
    with pytest.raises(AssertionError):
        verify_export_alignment(result, out)


def test_in_memory_aggregation_and_export_equal_csv_path(tmp_path):
    run, env = raw_fixture.TestRawDataAudit()._make_episode(tmp_path)
    before = set(run.rglob("*"))
    tables = process_env_dir(env, None, [60.0], 180, .55, 8)
    assert set(run.rglob("*")) == before
    process_env_dir(env, run / "derived" / env.parent.name / env.name, [60.0], 180, .55, 8)
    csv_bundle, memory_bundle = tmp_path / "csv_bundle", tmp_path / "memory_bundle"
    export_runs([run], csv_bundle, write_atomic=False)
    prepared = [(run.name + "__" + env.parent.name, tables["features"],
                 {int(row["window_index"]): row for row in tables["labels"]},
                 sorted(tables["events"], key=lambda row: row["start_s"]), tables["job_kpi"])]
    export_runs([run], memory_bundle, write_atomic=False, prepared_episodes=prepared)
    with np.load(csv_bundle / "episodes.npz") as disk, np.load(memory_bundle / "episodes.npz") as memory:
        assert set(disk.files) == set(memory.files)
        for key in disk.files:
            np.testing.assert_array_equal(disk[key], memory[key], err_msg=key)


def test_archive_keeps_old_bytes_and_unrelated_files(tmp_path):
    (tmp_path / "dataset.pt").write_bytes(b"old checkpoint data")
    (tmp_path / "notes.txt").write_text("keep")
    archive = archive_files(tmp_path, ["dataset.pt"], "before_dense.zip")
    assert not (tmp_path / "dataset.pt").exists()
    assert (tmp_path / "notes.txt").read_text() == "keep"
    assert not any(path.is_dir() for path in tmp_path.iterdir())
    with zipfile.ZipFile(archive) as saved:
        assert saved.read("dataset.pt") == b"old checkpoint data"
        assert "dataset.pt" in json.loads(saved.read("archive_manifest.json"))


def test_existing_archive_never_overwritten(tmp_path):
    (tmp_path / "dataset.pt").write_bytes(b"preserve")
    (tmp_path / "before.zip").write_bytes(b"previous archive")
    with pytest.raises(FileExistsError):
        archive_files(tmp_path, ["dataset.pt"], "before.zip")
    assert (tmp_path / "dataset.pt").read_bytes() == b"preserve"


def test_archive_rejects_external_or_directory_inputs(tmp_path):
    with pytest.raises(ValueError):
        archive_files(tmp_path, ["../other"], "before.zip")
    (tmp_path / "model").mkdir()
    with pytest.raises(ValueError):
        archive_files(tmp_path, ["model"], "before.zip")


def test_extension_keeps_old_test_and_validation_membership():
    old_rows = [{"run_id": "original", "env_id": 0, "episode_id": i, "run_dir": "/raw/old"} for i in range(3)]
    old = {name: {"group_ids": [identity(row)]} for name, row in zip(("train", "validation", "test"), old_rows)}
    new_rows = [{"run_id": "new", "env_id": 0, "episode_id": i, "run_dir": "/raw/new"} for i in range(5)]
    result = extend_splits(old, old_rows + new_rows, 42)
    assert {name: len(groups) for name, groups in result.items()} == {"train": 4, "validation": 2, "test": 2}
    for name, values in old.items():
        assert set(values["group_ids"]).issubset(result[name])
    with pytest.raises(ValueError, match="no longer passes"):
        extend_splits(old, old_rows[:2] + new_rows, 42)
    with pytest.raises(ValueError, match="Duplicate physical"):
        extend_splits(old, old_rows + new_rows + new_rows[:1], 42)


@pytest.mark.parametrize("model,batch_size", [("B4", 24), ("B5", 16)])
@pytest.mark.parametrize("variant", control.DENSE_VARIANTS)
def test_control_archives_stale_test_and_starts_fresh_validation_only(tmp_path, monkeypatch, model, batch_size, variant):
    repo = tmp_path / "BSTAN_isaac_factory"
    dataset, output = repo / "dataset", repo / "models"
    dataset.mkdir(parents=True)
    output.mkdir()
    (dataset / "dataset_manifest.json").write_text("{}")
    (output / "best.pt").write_bytes(b"previous checkpoint")
    (output / "metrics_test.json").write_text("previous test")
    (output / "notes.md").write_text("keep notes")
    directories = {p for p in repo.rglob("*") if p.is_dir()}
    monkeypatch.setattr(control.subprocess, "check_output", lambda command, **kwargs:
                        str(repo) if "--show-toplevel" in command else "dev_xwt" if "branch" in command else "abcdef")
    monkeypatch.setattr(control, "load_shared_dataset", lambda path:
                        ({}, {"shared_bundle_alignment": {"status": "passed"}}))
    # Feature construction is separately exercised against real tensors. This test
    # mocks the dataset and checks archival and launch boundaries for every arm.
    monkeypatch.setattr("factory_baselines.precursor.attach_precursor", lambda payload, *args: (payload, None))
    monkeypatch.setattr("factory_baselines.onset_history.attach_onset_history", lambda payload, *args: (payload, None))
    calls = []
    def train(**kwargs):
        calls.append(kwargs)
        assert not (output / "metrics_test.json").exists()
        assert not (output / "best.pt").exists()
        return {"status": "validation_completed"}
    monkeypatch.setattr(control, "train_torch_baseline", train)
    result = control.run_control(model, dataset, output, "dense", 42, "cpu", variant)
    assert result["status"] == "validation_completed"
    assert calls[0]["train_config"].evaluate_test is False
    assert calls[0]["train_config"].batch_size == batch_size
    assert "warm_start_checkpoint" not in calls[0]
    assert calls[0]["model_overrides"]["temporal_readout"] == (
        "last_attention" if variant == "temporal_attention" else "last_mean"
    )
    assert calls[0]["model_overrides"]["event_context"] == (variant == "graph_context")
    assert calls[0]["model_overrides"]["event_head"] == ("three_class" if variant == "three_class" else "binary")
    assert calls[0]["train_config"].training_profile == f"dense_{variant}_v2"
    assert calls[0]["loss_config"].event_will_upcoming_pos_weight == (
        12.0 if variant == "upcoming_weighted" else 4.0
    )
    assert (output / "notes.md").read_text() == "keep notes"
    assert {p for p in repo.rglob("*") if p.is_dir()} == directories
    with zipfile.ZipFile(output / "model_before_dense.zip") as archive:
        assert archive.read("best.pt") == b"previous checkpoint"
        assert archive.read("metrics_test.json") == b"previous test"
    with pytest.raises(FileExistsError):
        control.run_control(model, dataset, output, "dense", 42, "cpu")
    with pytest.raises(ValueError, match="existing directory"):
        control.run_control(model, dataset, output / "missing", "dense2", 42, "cpu")


@pytest.mark.parametrize("model", ["B4", "B5"])
def test_dense_candidates_are_single_variable_and_leave_scoring_unchanged(model):
    configurations = {}
    for variant in control.DENSE_VARIANTS:
        training, overrides, loss = control.dense_configuration(model, variant, 42, "cpu")
        training_values = asdict(training)
        training_values.pop("training_profile")
        configurations[variant] = (training_values, overrides, loss.to_dict())
    base, context, weighted, three_class = (configurations[name] for name in
                                          ("history_control", "graph_context", "upcoming_weighted", "three_class"))
    assert base[0] == context[0] == weighted[0] == three_class[0]
    assert base[0]["evaluate_test"] is False
    assert base[0]["event_oversample_factor"] == 1
    assert base[2] == context[2]
    assert {k for k in base[1] if base[1][k] != context[1][k]} == {"event_context"}
    assert base[1] == weighted[1]
    assert {k for k in base[2] if base[2][k] != weighted[2][k]} == {"event_will_upcoming_pos_weight"}
    assert base[2] == three_class[2]
    assert {k for k in base[1] if base[1][k] != three_class[1][k]} == {"event_head"}
    near, far, onset = (configurations[name] for name in ("near_precursor", "far_precursor", "onset_aux"))
    assert near[0] == far[0] == onset[0] == base[0]
    assert near[2] == far[2] == base[2]
    assert {k for k in near[1] if near[1][k] != far[1][k]} == {"event_precursor"}
    assert {**near[1], "event_onset_aux": True} == onset[1]
    assert {**near[2], "lambda_event_onset_aux": 1.0} == onset[2]
    joint = configurations["joint_onset"]
    assert joint[0] == onset[0] and joint[2] == onset[2]
    assert joint[1] == {**onset[1], "event_onset_joint": True}
    with pytest.raises(ValueError, match="registered dense variant"):
        control.dense_configuration(model, "unknown", 42, "cpu")
