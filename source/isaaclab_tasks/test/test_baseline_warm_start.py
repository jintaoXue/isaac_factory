"""Weights-only continuation guards, stage-zero selection, and trial budgets."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))

from factory_baselines import torch_trainer as trainer
from factory_baselines.dataset import build_factory_baseline_dataset
from factory_baselines.torch_losses import MultiTaskLossConfig
from factory_baselines.warm_start import load_warm_start_parent
from run_staged_baseline import ARMS, stage_configuration
from test_factory_baseline_dataset import TestFactoryBaselineDataset as DatasetFixture
import run_staged_baseline as staged


@pytest.fixture(scope="module")
def stage_parents(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("warm_parents")
    run = DatasetFixture()._make_run(root)
    dataset = root / "dataset"
    result = build_factory_baseline_dataset(
        run_dirs=[run], out_dir=dataset, derived_root=root / "derived",
        main_bundle=root / "main_bundle", window_size=60, stride=60,
        input_windows=12, horizon=180, seed=42,
    )
    models = {
        "b4_gcn_gru": {"gcn_hidden": 8, "gru_hidden": 8, "dropout": 0.0},
        "b5_gat_gru": {"gat_hidden": 8, "gat_heads": 2, "gru_hidden": 8, "dropout": 0.0},
    }
    for kind, overrides in models.items():
        trainer.train_torch_baseline(
            kind, dataset, root / kind, model_overrides=overrides,
            train_config=trainer.TorchTrainConfig(
                evaluate_test=False, batch_size=8, max_epochs=1, min_epochs=1,
                patience=1, device="cpu", seed=42,
            ),
        )
    return root, dataset, result, models


def parent_args(stage_parents, kind="b4_gcn_gru"):
    root, dataset, result, _ = stage_parents
    config = json.loads((root / kind / "config.json").read_text())
    return dict(
        model_kind=kind, model_config=config["model"], seed=42,
        dataset_manifest_sha256=hashlib.sha256((dataset / "dataset_manifest.json").read_bytes()).hexdigest(),
        train_sample_count=len(result["payload"]["split_indices"]["train"]),
    )


@pytest.mark.parametrize("kind", ("b4_gcn_gru", "b5_gat_gru"))
def test_loads_only_weights_with_hashed_complete_parent(stage_parents, kind):
    root, _, _, _ = stage_parents
    path = root / kind / "best.pt"
    state, proof = load_warm_start_parent(path, **parent_args(stage_parents, kind))
    source = torch.load(path, weights_only=False)
    assert source["optimizer_state_dict"]["state"]
    assert all(torch.equal(value, source["model_state_dict"][name]) for name, value in state.items())
    assert proof["mode"] == "weights_only_new_optimizer"
    assert proof["artifact_sha256"]["checkpoint"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert proof["epochs_trained"] == 1
    assert set(proof["artifact_sha256"]) == {"checkpoint", "config", "history", "summary", "validation"}


@pytest.mark.parametrize("field,value,match", [
    ("seed", 43, "seed"),
    ("model_kind", "b5_gat_gru", "model kind"),
    ("dataset_manifest_sha256", "0" * 64, "manifest"),
    ("model_config", {}, "model configuration"),
])
def test_rejects_wrong_identity(stage_parents, field, value, match):
    args = parent_args(stage_parents)
    args[field] = value
    with pytest.raises(ValueError, match=match):
        load_warm_start_parent(stage_parents[0] / "b4_gcn_gru/best.pt", **args)


@pytest.mark.parametrize("corruption", ["incomplete", "test_parent", "wrong_metrics", "nested", "nonfinite", "config"])
def test_rejects_invalid_parent_artifacts(stage_parents, tmp_path, corruption):
    source = stage_parents[0] / "b4_gcn_gru"
    target = tmp_path / "parent"
    shutil.copytree(source, target)
    path = target / "best.pt"
    if corruption == "incomplete":
        (target / "history.csv").write_text("epoch,train_total\n")
    elif corruption == "test_parent":
        summary = json.loads((target / "run_summary.json").read_text())
        summary["status"] = "completed"
        (target / "run_summary.json").write_text(json.dumps(summary))
    elif corruption == "wrong_metrics":
        metrics = json.loads((target / "metrics_validation.json").read_text())
        metrics["station_report"]["report_f1"] += 0.1
        (target / "metrics_validation.json").write_text(json.dumps(metrics))
    elif corruption == "config":
        config = json.loads((target / "config.json").read_text())
        config["training"]["batch_size"] += 1
        (target / "config.json").write_text(json.dumps(config))
    else:
        checkpoint = torch.load(path, weights_only=False)
        if corruption == "nested":
            checkpoint["metadata"]["warm_start_parent"] = {}
        else:
            next(iter(checkpoint["model_state_dict"].values())).fill_(float("nan"))
        torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        load_warm_start_parent(path, **parent_args(stage_parents))


@pytest.mark.parametrize("kind", ("b4_gcn_gru", "b5_gat_gru"))
def test_actual_second_stage_preserves_initial_weights_and_budget(stage_parents, tmp_path, kind):
    root, dataset, result, models = stage_parents
    source = root / kind / "best.pt"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "warm"
    summary = trainer.train_torch_baseline(
        kind, dataset, output, model_overrides=models[kind],
        train_config=trainer.TorchTrainConfig(
            evaluate_test=False, batch_size=8, max_epochs=1, min_epochs=1,
            patience=1, learning_rate=3.75e-5, device="cpu", seed=42,
        ),
        loss_config=MultiTaskLossConfig(event_will_upcoming_pos_weight=16),
        warm_start_checkpoint=source,
    )
    initial = torch.load(output / "initial.pt", weights_only=False)
    parent = torch.load(source, weights_only=False)
    assert initial["optimizer_state_dict"]["state"] == {}
    assert initial["epoch"] == 0
    assert all(torch.equal(value, parent["model_state_dict"][name])
               for name, value in initial["model_state_dict"].items())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert summary["status"] == "validation_completed"
    assert summary["initialization"] == "warm_start_weights_only"
    budget = summary["training_budget"]
    assert budget["cumulative_epochs_trained"] == 2
    assert budget["cumulative_optimizer_steps"] == 2 * budget["stage_optimizer_steps"]
    assert budget["cumulative_elapsed_seconds"] >= budget["stage_elapsed_seconds"]
    assert set(json.loads((output / "metrics.json").read_text())) == {"validation"}
    assert not list(output.glob("*test*"))
    saved = torch.load(output / "best.pt", weights_only=False)
    assert saved["metadata"]["training_budget"] == budget
    assert saved["metadata"]["warm_start_parent"]["artifact_sha256"]["checkpoint"] == before


def test_no_update_retains_stage_zero(stage_parents, tmp_path, monkeypatch):
    root, dataset, _, models = stage_parents
    def no_update(*args, **kwargs):
        return {"total": 0.0}
    monkeypatch.setattr(trainer, "_run_train_epoch", no_update)
    summary = trainer.train_torch_baseline(
        "b4_gcn_gru", dataset, tmp_path / "unchanged", model_overrides=models["b4_gcn_gru"],
        train_config=trainer.TorchTrainConfig(
            evaluate_test=False, batch_size=8, max_epochs=1, min_epochs=1, patience=1,
            device="cpu", seed=42, lr_schedule="none",
        ),
        warm_start_checkpoint=root / "b4_gcn_gru/best.pt",
    )
    assert summary["best_epoch"] == summary["checkpoint_epoch"] == 0
    assert not summary["warm_start_checkpoint_improved"]
    assert summary["best_validation_report_f1"] == summary["initial_validation_report_f1"]


def test_warm_tuning_rejects_test_and_parent_overwrite(stage_parents):
    root, dataset, _, _ = stage_parents
    parent = root / "b4_gcn_gru"
    with pytest.raises(ValueError, match="validation-only"):
        trainer.train_torch_baseline("b4_gcn_gru", dataset, root / "invalid_test",
                                    warm_start_checkpoint=parent / "best.pt")
    with pytest.raises(ValueError, match="overwrite"):
        trainer.train_torch_baseline(
            "b4_gcn_gru", dataset, parent, train_config=trainer.TorchTrainConfig(evaluate_test=False),
            warm_start_checkpoint=parent / "best.pt",
        )


@pytest.mark.parametrize("kind", ("B4", "B5"))
def test_preregistered_four_arms_preserve_architecture_and_other_losses(kind):
    repo = next(p for p in TOOLS.parents if (p / ".git").exists())
    snapshot = json.loads((repo / "doc/experiments/baseline_validation_v5_20260906_round4_complete.json").read_text())
    record = next(r for r in snapshot["runs"] if r["summary"]["baseline_id"] == kind
                  and "candidate_history/seed42" in r["run"])
    parent = record["configuration"]
    before = copy.deepcopy(parent)
    for arm in ARMS:
        config = stage_configuration(parent, arm, "test_stage", "cpu", 42)
        assert config["model"] == parent["model"]
        expected_loss = copy.deepcopy(parent["loss"])
        if arm.endswith("signal"):
            expected_loss["event_will_upcoming_pos_weight"] = 16
        assert config["loss"] == expected_loss
        assert not config["training"]["evaluate_test"]
        expected_lr = parent["training"]["learning_rate"] * (.25 if arm.startswith("warm") else 1)
        assert config["training"]["learning_rate"] == expected_lr
        assert config["training"]["max_epochs"] == (20 if arm.startswith("warm") else 80)
    assert parent == before


@pytest.fixture
def staged_study(tmp_path, monkeypatch):
    repo = next(p for p in TOOLS.parents if (p / ".git").exists())
    snapshot = json.loads((repo / "doc/experiments/baseline_validation_v5_20260906_round4_complete.json").read_text())
    dataset, parent_dir, output = (tmp_path / name for name in ("dataset", "parents", "output"))
    dataset.mkdir()
    parent_dir.mkdir()
    manifest = dataset / "dataset_manifest.json"
    manifest.write_text(json.dumps({"sample_counts": {"train": 13813}}))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    split = dataset / "episode_split_audit.json"
    split.write_text(json.dumps({"episode_split_match": True, "provenance": {
        "baseline_manifest": {"sha256": manifest_hash}
    }}))
    (dataset / "validation_contract_audit.json").write_text(json.dumps({
        "comparison_match": True, "split_audit_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
    }))
    selection = json.loads((repo / "doc/experiments/baseline_b4_representation_selection_20260906.json").read_text())
    (parent_dir / "selection.json").write_text(json.dumps(selection))
    for seed in (42, 43):
        record = next(r for r in snapshot["runs"] if r["summary"]["baseline_id"] == "B4"
                      and f"candidate_history/seed{seed}" in r["run"])
        dest = parent_dir / "candidate_history" / f"seed{seed}"
        dest.mkdir(parents=True)
        (dest / "config.json").write_text(json.dumps(record["configuration"]))
    calls = []
    def load_parent(path, **kwargs):
        assert path.parent.name == f"seed{kwargs['seed']}"
        assert kwargs["dataset_manifest_sha256"] == manifest_hash
        assert kwargs["train_sample_count"] == 13813
        return {}, {"seed": kwargs["seed"], "checkpoint": str(path)}
    def train(kind, source, dest, **kwargs):
        calls.append((kind, source, dest, kwargs))
    def select(args, **kwargs):
        assert len(calls) == 8
        assert args[1].endswith("select_baseline_tuning.py")
        assert args[-3:] == ["--expected_seeds", "42", "43"]
        calls.append("selection")
    monkeypatch.setattr(staged, "load_warm_start_parent", load_parent)
    monkeypatch.setattr(staged, "train_torch_baseline", train)
    monkeypatch.setattr(staged.subprocess, "check_output", lambda args, **kw:
                        "dev_xwt\n" if "branch" in args else
                        ("" if "status" in args else "a" * 40 + "\n"))
    monkeypatch.setattr(staged.subprocess, "run", select)
    return dataset, parent_dir, output, calls


def test_study_routes_all_trials_and_selects_only_after_completion(staged_study):
    dataset, parent_dir, output, calls = staged_study
    staged.run_study("B4", dataset, parent_dir, output, [42, 43], "cpu")
    assert len(calls) == 9 and calls[-1] == "selection"
    study = json.loads((output / "study_config.json").read_text())
    assert not study["test_evaluated"] and study["selection_split"] == "validation"
    assert len(study["trials"]) == 8
    for index, (kind, source, dest, kwargs) in enumerate(calls[:-1]):
        seed, arm = (42 if index < 4 else 43), ARMS[index % 4]
        assert kind == "b4_gcn_gru" and source == dataset
        assert dest == output / f"candidate_{arm}" / f"seed{seed}"
        assert kwargs["train_config"].seed == seed
        assert not kwargs["train_config"].evaluate_test
        expected_parent = parent_dir / "candidate_history" / f"seed{seed}" / "best.pt"
        assert kwargs["warm_start_checkpoint"] == (expected_parent if arm.startswith("warm") else None)
    with pytest.raises(FileExistsError):
        staged.run_study("B4", dataset, parent_dir, output, [42, 43], "cpu")
    assert len(calls) == 9


def test_hard_negative_study_routes_six_scratch_trials(staged_study, monkeypatch):
    dataset, parent_dir, output, calls = staged_study
    def select(args, **kwargs):
        assert len(calls) == 6
        assert args[1].endswith("select_baseline_tuning.py")
        calls.append("selection")
    monkeypatch.setattr(staged.subprocess, "run", select)
    staged.run_study("B4", dataset, parent_dir, output, [42, 43], "cpu", study="hard_negatives")
    assert calls[-1] == "selection" and len(calls) == 7
    protocol = json.loads((output / "study_config.json").read_text())
    assert protocol["protocol"] == "baseline_short_hot_negative_v1"
    assert len(protocol["trials"]) == 6 and "not a final causal benchmark" in protocol["comparison_scope"]
    for index, (_, _, _, kwargs) in enumerate(calls[:-1]):
        assert kwargs["warm_start_checkpoint"] is None
        assert kwargs["train_config"].max_epochs == 60
        assert not kwargs["train_config"].evaluate_test
        assert kwargs["loss_config"].event_short_hot_fp_multiplier == (1., 2., 4.)[index % 3]


@pytest.mark.parametrize("change", ["manifest", "audit", "duplicate_candidate", "duplicate_seed", "missing_run", "test_selection", "seeds", "branch", "dirty"])
def test_study_refuses_invalid_preflight(staged_study, change, monkeypatch):
    dataset, parent_dir, output, calls = staged_study
    seeds = [42, 43]
    if change == "manifest":
        (dataset / "dataset_manifest.json").write_text('{"sample_counts":{"train":1}}')
    elif change == "audit":
        path = dataset / "validation_contract_audit.json"
        audit = json.loads(path.read_text())
        audit["comparison_match"] = False
        path.write_text(json.dumps(audit))
    elif change == "seeds":
        seeds = [43, 44]
    elif change == "branch":
        monkeypatch.setattr(staged.subprocess, "check_output", lambda *args, **kwargs: "dev_tyx\n")
    elif change == "dirty":
        monkeypatch.setattr(staged.subprocess, "check_output", lambda args, **kwargs:
                            "dev_xwt\n" if "branch" in args else " M trainer.py\n")
    else:
        path = parent_dir / "selection.json"
        selection = json.loads(path.read_text())
        if change == "duplicate_candidate":
            selection["candidates"][1] = selection["candidates"][0]
        elif change == "duplicate_seed":
            selection["candidates"][0]["runs"].append(selection["candidates"][0]["runs"][0])
        elif change == "missing_run":
            selection["candidates"][0]["runs"].pop()
        else:
            selection["test_evaluated"] = True
        path.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        staged.run_study("B4", dataset, parent_dir, output, seeds, "cpu")
    assert not calls and not output.exists()


def test_study_stops_without_selection_on_training_failure(staged_study, monkeypatch):
    dataset, parent_dir, output, calls = staged_study
    def fail(*args, **kwargs):
        raise RuntimeError("simulated training failure")
    monkeypatch.setattr(staged, "train_torch_baseline", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        staged.run_study("B4", dataset, parent_dir, output, [42, 43], "cpu")
    assert not calls
    assert (output / "study_config.json").exists()
    assert not (output / "selection.json").exists()
