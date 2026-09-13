#!/usr/bin/env python3
"""Count new train/validation labels and check unchanged B4/B5 backbones.

Only the requested new JSON output is written, in an existing directory.
No optimizer is created, no test sample is indexed, and no old result is replaced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import default_collate

from factory_baselines import protocol_20260913 as protocol
from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.precursor import attach_precursor
from factory_baselines.torch_losses import compute_multitask_loss
from factory_baselines.torch_trainer import _model_spec, _model_inputs, _move_batch, _evaluate_loader
from train_dense_baseline_control import dense_configuration


SOURCE_MANIFEST = "e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration(model: str, max_start: int, seed: int, device: str):
    training, overrides, loss = dense_configuration(model, "near_precursor", seed, device)
    training = replace(
        training, evaluation_protocol=protocol.VERSION, event_max_start_windows=max_start,
        report_threshold_sweep=protocol.THRESHOLDS,
        training_profile=f"dense_matched_7b2ab39_start{max_start}_min8",
    )
    return training, overrides, loss


def run(dataset_dir: Path, output: Path, source_commit: str, device: str = "cpu") -> dict:
    repo = Path.cwd().resolve()
    if repo != Path("/home/sci/work/BSTAN_isaac_factory"):
        raise ValueError("Run only in the authorized server baseline checkout")
    dataset_dir, output = dataset_dir.resolve(), output.resolve()
    if not dataset_dir.is_dir() or not output.parent.is_dir() or not output.is_relative_to(dataset_dir):
        raise ValueError("Reuse the existing benchmark directory for the new audit JSON")
    if output.exists():
        raise FileExistsError("Read the existing preflight; do not overwrite or repeat it")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != source_commit:
        raise ValueError("Runtime source commit differs from the registered preflight")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "dev_xwt":
        raise ValueError("Wrong runtime branch")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Tracked runtime files must be clean")
    if sha(dataset_dir / "dataset_manifest.json") != SOURCE_MANIFEST:
        raise ValueError("This preflight is for the frozen 208 source only")
    protected = [dataset_dir / name for name in
                 ("dataset.pt", "dataset_manifest.json", "split_manifest.json", "normalization.json", "samples.csv", "nodes.csv")]
    protected = [path for path in protected if path.exists()]
    for model in ("b4", "b5"):
        for seed in (42, 43):
            path = dataset_dir / f"models/tuning/{model}_representation_v1/candidate_history/seed{seed}"
            protected.extend(item for item in path.iterdir() if item.is_file())
    before = {str(p.relative_to(dataset_dir)): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
    torch.set_num_threads(2)
    payload, manifest = load_shared_dataset(dataset_dir)
    assert tuple(payload["x"].shape[1:]) == (30, 38, 27)
    assert manifest["episode_counts"] == {"train": 138, "validation": 30, "test": 40}
    payload, input_contract = attach_precursor(payload, manifest, dataset_dir, "near", ("train", "validation"))
    result = {
        "status": "preflight_completed", "source_commit": source_commit,
        "reference_commit": protocol.REFERENCE_COMMIT, "source_manifest_sha256": SOURCE_MANIFEST,
        "source_episode_counts": manifest["episode_counts"],
        "source_sample_counts": manifest["sample_counts"],
        "test_indexed": False, "optimizer_steps": 0, "trained_checkpoint_created": False,
        "input_contract": input_contract, "tasks": {}, "model_checks": [],
        "training_budget_status": "preflight_only_existing_baseline_settings_not_a_registered_main_training_queue",
    }
    task_views = {}
    for max_start in (5, 10, 15):
        view, view_manifest = protocol.protocol_view(payload, manifest, max_start)
        task_views[max_start] = (view, view_manifest)
        task = {"contract": view["evaluation_contract"], "splits": {}}
        for split in ("train", "validation"):
            dataset = FactoryBaselineTensorDataset(view, view["split_indices"][split].tolist())
            stats = dict(samples=len(dataset), ongoing=0, upcoming=0, negative=0,
                         upcoming_start_counts=[0] * 20, first_upcoming_position=None)
            for position in range(len(dataset)):
                sample = dataset[position]
                valid = sample["occ_node_mask"].bool()
                positive = (sample["event_will"] > .5) & valid
                upcoming = positive & ~(sample["hist_last_hot"] > .5)
                stats["upcoming"] += int(upcoming.sum())
                stats["ongoing"] += int((positive & ~upcoming).sum())
                stats["negative"] += int((valid & ~positive).sum())
                counts = torch.bincount(sample["event_start"][upcoming], minlength=20).tolist()
                stats["upcoming_start_counts"] = [a + b for a, b in zip(stats["upcoming_start_counts"], counts)]
                if stats["first_upcoming_position"] is None and upcoming.any():
                    stats["first_upcoming_position"] = position
                if position and position % 5000 == 0:
                    print("LABEL_PROGRESS", max_start, split, position, flush=True)
            task["splits"][split] = stats
            print("LABEL_COUNTS", max_start, split, json.dumps(stats), flush=True)
        result["tasks"][str(max_start)] = task
    target_device = torch.device(device)
    for model_name, kind in (("B4", "b4_gcn_gru"), ("B5", "b5_gat_gru")):
        training, overrides, loss = configuration(model_name, 5, 42, device)
        cls, cfg_cls, _, _ = _model_spec(kind)
        values = dict(input_dim=27, global_dim=payload["global_features"].shape[-1],
                      num_nodes=38, num_causes=len(manifest["cause_classes"]), **overrides)
        torch.manual_seed(42)
        old = cls(cfg_cls(**values, max_remain_windows=15))
        torch.manual_seed(42)
        model = cls(cfg_cls(**values, max_remain_windows=20))
        backbone_keys = [key for key in old.state_dict() if not key.startswith("heads.")]
        assert backbone_keys and all(torch.equal(old.state_dict()[key], model.state_dict()[key]) for key in backbone_keys)
        del old
        model = model.to(target_device)
        loss = replace(loss, near_remain_windows=20, event_partition="history",
                       remain_progress_weight_floor=.25, remain_progress_weight_power=1.5,
                       cause_ignored_ids=tuple(i for i, name in enumerate(manifest["cause_classes"]) if name not in protocol.CAUSE_CLASSES))
        for max_start, (view, _) in task_views.items():
            dataset = FactoryBaselineTensorDataset(view, view["split_indices"]["train"].tolist())
            first = result["tasks"][str(max_start)]["splits"]["train"]["first_upcoming_position"]
            positions = list(range(4)) if first is None or first < 4 else [0, 1, 2, first]
            cpu = default_collate([dataset[i] for i in positions])
            batch = _move_batch(cpu, target_device)
            model.train(); model.zero_grad(set_to_none=True)
            outputs = model(**_model_inputs(batch, model))
            value, _ = compute_multitask_loss(outputs, batch, loss)
            value.backward()
            assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            class OneBatch:
                def __iter__(self):
                    return iter([cpu])
            loader = OneBatch(); loader.dataset = dataset
            metrics, _, _ = _evaluate_loader(
                model, loader, loss, torch.tensor(1., device=target_device), target_device,
                model.config.num_causes, cause_classes=manifest["cause_classes"],
                report_threshold_sweep=protocol.THRESHOLDS,
            )
            assert metrics["evaluation_contract"]["version"] == protocol.VERSION
            assert metrics["will15_f1"] == metrics["who_f1"]
            result["model_checks"].append({
                "model": model_name, "max_start": max_start, "device": str(target_device),
                "backbone_tensor_count_unchanged": len(backbone_keys),
                "loss_finite": True, "gradients_finite": True,
                "output_horizon": int(outputs["remain_hot_logit"].shape[1]),
                "sample_indices": cpu["sample_index"].tolist(),
                "evaluation_path_passed": True, "random_initialization_not_a_performance_result": True,
            })
            print("MODEL_PREFLIGHT", model_name, max_start, "passed", flush=True)
        del model
    after = {str(p.relative_to(dataset_dir)): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
    assert before == after
    result["protected_files_stat"] = before
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("MATCHED_PROTOCOL_PREFLIGHT_COMPLETE", output.stat().st_size, sha(output), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--device", default="cpu")
    run(**vars(parser.parse_args()))
