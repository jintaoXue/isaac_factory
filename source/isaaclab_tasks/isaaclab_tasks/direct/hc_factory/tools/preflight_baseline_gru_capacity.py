#!/usr/bin/env python3
"""Verify the registered GRU-width control against the saved near parent."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import torch
from torch.utils.data import default_collate
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.precursor import attach_precursor
from factory_baselines.torch_losses import MultiTaskLossConfig, compute_multitask_loss
from factory_baselines.torch_trainer import TorchTrainConfig, _model_inputs, _move_batch, _occupancy_type_masks
from train_dense_baseline_control import dense_configuration

TAG = "grucapacity20260913"
NEAR_SHA = "e71c84ded25a2b5fe861b45ecc12e2a0941193043a526654a9d8327a9d7a4b27"
CURRENT_SHA = "b599500b3dd973393d3a997569692959ae747f7a8eea022d2ad5db417f696c78"
HOLDOUT_SHA = "0071a30c1af9e8d557853270a62ba868d60c18f7f8804a0de3ff6042fb2a5707"
INVENTORY_SHA = "7d032b4dffde116e1e96790840e9b1562f103f874732204d9769c7cdacd1b67c"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    tools = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
    d = tools.parent / "output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    output = d / "baseline_gru_capacity_preflight20260913.json"
    assert d.is_dir() and not output.exists()
    evidence = {
        "baseline_dense_near_metrics_20260912.json": NEAR_SHA,
        "baseline_dense_readout_dropout_metrics_20260913.json": CURRENT_SHA,
        "baseline_episodeholdout20260913_final_verification.json": HOLDOUT_SHA,
        "baseline_gru_capacity_history_audit20260913.json": INVENTORY_SHA,
    }
    for name, h in evidence.items(): assert sha(d / name) == h
    near = json.loads((d / "baseline_dense_near_metrics_20260912.json").read_text())
    current = json.loads((d / "baseline_dense_readout_dropout_metrics_20260913.json").read_text())
    old_files = {}
    for run in current["runs"]:
        prefix = f"models/tuning/{run['model']}_representation_v1/candidate_history/seed{run['seed']}/"
        old_files.update({prefix + name: h for name, h in run["training"]["files_sha256"].items()})
    assert len(old_files) == 48 and len(current["dataset_files_stat"]) == 6
    names = ("preflight_baseline_gru_capacity.py", "run_baseline_gru_capacity.py", "train_dense_baseline_control.py",
        "diagnose_baseline_events.py", "factory_baselines/b4_gcn_gru.py", "factory_baselines/b5_gat_gru.py",
        "factory_baselines/torch_heads.py", "factory_baselines/torch_losses.py", "factory_baselines/torch_trainer.py",
        "factory_baselines/dataset.py", "factory_baselines/precursor.py", "factory_baselines/artifacts.py", "factory_bn_shared/remain.py")
    source_paths = [tools / name for name in names] + [tools.parent / "PDFormer/factory_bn/remain.py"]
    sources = {str(p.relative_to(repo)): sha(p) for p in source_paths}

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, h in sources.items(): assert sha(repo / name) == h
        for name, h in evidence.items(): assert sha(d / name) == h
        for name, h in old_files.items(): assert sha(d / name) == h
        for name, value in current["dataset_files_stat"].items():
            stat = (d / name).stat(); assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value

    guard()
    torch.set_num_threads(2); device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    payload, manifest = load_shared_dataset(d)
    assert tuple(payload["x"].shape[1:]) == (30, 38, 27)
    assert sha(d / "dataset_manifest.json") == "e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f"
    payload, contract = attach_precursor(payload, manifest, d, "near", ("train", "validation"))
    type_masks = _occupancy_type_masks(d, payload, device)
    selected, batches = {}, {}
    for model, count in (("B4", 24), ("B5", 16)):
        for split, n in (("train", 23859), ("validation", 5439)):
            dataset = FactoryBaselineTensorDataset(payload, payload["split_indices"][split])
            assert len(dataset) == n
            first_up = next(i for i, row in enumerate(dataset) if bool(((row["event_will"] > .5) & (row["event_start"] > 0) & row["occ_node_mask"].bool()).any()))
            indices = list(range(count)) if first_up < count else [*range(count - 1), first_up]
            cpu = default_collate([dataset[i] for i in indices]); key = model + "_" + split
            selected[key] = dict(sample_indices=cpu["sample_index"].tolist(), selection="first_batch_with_first_upcoming_before_prediction")
            batches[key] = _move_batch(cpu, device)
    checks, archives = [], []
    for model_name, cls, config_cls in (("B4", B4GcnGru, B4ModelConfig), ("B5", B5GatGru, B5ModelConfig)):
        for seed in (42, 43):
            previous = next(r for r in near["runs"] if r["model"] == model_name.lower() and r["seed"] == seed)
            out = d / f"models/tuning/{model_name.lower()}_representation_v1/candidate_history/seed{seed}"
            assert out.is_dir() and not (out / f"model_before_{TAG}.zip").exists() and not (out / f"dense_control_{TAG}.json").exists()
            archive_path = out / "model_before_onsetaux20260912.zip"
            with zipfile.ZipFile(archive_path) as archive:
                assert archive.testzip() is None
                for name, h in previous["files_sha256"].items(): assert hashlib.sha256(archive.read(name)).hexdigest() == h
            archives.append(dict(model=model_name, seed=seed, file=str(archive_path.relative_to(d)), sha256=sha(archive_path), verified_parent_files=previous["files_sha256"]))
            assert previous["config"]["metadata"]["input_feature_contract"] == contract
            parent_train, parent_overrides, parent_loss = dense_configuration(model_name, "near_precursor", seed, args.device)
            candidate_train, overrides, loss_config = dense_configuration(model_name, "gru_capacity32", seed, args.device)
            old = previous["config"]["training"]
            assert TorchTrainConfig(**{**old, "device": args.device, "report_threshold_sweep": tuple(old["report_threshold_sweep"])}) == parent_train
            assert replace(candidate_train, training_profile=parent_train.training_profile) == parent_train
            assert MultiTaskLossConfig.from_dict(previous["config"]["loss"]) == parent_loss == loss_config
            parent_config = config_cls.from_dict(previous["config"]["model"])
            assert parent_config == config_cls(input_dim=27, global_dim=0, num_nodes=38, **parent_overrides)
            config = config_cls(input_dim=27, global_dim=0, num_nodes=38, **overrides)
            assert config == replace(parent_config, gru_hidden=32)
            assert not config.event_onset_aux and not config.event_onset_joint and not config.history_graph_refine
            assert config.readout_dropout == 0 and not candidate_train.evaluate_test and candidate_train.max_epochs == 60
            torch.manual_seed(seed); parent = cls(parent_config)
            torch.manual_seed(seed); candidate = cls(config)
            spatial = [k for k in parent.state_dict() if k.startswith(("input_projection.", "gcn", "gat"))]
            assert spatial and all(torch.equal(parent.state_dict()[k], candidate.state_dict()[k]) for k in spatial)
            count = sum(p.numel() for p in candidate.parameters()); parent_count = sum(p.numel() for p in parent.parameters())
            assert (parent_count, count) == ((273054, 31230) if model_name == "B4" else (285982, 44158))
            assert sum(p.numel() for p in candidate.gru.parameters()) == 9408
            del parent
            candidate.to(device).eval()
            with torch.no_grad():
                validation_result = candidate(**_model_inputs(batches[model_name + "_validation"], candidate))
            assert validation_result["node_hidden"].shape == (len(batches[model_name + "_validation"]["x"]), 38, 32)
            assert validation_result["event_will_logit"].shape == validation_result["node_hidden"].shape[:2]
            assert all(torch.isfinite(v).all() for v in validation_result.values())
            del validation_result
            torch.cuda.reset_peak_memory_stats(device); candidate.train(); torch.manual_seed(seed + 1000)
            result = candidate(**_model_inputs(batches[model_name + "_train"], candidate))
            loss, components = compute_multitask_loss(result, batches[model_name + "_train"], loss_config, occupancy_type_masks=type_masks)
            assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in result.values()); loss.backward()
            gradients = {name: p.grad.abs().sum().item() for name, p in candidate.named_parameters() if p.grad is not None}
            for p in candidate.parameters():
                if p.grad is not None: assert torch.isfinite(p.grad).all()
            for name in ("gru.weight_ih_l0", "gru.weight_hh_l0", "history_readout.0.weight", "heads.event_will_head.0.weight",
                         "gcn1.linear.weight" if model_name == "B4" else "gat1.projection.weight"):
                assert gradients[name] > 0, name
            torch.cuda.synchronize(device)
            checks.append(dict(model=model_name, seed=seed, model_config=config.to_dict(), training_config=asdict(candidate_train), loss_config=loss_config.to_dict(),
                parameter_count=count, parent_parameter_count=parent_count, gru_parameter_count=9408,
                changed_model_config_fields=["gru_hidden"], spatial_initialization_equal=True,
                train_loss=loss.item(), gradients_l1=gradients, peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
            print("GRU_CAPACITY_PREFLIGHT_CASE", model_name, seed, "parameters", parent_count, "->", count, flush=True)
            del candidate, result, loss, components
            torch.cuda.empty_cache(); guard()
    record = dict(status="four_gru_capacity_real_208_preflight_cases_passed", source_commit=args.source_commit,
        runtime_source_sha256=sources, evidence_sha256=evidence, current_model_files_sha256=old_files,
        dataset_files_stat=current["dataset_files_stat"], manifest_sha256=sha(d / "dataset_manifest.json"),
        input_feature_contract=contract, selected_batches=selected, parent_archives=archives, checks=checks,
        current_model_files_unchanged=48, optimizer_steps=0, model_training_launched=False, test_evaluated=False, goal_met=False)
    with output.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    print("GRU_CAPACITY_PREFLIGHT_COMPLETE", output.stat().st_size, sha(output), flush=True)


if __name__ == "__main__": main()
