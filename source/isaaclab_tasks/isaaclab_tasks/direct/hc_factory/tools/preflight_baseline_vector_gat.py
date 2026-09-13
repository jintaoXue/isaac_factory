#!/usr/bin/env python3
"""Verify the registered B5 score ablation on frozen train/validation inputs."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import torch
from torch.utils.data import default_collate

from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.precursor import attach_precursor
from factory_baselines.torch_losses import MultiTaskLossConfig, compute_multitask_loss
from factory_baselines.torch_trainer import TorchTrainConfig, _model_inputs, _move_batch, _occupancy_type_masks
from train_dense_baseline_control import dense_configuration


JOINT_SHA = "c35d4d46a2a45d40f48381662278ea1655c0a1e0a02b8ec053a7018767dcae19"
NEAR_SHA = "e71c84ded25a2b5fe861b45ecc12e2a0941193043a526654a9d8327a9d7a4b27"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat(path):
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def restore_training_config(values):
    # JSON records tuples as arrays. Preserve the exact values and ordering;
    # normalize only this declared tuple field before dataclass comparison.
    return TorchTrainConfig(**{**values, "report_threshold_sweep": tuple(values["report_threshold_sweep"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = Path.cwd()
    if repo != Path("/home/sci/work/BSTAN_isaac_factory"):
        raise ValueError("Preflight only in the existing server BSTAN repository")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    assert branch == "dev_xwt"
    d, output = args.dataset_dir.resolve(), args.output.resolve()
    assert d.is_dir() and d.is_relative_to(repo) and output.parent == d and not output.exists()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    joint_path = d / "baseline_dense_joint_onset_metrics_20260913.json"
    near_path = d / "baseline_dense_near_metrics_20260912.json"
    assert sha(joint_path) == JOINT_SHA and sha(near_path) == NEAR_SHA
    joint, near = json.loads(joint_path.read_text()), json.loads(near_path.read_text())
    assert joint["diagnostics_completed"] and not joint["test_evaluated"]
    source_files = [Path(__file__), *[Path(__file__).parent / name for name in (
        "train_dense_baseline_control.py", "factory_baselines/b5_gat_gru.py",
        "factory_baselines/torch_heads.py", "factory_baselines/torch_losses.py",
        "factory_baselines/torch_trainer.py", "factory_baselines/dataset.py",
        "factory_baselines/precursor.py")]]
    source_hashes = {str(path.relative_to(repo)): sha(path) for path in source_files}

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
        assert sha(joint_path) == JOINT_SHA and sha(near_path) == NEAR_SHA
        for name, want in source_hashes.items():
            assert sha(repo / name) == want
        for name, want in joint["dataset_files_stat"].items():
            assert file_stat(d / name) == want
        for run in joint["runs"]:
            out = d / f"models/tuning/{run['model']}_representation_v1/candidate_history/seed{run['seed']}"
            for name, want in run["training"]["files_sha256"].items():
                assert sha(out / name) == want

    guard()
    for name, want in joint["dataset_files_sha256"].items():
        assert sha(d / name) == want
    torch.set_num_threads(2)
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    print("BUILDING_FROZEN_NEAR_INPUTS", flush=True)
    payload, manifest = load_shared_dataset(d)
    payload, contract = attach_precursor(payload, manifest, d, "near", ("train", "validation"))
    assert tuple(payload["x"].shape[1:]) == (30, 38, 27)
    selected, batches = {}, {}
    for split, count in (("train", 23859), ("validation", 5439)):
        dataset = FactoryBaselineTensorDataset(payload, payload["split_indices"][split])
        assert len(dataset) == count
        first_upcoming = None
        for index in range(len(dataset)):
            sample = dataset[index]
            if bool(((sample["event_will"] > .5) & (sample["event_start"] > 0) & sample["occ_node_mask"].bool()).any()):
                first_upcoming = index
                break
        assert first_upcoming is not None
        indices = list(range(16)) if first_upcoming < 16 else [*range(15), first_upcoming]
        cpu_batch = default_collate([dataset[index] for index in indices])
        selected[split] = dict(split_offsets=indices, sample_indices=cpu_batch["sample_index"].tolist(),
                               selection="first_15_plus_first_upcoming_before_any_prediction")
        batches[split] = _move_batch(cpu_batch, device)
    type_masks = _occupancy_type_masks(d, payload, device)
    checks = []
    for seed in (42, 43):
        previous = next(run for run in near["runs"] if run["model"] == "b5" and run["seed"] == seed)
        assert previous["config"]["metadata"]["input_feature_contract"] == contract
        out = d / f"models/tuning/b5_representation_v1/candidate_history/seed{seed}"
        with zipfile.ZipFile(out / "model_before_onsetaux20260912.zip") as archive:
            assert archive.testzip() is None
            for name, want in previous["files_sha256"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == want
        parent_train, parent_overrides, parent_loss = dense_configuration("B5", "near_precursor", seed, args.device)
        candidate_train, overrides, loss_config = dense_configuration("B5", "vector_gat", seed, args.device)
        old_train = restore_training_config(previous["config"]["training"])
        old_train.device = parent_train.device
        assert old_train == parent_train
        assert MultiTaskLossConfig.from_dict(previous["config"]["loss"]) == parent_loss == loss_config
        parent_config = B5ModelConfig.from_dict(previous["config"]["model"])
        registered_parent = B5ModelConfig(input_dim=27, global_dim=0, num_nodes=38, **parent_overrides)
        assert parent_config == registered_parent
        config = B5ModelConfig(input_dim=27, global_dim=0, num_nodes=38, **overrides)
        assert config == replace(parent_config, gat_score_mode="vector_additive")
        assert not config.event_onset_aux and not config.history_graph_refine and not candidate_train.evaluate_test
        torch.manual_seed(seed); parent = B5GatGru(parent_config)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(seed); model = B5GatGru(config)
        assert torch.equal(expected_rng, torch.get_rng_state())
        assert list(parent.state_dict()) == list(model.state_dict())
        for name, value in parent.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
        count = sum(p.numel() for p in model.parameters())
        assert count == sum(p.numel() for p in parent.parameters())
        parent.to(device).eval(); model.to(device).eval()
        with torch.no_grad():
            base_out = parent(**_model_inputs(batches["validation"], parent))
            new_out = model(**_model_inputs(batches["validation"], model))
        difference = (new_out["event_will_logit"] - base_out["event_will_logit"]).abs().max().item()
        assert difference > 0 and all(torch.isfinite(v).all() for v in new_out.values())
        del parent, base_out, new_out
        torch.cuda.reset_peak_memory_stats(device)
        model.train(); torch.manual_seed(seed + 1000)
        result = model(**_model_inputs(batches["train"], model))
        loss, components = compute_multitask_loss(result, batches["train"], loss_config,
                                                   occupancy_type_masks=type_masks)
        assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in result.values())
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                gradients[name] = parameter.grad.abs().sum().item()
        for name in ("gat1.attention_source", "gat2.attention_source", "gat1.projection.weight",
                     "gat2.projection.weight", "gru.weight_ih_l0", "heads.event_will_head.0.weight"):
            assert gradients[name] > 0, name
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        checks.append(dict(seed=seed,model_config=config.to_dict(),training_config=asdict(candidate_train),
                           loss_config=loss_config.to_dict(),parameter_count=count,additional_parameters=0,
                           initial_weights_and_rng_match=True,validation_initial_logit_max_abs_change=difference,
                           train_batch_loss=loss.item(),gradients_l1=gradients,peak_allocated_bytes=peak))
        print("REAL_BATCH_VERIFIED", seed, "PARAMETERS", count, "PEAK_MIB", peak / 1024**2, flush=True)
        del model, result, loss, components
        torch.cuda.empty_cache()
        guard()
    result = dict(status="registered_B5_vector_score_real_208_preflight_passed", source_commit=args.source_commit,
                  runtime_source_sha256=source_hashes, parent_near_sha256=NEAR_SHA, prior_joint_sha256=JOINT_SHA,
                  manifest_sha256=sha(d / "dataset_manifest.json"), input_feature_contract=contract,
                  selected_batches=selected, checks=checks, dataset_files_sha256=joint["dataset_files_sha256"],
                  dataset_files_stat=joint["dataset_files_stat"], current_28_files_unchanged=True,
                  test_evaluated=False, training_launched=False)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print("VECTOR_GAT_PREFLIGHT_PASSED", output.stat().st_size, sha(output), flush=True)


if __name__ == "__main__":
    main()
