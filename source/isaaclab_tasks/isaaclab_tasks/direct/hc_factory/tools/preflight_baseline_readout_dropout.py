#!/usr/bin/env python3
"""Validate the single temporal-readout regularization change before training."""

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


NEAR_SHA = "e71c84ded25a2b5fe861b45ecc12e2a0941193043a526654a9d8327a9d7a4b27"
CURRENT_AUDIT_SHA = "c1837f569cb50023ec8647a473440ca406b96be61b360463f9a174db9c2ce340"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    tools = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
    d = tools.parent / "output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    output = d / "baseline_readout_dropout_preflight20260913.json"
    assert d.is_dir() and not output.exists()
    near_path, audit_path = d / "baseline_dense_near_metrics_20260912.json", d / "baseline_schedule_strata_final_verification20260913.json"
    assert sha(near_path) == NEAR_SHA and sha(audit_path) == CURRENT_AUDIT_SHA
    near, audit = json.loads(near_path.read_text()), json.loads(audit_path.read_text())
    source_files = [tools / name for name in ("preflight_baseline_readout_dropout.py", "train_dense_baseline_control.py",
        "factory_baselines/b4_gcn_gru.py", "factory_baselines/b5_gat_gru.py", "factory_baselines/torch_heads.py",
        "factory_baselines/torch_losses.py", "factory_baselines/torch_trainer.py", "factory_baselines/dataset.py",
        "factory_baselines/precursor.py", "factory_baselines/artifacts.py")]
    sources = {str(p.relative_to(repo)): sha(p) for p in source_files}

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == args.source_commit
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, h in sources.items(): assert sha(repo / name) == h
        for name, h in audit["current_model_files_sha256"].items(): assert sha(d / name) == h
        for name, value in audit["dataset_files_stat"].items():
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
            cpu = default_collate([dataset[i] for i in indices])
            key = model + "_" + split
            selected[key] = dict(sample_indices=cpu["sample_index"].tolist(), selection="first_batch_with_first_upcoming_before_prediction")
            batches[key] = _move_batch(cpu, device)
    checks, archives = [], []
    for model_name, cls, config_cls in (("B4", B4GcnGru, B4ModelConfig), ("B5", B5GatGru, B5ModelConfig)):
        for seed in (42, 43):
            previous = next(r for r in near["runs"] if r["model"] == model_name.lower() and r["seed"] == seed)
            out = d / f"models/tuning/{model_name.lower()}_representation_v1/candidate_history/seed{seed}"
            assert out.is_dir() and not (out / "model_before_readoutdrop20260913.zip").exists() and not (out / "dense_control_readoutdrop20260913.json").exists()
            archive_path = out / "model_before_onsetaux20260912.zip"
            with zipfile.ZipFile(archive_path) as archive:
                assert archive.testzip() is None
                for name, h in previous["files_sha256"].items(): assert hashlib.sha256(archive.read(name)).hexdigest() == h
            archives.append(dict(model=model_name, seed=seed, file=str(archive_path.relative_to(d)), sha256=sha(archive_path), verified_parent_files=previous["files_sha256"]))
            assert previous["config"]["metadata"]["input_feature_contract"] == contract
            parent_train, parent_overrides, parent_loss = dense_configuration(model_name, "near_precursor", seed, args.device)
            candidate_train, overrides, loss_config = dense_configuration(model_name, "readout_dropout", seed, args.device)
            old = previous["config"]["training"]
            old_train = TorchTrainConfig(**{**old, "device": args.device, "report_threshold_sweep": tuple(old["report_threshold_sweep"])})
            assert old_train == parent_train
            assert asdict(replace(candidate_train, training_profile=parent_train.training_profile)) == asdict(parent_train)
            assert MultiTaskLossConfig.from_dict(previous["config"]["loss"]) == parent_loss == loss_config
            parent_config = config_cls.from_dict(previous["config"]["model"])
            assert parent_config == config_cls(input_dim=27, global_dim=0, num_nodes=38, **parent_overrides)
            config = config_cls(input_dim=27, global_dim=0, num_nodes=38, **overrides)
            assert config == replace(parent_config, readout_dropout=.2)
            assert not config.event_onset_aux and not config.event_onset_joint and not config.history_graph_refine
            assert not candidate_train.evaluate_test and candidate_train.max_epochs == 60
            torch.manual_seed(seed); parent = cls(parent_config); rng = torch.get_rng_state().clone()
            torch.manual_seed(seed); candidate = cls(config)
            assert torch.equal(rng, torch.get_rng_state())
            assert parent.state_dict().keys() == candidate.state_dict().keys()
            assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in parent.state_dict().items())
            count = sum(p.numel() for p in candidate.parameters()); assert count == sum(p.numel() for p in parent.parameters())
            parent.to(device).eval(); candidate.to(device).eval()
            with torch.no_grad():
                before = parent(**_model_inputs(batches[model_name + "_validation"], parent))
                after = candidate(**_model_inputs(batches[model_name + "_validation"], candidate))
            assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
            del parent, before, after
            torch.cuda.reset_peak_memory_stats(device); candidate.train(); torch.manual_seed(seed + 1000)
            captured = []
            def observe(_module, inputs, result): captured.append((inputs[0], result))
            hook = candidate.readout_dropout.register_forward_hook(observe)
            result = candidate(**_model_inputs(batches[model_name + "_train"], candidate)); hook.remove()
            before, after = captured[0]
            valid = batches[model_name + "_train"]["node_mask"].bool()[:, :, None].expand_as(before)
            dropped = valid & (before != 0) & (after == 0); kept = valid & (after != 0)
            assert dropped.any() and kept.any()
            torch.testing.assert_close(after[kept], before[kept] / .8, rtol=1e-6, atol=1e-6)
            loss, components = compute_multitask_loss(result, batches[model_name + "_train"], loss_config, occupancy_type_masks=type_masks)
            assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in result.values()); loss.backward()
            gradients = {name: p.grad.abs().sum().item() for name, p in candidate.named_parameters() if p.grad is not None}
            for p in candidate.parameters():
                if p.grad is not None: assert torch.isfinite(p.grad).all()
            for name in ("gru.weight_ih_l0", "history_readout.0.weight", "heads.event_will_head.0.weight",
                         "gcn1.linear.weight" if model_name == "B4" else "gat1.projection.weight"):
                assert gradients[name] > 0, name
            torch.cuda.synchronize(device)
            checks.append(dict(model=model_name, seed=seed, model_config=config.to_dict(), training_config=asdict(candidate_train), loss_config=loss_config.to_dict(),
                parameter_count=count, additional_parameters=0, initialization_and_rng_equal=True, initial_eval_all_outputs_equal=True,
                valid_readout_coordinates=int(valid.sum()), dropped_coordinates=int(dropped.sum()), train_loss=loss.item(), gradients_l1=gradients,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
            print("READOUT_PREFLIGHT_CASE", model_name, seed, "parameters", count, "drop_fraction", float(dropped.sum() / valid.sum()), flush=True)
            del candidate, result, loss, components, captured, before, after, valid, dropped, kept
            torch.cuda.empty_cache(); guard()
    record = dict(status="four_readout_dropout_real_208_preflight_cases_passed", source_commit=args.source_commit,
        runtime_source_sha256=sources, parent_near_sha256=NEAR_SHA, source_current_model_audit_sha256=CURRENT_AUDIT_SHA,
        current_model_files_sha256=audit["current_model_files_sha256"], dataset_files_stat=audit["dataset_files_stat"],
        manifest_sha256=sha(d / "dataset_manifest.json"), input_feature_contract=contract, selected_batches=selected, parent_archives=archives, checks=checks,
        current_model_files_unchanged=28, model_training_launched=False, test_evaluated=False, goal_met=False)
    with output.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    print("READOUT_DROPOUT_PREFLIGHT_COMPLETE", output.stat().st_size, sha(output), flush=True)


if __name__ == "__main__": main()
