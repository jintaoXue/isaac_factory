#!/usr/bin/env python3
"""Run a finite, preregistered B4/B5 training-procedure comparison."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from factory_baselines import MultiTaskLossConfig, TorchTrainConfig, train_torch_baseline
from factory_baselines.warm_start import load_warm_start_parent


ARMS = ("scratch_base", "scratch_signal", "warm_base", "warm_signal")
HARD_NEGATIVE_ARMS = {"weight1_control": 1.0, "weight2": 2.0, "weight4": 4.0}
SAMPLING_ARMS = {
    "uniform_control": ("any_event", 1.0),
    "event4": ("any_event", 4.0),
    "upcoming4": ("upcoming", 4.0),
}


def stage_configuration(parent: dict, arm: str, profile: str, device: str, seed: int) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown staged-training arm: {arm}")
    config = copy.deepcopy({key: parent[key] for key in ("model", "training", "loss")})
    training = config["training"]
    if training["max_epochs"] != 60 or training["evaluate_test"]:
        raise ValueError("The preregistered parent must be a validation-only 60-epoch-cap run")
    if config["model"]["temporal_readout"] != "last_mean" or config["model"]["node_embedding"] != 0:
        raise ValueError("This comparison requires the selected history-only representation")
    if config["model"]["event_context"] or config["loss"]["event_focal_gamma"] != 0:
        raise ValueError("Context and focal are not part of this comparison")
    expected_loss = {
        "event_will_upcoming_pos_weight": 4.0, "event_will_ongoing_pos_weight": 3.0,
        "event_will_fp_weight": 2.0, "lambda_event_will": 2.5,
    }
    if any(config["loss"][key] != value for key, value in expected_loss.items()):
        raise ValueError("Parent event loss differs from the preregistered baseline")
    training.update(training_profile=profile, seed=seed, device=device, evaluate_test=False)
    if arm.startswith("warm_"):
        training.update(max_epochs=20, min_epochs=10, patience=10)
        training["learning_rate"] *= 0.25
    else:
        training["max_epochs"] = 80
    if arm.endswith("_signal"):
        config["loss"]["event_will_upcoming_pos_weight"] = 16.0
    return config


def hard_negative_configuration(parent: dict, arm: str, profile: str, device: str, seed: int) -> dict:
    if arm not in HARD_NEGATIVE_ARMS:
        raise ValueError(f"Unknown hard-negative arm: {arm}")
    config = stage_configuration(parent, "scratch_base", profile, device, seed)
    config["training"]["max_epochs"] = 60
    config["loss"]["event_short_hot_fp_multiplier"] = HARD_NEGATIVE_ARMS[arm]
    return config


def sampling_configuration(parent: dict, arm: str, profile: str, device: str, seed: int) -> dict:
    if arm not in SAMPLING_ARMS:
        raise ValueError(f"Unknown sampling arm: {arm}")
    config = stage_configuration(parent, "scratch_base", profile, device, seed)
    config["training"]["max_epochs"] = 60
    target, factor = SAMPLING_ARMS[arm]
    config["training"].update(event_oversample_target=target, event_oversample_factor=factor)
    return config


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        return self.log.write(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_study(model: str, dataset_dir: Path, parent_dir: Path, output_dir: Path,
              seeds: list[int], device: str, *, study: str = "staged") -> None:
    if study not in {"staged", "hard_negatives", "sampling"}:
        raise ValueError(f"Unknown study: {study}")
    tools_dir = Path(__file__).resolve().parent
    repo = next(path for path in tools_dir.parents if (path / ".git").exists())
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != "dev_xwt":
        raise ValueError("Staged experiments must run on dev_xwt")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True
    ).strip():
        raise ValueError("Commit tracked changes before starting an immutable staged round")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if len(seeds) != 2 or set(seeds) != {42, 43}:
        raise ValueError("This preregistered study requires parent seeds 42 and 43 exactly once")
    manifest_bytes = (dataset_dir / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    split_bytes = (dataset_dir / "episode_split_audit.json").read_bytes()
    split_audit = json.loads(split_bytes)
    audit = json.loads((dataset_dir / "validation_contract_audit.json").read_bytes())
    if not split_audit["episode_split_match"] or not audit["comparison_match"]:
        raise ValueError("Resolve the common validation contract audit first")
    if audit["split_audit_sha256"] != hashlib.sha256(split_bytes).hexdigest():
        raise ValueError("Split audit changed after validation")
    if split_audit["provenance"]["baseline_manifest"]["sha256"] != manifest_hash:
        raise ValueError("Dataset manifest changed after validation")
    selection_bytes = (parent_dir / "selection.json").read_bytes()
    selection = json.loads(selection_bytes)
    if selection["status"] != "validation_selection_completed" or selection["test_evaluated"]:
        raise ValueError("Parent selection must be complete and validation-only")
    if selection["selected_candidate"] != "candidate_history":
        raise ValueError("Expected the complete history-only representation winner")
    expected_candidates = {"candidate_control", "candidate_identity",
                           "candidate_history", "candidate_identity_history"}
    if (len(selection["candidates"]) != 4
        or {candidate["candidate"] for candidate in selection["candidates"]} != expected_candidates
        or any(
        len(candidate["runs"]) != len(seeds)
        or {run["seed"] for run in candidate["runs"]} != set(seeds)
        for candidate in selection["candidates"]
    )):
        raise ValueError("Parent representation round is incomplete")
    kind = {"B4": "b4_gcn_gru", "B5": "b5_gat_gru"}[model]
    parents, configs, provenance = {}, {}, {}
    for seed in seeds:
        parent = parent_dir / selection["selected_candidate"] / f"seed{seed}" / "best.pt"
        config = json.loads((parent.parent / "config.json").read_text())
        state, proof = load_warm_start_parent(
            parent, model_kind=kind, model_config=config["model"], seed=seed,
            dataset_manifest_sha256=manifest_hash,
            train_sample_count=int(manifest["sample_counts"]["train"]),
        )
        del state
        parents[seed], configs[seed], provenance[seed] = parent, config, proof
    trials = []
    studies = {
        "staged": (ARMS, stage_configuration, "baseline_staged_training_v1"),
        "hard_negatives": (HARD_NEGATIVE_ARMS, hard_negative_configuration, "baseline_short_hot_negative_v1"),
        "sampling": (SAMPLING_ARMS, sampling_configuration, "baseline_event_sampling_v1"),
    }
    arms, configure, protocol = studies[study]
    for seed in seeds:
        for arm in arms:
            config = configure(configs[seed], arm, f"{output_dir.name}_{arm}", device, seed)
            trials.append({"seed": seed, "arm": arm, "configuration": config,
                           "warm_start_checkpoint": str(parents[seed]) if study == "staged" and arm.startswith("warm_") else None})
    output_dir.mkdir(parents=True)
    study_record = {
        "protocol": protocol,
        "selection_split": "validation",
        "test_evaluated": False, "model": model, "seeds": seeds,
        "dataset_manifest_sha256": manifest_hash,
        "parent_selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "parents": provenance, "trials": trials,
    }
    if study in {"hard_negatives", "sampling"}:
        study_record.update(
            initialization="from scratch; parent files supply configuration/provenance, not weights",
            comparison_scope="development under the frozen v5 scoring contract; not a final causal benchmark",
            post_selection_audit="repeat frozen-model observed-prefix decoding sensitivity before interpretation",
        )
    if study == "sampling":
        study_record.update(
            sampling_scope="training windows only; validation/test order, population and scoring unchanged",
            draws_per_epoch=int(manifest["sample_counts"]["train"]),
            importance_correction=False,
            interpretation="Weighted replacement changes all tasks' training exposure, not just event loss. "
                           "Epoch draw count is fixed; unique windows seen can differ. No new independent data.",
        )
    (output_dir / "study_config.json").write_text(json.dumps(study_record, indent=2) + "\n")
    for trial in trials:
        arm, seed, config = trial["arm"], trial["seed"], trial["configuration"]
        dest = output_dir / f"candidate_{arm}" / f"seed{seed}"
        dest.mkdir(parents=True)
        print(f"\n{model} {study} candidate={arm} seed={seed} validation-only", flush=True)
        with (dest / "training.log").open("w") as log, redirect_stdout(_Tee(sys.stdout, log)):
            train_torch_baseline(
                kind, dataset_dir, dest, model_overrides=config["model"],
                train_config=TorchTrainConfig(**config["training"]),
                loss_config=MultiTaskLossConfig.from_dict(config["loss"]),
                warm_start_checkpoint=(Path(trial["warm_start_checkpoint"])
                                       if trial["warm_start_checkpoint"] else None),
            )
    subprocess.run([
        sys.executable, str(tools_dir / "select_baseline_tuning.py"),
        "--tuning_dir", str(output_dir), "--expected_seeds", *map(str, seeds),
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("B4", "B5"), required=True)
    parser.add_argument("--study", choices=("staged", "hard_negatives", "sampling"), default="staged")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--parent_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run_study(args.model, args.dataset_dir.resolve(), args.parent_dir.resolve(),
              args.output_dir.resolve(), args.seeds, args.device, study=args.study)


if __name__ == "__main__":
    main()
