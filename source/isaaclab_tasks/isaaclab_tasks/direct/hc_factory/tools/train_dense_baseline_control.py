#!/usr/bin/env python3
"""Run fresh dense B4/B5 controls in existing directories, archiving prior files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from factory_baselines import MultiTaskLossConfig, TorchTrainConfig, train_torch_baseline
from factory_baselines.artifacts import archive_files
from factory_baselines.dataset import load_shared_dataset
from factory_bn_shared.bundle import file_hash


DENSE_VARIANTS = ("history_control", "graph_context", "upcoming_weighted", "three_class", "near_precursor", "far_precursor", "onset_aux", "temporal_attention", "history_graph_refine", "event_fbeta", "joint_onset")


def dense_configuration(model: str, variant: str, seed: int, device: str) -> tuple:
    if model not in {"B4", "B5"}:
        raise ValueError("Expected B4 or B5")
    if variant not in DENSE_VARIANTS:
        raise ValueError("Expected a registered dense variant")
    b4 = model == "B4"
    training = TorchTrainConfig(
        training_profile=f"dense_{variant}_v2", evaluate_test=False,
        seed=seed, device=device, batch_size=24 if b4 else 16,
        max_epochs=60, min_epochs=10 if b4 else 15, patience=10 if b4 else 20,
        learning_rate=3e-4 if b4 else 1.5e-4, weight_decay=1e-2,
        event_oversample_factor=1.0,
    )
    overrides = dict(gru_hidden=128, gru_layers=1, dropout=.2,
                     temporal_readout="last_mean", node_embedding=0,
                     event_context=variant == "graph_context",
                     event_head="three_class" if variant == "three_class" else "binary")
    if variant in {"near_precursor", "far_precursor", "onset_aux", "temporal_attention", "history_graph_refine", "event_fbeta", "joint_onset"}:
        overrides["event_precursor"] = "near_far" if variant == "far_precursor" else "near"
    if variant == "temporal_attention":
        overrides["temporal_readout"] = "last_attention"
    if variant in {"onset_aux", "joint_onset"}:
        overrides["event_onset_aux"] = True
    if variant == "joint_onset":
        overrides["event_onset_joint"] = True
    if variant == "history_graph_refine":
        overrides["history_graph_refine"] = True
    overrides.update({"gcn_hidden": 64} if b4 else {"gat_hidden": 64, "gat_heads": 4})
    loss = MultiTaskLossConfig(
        event_will_upcoming_pos_weight=12.0 if variant == "upcoming_weighted" else 4.0,
        lambda_event_onset_aux=1.0 if variant in {"onset_aux", "joint_onset"} else 0.0,
        event_fbeta_weight=.8 if variant == "event_fbeta" else 0.0,
    )
    return training, overrides, loss


TRAINING_FILES = [
    "best.pt", "fallback_best.pt", "last.pt", "config.json", "history.csv",
    "run_summary.json", "metrics.json", "metrics_initial_validation.json",
    "confusion_matrix.csv",
] + [
    f"{stem}_{split}.{suffix}"
    for split in ("train", "validation", "test")
    for stem, suffix in (
        ("metrics", "json"), ("predictions", "csv"),
        ("occupancy_events", "csv"), ("confusion_matrix", "csv"),
    )
]


def run_control(model: str, dataset_dir: Path, output_dir: Path, archive_tag: str,
                seed: int, device: str, variant: str = "history_control") -> dict:
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != "dev_xwt" or repo.name != "BSTAN_isaac_factory":
        raise ValueError("Run only in the server BSTAN_isaac_factory repo on dev_xwt")
    dataset_dir, output_dir = dataset_dir.resolve(), output_dir.resolve()
    for directory in (dataset_dir, output_dir):
        if not directory.is_dir() or not directory.is_relative_to(repo):
            raise ValueError(f"Reuse an existing directory inside BSTAN_isaac_factory: {directory}")
    train_config, overrides, loss_config = dense_configuration(model, variant, seed, device)
    if not archive_tag or not all(c.isalnum() or c in "_-" for c in archive_tag):
        raise ValueError("Invalid archive tag")
    record_path = output_dir / f"dense_control_{archive_tag}.json"
    archive_path = output_dir / f"model_before_{archive_tag}.zip"
    if record_path.exists() or archive_path.exists():
        raise FileExistsError("This archive tag already exists; do not overwrite an earlier run")
    payload, manifest = load_shared_dataset(dataset_dir)
    if manifest.get("shared_bundle_alignment", {}).get("status") != "passed":
        raise ValueError("Complete the canonical export alignment audit before training")
    from factory_baselines.precursor import attach_precursor
    payload, feature_contract = attach_precursor(
        payload, manifest, dataset_dir, overrides.get("event_precursor", "none"), ("train", "validation"),
    )
    from factory_baselines.onset_history import attach_onset_history
    payload, onset_history_contract = attach_onset_history(
        payload, manifest, dataset_dir, overrides.get("event_onset_joint", False), ("train", "validation"),
    )
    del payload
    saved = None
    if any((output_dir / name).exists() for name in TRAINING_FILES):
        saved = str(archive_files(output_dir, TRAINING_FILES, archive_path.name))
    record = {
        "status": "started", "baseline_id": model, "seed": seed, "variant": variant,
        "dataset_manifest_sha256": file_hash(dataset_dir / "dataset_manifest.json"),
        "prior_model_archive": saved, "initialization": "from_scratch",
        "selection_split": "validation", "test_evaluated": False,
        "comparison_role": "exploratory_upcoming_optimization_not_verified_main_cohort",
        "event_supervision_partition": "positive_start_zero_ongoing_positive_start_greater_zero_upcoming",
        "event_head": overrides["event_head"],
        "temporal_readout": overrides["temporal_readout"],
        "parent_control": ("onset_aux" if variant == "joint_onset" else "near_precursor"
                           if variant in {"far_precursor", "onset_aux", "temporal_attention", "history_graph_refine", "event_fbeta"} else None),
        "event_soft_fbeta": {
            "enabled": variant == "event_fbeta",
            "inside_event_loss_weight": loss_config.event_fbeta_weight,
            "beta": loss_config.event_fbeta_beta,
            "main_reference_commit": "20c40e230aedee6aef2429d352413fbcf0fa571a",
            "scope": "equal_all_events_and_upcoming_soft_Fbeta_using_existing_BCE_weights",
        },
        "history_graph_refinement": {
            "enabled": variant == "history_graph_refine",
            "mechanism": "extra_residual_GCN_or_GAT_on_pooled_GRU_histories_zero_output_projection",
            "additional_attention_dropout": 0.0,
        },
        "onset_auxiliary": {
            "enabled": variant in {"onset_aux", "joint_onset"}, "coefficient": loss_config.lambda_event_onset_aux,
            "loss": "half_mean_upcoming_BCE_plus_half_mean_negative_BCE_per_batch_absent_class_zero",
            "ongoing_targets_excluded": True, "used_for_report_decision": variant == "joint_onset",
            "parent_control": ("onset_aux" if variant == "joint_onset" else
                               "near_precursor" if variant == "onset_aux" else None),
        },
        "input_feature_contract": feature_contract,
        "event_classification_loss": "three_class_cross_entropy" if variant == "three_class" else "binary_cross_entropy",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    }
    if onset_history_contract is not None:
        record["onset_history_contract"] = onset_history_contract
        record["joint_onset_reporting"] = {
            "score": "observed_history_hot_then_continue_else_max_continue_onset_logits",
            "score_used_in_main_event_loss": True,
            "legacy_hist_last_hot_added_to_model_input": False,
            "original_labels_decoder_threshold_selection_unchanged": True,
        }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        summary = train_torch_baseline(
            model_kind="b4_gcn_gru" if model == "B4" else "b5_gat_gru",
            dataset_dir=dataset_dir, output_dir=output_dir,
            model_overrides=overrides, train_config=train_config,
            loss_config=loss_config,
        )
    except Exception as error:
        record.update(status="failed", error=str(error))
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise
    record.update(status=summary["status"], summary=summary)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("B4", "B5"), required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--archive_tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=DENSE_VARIANTS, default="history_control")
    print(json.dumps(run_control(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
