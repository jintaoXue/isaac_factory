#!/usr/bin/env python3
"""Diagnose train/validation event errors without changing scoring or selecting models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.metrics import _binary_metrics
from factory_baselines.evaluation import event_rule_kwargs
from factory_baselines.torch_trainer import (
    _manifest_hash,
    _model_inputs,
    _resolve_device,
    load_checkpoint,
)
from factory_bn_shared.remain import node_event_targets, station_report_metrics


def summarize_schedule_strata(arrays, sample_rows, node_ids, audit, support, historical, split, threshold):
    """Join retrospective strata after prediction; do not filter reports or choose a threshold."""
    if split not in {"train", "validation"}:
        raise ValueError("Schedule strata are restricted to train/validation")
    selected = [row for row in sample_rows if row["split"] == split]
    lookup = {int(row["sample_index"]): row for row in selected}
    indices = np.asarray(arrays["sample_index"]).reshape(-1)
    if len(lookup) != len(selected) or len(set(indices.tolist())) != len(indices) or set(indices.tolist()) != set(lookup):
        raise ValueError("Prediction/sample-index identities differ")
    episodes = {ep["group_id"]: ep for ep in support["episodes"]}
    compatibility = {row["group_id"]: row["match_class"] for row in historical["episodes"]}
    plans = {row["group_id"]: row for row in audit["episode_plan_reconstruction"]}
    train_support = Counter((event["resource_id"], ep["scenario_id"])
        for ep in support["episodes"] if ep["split"] == "train" for event in ep["upcoming_onsets"])
    targets = {}
    for row in audit["upcoming_targets"]:
        if row["split"] != split:
            continue
        key = (row["group_id"], row["resource_id"], row["onset_window_index"], row["start_index"])
        if key in targets:
            raise ValueError("Duplicate audited upcoming anchor")
        targets[key] = row
    valid = arrays["occ_node_mask"] > .5
    positive = valid & (arrays["event_will"] > .5)
    upcoming = positive & (arrays["event_start"] > 0)
    negative = valid & ~positive
    probability = arrays["will_probability"]
    decoded = np.where(arrays["hist_last_hot"] > .5, 0, arrays["predicted_start"])
    if probability.shape != (len(indices), len(node_ids)) or not np.isfinite(probability).all():
        raise ValueError("Invalid node probability shape/values")
    family = []
    records, seen = [], set()
    for position, sample_index in enumerate(indices):
        sample = lookup[int(sample_index)]
        group = sample["group_id"]
        ep, plan = episodes[group], plans[group]
        if ep["split"] != split or plan["split"] != split or ep["scenario_id"] != sample["scenario_id"]:
            raise ValueError("Episode/split/scenario mismatch")
        if plan["schedule_mode"] == "resample_per_episode":
            label = compatibility[group]
        else:
            if group in compatibility:
                raise ValueError("Unexpected reconstructed non-resampled episode")
            label = "no_resampled_schedule"
        family.append(label)
        for node in np.flatnonzero(upcoming[position]):
            start = int(arrays["event_start"][position, node])
            first_s = float(sample["first_future_start_s"])
            first = round(first_s / 60)
            if first_s != first * 60 or start not in (1, 2):
                raise ValueError("Upcoming window contract differs")
            key = (group, node_ids[node], first + start, start)
            if key not in targets or key in seen:
                raise ValueError("Predicted target identity differs from frozen onset audit")
            seen.add(key)
            target = targets[key]
            if target["first_future_time_s"] != first_s:
                raise ValueError("Forecast time differs from runtime audit")
            p = float(probability[position, node])
            timing_ok = abs(int(decoded[position, node]) - start) <= 3
            count = train_support[(node_ids[node], ep["scenario_id"])]
            support_bin = "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4-10" if count <= 10 else ">10"
            records.append(dict(sample_index=int(sample_index), group_id=group, resource_id=node_ids[node],
                onset_window_index=first + start, target_start=start, generator_compatibility=label,
                future_local_runtime_start=target["matched_future_local_runtime_start"],
                no_prior_runtime_start=target["no_prior_recorded_runtime_start"],
                train_joint_unique_onset_support=count, train_joint_support_bin=support_bin,
                probability=p, predicted_start=int(arrays["predicted_start"][position, node]),
                decoded_start=int(decoded[position, node]), report_hit=bool(p >= threshold and timing_ok),
                probability_miss=bool(p < threshold), timing_miss=bool(p >= threshold and not timing_ok)))
    if seen != set(targets):
        raise ValueError("Incomplete audited upcoming target coverage")

    def counts(rows):
        identities = {(x["group_id"], x["resource_id"], x["onset_window_index"]) for x in rows}
        hit_ids = {(x["group_id"], x["resource_id"], x["onset_window_index"]) for x in rows if x["report_hit"]}
        return dict(window_targets=len(rows), unique_onsets=len(identities), report_hits=sum(x["report_hit"] for x in rows),
            unique_onsets_with_any_hit=len(hit_ids), probability_misses=sum(x["probability_miss"] for x in rows),
            timing_misses=sum(x["timing_miss"] for x in rows),
            probability_q10_q50_q90=np.quantile([x["probability"] for x in rows], [.1, .5, .9]).tolist() if rows else None)

    groups = {}
    for field in ("generator_compatibility", "future_local_runtime_start", "train_joint_support_bin"):
        groups[field] = {str(value): counts([x for x in records if x[field] == value])
                         for value in sorted({x[field] for x in records}, key=str)}
    family = np.asarray(family)
    ranking = {}
    for label in sorted(set(family.tolist())):
        mask = np.broadcast_to((family == label)[:, None], upcoming.shape) & (upcoming | negative)
        m = _binary_metrics(upcoming[mask].astype(np.int64), probability[mask])
        ranking[label] = dict(positive_count=m["positive_count"], negative_count=m["negative_count"],
            tie_aware_average_precision=m["pr_auc"], roc_auc=m["roc_auc"])
    return dict(version="factory_frozen_upcoming_schedule_strata_v1", saved_threshold=threshold,
        summary=counts(records), groups=groups, compatibility_upcoming_vs_negative=ranking, upcoming_targets=records,
        scope="Post-prediction diagnosis only. No future schedule, support count, or label enters inference, filtering, threshold selection or checkpoint selection.",
        limitations=["Compatibility does not uniquely identify the historical collector version.",
            "Runtime temporal coincidence does not establish causal unpredictability.",
            "Training support counts include the target training episode; validation support counts use training episodes only.",
            "Unique onsets may span multiple strata because anchors have different observation cutoffs.",
            "These current B4 joint-onset and B5 vector-GAT checkpoints are distinct negative ablations, not an architecture-isolated B4/B5 comparison."])


def attach_schedule_strata(report, arrays, dataset_dir, manifest, split, threshold):
    expected = {
        "baseline_upcoming_schedule_support20260913.json": "ebeead2dff053e3b8041ad563d3e95519c5a58b3b91c4006b0586cc6e3d86d02",
        "dense_event_support20260912.json": "56d2dc07169d12e420ed1d2877bcf3290d2cd53d0abdd317ea6581845b32cd93",
        "baseline_upcoming_schedule_historical_match20260913.json": "4890663f63dbb570f4e5591cea441f6380ac39911ed20e785cfcbaa3587cc193",
    }
    sources = []
    for name, expected_hash in expected.items():
        content = (dataset_dir / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("Frozen schedule source changed: " + name)
        sources.append(json.loads(content))
    if sources[0]["manifest_sha256"] != hashlib.sha256((dataset_dir / "dataset_manifest.json").read_bytes()).hexdigest():
        raise ValueError("Schedule audit and model dataset manifests differ")
    with (dataset_dir / "model_sample_index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = summarize_schedule_strata(arrays, rows, manifest["node_ids"], *sources, split, threshold)
    canonical = next(row for row in report["thresholds"] if row["threshold"] == threshold)
    for key, original in (("window_targets", "n_true_upcoming"), ("probability_misses", "upcoming_probability_misses"), ("timing_misses", "upcoming_timing_misses")):
        if result["summary"][key] != canonical[original]:
            raise ValueError("Stratum totals differ from canonical upcoming metrics")
    if result["summary"]["report_hits"] != canonical["n_true_upcoming"] - canonical["upcoming_probability_misses"] - canonical["upcoming_timing_misses"]:
        raise ValueError("Stratum hits differ from canonical upcoming report")
    result["source_files_sha256"] = expected
    result["sample_index_sha256"] = hashlib.sha256((dataset_dir / "model_sample_index.csv").read_bytes()).hexdigest()
    report["schedule_strata"] = result


def load_diagnostic_checkpoint(path: Path, device: torch.device, archive_member: str | None):
    with path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {"checkpoint_file_sha256": source_hash, "checkpoint_archive_member": archive_member}
    if archive_member is None:
        model, checkpoint = load_checkpoint(path, device)
        return model, checkpoint, provenance
    if Path(archive_member).name != archive_member or archive_member in {"", ".", ".."}:
        raise ValueError("Checkpoint archive member must be a direct-child filename")
    with zipfile.ZipFile(path) as archive:
        for name in ("archive_manifest.json", archive_member):
            if archive.namelist().count(name) != 1:
                raise ValueError(f"Expected one archive member: {name}")
        manifest = json.loads(archive.read("archive_manifest.json"))
        content = archive.read(archive_member)
    member_hash = hashlib.sha256(content).hexdigest()
    if manifest.get(archive_member) != member_hash:
        raise ValueError("Archived checkpoint checksum does not match its manifest")
    provenance["checkpoint_member_sha256"] = member_hash
    # Read the verified member in memory; do not restore over a live training directory.
    model, checkpoint = load_checkpoint(io.BytesIO(content), device)
    return model, checkpoint, provenance


def attach_node_catalog(report: dict, catalog_path: Path, manifest: dict) -> None:
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        catalog = list(csv.DictReader(stream))
    by_index = {int(node["node_index"]): node for node in catalog}
    if len(by_index) != len(catalog) or set(by_index) != set(range(len(manifest["node_ids"]))):
        raise ValueError("Node catalog indices do not match the dataset manifest")
    for index, node_id in enumerate(manifest["node_ids"]):
        node = by_index[index]
        if (node["resource_id"] != node_id or node["resource_type"] !=
                manifest["resource_types"][int(node["resource_type_index"])]):
            raise ValueError("Node catalog identities/types differ from the dataset manifest")
    for row in report["thresholds"]:
        for node in row["per_node"]:
            source = by_index[node["node_index"]]
            node.update(resource_id=source["resource_id"], resource_type=source["resource_type"])


def summarize_event_kinds(arrays: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> dict:
    classes = ("none", "ongoing", "upcoming")
    valid = groups["negative"] | groups["ongoing"] | groups["upcoming"]
    probabilities = arrays["event_kind_probability"]
    if probabilities.shape != (*valid.shape, 3):
        raise ValueError("Event class probabilities must have shape (samples, nodes, 3)")
    scores = probabilities[valid]
    if (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or not np.allclose(scores.sum(-1), 1.0, atol=1e-6)):
        raise ValueError("Invalid event class probability distribution")
    if not np.allclose(scores[:, 1:].sum(-1), arrays["will_probability"][valid], atol=1e-6):
        raise ValueError("Event class probabilities do not match the common event probability")
    target = np.where(groups["ongoing"], 1, np.where(groups["upcoming"], 2, 0))[valid]
    predicted = scores.argmax(-1)
    confusion = np.bincount(target * 3 + predicted, minlength=9).reshape(3, 3)
    per_class = {}
    for index, name in enumerate(classes):
        selected = target == index
        count = int(selected.sum())
        per_class[name] = {
            "count": count,
            "argmax_recall": float(confusion[index, index] / count) if count else None,
            "mean_predicted_probabilities": scores[selected].mean(0).tolist() if count else None,
        }
    upcoming_or_negative = target != 1
    upcoming_ranking = _binary_metrics(
        (target[upcoming_or_negative] == 2).astype(np.int64),
        scores[upcoming_or_negative, 2],
    )
    return {
        "class_order": list(classes),
        "sample_count": len(target),
        "confusion_rows_true_columns_argmax": confusion.tolist(),
        "argmax_accuracy": float((target == predicted).mean()) if len(target) else None,
        "per_true_class": per_class,
        "upcoming_class_vs_negative": {
            "positive_count": upcoming_ranking["positive_count"],
            "negative_count": upcoming_ranking["negative_count"],
            "tie_aware_average_precision": upcoming_ranking["pr_auc"],
            "roc_auc": upcoming_ranking["roc_auc"],
        },
        "scope": "Retrospective subtype diagnostic, not the event report decision. "
                 "Report probabilities remain the sum of ongoing and upcoming probabilities; "
                 "argmax classes and subtype scores do not select thresholds or checkpoints.",
    }


def summarize_events(arrays: dict[str, np.ndarray], thresholds: list[float]) -> dict:
    valid = arrays["occ_node_mask"] > 0.5
    will, start_target, _ = node_event_targets(
        arrays["y_hot"], remain_mask=arrays["remain_mask"],
        occ_node_mask=arrays["occ_node_mask"], hist_last_hot=arrays["hist_last_hot"],
        **event_rule_kwargs(8),
    )
    if not np.array_equal(will[valid], arrays["event_will"][valid]) or not np.array_equal(
        start_target[will > .5], arrays["event_start"][will > .5]
    ):
        raise ValueError("Diagnostic targets differ from the current event contract")
    positive = (arrays["event_will"] > 0.5) & valid
    start = arrays["event_start"]
    groups = {
        "ongoing": positive & (start == 0),
        "upcoming": positive & (start > 0),
        "negative": valid & ~positive,
    }
    probability = arrays["will_probability"]
    decoded_start = np.where(
        arrays["hist_last_hot"] > 0.5, 0, arrays["predicted_start"]
    )
    future_observed = arrays["remain_mask"] > 0.5
    short_horizon = np.broadcast_to(future_observed.sum(axis=1)[:, None] < 8, valid.shape)
    any_future_hot = ((arrays["y_hot"] > 0.5) & future_observed[:, :, None]).any(axis=1)
    negative_kinds = {
        "short_observed_horizon": groups["negative"] & short_horizon,
        "hot_without_qualifying_event": groups["negative"] & ~short_horizon & any_future_hot,
        "no_future_hot": groups["negative"] & ~short_horizon & ~any_future_hot,
    }
    historical_hot = arrays["hist_last_hot"] > 0.5
    restarted = groups["upcoming"] & historical_hot
    output = {
        "groups": {}, "thresholds": [], "ranking": {},
        "training_partition_audit": {
            "definition": "positive start==0 ongoing; positive start>0 upcoming",
            "upcoming_with_hot_history": int(restarted.sum()),
            "upcoming_with_hot_history_windows": int(restarted.any(axis=1).sum()),
            "ongoing_with_cold_history": int((groups["ongoing"] & ~historical_hot).sum()),
            "note": "The previous hist_hot-based loss grouped restarted upcoming as ongoing "
                    "and omitted their start loss. Counts audit potential exposure, not effect size.",
        },
    }
    ranking_groups = {
        "all_events": (positive, valid),
        "ongoing_vs_negative": (groups["ongoing"], groups["ongoing"] | groups["negative"]),
        "upcoming_vs_negative": (groups["upcoming"], groups["upcoming"] | groups["negative"]),
        "events_vs_short_hot_negative": (
            positive, positive | negative_kinds["hot_without_qualifying_event"]
        ),
    }
    for name, (target, mask) in ranking_groups.items():
        metrics = _binary_metrics(target[mask].astype(np.int64), probability[mask])
        count = int(mask.sum())
        prevalence = float(target[mask].mean()) if count else None
        output["ranking"][name] = {
            "sample_count": count,
            "positive_count": metrics["positive_count"],
            "negative_count": metrics["negative_count"],
            "positive_rate": prevalence,
            "tie_aware_average_precision": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "ap_over_prevalence": (
                metrics["pr_auc"] / prevalence if prevalence else None
            ),
        }
    for name, mask in groups.items():
        values = probability[mask]
        output["groups"][name] = {
            "count": int(mask.sum()),
            "will_q10_q50_q90": (
                np.quantile(values, [0.1, 0.5, 0.9]).tolist() if values.size else None
            ),
            "hist_hot_count": int((mask & (arrays["hist_last_hot"] > 0.5)).sum()),
        }
        if name != "negative":
            errors = np.abs(decoded_start[mask] - start[mask])
            output["groups"][name]["start_within_tolerance_rate"] = (
                float((errors <= 3).mean()) if errors.size else None
            )
    for threshold in thresholds:
        report = station_report_metrics(
            arrays["y_hot"], probability, arrays["predicted_start"],
            arrays["predicted_duration"], arrays["remain_mask"],
            arrays["occ_node_mask"], threshold=threshold,
            **event_rule_kwargs(8), start_tol_windows=3,
            hist_last_hot=arrays["hist_last_hot"], force_ongoing_will=False,
        )
        row = {"threshold": threshold, **report}
        predicted = probability >= threshold
        for name in ("ongoing", "upcoming"):
            mask = groups[name]
            probability_miss = mask & ~predicted
            timing_miss = mask & predicted & (np.abs(decoded_start - start) > 3)
            row[f"{name}_probability_misses"] = int(probability_miss.sum())
            row[f"{name}_timing_misses"] = int(timing_miss.sum())
        row["false_positive_stations"] = int((predicted & groups["negative"]).sum())
        timing_error = positive & predicted & (np.abs(decoded_start - start) > 3)
        hits = positive & predicted & ~timing_error
        false_alarm = (predicted & groups["negative"]) | timing_error
        counts = {"true_event_wrong_start": int(timing_error.sum())}
        historical_hot = arrays["hist_last_hot"] > 0.5
        for kind, mask in negative_kinds.items():
            for state, hist_mask in (("historically_hot", historical_hot),
                                     ("historically_cold", ~historical_hot)):
                counts[f"{kind}_{state}"] = int((predicted & mask & hist_mask).sum())
        row["report_false_alarm_breakdown"] = counts
        row["report_false_alarm_count"] = int(false_alarm.sum())
        row["per_node"] = [
            {
                "node_index": index,
                "predicted": int((predicted[:, index] & valid[:, index]).sum()),
                "true": int(positive[:, index].sum()),
                "report_hits": int(hits[:, index].sum()),
                "false_alarms": int(false_alarm[:, index].sum()),
                "wrong_start": int(timing_error[:, index].sum()),
                "negative_subgroups": {
                    name: int((predicted[:, index] & mask[:, index]).sum())
                    for name, mask in negative_kinds.items()
                },
            }
            for index in range(valid.shape[1]) if bool(valid[:, index].any())
        ]
        row["predicted_duration_q25_q50_q75"] = (
            np.quantile(arrays["predicted_duration"][predicted & valid], [.25, .5, .75]).tolist()
            if bool((predicted & valid).any()) else None
        )
        if sum(counts.values()) != int(false_alarm.sum()):
            raise AssertionError("False-alarm diagnostic groups must be disjoint and exhaustive")
        expected_precision = float(hits.sum()) / max(int((predicted & valid).sum()), 1)
        if not np.isclose(expected_precision, row["report_precision"]):
            raise ValueError("Event diagnostic targets differ from canonical report metrics")
        output["thresholds"].append(row)
    if "event_kind_probability" in arrays:
        output["event_kind_diagnostics"] = summarize_event_kinds(arrays, groups)
    if "event_onset_probability" in arrays:
        score = arrays["event_onset_probability"]
        if score.shape != valid.shape:
            raise ValueError("Onset probabilities must share the event sample/node grid")
        if not np.isfinite(score[valid]).all() or (score[valid] < 0).any() or (score[valid] > 1).any():
            raise ValueError("Invalid onset probabilities")
        selected = groups["upcoming"] | groups["negative"]
        metrics = _binary_metrics(groups["upcoming"][selected].astype(np.int64), score[selected])
        output["onset_auxiliary_diagnostics"] = {
            "upcoming_count": metrics["positive_count"], "negative_count": metrics["negative_count"],
            "upcoming_vs_negative_ap": metrics["pr_auc"], "upcoming_vs_negative_auc": metrics["roc_auc"],
            "score_q10_q50_q90": {
                name: np.quantile(score[mask], [.1, .5, .9]).tolist() if mask.any() else None
                for name, mask in groups.items() if name in {"upcoming", "negative"}
            },
            "scope": (
                "Onset participates in the jointly trained canonical event score; this separate ranking is diagnostic only."
                if "event_continue_probability" in arrays else
                "Training-only auxiliary head; independent ranking diagnosis, excluded from canonical report decisions and checkpoint selection."
            ),
        }
    return output


def observe_joint_onset(result: dict[str, torch.Tensor], history: torch.Tensor) -> dict:
    """Read both existing heads from one forward; verify the actual joint score."""
    names = ("event_will_continue_logit", "event_onset_logit", "event_will_logit")
    if any(name not in result for name in names):
        raise ValueError("Joint observation requires continue, onset and combined logits")
    cont, onset, combined = (result[name] for name in names)
    if history.shape != cont.shape or onset.shape != cont.shape or combined.shape != cont.shape:
        raise ValueError("Joint observation requires matching sample/node grids")
    if any(not bool(torch.isfinite(x).all()) for x in (history, cont, onset, combined)):
        raise ValueError("Non-finite joint observation")
    if bool(((history != 0) & (history != 1)).any()):
        raise ValueError("Joint observation requires binary observed history")
    cold = history <= .5
    expected = torch.where(cold, torch.maximum(cont, onset), cont)
    if not torch.equal(combined, expected):
        raise ValueError("Observed score differs from the registered joint rule")
    return {
        "event_continue_probability": cont.sigmoid(),
        "event_history_hot": history,
        "joint_onset_selected": cold & (onset > cont),
        "joint_cold_logit_tie": cold & (onset == cont),
    }


def summarize_joint_onset(arrays: dict[str, np.ndarray], threshold: float) -> dict:
    """Describe onset's extra reports at the saved threshold without selecting a policy."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid saved joint threshold")
    valid = arrays["occ_node_mask"] > .5
    keys = ("will_probability", "event_continue_probability", "event_onset_probability",
            "event_history_hot", "joint_onset_selected", "joint_cold_logit_tie")
    for key in keys:
        if key not in arrays or arrays[key].shape != valid.shape:
            raise ValueError("Missing or mismatched joint observation grid")
        values = arrays[key][valid]
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Invalid joint observation values")
        if key in keys[3:] and not np.isin(values, [0, 1]).all():
            raise ValueError("Joint gate and branch flags must be binary")
    combined, cont, onset, history, selected, tie = (arrays[key] for key in keys)
    cold = history <= .5
    expected = np.where(cold, np.maximum(cont, onset), cont)
    if not np.allclose(combined[valid], expected[valid], rtol=0, atol=1e-7):
        raise ValueError("Joint probabilities differ from the registered rule")
    selected, tie = selected > .5, tie > .5
    if ((selected | tie) & ~cold & valid).any() or (selected & tie & valid).any():
        raise ValueError("Invalid joint branch selection")
    positive = (arrays["event_will"] > .5) & valid
    groups = dict(ongoing=positive & (arrays["event_start"] == 0),
                  upcoming=positive & (arrays["event_start"] > 0), negative=valid & ~positive)
    decoded_start = np.where(arrays["hist_last_hot"] > .5, 0, arrays["predicted_start"])
    timely = np.abs(decoded_start - arrays["event_start"]) <= 3
    extra = valid & (combined >= threshold) & (cont < threshold)
    if (valid & (combined < threshold) & (cont >= threshold)).any():
        raise ValueError("Joint max must not remove continue reports")
    group_stats = {}
    for name, mask in groups.items():
        lift = (combined - cont)[mask]
        group_stats[name] = dict(
            count=int(mask.sum()), observed_hot=int((mask & ~cold).sum()),
            onset_selected=int((mask & selected).sum()), cold_logit_ties=int((mask & tie).sum()),
            probability_lift_q10_q50_q90=np.quantile(lift, [.1, .5, .9]).tolist() if lift.size else None,
            extra_reports=int((mask & extra).sum()),
            extra_hits=int((mask & extra & positive & timely).sum()),
            extra_false_alarms=int((mask & extra & (~positive | ~timely)).sum()),
        )
    ranked = groups["upcoming"] | groups["negative"]
    ranking = {}
    for name, score in (("continue", cont), ("onset", onset), ("joint", combined)):
        metrics = _binary_metrics(groups["upcoming"][ranked].astype(np.int64), score[ranked])
        ranking[name] = dict(upcoming_vs_negative_ap=metrics["pr_auc"],
                             upcoming_vs_negative_auc=metrics["roc_auc"])
    # Each policy uses the same checkpoint, threshold, start/duration and legacy decoder.
    baseline = {k: v for k, v in arrays.items() if k not in keys[1:]}
    joint_report = summarize_events(baseline, [threshold])["thresholds"][0]
    continue_report = summarize_events({**baseline, "will_probability": cont}, [threshold])["thresholds"][0]
    return dict(
        version="factory_frozen_joint_onset_observation_v1", saved_threshold=threshold,
        groups=group_stats, ranking=ranking,
        reports_at_saved_threshold=dict(joint=joint_report, continue_only=continue_report),
        scope="Read-only output ablation within the jointly trained checkpoint. Extra reports are compared at its saved threshold. The continue-only result is not an independently trained baseline, and no threshold, checkpoint or reporting policy is selected or changed.",
    )


def predicted_hot_run_score(probability: np.ndarray) -> np.ndarray:
    """Score an eight-minute predicted run starting at 0, 1 or 2; no target inputs."""
    probability = np.asarray(probability)
    if probability.ndim != 3 or probability.shape[1] < 10:
        raise ValueError("Expected (samples, at least ten forecast windows, nodes)")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Expected finite predicted probabilities in [0, 1]")
    return np.stack([probability[:, start:start+8].min(axis=1)
                     for start in (0, 1, 2)]).max(axis=0)


def cold_onset_report_probability(event: np.ndarray, onset: np.ndarray,
                                  history_hot: np.ndarray) -> np.ndarray:
    """A frozen diagnostic policy using predictions and observed history only."""
    event, onset, history_hot = (np.asarray(x) for x in (event, onset, history_hot))
    if event.ndim != 2 or onset.shape != event.shape or history_hot.shape != event.shape:
        raise ValueError("Expected matching (samples, nodes) event/onset/history grids")
    for value in (event, onset, history_hot):
        if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Expected finite probabilities and history-hot values in [0, 1]")
    return np.where(history_hot > .5, event, np.maximum(event, onset))


def summarize_onset_reports(arrays: dict[str, np.ndarray], thresholds: list[float],
                            saved_threshold: float) -> dict:
    """Paired canonical scoring of fixed policies; no selection or deployment."""
    if "event_onset_probability" not in arrays:
        raise ValueError("Onset report comparison requires an independent onset head")
    if "event_kind_probability" in arrays:
        raise ValueError("Onset report comparison requires the binary event head")
    grid = sorted(set([*thresholds, saved_threshold]))
    if not grid or any(not np.isfinite(t) or not 0 <= t <= 1 for t in grid):
        raise ValueError("Expected finite report thresholds in [0, 1]")
    fused = cold_onset_report_probability(
        arrays["will_probability"], arrays["event_onset_probability"], arrays["hist_last_hot"],
    )
    policies = {}
    for name, score in (("event_only", arrays["will_probability"]), ("cold_onset_max", fused)):
        report = summarize_events({**arrays, "will_probability": score}, grid)
        policies[name] = {k: report[k] for k in ("ranking", "thresholds")}
    return {
        "version": "factory_frozen_onset_report_comparison_v1",
        "score_definition": "event probability when history_hot>0.5; otherwise max(event, onset)",
        "score_inputs": "Two predicted probabilities and observed last-history hot only; no future labels, horizon mask, true start, or predicted completion-time filtering",
        "threshold_source": "Frozen checkpoint training threshold sweep plus its saved report threshold; same grid for both policies",
        "thresholds": grid, "saved_report_threshold": saved_threshold,
        "policies": policies,
        "scope": "Fixed weights, shared predicted start/duration and canonical matching. Curves are diagnostics; no threshold or checkpoint is selected, no original report is replaced. No separate onset calibration or training with dual-head checkpoint selection is performed.",
    }


def independent_onset_report_mask(event: np.ndarray, onset: np.ndarray,
                                  history_hot: np.ndarray, event_threshold: float | None,
                                  onset_threshold: float | None) -> np.ndarray:
    """Two scalar thresholds; None disables that branch. No target inputs."""
    event, onset, history_hot = (np.asarray(x) for x in (event, onset, history_hot))
    if event.ndim != 2 or onset.shape != event.shape or history_hot.shape != event.shape:
        raise ValueError("Expected matching event/onset/history grids")
    for value in (event, onset, history_hot):
        if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Expected finite probabilities and history-hot values in [0, 1]")
    for threshold in (event_threshold, onset_threshold):
        if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("Expected a threshold in [0, 1] or None to disable the branch")
    reported = np.zeros(event.shape, dtype=bool)
    if event_threshold is not None:
        reported |= event >= event_threshold
    if onset_threshold is not None:
        reported |= (history_hot <= .5) & (onset >= onset_threshold)
    return reported


class _OnsetPrefixTree:
    """Dynamic score-tie groups; rightmost prefix satisfying 5*hits-4*reports.

    For a fixed event threshold, additional reports are a prefix in descending
    onset score. Its hit/upcoming counts cannot decrease with prefix length.
    The rightmost precision-feasible prefix therefore maximizes upcoming hits;
    it also has the most total hits among feasible prefixes. All score ties
    enter together. Integer slack implements precision>=0.8 without rounding.
    """

    def __init__(self, counts: np.ndarray, hits: np.ndarray, upcoming: np.ndarray):
        self.groups = len(counts)
        self.size = 1 << max(self.groups - 1, 0).bit_length()
        self.count = [0] * (2 * self.size)
        self.hit = [0] * (2 * self.size)
        self.up = [0] * (2 * self.size)
        self.max_prefix = [0] * (2 * self.size)
        for i, (n, h, u) in enumerate(zip(counts, hits, upcoming)):
            leaf = self.size + i
            self.count[leaf], self.hit[leaf], self.up[leaf] = int(n), int(h), int(u)
            self.max_prefix[leaf] = max(0, 5 * int(h) - 4 * int(n))
        for node in range(self.size - 1, 0, -1):
            self._pull(node)

    def _pull(self, node: int) -> None:
        left, right = 2 * node, 2 * node + 1
        self.count[node] = self.count[left] + self.count[right]
        self.hit[node] = self.hit[left] + self.hit[right]
        self.up[node] = self.up[left] + self.up[right]
        self.max_prefix[node] = max(
            self.max_prefix[left],
            5 * self.hit[left] - 4 * self.count[left] + self.max_prefix[right],
        )

    def remove(self, group: int, hit: bool, upcoming: bool) -> None:
        node = self.size + group
        self.count[node] -= 1
        self.hit[node] -= int(hit)
        self.up[node] -= int(upcoming)
        if not 0 <= self.up[node] <= self.hit[node] <= self.count[node]:
            raise AssertionError("Invalid remaining onset score group")
        self.max_prefix[node] = max(0, 5 * self.hit[node] - 4 * self.count[node])
        node //= 2
        while node:
            self._pull(node)
            node //= 2

    def rightmost(self, required_slack: int) -> tuple[int, int, int, int] | None:
        if self.max_prefix[1] < required_slack:
            return None
        node, count, hit, up = 1, 0, 0, 0
        while node < self.size:
            left, right = 2 * node, 2 * node + 1
            left_slack = 5 * (hit + self.hit[left]) - 4 * (count + self.count[left])
            if left_slack + self.max_prefix[right] >= required_slack:
                count += self.count[left]
                hit += self.hit[left]
                up += self.up[left]
                node = right
            else:
                node = left
        end = node - self.size
        if 5 * (hit + self.hit[node]) - 4 * (count + self.count[node]) >= required_slack:
            count += self.count[node]
            hit += self.hit[node]
            up += self.up[node]
            end += 1
        return min(end, self.groups), count, hit, up


def summarize_onset_frontier(arrays: dict[str, np.ndarray], saved_threshold: float) -> dict:
    """Exact empirical maximum over two scalar thresholds; explicitly an oracle.

    Future targets evaluate the frontier, never the alarm mask. This does not
    calibrate or select a deployable threshold and cannot estimate unseen-data
    performance. Runtime is O(N log N), rather than a quadratic threshold grid.
    """
    if "event_onset_probability" not in arrays:
        raise ValueError("Onset frontier requires an independent onset head")
    if "event_kind_probability" in arrays:
        raise ValueError("Onset frontier requires the binary event head")
    if not np.isfinite(saved_threshold) or not 0 <= saved_threshold <= 1:
        raise ValueError("Invalid saved threshold")
    valid = arrays["occ_node_mask"] > .5
    safe = []
    for key in ("will_probability", "event_onset_probability", "hist_last_hot"):
        value = np.asarray(arrays[key])
        if value.shape != valid.shape:
            raise ValueError("Expected the same valid sample/node grid")
        if not np.isfinite(value[valid]).all() or (value[valid] < 0).any() or (value[valid] > 1).any():
            raise ValueError("Invalid valid-node probability/history value")
        safe.append(np.where(valid, value, 0))
    event_grid, onset_grid, history_grid = safe
    positive, target_start, _ = node_event_targets(
        arrays["y_hot"], remain_mask=arrays["remain_mask"],
        occ_node_mask=arrays["occ_node_mask"], hist_last_hot=arrays["hist_last_hot"],
        **event_rule_kwargs(8),
    )
    if not np.array_equal(positive[valid], arrays["event_will"][valid]) or not np.array_equal(
        target_start[(positive > .5) & valid], arrays["event_start"][(positive > .5) & valid]
    ):
        raise ValueError("Frontier targets differ from the current event contract")
    positive = (positive > .5) & valid
    upcoming = positive & (target_start > 0)
    decoded_start = np.where(history_grid > .5, 0, arrays["predicted_start"])
    hits = positive & (np.abs(decoded_start - target_start) <= 3)
    event, onset, cold = event_grid[valid], onset_grid[valid], history_grid[valid] <= .5
    hit, up = hits[valid], (hits & upcoming)[valid]
    n_true, n_up = int(positive.sum()), int(upcoming.sum())
    levels = np.unique(onset[cold])[::-1]
    groups = np.full(len(event), -1, dtype=np.int64)
    groups[cold] = np.searchsorted(-levels, -onset[cold])
    counts = np.bincount(groups[cold], minlength=len(levels))
    hit_counts = np.bincount(groups[cold & hit], minlength=len(levels))
    up_counts = np.bincount(groups[cold & up], minlength=len(levels))
    tree = _OnsetPrefixTree(counts, hit_counts, up_counts)
    best = {"saved_event_threshold_P80": None, "all_thresholds_P80": None,
            "all_thresholds_P80_R70": None}
    base_n = base_h = base_u = 0

    def inspect(event_threshold: float | None, fixed: bool = False) -> None:
        prefix = tree.rightmost(4 * base_n - 5 * base_h)
        if prefix is None:
            return
        end, added_n, added_h, added_u = prefix
        n, h, u = base_n + added_n, base_h + added_h, base_u + added_u
        if n == 0:
            return  # Canonical precision is zero for an empty report set.
        if 5 * h < 4 * n:
            raise AssertionError("Frontier precision feasibility failed")
        keys = ["saved_event_threshold_P80"] if fixed else ["all_thresholds_P80"]
        if not fixed and 10 * h >= 7 * n_true:
            keys.append("all_thresholds_P80_R70")
        for key in keys:
            # Keep one witness for maximal upcoming hits; do not claim F1-optimal ties.
            if best[key] is None or u > best[key]["upcoming_hits"]:
                best[key] = dict(
                    event_threshold=event_threshold,
                    onset_threshold=float(levels[end - 1]) if end else None,
                    n_pred_who=n, n_matched_report=h, upcoming_hits=u,
                    ongoing_hits=h-u, report_false_alarm_count=n-h,
                    report_precision=h/n, report_recall=h/max(n_true, 1),
                    report_f1=2*h/max(n+n_true, 1), report_recall_upcoming=u/max(n_up, 1),
                )

    inspect(None)
    order = np.argsort(-event, kind="stable")
    sorted_event = event[order]
    saved_reports = event >= saved_threshold  # Preserve NumPy's array/scalar dtype semantics.
    ends = (np.flatnonzero(np.r_[sorted_event[1:] != sorted_event[:-1], True]) + 1
            if len(order) else np.array([], dtype=int))
    fixed_seen, begin = False, 0
    for end in ends:
        score = sorted_event[begin]
        if not fixed_seen and not saved_reports[order[begin]]:
            inspect(saved_threshold, fixed=True)
            fixed_seen = True
        entered = order[begin:end]
        base_n += len(entered)
        base_h += int(hit[entered].sum())
        base_u += int(up[entered].sum())
        for index in entered[cold[entered]]:
            tree.remove(int(groups[index]), bool(hit[index]), bool(up[index]))
        inspect(float(score))
        begin = end
    if not fixed_seen:
        inspect(saved_threshold, fixed=True)
    for point in best.values():
        if point is None:
            continue
        mask = independent_onset_report_mask(
            event_grid, onset_grid, history_grid, point["event_threshold"], point["onset_threshold"],
        )
        report = station_report_metrics(
            arrays["y_hot"], mask.astype(np.float32), arrays["predicted_start"],
            arrays["predicted_duration"], arrays["remain_mask"], arrays["occ_node_mask"],
            threshold=.5, **event_rule_kwargs(8), start_tol_windows=3,
            hist_last_hot=arrays["hist_last_hot"], force_ongoing_will=False,
        )
        for key in ("n_pred_who", "n_matched_report", "report_precision", "report_recall",
                    "report_f1", "report_recall_upcoming"):
            if abs(report[key] - point[key]) > 1e-10:
                raise AssertionError(f"Frontier witness differs from canonical report: {key}")
        point["canonical_report"] = report
    return {
        "version": "factory_frozen_independent_onset_frontier_v1",
        "policy": "event>=event_threshold OR (observed history_hot<=0.5 AND onset>=onset_threshold); None disables a branch",
        "coverage": "All distinct report sets from scalar thresholds in [0,1], plus disabled branches; score ties are atomic",
        "valid_targets": int(valid.sum()), "true_events": n_true, "true_upcoming": n_up,
        "distinct_event_scores": len(ends), "distinct_cold_onset_scores": len(levels),
        "event_states_examined": len(ends) + 1, "saved_event_threshold": saved_threshold,
        "precision_floor": .8, "additional_recall_floor": .7, "empirical_maxima": best,
        "scope": "Label-informed empirical oracle on this split and frozen checkpoint only. "
                 "Witness thresholds are not selected or deployed and are not an unbiased estimate of "
                 "calibrated generalization. Shared predicted start/duration, target definition and denominator "
                 "are unchanged. Does not bound new weights, other fusion functions or another backbone.",
    }


def summarize_prediction_heads(arrays: dict[str, np.ndarray], event_threshold: float) -> dict:
    """Retrospective score comparison, leaving event reports and selection untouched."""
    event = arrays["will_probability"]
    hot = predicted_hot_run_score(arrays["predicted_hot_probability"])
    if hot.shape != event.shape:
        raise ValueError("Event and predicted-hot scores refer to different sample/node grids")
    valid = arrays["occ_node_mask"] > .5
    positive = (arrays["event_will"] > .5) & valid
    upcoming = positive & (arrays["event_start"] > 0)
    negative = valid & ~positive
    ranked = upcoming | negative
    heads = {}
    for name, score in (("event_head", event), ("predicted_hot_run", hot)):
        metrics = _binary_metrics(upcoming[ranked].astype(np.int64), score[ranked])
        heads[name] = {
            "upcoming_count": int(upcoming.sum()), "negative_count": int(negative.sum()),
            "upcoming_vs_negative_ap": metrics["pr_auc"],
            "upcoming_vs_negative_auc": metrics["roc_auc"],
            "upcoming_score_q10_q50_q90": np.quantile(score[upcoming], [.1, .5, .9]).tolist() if upcoming.any() else None,
            "negative_score_q10_q50_q90": np.quantile(score[negative], [.1, .5, .9]).tolist() if negative.any() else None,
        }
    hot_threshold = .45  # Pinned main configuration; a diagnostic point, not selected here.
    event_alarm, hot_alarm = event >= event_threshold, hot >= hot_threshold
    overlap = {}
    for name, group in (("upcoming", upcoming), ("negative", negative)):
        overlap[name] = {
            "both": int((group & event_alarm & hot_alarm).sum()),
            "event_only": int((group & event_alarm & ~hot_alarm).sum()),
            "hot_only": int((group & ~event_alarm & hot_alarm).sum()),
            "neither": int((group & ~event_alarm & ~hot_alarm).sum()),
        }
        if sum(overlap[name].values()) != int(group.sum()):
            raise AssertionError("Prediction-head overlap must partition each diagnostic group")
    return {
        "version": "factory_frozen_head_comparison_v1",
        "hot_score_definition": "max over starts 0,1,2 of the minimum predicted hot probability in eight consecutive forecast windows",
        "hot_score_inputs": "predicted hot probabilities only; no ground-truth masks, history-hot state, event times, or predicted completion-time filtering",
        "heads": heads, "overlap": overlap,
        "event_threshold": event_threshold, "hot_threshold": hot_threshold,
        "scope": "Same frozen model and samples. AP excludes ongoing targets. Threshold counts measure score crossings, not canonical report hits. No threshold, checkpoint or alarm policy is selected or changed.",
    }


@torch.no_grad()
def predict_with_temporal_observation(model: torch.nn.Module, inputs: dict) -> tuple[dict, dict]:
    """Observe a single frozen forward; historical tensors alone determine weights."""
    pool = getattr(model, "history_attention", None)
    if pool is None or model.training:
        raise ValueError("Temporal observation requires a frozen attention-readout model in eval mode")
    observed = {}

    def capture(module, args, pooled):
        history = args[0]
        scores = torch.einsum("bth,h->bt", history, module.query) / (module.query.numel() ** .5)
        mean = history.mean(dim=1)
        observed.update(
            temporal_attention_weights=scores.softmax(dim=1),
            temporal_pool_shift_l2=(pooled - mean).norm(dim=-1),
            temporal_pool_mean_l2=mean.norm(dim=-1),
        )

    handle = pool.register_forward_hook(capture)
    try:
        result = model(**inputs)
    finally:
        handle.remove()
    if not observed:
        raise ValueError("The registered temporal attention module was not used")
    batch, nodes = inputs["node_mask"].shape
    return result, {
        key: value.reshape(batch, nodes, *value.shape[1:])
        for key, value in observed.items()
    }


def summarize_temporal_attention(arrays: dict[str, np.ndarray], query_l2: float) -> dict:
    """Describe attention by retrospective event groups, without changing reports."""
    valid = arrays["occ_node_mask"] > .5
    weights = arrays["temporal_attention_weights"]
    shift, mean = arrays["temporal_pool_shift_l2"], arrays["temporal_pool_mean_l2"]
    if weights.ndim != 3 or weights.shape[:2] != valid.shape or weights.shape[-1] < 1:
        raise ValueError("Attention weights must share the sample/node grid and contain history steps")
    if shift.shape != valid.shape or mean.shape != valid.shape:
        raise ValueError("Temporal pool norms must share the sample/node grid")
    if not np.isfinite(query_l2) or query_l2 < 0:
        raise ValueError("Invalid query norm")
    w = weights[valid]
    if (not np.isfinite(w).all() or (w < 0).any() or (w > 1).any()
            or not np.allclose(w.sum(-1), 1., atol=1e-6)):
        raise ValueError("Invalid attention probability distribution")
    for value in (shift, mean):
        if not np.isfinite(value[valid]).all() or (value[valid] < 0).any():
            raise ValueError("Invalid temporal pool norm")
    positive = (arrays["event_will"] > .5) & valid
    masks = {"ongoing": positive & (arrays["event_start"] == 0),
             "upcoming": positive & (arrays["event_start"] > 0),
             "negative": valid & ~positive}
    steps = weights.shape[-1]
    groups = {}
    for name, mask in masks.items():
        selected = weights[mask].astype(np.float64)
        row = {"count": int(mask.sum()), "mean_weights_oldest_to_newest": None}
        if mask.any():
            entropy = -(selected * np.log(np.maximum(selected, 1e-300))).sum(-1)
            statistics = {
                "max_weight": selected.max(-1),
                "total_variation_from_uniform": .5 * np.abs(selected - 1. / steps).sum(-1),
                "normalized_entropy": entropy / np.log(steps) if steps > 1 else np.zeros_like(entropy),
                "pool_shift_l2": shift[mask],
                "pool_shift_over_mean_l2": shift[mask] / np.maximum(mean[mask], 1e-8),
            }
            row["mean_weights_oldest_to_newest"] = selected.mean(0).tolist()
            row.update({key + "_q10_q50_q90": np.quantile(value, [.1, .5, .9]).tolist()
                        for key, value in statistics.items()})
        groups[name] = row
    return {
        "history_steps": steps, "query_l2": float(query_l2), "groups": groups,
        "relative_shift_denominator_floor": 1e-8,
        "scope": "One original forward. Weights use historical GRU states and the learned query only; "
                 "future labels only stratify statistics. No predictions, thresholds or checkpoint selection "
                 "change. Attention weights are not causal feature importance. Single-step normalized entropy is zero.",
    }


@torch.no_grad()
def predict_with_history_graph_observation(model: torch.nn.Module, inputs: dict) -> tuple[dict, dict]:
    """Measure the applied history-graph residual in one unmodified eval forward."""
    block = getattr(model, "history_graph", None)
    if block is None or model.training:
        raise ValueError("History graph observation requires a refinement model in eval mode")
    observed = {}

    def capture(module, args, refined):
        if observed:
            raise ValueError("Expected a single history graph invocation")
        history, _, mask = args
        baseline = history * mask[:, :, None].to(history.dtype)
        observed.update(history_graph_base_l2=baseline.norm(dim=-1),
                        history_graph_refined_l2=refined.norm(dim=-1),
                        history_graph_shift_l2=(refined - baseline).norm(dim=-1))

    handle = block.register_forward_hook(capture)
    try:
        result = model(**inputs)
    finally:
        handle.remove()
    if not observed:
        raise ValueError("The registered history graph module was not used")
    return result, observed


def summarize_history_graph(arrays: dict[str, np.ndarray], output_weight_l2: float) -> dict:
    """Describe learned residual size, not its causal contribution to predictions."""
    valid = arrays["occ_node_mask"] > .5
    norms = {name: arrays["history_graph_" + name + "_l2"] for name in ("base", "refined", "shift")}
    if valid.ndim != 2 or any(value.shape != valid.shape for value in norms.values()):
        raise ValueError("History graph norms must share the sample/node grid")
    if not np.isfinite(output_weight_l2) or output_weight_l2 < 0:
        raise ValueError("Invalid history graph output weight norm")
    for value in norms.values():
        if not np.isfinite(value[valid]).all() or (value[valid] < 0).any():
            raise ValueError("Invalid history graph norm")
    positive = (arrays["event_will"] > .5) & valid
    masks = {"ongoing": positive & (arrays["event_start"] == 0),
             "upcoming": positive & (arrays["event_start"] > 0),
             "negative": valid & ~positive}
    groups = {}
    for name, mask in masks.items():
        row = {"count": int(mask.sum()), "base_norm_below_floor_count": 0}
        if mask.any():
            row["base_norm_below_floor_count"] = int((norms["base"][mask] < 1e-8).sum())
            stats = {key + "_l2": value[mask].astype(np.float64) for key, value in norms.items()}
            stats["shift_over_base_l2"] = stats["shift_l2"] / np.maximum(stats["base_l2"], 1e-8)
            row.update({key + "_q10_q50_q90": np.quantile(value, [.1, .5, .9]).tolist()
                        for key, value in stats.items()})
        groups[name] = row
    return dict(output_projection_weight_l2=float(output_weight_l2), groups=groups,
                relative_shift_denominator_floor=1e-8,
                scope="One original eval forward; norms measure the actually applied residual. "
                      "Future labels only stratify statistics. Output projection was initialized to zero. "
                      "Nonzero updates show an active path, not beneficial or causal predictive information; "
                      "no predictions, thresholds or checkpoint selection are changed.")


@torch.no_grad()
def shared_neighbor_top_reversals(attention: torch.Tensor, edges: torch.Tensor,
                                  batch_size: int, steps: int) -> dict:
    """Count strict changes of the top common neighbor between query pairs.

    Read actual pre-dropout softmax weights. Chunk query pairs to avoid keeping
    the full time/query/query/neighbor/head grid in memory. Ties and differences
    at most 1e-7 are unresolved; masked neighbors never enter a comparison.
    """
    if (attention.ndim != 4 or edges.shape != attention.shape[:3]
            or attention.shape[1] != attention.shape[2]
            or attention.shape[0] != batch_size * steps):
        raise ValueError("Invalid attention/edge/history grid")
    if edges.dtype != torch.bool or not torch.isfinite(attention).all():
        raise ValueError("Invalid attention observation")
    rows, nodes, _, heads = attention.shape
    if ((attention < 0).any() or (attention > 1).any()
            or not torch.allclose(attention.sum(2)[edges.any(2)],
                                  torch.ones_like(attention.sum(2)[edges.any(2)]), atol=1e-6)):
        raise ValueError("Invalid attention probability distribution")
    pairs = torch.triu_indices(nodes, nodes, 1, device=attention.device)
    eligible_query = torch.zeros(rows, nodes, heads, dtype=torch.long, device=attention.device)
    reversed_query = torch.zeros_like(eligible_query)
    eligible_pair = torch.zeros(rows, heads, dtype=torch.long, device=attention.device)
    reversed_pair = torch.zeros_like(eligible_pair)
    for offset in range(0, pairs.shape[1], 32):
        left, right = pairs[:, offset:offset + 32]
        common = edges[:, left] & edges[:, right]
        eligible = (common.sum(2) >= 2).unsqueeze(-1).expand(-1, -1, heads)
        a = attention[:, left].masked_fill(~common.unsqueeze(-1), -1.)
        b = attention[:, right].masked_fill(~common.unsqueeze(-1), -1.)
        a_top, a_index = a.max(2, keepdim=True)
        b_top, b_index = b.max(2, keepdim=True)
        # A strict cross-preference establishes a real ordering reversal even
        # if there are additional tied top neighbors within either query.
        reversed_order = eligible & ((a_top - a.gather(2, b_index)).squeeze(2) > 1e-7)
        reversed_order &= ((b_top - b.gather(2, a_index)).squeeze(2) > 1e-7)
        for index in (left, right):
            eligible_query.index_add_(1, index, eligible.long())
            reversed_query.index_add_(1, index, reversed_order.long())
        eligible_pair += eligible.sum(1)
        reversed_pair += reversed_order.sum(1)
    return dict(
        query_eligible=eligible_query.reshape(batch_size, steps, nodes, heads).sum((1, 3)),
        query_reversed=reversed_query.reshape(batch_size, steps, nodes, heads).sum((1, 3)),
        pair_eligible_by_head=eligible_pair.reshape(batch_size, steps, heads).sum(1),
        pair_reversed_by_head=reversed_pair.reshape(batch_size, steps, heads).sum(1),
    )


@torch.no_grad()
def predict_with_gat_ranking_observation(model: torch.nn.Module, inputs: dict) -> tuple[dict, dict]:
    """Observe both original GAT forwards; do not recompute or replace attention."""
    from factory_baselines.b5_gat_gru import DenseGraphAttention
    layers = {name: getattr(model, name, None) for name in ("gat1", "gat2")}
    if model.training or any(not isinstance(layer, DenseGraphAttention) for layer in layers.values()):
        raise ValueError("GAT ranking observation requires a B5 model in eval mode")
    batch_size, steps = inputs["x"].shape[:2]
    observed, captured, handles = {}, {}, []

    def capture_weights(name):
        def hook(module, args, output):
            if name in captured:
                raise ValueError("Expected one attention invocation per GAT layer")
            captured[name] = args[0].detach()
        return hook

    def capture_layer(name):
        def hook(module, args, output):
            _, adjacency, mask = args
            edges = adjacency.bool() & mask.bool()[:, :, None] & mask.bool()[:, None, :]
            stats = shared_neighbor_top_reversals(captured[name], edges, batch_size, steps)
            observed.update({name + "_" + key: value for key, value in stats.items()})
        return hook

    try:
        for name, layer in layers.items():
            handles.append(layer.dropout.register_forward_hook(capture_weights(name)))
            handles.append(layer.register_forward_hook(capture_layer(name)))
        result = model(**inputs)
        if set(captured) != set(layers):
            raise ValueError("Expected both original GAT layers to run")
    finally:
        for handle in handles:
            handle.remove()
    return result, observed


def summarize_gat_ranking(arrays: dict[str, np.ndarray], score_mode: str) -> dict:
    valid = arrays["occ_node_mask"] > .5
    positive = (arrays["event_will"] > .5) & valid
    groups = dict(upcoming=positive & (arrays["event_start"] > 0),
                  ongoing=positive & (arrays["event_start"] == 0), negative=valid & ~positive)
    layers = {}
    for name in ("gat1", "gat2"):
        eligible, reversed_order = (arrays[name + "_query_" + key] for key in ("eligible", "reversed"))
        pe, pr = (arrays[name + "_pair_" + key + "_by_head"] for key in ("eligible", "reversed"))
        if eligible.shape != valid.shape or reversed_order.shape != valid.shape or pe.shape != pr.shape:
            raise ValueError("Invalid GAT ranking summary grid")
        for e, r in ((eligible, reversed_order), (pe, pr)):
            if (not np.isfinite(e).all() or not np.isfinite(r).all()
                    or (r < 0).any() or (e < r).any() or (e != np.floor(e)).any()
                    or (r != np.floor(r)).any()):
                raise ValueError("Invalid GAT ranking counts")
        if not np.array_equal(eligible.sum(1), 2 * pe.sum(1)) or not np.array_equal(reversed_order.sum(1), 2 * pr.sum(1)):
            raise ValueError("Query incidence must count both endpoints of each pair")
        rows = {}
        for group_name, mask in groups.items():
            e, r = int(eligible[mask].sum()), int(reversed_order[mask].sum())
            rows[group_name] = dict(count=int(mask.sum()),
                nodes_with_eligible_pair=int((eligible[mask] > 0).sum()),
                nodes_with_any_strict_reversal=int((reversed_order[mask] > 0).sum()),
                eligible_query_pair_time_head_incidence=e,
                strict_reversal_query_pair_time_head_incidence=r,
                reversal_fraction=r / e if e else None)
        layers[name] = dict(groups=rows, eligible_pair_time_by_head=pe.sum(0).tolist(),
                            strict_reversal_pair_time_by_head=pr.sum(0).tolist())
    return dict(version="factory_frozen_gat_common_neighbor_top_v1", score_mode=score_mode,
        absolute_attention_difference_tolerance=1e-7, layers=layers,
        scope="Both actual GAT layers, all input history frames, heads and unordered query pairs. Compare only shared valid neighbors, requiring at least two. A reversal requires strict cross-preference between the two top choices. Counts demonstrate query-dependent ranking use, not causal importance or a recall benefit. Zero does not exclude changes among lower-ranked neighbors or differences below tolerance. Labels only stratify target-node statistics; no prediction, threshold or checkpoint changes.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive_member", help="Read this checkpoint member from a verified archive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compare_hot_head", action="store_true", help="Compare existing event and hot forecast outputs without changing reports")
    parser.add_argument("--compare_onset_report", action="store_true", help="Score frozen event-only and history-cold onset-max policies without selecting or changing official reports")
    parser.add_argument("--inspect_onset_frontier", action="store_true", help="Compute a labelled empirical oracle over independent event/onset thresholds; no calibration or deployment")
    parser.add_argument("--inspect_schedule_strata", action="store_true", help="Join frozen upcoming schedule/support strata after prediction without changing reports")
    observation = parser.add_mutually_exclusive_group()
    observation.add_argument("--inspect_temporal_attention", action="store_true", help="Observe frozen temporal pooling weights and representation changes without changing predictions")
    observation.add_argument("--inspect_history_graph", action="store_true", help="Observe applied post-GRU graph residuals without changing predictions")
    observation.add_argument("--inspect_gat_ranking", action="store_true", help="Observe strict top-choice reversals over common neighbors in both original B5 GAT layers")
    parser.add_argument("--source_commit", help="Pinned diagnostic Git source when executing this file via stdin")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[
        0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.68, 0.75, 0.80, 0.85, 0.90, 0.95,
    ])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError("Reuse an existing output directory")
    source_path = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/diagnose_baseline_events.py"
    source = (subprocess.check_output(["git", "show", f"{args.source_commit}:{source_path}"])
              if args.source_commit else Path(__file__).read_bytes())
    torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    payload, manifest = load_shared_dataset(args.dataset_dir)
    model, checkpoint, provenance = load_diagnostic_checkpoint(
        args.checkpoint, device, args.archive_member,
    )
    if (args.compare_onset_report or args.inspect_onset_frontier) and not checkpoint["model_config"].get("event_onset_aux", False):
        raise ValueError("Onset report comparison requires a checkpoint with an independent onset head")
    if args.inspect_temporal_attention and getattr(model, "history_attention", None) is None:
        raise ValueError("Temporal observation requires an attention-readout checkpoint")
    if args.inspect_history_graph and getattr(model, "history_graph", None) is None:
        raise ValueError("History graph observation requires a refinement checkpoint")
    if checkpoint["metadata"]["dataset_manifest_sha256"] != _manifest_hash(
        args.dataset_dir / "dataset_manifest.json"
    ):
        raise ValueError("Checkpoint and dataset manifest hashes do not match")
    if int(manifest["event_min_windows"]) != 8:
        raise ValueError("This diagnostic requires the canonical 8-window event contract")
    from factory_baselines.precursor import attach_precursor
    payload, _ = attach_precursor(
        payload, manifest, args.dataset_dir, checkpoint["model_config"].get("event_precursor", "none"),
        (args.split,), checkpoint["metadata"].get("input_feature_contract"),
    )
    from factory_baselines.onset_history import attach_onset_history
    joint = checkpoint["model_config"].get("event_onset_joint", False)
    expected_onset = checkpoint["metadata"].get("onset_history_contract")
    if joint and expected_onset is None:
        raise ValueError("Joint checkpoint is missing its observed-onset-history contract")
    payload, _ = attach_onset_history(payload, manifest, args.dataset_dir, joint,
                                      (args.split,), expected_onset)
    loader = DataLoader(
        FactoryBaselineTensorDataset(payload, payload["split_indices"][args.split].tolist()),
        batch_size=args.batch_size, shuffle=False,
    )
    collected: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            observed = {}
            inputs = _model_inputs(batch, model)
            if args.inspect_temporal_attention:
                result, observed = predict_with_temporal_observation(model, inputs)
            elif args.inspect_history_graph:
                result, observed = predict_with_history_graph_observation(model, inputs)
            elif args.inspect_gat_ranking:
                result, observed = predict_with_gat_ranking_observation(model, inputs)
            else:
                result = model(**inputs)
            values = {key: batch[key] for key in (
                "sample_index", "y_hot", "remain_mask", "occ_node_mask",
                "hist_last_hot", "event_will", "event_start",
            )}
            values.update(
                will_probability=result["event_will_logit"].sigmoid(),
                predicted_start=result["event_start_logit"].argmax(-1),
                predicted_duration=result["event_duration"],
            )
            if "event_kind_logits" in result:
                values["event_kind_probability"] = result["event_kind_logits"].softmax(-1)
            if "event_onset_logit" in result:
                values["event_onset_probability"] = result["event_onset_logit"].sigmoid()
            if joint:
                values.update(observe_joint_onset(result, batch["event_history_hot"]))
            if args.compare_hot_head:
                values["predicted_hot_probability"] = result["remain_hot_logit"].sigmoid()
            values.update(observed)
            for key, value in values.items():
                collected.setdefault(key, []).append(value.cpu().numpy())
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    chosen_threshold = float(checkpoint["metadata"]["event_report_threshold"])
    thresholds = sorted(set([*args.thresholds, chosen_threshold]))
    report = summarize_events(arrays, thresholds)
    if args.inspect_schedule_strata:
        attach_schedule_strata(report, arrays, args.dataset_dir, manifest, args.split, chosen_threshold)
    if joint:
        report["joint_onset_diagnostics"] = summarize_joint_onset(arrays, chosen_threshold)
    if args.inspect_temporal_attention:
        report["temporal_attention_observation"] = summarize_temporal_attention(
            arrays, float(model.history_attention.query.norm().item()),
        )
    if args.inspect_history_graph:
        report["history_graph_observation"] = summarize_history_graph(
            arrays, float(model.history_graph.output.weight.norm().item()),
        )
    if args.inspect_gat_ranking:
        report["gat_ranking_observation"] = summarize_gat_ranking(arrays, model.config.gat_score_mode)
    if args.compare_hot_head:
        report["prediction_head_comparison"] = summarize_prediction_heads(arrays, chosen_threshold)
    if args.compare_onset_report:
        report["onset_report_comparison"] = summarize_onset_reports(
            arrays, checkpoint["train_config"]["report_threshold_sweep"], chosen_threshold,
        )
    if args.inspect_onset_frontier:
        report["independent_onset_frontier"] = summarize_onset_frontier(arrays, chosen_threshold)
    attach_node_catalog(report, args.dataset_dir / "node_catalog.csv", manifest)
    if args.compare_onset_report:
        for policy in report["onset_report_comparison"]["policies"].values():
            attach_node_catalog(policy, args.dataset_dir / "node_catalog.csv", manifest)
    report.update(
        **provenance,
        split=args.split, test_evaluated=False,
        checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint["epoch"],
        dataset_manifest_sha256=checkpoint["metadata"]["dataset_manifest_sha256"],
        sample_count=len(arrays["sample_index"]),
        saved_report_threshold=chosen_threshold,
        diagnostic_source_commit=args.source_commit,
        diagnostic_source_sha256=hashlib.sha256(source).hexdigest(),
        diagnostic_scope="Future labels and horizon length are retrospective diagnostic strata, "
                         "not deployable features or proposed alarm filters. No threshold is selected here.",
        ranking_scope="The common event-existence score is ranked in each retrospective subgroup. "
                      "Optional event_kind_diagnostics separately inspect subtype probabilities. "
                      "Neither diagnostic replaces the canonical checkpoint-selection metric.",
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["groups"], indent=2), flush=True)
    print(json.dumps(report["ranking"], indent=2), flush=True)
    print(json.dumps(report["training_partition_audit"], indent=2), flush=True)
    if args.compare_hot_head:
        print(json.dumps(report["prediction_head_comparison"], indent=2), flush=True)
    if "onset_auxiliary_diagnostics" in report:
        print(json.dumps(report["onset_auxiliary_diagnostics"], indent=2), flush=True)
    if "joint_onset_diagnostics" in report:
        observed = report["joint_onset_diagnostics"]
        print("Joint onset observation:", json.dumps({
            "groups": observed["groups"], "ranking": observed["ranking"],
        }), flush=True)
    if "onset_report_comparison" in report:
        comparison = report["onset_report_comparison"]
        for name, policy in comparison["policies"].items():
            row = next(r for r in policy["thresholds"] if r["threshold"] == chosen_threshold)
            print("Onset report probe at saved threshold:", name, json.dumps({
                key: row[key] for key in ("report_precision", "report_recall", "report_f1",
                                         "report_recall_upcoming", "report_false_alarm_count")
            }), flush=True)
    if "independent_onset_frontier" in report:
        for constraint, point in report["independent_onset_frontier"]["empirical_maxima"].items():
            print("Empirical onset frontier (not selected):", constraint, json.dumps(
                {key: value for key, value in point.items() if key != "canonical_report"}
                if point is not None else None
            ), flush=True)
    print("threshold P R F1 upcoming_R upcoming_probability_misses upcoming_timing_misses")
    for row in report["thresholds"]:
        print(row["threshold"], *[round(row[k], 4) for k in (
            "report_precision", "report_recall", "report_f1", "report_recall_upcoming",
        )], row["upcoming_probability_misses"], row["upcoming_timing_misses"], flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
