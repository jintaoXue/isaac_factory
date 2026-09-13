#!/usr/bin/env python3
"""Capture existing encoder and event-head inputs for reusable frozen diagnostics."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


RUNTIME = "979c680fb4d1919ae760cdfb0038f69fb7cc6708"
SOURCE_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/export_frozen_event_representations.py"
FINAL_AUDIT_SHA = "c1837f569cb50023ec8647a473440ca406b96be61b360463f9a174db9c2ce340"


def observe_event_input(model, inputs):
    """Observe the actual input to the existing head, without replacing outputs."""
    captured = []

    def capture(_module, args):
        captured.append(args[0].detach())

    handle = model.heads.event_will_head.register_forward_pre_hook(capture)
    try:
        result = model(**inputs)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != result["node_hidden"].shape:
        raise ValueError("Unexpected event-head execution or shape")
    return result, captured[0]


def pack_representation_rows(batch_cpu, result, event_hidden):
    """Keep all valid target nodes, with labels used only to annotate rows.

    Raw summaries use the actual normalized X on CPU for both models. They are
    fixed last/mean/population-std/last-minus-first statistics of 21 channels.
    """
    valid = batch_cpu["occ_node_mask"].numpy() > .5
    sample, node = np.nonzero(valid)
    will = batch_cpu["event_will"].numpy() > .5
    start = batch_cpu["event_start"].numpy()
    kind = np.where(will, np.where(start > 0, 2, 1), 0).astype(np.int8)
    x = batch_cpu["x"][..., :21]
    raw = torch.cat((x[:, -1], x.mean(1), x.std(1, unbiased=False), x[:, -1] - x[:, 0]), dim=-1)
    fields = dict(sample_index=batch_cpu["sample_index"].numpy()[sample], node_index=node.astype(np.int16),
        label_kind=kind[valid], target_start=start[valid].astype(np.int16),
        history_last_hot=batch_cpu["hist_last_hot"].numpy()[valid],
        raw_summary=raw.numpy()[valid], backbone=result["node_hidden"].detach().cpu().numpy()[valid],
        event_input=event_hidden.detach().cpu().numpy()[valid],
        probability=result["event_will_logit"].sigmoid().detach().cpu().numpy()[valid],
        predicted_start=result["event_start_logit"].argmax(-1).detach().cpu().numpy()[valid])
    for name, value in fields.items():
        if len(value) != len(node) or not np.isfinite(value).all(): raise ValueError("Invalid representation field: " + name)
    return fields


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def verify_old_report(actual, expected, threshold):
    a = next(row for row in actual["thresholds"] if row["threshold"] == threshold)
    e = next(row for row in expected["thresholds"] if row["threshold"] == threshold)
    for key, value in e.items():
        if key.startswith(("n_", "who_", "report_")) and key != "report_threshold_used":
            assert (abs(a[key] - value) < 1e-10 if isinstance(value, (int, float)) else a[key] == value), key
    for name in ("ongoing", "upcoming", "negative"):
        assert actual["groups"][name]["count"] == expected["groups"][name]["count"]
    assert abs(actual["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"] - expected["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"]) <= 1e-8


def main():
    import diagnose_baseline_events as diagnostic
    from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
    from factory_baselines.onset_history import attach_onset_history
    from factory_baselines.precursor import attach_precursor
    from factory_baselines.torch_trainer import _model_inputs
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--model", choices=("b4", "b5"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    source = subprocess.check_output(["git", "show", args.source_commit + ":" + SOURCE_PATH])
    source_hash = hashlib.sha256(source).hexdigest()
    audit_path = d / "baseline_schedule_strata_final_verification20260913.json"
    assert sha(audit_path) == FINAL_AUDIT_SHA; audit = json.loads(audit_path.read_text())

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, h in audit["current_model_files_sha256"].items(): assert sha(d / name) == h, name
        for name, value in audit["dataset_files_stat"].items():
            stat = (d / name).stat(); assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value, name

    guard()
    whole = next(row for row in audit["models"] if row["model"] == args.model)
    assert sha(d / whole["file"]) == whole["sha256"]
    reference = next(row for row in whole["runs"] if row["seed"] == 42 and row["checkpoint"] == "last" and row["split"] == args.split)
    assert sha(d / reference["file"]) == reference["file_sha256"]
    old = json.loads((d / reference["file"]).read_text())
    stem = f"baseline_repr_{args.model}s42_last_{args.split}20260913"
    archive, summary_path = d / (stem + ".npz"), d / (stem + ".json")
    threshold = old["saved_report_threshold"]
    if archive.exists():
        with np.load(archive, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata_json"].item()))
            assert metadata["source_commit"] == args.source_commit and metadata["source_sha256"] == source_hash
            assert metadata["original_file_sha256"] == reference["file_sha256"]
            assert metadata["checkpoint_file_sha256"] == old["checkpoint_file_sha256"]
            verify_old_report(metadata["canonical_report"], old, threshold)
            assert len(stored["sample_index"]) == metadata["row_count"]
        guard()
        if summary_path.exists():
            saved = json.loads(summary_path.read_text()); assert saved["archive_sha256"] == sha(archive)
            assert saved["metadata"] == metadata
        else:
            with summary_path.open("x") as f: json.dump(dict(metadata=metadata, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size), f, indent=2); f.write("\n")
        print("REPRESENTATION_REUSED", args.model, args.split, archive.stat().st_size, sha(archive), flush=True)
        return
    assert not summary_path.exists()
    torch.set_num_threads(2)
    device, batch_size = (torch.device("cpu"), 32) if args.model == "b4" else (torch.device("cuda:0"), 16)
    payload, manifest = load_shared_dataset(d)
    checkpoint_path = d / f"models/tuning/{args.model}_representation_v1/candidate_history/seed42/last.pt"
    assert sha(checkpoint_path) == old["checkpoint_file_sha256"]
    model, checkpoint, provenance = diagnostic.load_diagnostic_checkpoint(checkpoint_path, device, None)
    assert checkpoint["epoch"] == old["epoch"] and checkpoint["metadata"]["event_report_threshold"] == threshold
    assert checkpoint["metadata"]["dataset_manifest_sha256"] == old["dataset_manifest_sha256"] == sha(d / "dataset_manifest.json")
    payload, _ = attach_precursor(payload, manifest, d, checkpoint["model_config"].get("event_precursor", "none"),
        (args.split,), checkpoint["metadata"].get("input_feature_contract"))
    joint = checkpoint["model_config"].get("event_onset_joint", False)
    payload, _ = attach_onset_history(payload, manifest, d, joint, (args.split,), checkpoint["metadata"].get("onset_history_contract"))
    loader = DataLoader(FactoryBaselineTensorDataset(payload, payload["split_indices"][args.split].tolist()), batch_size=batch_size, shuffle=False)
    collected, tables = {}, {}
    model.eval()
    with torch.no_grad():
        for position, cpu in enumerate(loader):
            batch = {key: value.to(device) for key, value in cpu.items()}
            result, event_input = observe_event_input(model, _model_inputs(batch, model))
            for key, value in pack_representation_rows(cpu, result, event_input).items(): tables.setdefault(key, []).append(value)
            values = {key: cpu[key].numpy() for key in ("sample_index", "y_hot", "remain_mask", "occ_node_mask", "hist_last_hot", "event_will", "event_start")}
            values.update(will_probability=result["event_will_logit"].sigmoid().cpu().numpy(), predicted_start=result["event_start_logit"].argmax(-1).cpu().numpy(), predicted_duration=result["event_duration"].cpu().numpy())
            for key, value in values.items(): collected.setdefault(key, []).append(value)
            if position % 200 == 0: print("REPRESENTATION_BATCH", args.model, args.split, position, len(loader), flush=True)
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    report = diagnostic.summarize_events(arrays, [threshold]); diagnostic.attach_node_catalog(report, d / "node_catalog.csv", manifest)
    verify_old_report(report, old, threshold)
    table = {k: np.concatenate(v) for k, v in tables.items()}
    assert len(arrays["sample_index"]) == old["sample_count"]
    assert len(set(zip(table["sample_index"].tolist(), table["node_index"].tolist()))) == len(table["sample_index"])
    counts = np.bincount(table["label_kind"], minlength=3).tolist()
    assert counts == ([297152, 4191, 595] if args.split == "train" else [67331, 950, 145])
    metadata = dict(status="frozen_event_representations_exported_with_original_scores_reproduced", source_commit=args.source_commit,
        source_sha256=source_hash, runtime_commit=RUNTIME, runtime_diagnostic_sha256=sha(Path(inspect.getfile(diagnostic))),
        model=args.model, seed=42, checkpoint="last", split=args.split, epoch=checkpoint["epoch"],
        checkpoint_file_sha256=old["checkpoint_file_sha256"], dataset_manifest_sha256=old["dataset_manifest_sha256"],
        original_file=reference["file"], original_file_sha256=reference["file_sha256"], saved_threshold=threshold,
        sample_count=old["sample_count"], row_count=len(table["sample_index"]), label_kind_order=["negative", "ongoing", "upcoming"], label_counts=counts,
        representations={key: list(table[key].shape) for key in ("raw_summary", "backbone", "event_input")}, canonical_report=report,
        scope="Captured during unchanged forward. Raw summary uses actual 30-window normalized X; backbone is masked GRU readout; event_input is actual event-will head input after optional context/precursor. No labels enter these representations. Last weights are diagnostic only.",
        model_training=False, test_evaluated=False, goal_met=False)
    guard()
    # Keep the canonical/provenance record inside the archive too, so a summary
    # write interruption never requires repeating the completed model forward.
    with archive.open("xb") as f: np.savez_compressed(f, **table, metadata_json=np.asarray(json.dumps(metadata)))
    saved = dict(metadata=metadata, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)
    with summary_path.open("x") as f: json.dump(saved, f, indent=2); f.write("\n")
    guard()
    print("REPRESENTATION_COMPLETE", args.model, args.split, archive.stat().st_size, saved["archive_sha256"], flush=True)


if __name__ == "__main__": main()
