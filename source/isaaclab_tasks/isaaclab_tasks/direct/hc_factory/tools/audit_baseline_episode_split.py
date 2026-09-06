#!/usr/bin/env python3
"""Compare physical episode identities using the main experiment's actual splitter."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from factory_bn_shared.bundle import file_hash, main_episode_identities


SPLITS = ("train", "validation", "test")


def assignment(splits: dict[str, list[str]]) -> dict[str, str]:
    result = {}
    for split in SPLITS:
        for group in splits[split]:
            if group in result:
                raise ValueError(f"Episode repeated within/across splits: {group}")
            result[group] = split
    return result


def compare_splits(baseline: dict, main: dict) -> dict:
    baseline_map, main_map = assignment(baseline), assignment(main)
    common = set(baseline_map) & set(main_map)
    moved = [
        {"group_id": group, "baseline": baseline_map[group], "main": main_map[group]}
        for group in sorted(common)
        if baseline_map[group] != main_map[group]
    ]
    baseline_only = sorted(set(baseline_map) - set(main_map))
    main_only = sorted(set(main_map) - set(baseline_map))
    return {
        "episode_split_match": not (moved or baseline_only or main_only),
        "baseline_only": baseline_only,
        "main_only": main_only,
        "changed_split": moved,
        "counts": {
            split: {
                "baseline": len(baseline[split]),
                "main": len(main[split]),
                "intersection": len(set(baseline[split]) & set(main[split])),
            }
            for split in SPLITS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--main_bundle", type=Path, required=True)
    parser.add_argument("--main_checkpoint", type=Path, required=True)
    parser.add_argument("--pdformer_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    splitter_path = args.pdformer_root.resolve() / "factory_bn" / "dataset.py"
    sys.path.insert(0, str(args.pdformer_root.resolve()))
    module = importlib.import_module("factory_bn.dataset")
    if Path(module.__file__).resolve() != splitter_path:
        raise ValueError("Imported main splitter from an unexpected checkout")

    import torch

    # Only use a trusted checkpoint produced by the user's own experiment.
    checkpoint = torch.load(args.main_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    params = {
        "train_ratio": float(config.get("train_rate", 0.7)),
        "val_ratio": float(config.get("eval_rate", 0.15)),
        "seed": int(config.get("seed", 42)),
        "train_only_contains": list(config.get("train_only_contains") or []),
    }
    meta_path = args.main_bundle / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bundle_path = args.main_bundle / "episodes.npz"
    with np.load(bundle_path, allow_pickle=False) as bundle:
        names = [str(name) for name in bundle["episode_names"].tolist()]
    if len(names) != len(set(names)) or set(names) != set(meta["episodes"]):
        raise ValueError("Main bundle NPZ/meta episode inventories differ or contain duplicates")
    identities = main_episode_identities(meta, names)
    main_named = dict(zip(SPLITS, module.split_episodes_by_name(names, **params), strict=True))
    main_groups = {
        split: sorted(identities[name] for name in main_named[split]) for split in SPLITS
    }
    split_path = args.baseline_dir / "split_manifest.json"
    manifest_path = args.baseline_dir / "dataset_manifest.json"
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_split = json.loads(split_path.read_text(encoding="utf-8"))
    baseline_groups = {split: baseline_split[split]["group_ids"] for split in SPLITS}
    declared = [row["group_id"] for row in baseline_manifest["source_episodes"]]
    if len(set(declared)) != len(declared) or set(declared) != set(assignment(baseline_groups)):
        raise ValueError("Baseline source episode inventory does not match split manifest")

    result = compare_splits(baseline_groups, main_groups)
    result.update({
        "audit_scope": "episode identity and split only; no label or metric comparison",
        "test_metrics_read": False,
        "main_split_evidence": "reconstructed from checkpoint config, current bundle and supplied splitter",
        "limitation": "Historical split is not proved if the bundle or splitter changed after training. "
        "Matching episode IDs does not prove unchanged raw bytes, labels, features or sample anchors.",
        "main_split_parameters": params,
        "main_run_aliases": dict(zip(meta["run_names"], meta["run_dirs"], strict=True)),
        "main_episode_mapping": identities,
        "main_splits": main_groups,
        "baseline_splits": baseline_groups,
        "provenance": {
            name: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for name, path in {
                "main_splitter": splitter_path,
                "main_checkpoint": args.main_checkpoint,
                "main_bundle": bundle_path,
                "main_meta": meta_path,
                "baseline_manifest": manifest_path,
                "baseline_split": split_path,
            }.items()
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "episode_split_match", "counts", "baseline_only", "main_only", "changed_split"
    )}, indent=2, ensure_ascii=False))
    print(f"Audit: {args.output}")
    if not result["episode_split_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
