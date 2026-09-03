#!/usr/bin/env python3
"""Merge per-seed HcFactory_TPA_Eval rule runs into one 5seed×N run, then archive sources.

Example:
  python tools/merge_wandb_rule_eval.py --dry-run
  python tools/merge_wandb_rule_eval.py --apply
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Any

import wandb


ENTITY = os.environ.get("HC_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY") or "rl-driving"
PROJECT = os.environ.get("HC_WANDB_TEST_PROJECT") or "HcFactory_TPA_Eval"

# Prefer the completed 2ep protocol (matches current batch_train.sh defaults).
K10_MERGE = {
    "out_name": "rule_K10_N16_T64000_5seed_x2",
    "tags": ["K10", "N16", "merged-from-seeds", "multi", "x2"],
    "config": {
        "algo": "rule_based",
        "t_max": 64000,
        "test_seeds": [42, 43, 44, 45, 46],
        "test_times": 2,
        "train_n_products": 16,
        "max_parallel_cd_dispatch": 10,
        "legacy_mode_label": "multi",
    },
    "sources": [
        (42, "vovq7sf8"),
        (43, "5eu5dpz1"),
        (44, "zjr0906a"),
        (45, "45ldi1ea"),
        (46, "i830xmsj"),
    ],
}

# Already merged as rule_K1_N16_T64000_5seed_x2; only archive leftovers.
ARCHIVE_ONLY = [
    # K1 remaining un-prefixed per-seed runs
    "soelnokj",  # seed44
    "2d36gfad",  # seed45
    "tze8g4md",  # seed46
    # K10 older 4ep protocol
    "txucsfnn",  # seed42_4ep
    "mx1u094h",  # seed43_4ep
    "eiwqi304",  # seed44_4ep crashed
]


def _finite(v: Any) -> bool:
    try:
        return v is not None and not (isinstance(v, float) and math.isnan(v))
    except Exception:
        return False


def _episode_rows(run) -> list[dict[str, Any]]:
    hist = run.history(samples=20000)
    if "MetricCore/episode" not in hist.columns:
        return []
    ep = hist[hist["MetricCore/episode"].notna()].sort_values("MetricCore/episode")
    rows: list[dict[str, Any]] = []
    for _, row in ep.iterrows():
        payload = {k: row[k] for k in hist.columns if _finite(row[k]) and not str(k).startswith("_")}
        rows.append(payload)
    return rows


def _mirror_fullorder(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for k, v in list(payload.items()):
        if k.startswith("MetricCore/"):
            out.setdefault(k.replace("MetricCore/", "MetricFullorderCore/", 1), v)
        elif k.startswith("MetricPeak/"):
            out.setdefault(k.replace("MetricPeak/", "MetricFullorderPeak/", 1), v)
    return out


def _rebuild_cumulative(episodes: list[tuple[int, int, dict[str, Any]]], t_max: int) -> list[dict[str, Any]]:
    makespans: list[float] = []
    success_ms: list[float] = []
    n_success = 0
    rebuilt: list[dict[str, Any]] = []
    for i, (seed, seed_ep_idx, src) in enumerate(episodes, start=1):
        ms = float(src.get("MetricCore/05_makespan"))
        n_fin = int(src.get("MetricCore/06_n_finished") or 0)
        success = ms < float(t_max) and n_fin > 0
        if success:
            n_success += 1
            success_ms.append(ms)
        makespans.append(ms)
        mean_ms = sum(makespans) / len(makespans)
        mean_ok = sum(success_ms) / len(success_ms) if success_ms else None
        success_rate = n_success / float(i)

        payload = {
            "MetricCore/episode": i,
            "MetricCore/01_normalized_makespan": float(src.get("MetricCore/01_normalized_makespan", ms / t_max)),
            "MetricCore/02_mean_per_product_span": float(
                src.get("MetricCore/02_mean_per_product_span", ms / max(1, n_fin))
            ),
            "MetricCore/03_success_rate": success_rate,
            "MetricCore/04_normalized_return": float(src.get("MetricCore/04_normalized_return", 0.0)),
            "MetricCore/05_makespan": ms,
            "MetricCore/06_n_finished": n_fin,
            "MetricCore/07_finished_abs": int(src.get("MetricCore/07_finished_abs") or 0),
            "MetricCore/08_ep_return": float(src.get("MetricCore/08_ep_return", 0.0)),
            "MetricCore/09_mean_makespan": mean_ms,
            "Eval/seed": int(seed),
            "Eval/seed_episode_idx": int(seed_ep_idx),
            "Train/step": int(src.get("Train/step") or ms),
        }
        if mean_ok is not None:
            payload["MetricCore/10_mean_makespan_success"] = mean_ok
        for key in (
            "MetricPeak/03_peak_producing",
            "MetricPeak/04_peak_ongoing_product",
            "MetricPeak/05_peak_ongoing_human",
            "MetricPeak/06_peak_ongoing_robot",
            "MetricHuman/ep_mean_fatigue",
            "MetricHuman/ep_end_mean_fatigue",
            "MetricTrain/02_steps_per_min",
            "MetricTrain/03_wall_time_min",
            "MetricTrain/04_wall_time_hour",
        ):
            if key in src and _finite(src[key]):
                payload[key] = src[key]
        payload["MetricHuman/episode"] = i
        payload = _mirror_fullorder(payload)
        rebuilt.append(payload)
    return rebuilt


def archive_run(api: wandb.Api, run_id: str, *, dry_run: bool) -> None:
    path = f"{ENTITY}/{PROJECT}/{run_id}"
    run = api.run(path)
    if str(run.name).startswith("zzz_archived_"):
        print(f"[skip] already archived: {run.name} ({run_id})")
        return
    new_name = f"zzz_archived_{run.name}"
    print(f"[archive] {run.name} -> {new_name} ({run_id})")
    if dry_run:
        return
    tags = list(run.tags or [])
    if "archived" not in tags:
        tags.append("archived")
    run.name = new_name
    run.tags = tags
    run.update()


def merge_group(api: wandb.Api, spec: dict[str, Any], *, dry_run: bool) -> None:
    # Skip if target already exists.
    existing = [r for r in api.runs(f"{ENTITY}/{PROJECT}") if r.name == spec["out_name"]]
    if existing:
        print(f"[skip] target already exists: {spec['out_name']} ({existing[0].id})")
        for seed, rid in spec["sources"]:
            archive_run(api, rid, dry_run=dry_run)
        return

    collected: list[tuple[int, int, dict[str, Any]]] = []
    source_names: list[str] = []
    source_ids: list[str] = []
    for seed, rid in spec["sources"]:
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        rows = _episode_rows(run)
        if not rows:
            raise RuntimeError(f"no episode rows in {run.name} ({rid})")
        print(f"[source] seed={seed} {run.name} episodes={len(rows)}")
        source_names.append(run.name)
        source_ids.append(rid)
        for local_i, row in enumerate(rows):
            collected.append((seed, local_i, row))

    t_max = int(spec["config"]["t_max"])
    payloads = _rebuild_cumulative(collected, t_max=t_max)
    makespans = [p["MetricCore/05_makespan"] for p in payloads]
    success_flags = [p["MetricCore/05_makespan"] < t_max for p in payloads]
    success_ms = [m for m, ok in zip(makespans, success_flags) if ok]
    summary = {
        "n_episodes": len(payloads),
        "n_seeds": len(spec["sources"]),
        "success_rate": sum(success_flags) / float(len(payloads)),
        "mean_makespan": sum(makespans) / float(len(makespans)),
        "mean_makespan_success": (sum(success_ms) / float(len(success_ms))) if success_ms else None,
        "merged_from_run_ids": source_ids,
        "merged_from_names": source_names,
    }
    print(f"[merge] -> {spec['out_name']} n={summary['n_episodes']} success={summary['success_rate']:.3f} "
          f"mean_ms={summary['mean_makespan']:.1f}")
    if dry_run:
        for p in payloads:
            print(
                f"  ep{p['MetricCore/episode']:02d} seed={p['Eval/seed']} "
                f"ms={p['MetricCore/05_makespan']:.0f} sr={p['MetricCore/03_success_rate']:.2f}"
            )
        return

    cfg = dict(spec["config"])
    cfg["merged_from_run_ids"] = source_ids
    cfg["merged_from_names"] = source_names
    wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=spec["out_name"],
        tags=spec["tags"],
        config=cfg,
        job_type="merged-eval",
        reinit=True,
    )
    for payload in payloads:
        wandb.log(payload)
    for k, v in summary.items():
        wandb.run.summary[k] = v
    wandb.finish()

    for _, rid in spec["sources"]:
        archive_run(api, rid, dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.apply:
        raise SystemExit("pass --dry-run or --apply")

    dry = bool(args.dry_run)
    api = wandb.Api()
    print(f"target={ENTITY}/{PROJECT} dry_run={dry}")
    merge_group(api, K10_MERGE, dry_run=dry)
    print("[archive leftovers]")
    for rid in ARCHIVE_ONLY:
        archive_run(api, rid, dry_run=dry)


if __name__ == "__main__":
    main()
