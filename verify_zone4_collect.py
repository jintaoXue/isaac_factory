#!/usr/bin/env python3
"""Verify a zone4 1-episode collect: fleet use, 18-order completion, deadlock, BN labels."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/sci/work/isaac_factory")
BASE = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
TAGS = [
    "zone4_norm",
    "zone4_machine1.0",
    "zone4_human1.0",
    "zone4_logistics1.0",
    "zone4_material1.0",
]


def resolve(tag: str) -> Path | None:
    p = BASE / tag
    if p.is_symlink() or p.is_dir():
        return p.resolve() if p.exists() else None
    return None


def job_stats(ep: Path) -> dict:
    p = ep / "env_00" / "job_trace.csv"
    if not p.exists():
        return {"ok": False, "reason": "no job_trace"}
    rows = list(csv.DictReader(p.open()))
    jobs = defaultdict(lambda: {"events": [], "tmax": 0.0})
    for r in rows:
        j = r.get("job_id") or r.get("product_index")
        ev = r.get("event") or ""
        t = float(r.get("logic_time_s") or r.get("time_step") or 0)
        jobs[j]["events"].append(ev)
        jobs[j]["tmax"] = max(jobs[j]["tmax"], t)
    n_complete = sum(1 for v in jobs.values() if "stage_complete" in v["events"] or "process_end" in v["events"] and "stage_complete" in v["events"])
    n_stage = sum(1 for v in jobs.values() if "stage_complete" in v["events"])
    tmax = max((v["tmax"] for v in jobs.values()), default=0.0)
    return {
        "ok": True,
        "n_rows": len(rows),
        "n_jobs": len(jobs),
        "n_stage_complete": n_stage,
        "tmax": tmax,
        "events": dict(Counter(r.get("event") or "" for r in rows)),
    }


def fleet_use(ep: Path) -> dict:
    p = ep / "env_00" / "resource_event_log.jsonl"
    gantries = set()
    agvs = set()
    deadlock = 0
    last_t = 0.0
    if not p.exists():
        return {"gantries": [], "agvs": [], "deadlock_events": 0, "last_t": 0}
    with p.open() as f:
        for line in f:
            o = json.loads(line)
            last_t = max(last_t, float(o.get("logic_time_s") or o.get("time_step") or 0))
            rid = str(o.get("resource_id") or "")
            rtype = str(o.get("resource_type") or "")
            raw_to = str(o.get("raw_to_state") or "")
            to_st = str(o.get("to_state") or "")
            if rtype == "gantry" or rid.startswith("gantry_"):
                if to_st in ("PROCESSING",) or raw_to.startswith("working_"):
                    gantries.add(rid)
            if rtype == "transport_robot" or rid.startswith("robot_"):
                if to_st in ("PROCESSING",) or raw_to.startswith("working_"):
                    agvs.add(rid)
            notes = str(o.get("reason") or "") + str(o.get("disturbance_type") or "")
            if "deadlock" in notes.lower() or "deadlock" in rid.lower():
                deadlock += 1
    dist = ep / "env_00" / "disturbance_log.csv"
    if dist.exists():
        with dist.open() as f:
            for r in csv.DictReader(f):
                if "deadlock" in (r.get("disturbance_type") or "").lower():
                    deadlock += 1
    cfg = {}
    cfgp = ep / "env_00" / "episode_config.csv"
    if cfgp.exists():
        rows = list(csv.DictReader(cfgp.open()))
        cfg = rows[0] if rows else {}
    return {
        "gantries": sorted(gantries),
        "agvs": sorted(agvs),
        "deadlock_events": deadlock,
        "last_t": last_t,
        "cfg_dim": cfg.get("disturbance_dim"),
        "cfg_I": cfg.get("disturbance_intensity"),
        "robot_config": cfg.get("robot_config"),
        "gantry_config": cfg.get("gantry_config"),
        "human_config": cfg.get("human_config"),
        "production_done": cfg.get("production_done"),
    }


def derived_stats(run: Path) -> dict | None:
    d = run / "derived" / "episode_00" / "env_00"
    if not (d / "pipeline_summary.json").exists():
        return None
    s = json.loads((d / "pipeline_summary.json").read_text())
    n_hot = n_will = n_rows = 0
    lab = d / "bottleneck_label.csv"
    if lab.exists():
        with lab.open() as f:
            for r in csv.DictReader(f):
                n_rows += 1
                n_hot += int(float(r.get("is_hot") or r.get("y_hot") or 0) != 0)
                n_will += int(float(r.get("will_bn_180s") or r.get("y_will") or 0) != 0)
    return {
        "n_events": s.get("n_events"),
        "n_disturbance_l2": s.get("n_disturbance_l2"),
        "n_feature_rows": s.get("n_feature_rows"),
        "n_label_rows": n_rows,
        "n_hot": n_hot,
        "n_will": n_will,
        "horizon_s": s.get("horizon_s"),
    }


def main() -> int:
    print(f"{'tag':22s} {'jobs':>4} {'done':>4} {'tmax':>8} {'g':>12} {'agv':>20} {'dead':>4} {'l2':>4} {'hot':>6}")
    any_missing = False
    for tag in TAGS:
        run = resolve(tag)
        if run is None:
            print(f"{tag:22s}  MISSING")
            any_missing = True
            continue
        ep = run / "episode_00"
        js = job_stats(ep)
        fl = fleet_use(ep)
        der = derived_stats(run)
        g = ",".join(fl["gantries"]) or "-"
        a = ",".join(fl["agvs"]) or "-"
        done = js.get("n_stage_complete", 0) if js.get("ok") else 0
        nj = js.get("n_jobs", 0) if js.get("ok") else 0
        tmax = js.get("tmax", 0) if js.get("ok") else 0
        l2 = der.get("n_disturbance_l2") if der else "-"
        hot = der.get("n_hot") if der else "-"
        print(
            f"{tag:22s} {nj:4d} {done:4d} {tmax:8.0f} {g:>12} {a:>20} "
            f"{fl['deadlock_events']:4d} {str(l2):>4} {str(hot):>6}"
        )
        print(f"  dir={run.name} dim={fl.get('cfg_dim')} I={fl.get('cfg_I')} "
              f"robots={fl.get('robot_config')} gantry={fl.get('gantry_config')} humans={fl.get('human_config')}")
        if der:
            print(f"  derived events={der['n_events']} feat_rows={der['n_feature_rows']} "
                  f"will={der['n_will']} hot={der['n_hot']}")
        else:
            print("  derived: not aggregated yet")
    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
