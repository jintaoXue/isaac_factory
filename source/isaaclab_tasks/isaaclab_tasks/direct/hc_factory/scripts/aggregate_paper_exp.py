#!/usr/bin/env python3
"""Aggregate paper_exp raw JSON into summary files for ICCBEI plotting.

Usage (from isaac_factory root):
  python source/.../scripts/aggregate_paper_exp.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "output" / "paper_exp"
METRICS = PAPER / "metrics"
TPA = PAPER / "tpa"
RUNS = PAPER / "perception_runs"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def aggregate_tpa() -> dict:
    grid = {}
    for cell in sorted(TPA.glob("Nh*_O*")):
        rows = [r for r in _load_jsonl(cell / "episodes.jsonl") if r.get("success")]
        if not rows:
            grid[cell.name] = {"success_n": 0, "makespan_mean": None, "idle_h_mean": None, "idle_m_mean": None}
            continue
        grid[cell.name] = {
            "success_n": len(rows),
            "makespan_mean": statistics.mean(r["makespan"] for r in rows),
            "idle_h_mean": statistics.mean(r["idle_h"] for r in rows),
            "idle_m_mean": statistics.mean(r["idle_m"] for r in rows),
            "episodes": rows,
        }
    out = {"grid": grid}
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "tpa_grid.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def aggregate_perception_metrics() -> dict:
    """Collect any metrics/*.json written by perception eval jobs."""
    files = sorted((METRICS).glob("*.json"))
    bag = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in files if p.name != "tpa_grid.json"}
    # learning curves from history.json
    curves = {}
    for hist in RUNS.glob("*/history.json"):
        curves[hist.parent.name] = json.loads(hist.read_text(encoding="utf-8"))
    out = {"metrics_files": bag, "run_histories": curves}
    (METRICS / "perception_bundle.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    tpa = aggregate_tpa()
    perc = aggregate_perception_metrics()
    print(f"[aggregate] tpa cells={len(tpa['grid'])} perception_files={len(perc['metrics_files'])}")
    print(f"[aggregate] wrote under {METRICS}")


if __name__ == "__main__":
    main()
