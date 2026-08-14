"""CSV/JSONL helpers and run-directory discovery."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

def _discover_env_dirs(run_dir: Path, env_id: int | None) -> list[Path]:
    """Return env_* dirs under run_dir or under episode_*/ subfolders."""
    nested = sorted(run_dir.glob("episode_*/env_*"))
    if nested:
        env_dirs = nested
    else:
        env_dirs = sorted(run_dir.glob("env_*"))
    if env_id is not None:
        env_dirs = [d for d in env_dirs if d.name == f"env_{env_id:02d}"]
    return env_dirs


def _derived_out_dir(out_root: Path, run_dir: Path, env_dir: Path) -> Path:
    """Mirror episode nesting under derived/, e.g. derived/episode_00/env_00/."""
    try:
        rel = env_dir.relative_to(run_dir)
    except ValueError:
        return out_root / env_dir.name
    return out_root / rel


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _f(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _i(val: Any, default: int | None = None) -> int | None:
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
