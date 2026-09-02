#!/usr/bin/env python3
"""Summarize an explore catalog: coverage by n_finished, disk size, unique keys."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CURRICULUM_START_NFIN = (8, 6, 4, 2, 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect explore catalog coverage")
    p.add_argument(
        "root",
        nargs="?",
        default="env_checkpoints/random_explore/N10_T40000",
        help="Catalog root (contains catalog.jsonl)",
    )
    args = p.parse_args()
    root = Path(args.root)
    cat_path = root / "catalog.jsonl"
    if not cat_path.is_file():
        raise SystemExit(f"Missing {cat_path}")

    rows: list[dict] = []
    keys: set[str] = set()
    with cat_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            if row.get("key"):
                keys.add(str(row["key"]))

    by_nfin = Counter(int(r.get("n_finished", -1)) for r in rows)
    best_by_nfin: dict[int, dict] = {}
    for r in rows:
        n = int(r.get("n_finished", -1))
        prev = best_by_nfin.get(n)
        if prev is None or int(r.get("t", 0)) >= int(prev.get("t", 0)):
            best_by_nfin[n] = r

    pkl_bytes = 0
    pkl_count = 0
    ckpt_dir = root / "rounds"
    if ckpt_dir.is_dir():
        for pkl in ckpt_dir.rglob("*.pkl"):
            pkl_count += 1
            try:
                pkl_bytes += pkl.stat().st_size
            except OSError:
                pass

    print(f"catalog root: {root.resolve()}")
    print(f"jsonl rows: {len(rows)}  unique progress keys: {len(keys)}  pkl files: {pkl_count}")
    print(f"disk (pkls): {pkl_bytes / (1024 * 1024):.1f} MiB")
    print()
    print("n_finished coverage (curriculum start_nfin needs 8,6,4,2; 0=empty shop):")
    for n in sorted(by_nfin):
        mark = " <-- curriculum" if n in CURRICULUM_START_NFIN and n > 0 else ""
        print(f"  nfin={n:2d}: {by_nfin[n]:4d} rows{mark}")
    print()
    missing = [n for n in (8, 6, 4, 2) if n not in by_nfin]
    if missing:
        print(f"MISSING start_nfin buckets: {missing}  -> increase HC_EXPLORE_EPISODES or re-run explore")
    else:
        print("OK: all curriculum start_nfin buckets (8,6,4,2) present")
    print()
    print("pick_by_nfin would select (latest row per bucket):")
    for n in (8, 6, 4, 2, 0):
        row = best_by_nfin.get(n)
        if row:
            print(f"  nfin={n:2d}: {row.get('path')}  key={row.get('key')}  t={row.get('t')}")
        elif n == 0:
            print(f"  nfin={n:2d}: (empty shop, no catalog needed)")
        else:
            print(f"  nfin={n:2d}: MISSING")


if __name__ == "__main__":
    main()
