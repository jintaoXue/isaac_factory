"""Count window-level cause vs per-station seed clusters on dense_i1."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from factory_bn.cause_cluster import CAUSE_ALIGNED_NAMES, seed_cluster_ids
from factory_bn.causes import ROOT_CAUSE_CLASSES
from factory_bn.dataset import load_factory_bn_bundle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "raw_data" / "dense_i1"


def main() -> None:
    bundle = load_factory_bn_bundle(DATA)
    classes = list(bundle.get("cause_classes") or ROOT_CAUSE_CLASSES)
    win = Counter()
    seed = Counter()
    n_win = 0
    n_seed = 0
    for ep in bundle["episodes"]:
        cause = np.asarray(ep.get("cause", []), dtype=np.int64)
        for cid in cause.reshape(-1):
            n_win += 1
            if int(cid) >= 0:
                name = classes[int(cid)] if int(cid) < len(classes) else str(cid)
                win[name] += 1
            else:
                win["<unlabeled>"] += 1
        feats = np.asarray(ep["features"], dtype=np.float32)
        ids = seed_cluster_ids(feats, window_size_s=60.0)
        n_seed += int(ids.size)
        for i, name in enumerate(CAUSE_ALIGNED_NAMES):
            seed[name] += int((ids == i).sum())
    print("window_cause (one label per window, bottleneck node only)")
    for k, v in win.most_common():
        print(f"  {k:24s} {v:7d}  {v / max(n_win, 1):.4f}")
    print("station_seed_cluster (every station, every minute)")
    for name in CAUSE_ALIGNED_NAMES:
        v = seed[name]
        print(f"  {name:24s} {v:7d}  {v / max(n_seed, 1):.4f}")


if __name__ == "__main__":
    main()
