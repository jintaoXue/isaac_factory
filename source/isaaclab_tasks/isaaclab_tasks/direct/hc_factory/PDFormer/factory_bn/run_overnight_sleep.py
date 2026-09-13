"""Overnight queue while sleeping.

1) cluster_up 100ep (upcoming-oriented clustering on start<=2, >=8)
2) cluster_up_ctrl 100ep (same levers, no cluster)
3) start5_min10_cluster40 (start<=5, >=10 + cluster)
4) start5_min10_nocluster40 (same, no cluster)

Run from PDFormer root::

    python -u -m factory_bn.run_overnight_sleep
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

RUNS = [
    {
        "name": "cluster_up_100",
        "config": "factory_bn/configs/FactoryBN_dense_prefix8_cluster_up.json",
        "save_dir": "libcity/cache/model_cache/dense_i1_a1_prefix8_cluster_up",
        "wandb_name": "dense_i1_a1_prefix8_cluster_up",
        "max_epoch": "100",
    },
    {
        "name": "cluster_up_ctrl_100",
        "config": "factory_bn/configs/FactoryBN_dense_prefix8_cluster_up_ctrl.json",
        "save_dir": "libcity/cache/model_cache/dense_i1_a1_prefix8_cluster_up_ctrl",
        "wandb_name": "dense_i1_a1_prefix8_cluster_up_ctrl",
        "max_epoch": "100",
    },
    {
        "name": "start5_cluster40",
        "config": "factory_bn/configs/FactoryBN_dense_start5_min10_cluster40.json",
        "save_dir": "libcity/cache/model_cache/dense_i1_a1_start5_min10_cluster40",
        "wandb_name": "dense_i1_a1_start5_min10_cluster40",
        "max_epoch": "40",
    },
    {
        "name": "start5_nocluster40",
        "config": "factory_bn/configs/FactoryBN_dense_start5_min10_nocluster40.json",
        "save_dir": "libcity/cache/model_cache/dense_i1_a1_start5_min10_nocluster40",
        "wandb_name": "dense_i1_a1_start5_min10_nocluster40",
        "max_epoch": "40",
    },
]


def main() -> int:
    for i, run in enumerate(RUNS, 1):
        print(f"\n{'=' * 72}\n[{i}/{len(RUNS)}] START {run['name']}\n{'=' * 72}\n", flush=True)
        cmd = [
            PY,
            "-u",
            "-m",
            "factory_bn.train",
            "--config",
            run["config"],
            "--data_dir",
            "raw_data/dense_i1",
            "--save_dir",
            run["save_dir"],
            "--device",
            "cuda",
            "--max_epoch",
            run["max_epoch"],
            "--wandb_project",
            "FactoryBN_PDFormer",
            "--wandb_name",
            run["wandb_name"],
        ]
        print("[cmd]", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT))
        print(
            f"\n[{i}/{len(RUNS)}] END {run['name']} exit={proc.returncode}\n",
            flush=True,
        )
        if proc.returncode != 0:
            print(f"[warn] {run['name']} failed; continuing to next run", flush=True)
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
