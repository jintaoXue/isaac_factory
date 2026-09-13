"""Sequential overnight trains: heuristic+cluster, then start5/min10.

Run from PDFormer root::

    python -u -m factory_bn.run_overnight_ep100
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

RUNS = [
    {
        "name": "heur_cluster",
        "config": "factory_bn/configs/FactoryBN_dense_prefix8_heur_cluster.json",
        "save_dir": "libcity/cache/model_cache/dense_i1_a1_prefix8_heur_cluster_ep100",
        "wandb_name": "dense_i1_a1_prefix8_heur_cluster_ep100",
    },
    {
        "name": "start5_min10",
        "config": "factory_bn/configs/FactoryBN_dense_start5_min10.json",
        "save_dir": "libcity/cache/model_cache/dense_i1_a1_start5_min10_ep100",
        "wandb_name": "dense_i1_a1_start5_min10_ep100",
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
            "100",
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
            return int(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
