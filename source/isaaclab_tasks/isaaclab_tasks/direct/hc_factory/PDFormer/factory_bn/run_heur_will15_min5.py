"""Run the section 12.1 heuristic-cluster model on will_15/Min5."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
CONFIG = "factory_bn/configs/FactoryBN_dense_heur_will15_min5_hier.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["smoke", "screen", "main"], default="screen"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epoch", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    default_epochs = {"smoke": 1, "screen": 20, "main": 100}
    epochs = int(args.max_epoch or default_epochs[args.phase])
    save_dir = (
        f"libcity/cache/model_cache/dense_i1_heur_will15_min5_hier_"
        f"{args.phase}_seed{args.seed}"
    )
    run_name = f"dense_i1_heur_will15_min5_hier_{args.phase}_seed{args.seed}"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "factory_bn.train",
        "--config",
        CONFIG,
        "--save_dir",
        save_dir,
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--max_epoch",
        str(epochs),
        "--wandb_project",
        "FactoryBN_PDFormer",
        "--wandb_name",
        run_name,
    ]
    print("[run]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
