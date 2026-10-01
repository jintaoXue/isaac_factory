"""Run fair multi-entity ablations at Start<=5/10/15.

All arms use the same seed, split, targets, and 50-epoch budget. Every Start
cap is independently cold-started; checkpoints are never transferred between
horizons or arms.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
STARTS = (5, 10, 15)
ARMS = ("full", "noinfo", "nocross", "machineonly")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--data_dir", default="raw_data/dense_i1")
    parser.add_argument(
        "--skip_complete",
        action="store_true",
        help="Skip a run when last_metrics.json already exists.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")

    for arm in args.arm or list(ARMS):
        for start in STARTS:
            name = (
                f"dense_i1_entity_{arm}_start{start}_min8_"
                f"cold_ep{args.epochs}_seed{args.seed}"
            )
            save_rel = Path("libcity/cache/model_cache") / name
            save_abs = ROOT / save_rel
            metrics_path = save_abs / "last_metrics.json"
            ckpt_path = save_abs / "BNPDFormer_best.pt"
            if args.skip_complete and metrics_path.is_file() and ckpt_path.is_file():
                print(f"[skip complete] {name}", flush=True)
                continue

            cmd = [
                sys.executable,
                "-u",
                "-m",
                "factory_bn.train",
                "--config",
                f"factory_bn/configs/FactoryBN_dense_entity_{arm}_start{start}_min8.json",
                "--data_dir",
                args.data_dir,
                "--save_dir",
                str(save_rel),
                "--device",
                "cuda",
                "--seed",
                str(args.seed),
                "--max_epoch",
                str(args.epochs),
                "--wandb_project",
                "FactoryBN_PDFormer",
                "--wandb_name",
                name.removeprefix("dense_i1_"),
            ]
            cmd.extend(["--init_ckpt", ""])
            print(f"[{arm} Start<={start}]", " ".join(cmd), flush=True)
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                return int(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
