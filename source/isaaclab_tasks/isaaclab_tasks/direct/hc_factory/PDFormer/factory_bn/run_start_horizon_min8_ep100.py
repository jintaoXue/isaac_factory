"""Run matched start<=5/10/15, Min8, 100-epoch experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import argparse
import torch


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "main"], default="screen")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    epochs = 15 if args.phase == "screen" else 100
    previous_ckpt: Path | None = None
    for max_start in (5, 10, 15):
        name = (
            f"dense_i1_12_3_start{max_start}_min8_cause4_opt_"
            f"{args.phase}_seed{args.seed}"
        )
        save_rel = Path("libcity/cache/model_cache") / name
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "factory_bn.train",
            "--config",
            f"factory_bn/configs/FactoryBN_dense_12_3_start{max_start}_min8_opt.json",
            "--data_dir",
            "raw_data/dense_i1",
            "--save_dir",
            str(save_rel),
            "--device",
            "cuda",
            "--seed",
            str(args.seed),
            "--max_epoch",
            str(epochs),
            "--wandb_project",
            "FactoryBN_PDFormer",
            "--wandb_name",
            name,
        ]
        if previous_ckpt is not None:
            cmd.extend(["--init_ckpt", str(previous_ckpt)])
        print(f"[start<={max_start} Min8]", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            return int(result.returncode)
        previous_ckpt = ROOT / save_rel / "BNPDFormer_best.pt"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
