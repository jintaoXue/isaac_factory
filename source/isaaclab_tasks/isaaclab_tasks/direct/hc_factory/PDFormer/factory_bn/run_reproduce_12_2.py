"""Reproduce section 12.2 base -> v2 -> v3 with isolated checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
RUNS = (
    ("base", "FactoryBN_dense_start5_min10.json", 100),
    ("v2", "FactoryBN_dense_start5_min10_v2.json", 40),
    ("v3", "FactoryBN_dense_start5_min10_v3.json", 60),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "main"], default="main")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")

    suffix = f"repro12_2_{args.phase}_seed{args.seed}"
    previous_ckpt: Path | None = None
    for name, config_name, full_epochs in RUNS:
        save_rel = Path("libcity/cache/model_cache") / f"{name}_{suffix}"
        save_abs = ROOT / save_rel
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "factory_bn.train",
            "--config",
            f"factory_bn/configs/{config_name}",
            "--data_dir",
            "raw_data/dense_i1",
            "--save_dir",
            str(save_rel),
            "--device",
            "cuda",
            "--seed",
            str(args.seed),
            "--max_epoch",
            str(1 if args.phase == "smoke" else full_epochs),
            "--wandb_project",
            "FactoryBN_PDFormer",
            "--wandb_name",
            f"dense_i1_start5_min10_{name}_{suffix}",
        ]
        if previous_ckpt is not None:
            if not previous_ckpt.is_file():
                raise SystemExit(f"Missing dependency checkpoint: {previous_ckpt}")
            cmd.extend(["--init_ckpt", str(previous_ckpt)])
        print(f"[12.2 {name}]", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            return int(result.returncode)
        previous_ckpt = save_abs / "BNPDFormer_best.pt"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
