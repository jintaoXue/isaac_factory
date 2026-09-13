"""Serial GNN-structure ablations: start<=5/10/15, Min8.

Each arm trains start5 → start10 → start15, warm-starting from the previous
best checkpoint. The default skips ``full`` because its three official runs
already exist, then runs ablations in evidence-value priority order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
STARTS = (5, 10, 15)
ARMS = {
    "nograph": "FactoryBN_dense_struct_nograph_start{start}_min8.json",
    "nogroup": "FactoryBN_dense_struct_nogroup_start{start}_min8.json",
    "nosem": "FactoryBN_dense_struct_nosem_start{start}_min8.json",
    "nopattern": "FactoryBN_dense_struct_nopattern_start{start}_min8.json",
    "full": "FactoryBN_dense_12_3_start{start}_min8_opt.json",
}
DEFAULT_ARMS = ("nograph", "nogroup", "nosem", "nopattern")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "main"], default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arm",
        action="append",
        choices=list(ARMS),
        help=(
            "Repeat to pick a subset. Default priority: "
            "nograph, nogroup, nosem, nopattern; full is already available."
        ),
    )
    parser.add_argument(
        "--data_dir",
        default="raw_data/dense_i1",
        help="Keep the official pack for the paper table.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    epochs = 15 if args.phase == "screen" else 100
    arms = args.arm or list(DEFAULT_ARMS)
    for arm in arms:
        previous_ckpt: Path | None = None
        for max_start in STARTS:
            cfg_name = ARMS[arm].format(start=max_start)
            name = (
                f"dense_i1_struct_{arm}_start{max_start}_min8_"
                f"{args.phase}_seed{args.seed}"
            )
            save_rel = Path("libcity/cache/model_cache") / name
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "factory_bn.train",
                "--config",
                f"factory_bn/configs/{cfg_name}",
                "--data_dir",
                args.data_dir,
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
            print(f"[{arm} start<={max_start} Min8]", " ".join(cmd), flush=True)
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                return int(result.returncode)
            previous_ckpt = ROOT / save_rel / "BNPDFormer_best.pt"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
