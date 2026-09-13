"""Leakage-free Start<15/Min5 audit, smoke test, and overnight experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from factory_bn.train import _load_config


CONFIGS = {
    "base": "FactoryBN_clean_start15_min5_base.json",
    "hazard": "FactoryBN_clean_start15_min5_hazard.json",
    "hier": "FactoryBN_clean_will15_min5_hier.json",
}


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["audit", "smoke", "screen", "main", "all"], default="all"
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(CONFIGS), default=["base", "hazard"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 21])
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B explicitly; experiments log to W&B by default.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_dir = root / "factory_bn" / "configs"
    if args.phase in {"audit", "all"}:
        _run(
            [
                sys.executable,
                "-u",
                "-m",
                "factory_bn.diag_label_contract",
                "--data_dir",
                "raw_data/n10_i1_all_usable",
                "--max_start",
                "14",
                "--occupancy_horizon",
                "20",
                "--min_episode_jobs_total",
                "10",
            ]
        )
    if args.phase in {"smoke", "main", "all"} and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    if args.phase == "audit":
        return

    models = ["hier"] if args.phase in {"smoke", "screen"} else args.models
    seeds = [42] if args.phase in {"smoke", "screen"} else args.seeds
    if args.phase == "smoke":
        epochs = 1
    elif args.phase == "screen":
        epochs = args.max_epoch if args.max_epoch is not None else 12
    else:
        epochs = args.max_epoch
    for seed in seeds:
        for model_name in models:
            config_path = cfg_dir / CONFIGS[model_name]
            cfg = _load_config(config_path)
            base_save = Path(str(cfg["save_dir"]))
            suffix = "_smoke" if args.phase == "smoke" else f"_seed{seed}"
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "factory_bn.train",
                "--config",
                str(config_path),
                "--save_dir",
                str(base_save) + suffix,
                "--seed",
                str(seed),
                "--device",
                "cuda",
                "--wandb_name",
                f"{cfg.get('wandb_name', model_name)}_{args.phase}_seed{seed}",
            ]
            if epochs is not None:
                cmd.extend(["--max_epoch", str(epochs)])
            if args.no_wandb:
                cmd.append("--no_wandb")
            _run(cmd)


if __name__ == "__main__":
    main()
