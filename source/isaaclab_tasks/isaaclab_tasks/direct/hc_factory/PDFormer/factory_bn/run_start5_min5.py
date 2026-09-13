"""Reproducible label audit, CUDA smoke, and Start5/Min5 experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from factory_bn.train import _load_config


CONFIGS = {
    "base": "FactoryBN_dense_start5_min5_base.json",
    "hazard": "FactoryBN_dense_start5_min5_hazard.json",
    "hazard_nocluster": "FactoryBN_dense_start5_min5_hazard_nocluster.json",
    "legacy": "FactoryBN_dense_start5_min5_legacy.json",
}


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "smoke", "main", "all"], default="all")
    parser.add_argument("--models", nargs="+", choices=sorted(CONFIGS), default=["base", "hazard"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    cfg_dir = root / "factory_bn" / "configs"
    if args.phase in {"audit", "all"}:
        _run([sys.executable, "-u", "-m", "factory_bn.diag_label_contract"])
    if args.phase in {"smoke", "main", "all"} and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    if args.phase == "smoke":
        models = ["hazard"]
        seeds = [42]
        epochs = 1
    else:
        models = args.models
        seeds = args.seeds
        epochs = args.max_epoch
    if args.phase not in {"smoke", "main", "all"}:
        return
    for model_name in models:
        config_path = cfg_dir / CONFIGS[model_name]
        cfg = _load_config(config_path)
        base_save = Path(str(cfg["save_dir"]))
        for seed in seeds:
            suffix = "_smoke" if args.phase == "smoke" else f"_seed{seed}"
            save_dir = str(base_save) + suffix
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "factory_bn.train",
                "--config",
                str(config_path),
                "--save_dir",
                save_dir,
                "--seed",
                str(seed),
                "--device",
                "cuda",
            ]
            if epochs is not None:
                cmd.extend(["--max_epoch", str(epochs)])
            if not args.wandb:
                cmd.append("--no_wandb")
            _run(cmd)


if __name__ == "__main__":
    main()
