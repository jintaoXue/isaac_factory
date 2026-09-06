"""Sequential prefix8 follow-ups: supervised, remain-rate, STGNPP."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    (
        "factory_bn/configs/FactoryBN_dense_prefix8_sup.json",
        "libcity/cache/model_cache/dense_i1_a1_prefix8_sup",
    ),
    (
        "factory_bn/configs/FactoryBN_dense_prefix8_remain.json",
        "libcity/cache/model_cache/dense_i1_a1_prefix8_remain",
    ),
    (
        "factory_bn/configs/FactoryBN_dense_prefix8_stgnpp.json",
        "libcity/cache/model_cache/dense_i1_a1_prefix8_stgnpp",
    ),
)


def main() -> int:
    for cfg, save in RUNS:
        cmd = [
            sys.executable,
            "-m",
            "factory_bn.train",
            "--config",
            cfg,
            "--data_dir",
            "raw_data/dense_i1",
            "--save_dir",
            save,
            "--device",
            "cuda",
            "--no_wandb",
        ]
        print(f"[ablation] start {save}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"[ablation] failed {save} code={proc.returncode}", flush=True)
            return proc.returncode
        print(f"[ablation] done {save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
