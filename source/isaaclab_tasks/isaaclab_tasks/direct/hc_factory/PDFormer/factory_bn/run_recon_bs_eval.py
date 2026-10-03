"""Run eval_recon_bs on every checkpoint under model_cache, then summarize.

    cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer
    python factory_bn/run_recon_bs_eval.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "libcity/cache/model_cache"
OUT = ROOT / "libcity/cache/recon_eval"


def main() -> None:
    runs = sorted(p for p in CACHE.iterdir() if (p / "BNPDFormer_best.pt").is_file())
    print(f"[runner] {len(runs)} checkpoints", flush=True)
    failed = []
    for i, run in enumerate(runs, 1):
        print(f"[runner] ({i}/{len(runs)}) {run.name}", flush=True)
        rc = subprocess.call(
            [sys.executable, "-m", "factory_bn.eval_recon_bs", "eval", "--run_dir", str(run), "--out_root", str(OUT)],
            cwd=ROOT,
        )
        if rc != 0:
            failed.append(run.name)
    subprocess.check_call(
        [sys.executable, "-m", "factory_bn.eval_recon_bs", "summarize", "--out_root", str(OUT)], cwd=ROOT
    )
    print(f"[runner] done. failed={failed}", flush=True)


if __name__ == "__main__":
    main()
