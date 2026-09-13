"""Same-label decode / old-ckpt evals, then remain-only finetune."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "factory_bn" / "configs"
SRC = CFG_DIR / "FactoryBN_dense_f1_p80.json"
PREFIX8 = "libcity/cache/model_cache/dense_i1_a1_prefix8/BNPDFormer_best.pt"

EVALS = (
    (
        "FactoryBN_dense_ablate_noprefix.json",
        "libcity/cache/model_cache/dense_i1_a1_ablate_noprefix",
        {"event_decode_prefix": False, "wandb_name": "ablate_noprefix"},
    ),
    (
        "FactoryBN_dense_ablate_persist.json",
        "libcity/cache/model_cache/dense_i1_a1_ablate_persist",
        {"event_report_ongoing_only": True, "wandb_name": "ablate_persist"},
    ),
    (
        "FactoryBN_dense_ablate_nounion.json",
        "libcity/cache/model_cache/dense_i1_a1_ablate_nounion",
        {
            "event_union_occupancy": False,
            "event_union_upcoming": False,
            "wandb_name": "ablate_nounion",
        },
    ),
    (
        "FactoryBN_dense_ablate_noforce.json",
        "libcity/cache/model_cache/dense_i1_a1_ablate_noforce",
        {"force_ongoing_will": False, "wandb_name": "ablate_noforce"},
    ),
    (
        "FactoryBN_dense_ablate_f180.json",
        "libcity/cache/model_cache/dense_i1_a1_ablate_f180",
        {
            "event_decode_prefix": False,
            "init_ckpt": "libcity/cache/model_cache/dense_i1_a1_f180/BNPDFormer_best.pt",
            "wandb_name": "ablate_f180",
        },
    ),
    (
        "FactoryBN_dense_ablate_near5.json",
        "libcity/cache/model_cache/dense_i1_a1_ablate_near5",
        {
            "event_decode_prefix": False,
            "init_ckpt": "libcity/cache/model_cache/dense_i1_a1_near5/BNPDFormer_best.pt",
            "wandb_name": "ablate_near5",
        },
    ),
    (
        "FactoryBN_dense_ablate_v3ft.json",
        "libcity/cache/model_cache/dense_i1_a1_ablate_v3ft",
        {
            "event_decode_prefix": False,
            "init_ckpt": "libcity/cache/model_cache/dense_i1_a1_v3ft/BNPDFormer_best.pt",
            "wandb_name": "ablate_v3ft",
        },
    ),
)


def _base() -> dict:
    return json.loads(SRC.read_text(encoding="utf-8"))


def write_eval_configs() -> None:
    base = _base()
    for name, _save, overrides in EVALS:
        cfg = deepcopy(base)
        cfg.update(
            {
                "max_epoch": 0,
                "patience": 1,
                "init_ckpt": PREFIX8,
                "unfreeze_last_encoder_blocks": 0,
                "oversample_event_windows": 1.0,
            }
        )
        cfg.update(overrides)
        (CFG_DIR / name).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def write_remain_config() -> Path:
    cfg = deepcopy(_base())
    cfg.update(
        {
            "remain_len_use_rate": True,
            "w_remain_len": 4.0,
            "freeze_except_remain_len": True,
            "unfreeze_last_encoder_blocks": 0,
            "init_ckpt": PREFIX8,
            "max_epoch": 8,
            "patience": 8,
            "wandb_name": "dense_i1_a1_prefix8_remain_frozen",
        }
    )
    path = CFG_DIR / "FactoryBN_dense_prefix8_remain_frozen.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def _run(cfg: str, save: str) -> int:
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
        "--max_epoch",
        "0" if "ablate" in save else "8",
    ]
    print(f"[real-ablate] start {save}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"[real-ablate] done {save} code={proc.returncode}", flush=True)
    return int(proc.returncode)


def main() -> int:
    write_eval_configs()
    write_remain_config()
    for name, save, _ in EVALS:
        code = _run(f"factory_bn/configs/{name}", save)
        if code != 0:
            return code
    return _run(
        "factory_bn/configs/FactoryBN_dense_prefix8_remain_frozen.json",
        "libcity/cache/model_cache/dense_i1_a1_prefix8_remain_frozen",
    )


if __name__ == "__main__":
    raise SystemExit(main())
