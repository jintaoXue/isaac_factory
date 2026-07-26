"""ICCBEI paper experiment runtime config and path helpers.

Overrides are applied BEFORE env/managers are constructed (see train.py).
Raw JSON outputs are consumed by ICCBEI2027 for plotting/tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HC_FACTORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_EXP_ROOT = HC_FACTORY_ROOT / "output" / "paper_exp"

# Source setting used for perception training (main.tex)
SOURCE_NH = 5
SOURCE_O = 5

NH_GRID = (1, 2, 3, 4, 5)
O_GRID = (1, 2, 3, 4, 5)

# How many successful episodes to log per (Nh, O) for TPA metrics
TPA_EPISODES_PER_CELL = 3
# Perception collect: source train split needs 6; OOD test cells need 1
SOURCE_COLLECT_EPISODES = 6
OOD_COLLECT_EPISODES = 1

SCENE_HUMAN_SLOTS = 5  # USD has human_00..04


def setting_name(num_humans: int, product_order: int) -> str:
    return f"Nh{int(num_humans)}_O{int(product_order)}"


def dataset_dir(num_humans: int, product_order: int) -> Path:
    return PAPER_EXP_ROOT / "datasets" / setting_name(num_humans, product_order)


def tpa_metrics_dir(num_humans: int, product_order: int) -> Path:
    return PAPER_EXP_ROOT / "tpa" / setting_name(num_humans, product_order)


def runs_dir() -> Path:
    return PAPER_EXP_ROOT / "perception_runs"


def metrics_dir() -> Path:
    return PAPER_EXP_ROOT / "metrics"


def ensure_dirs() -> None:
    for p in (PAPER_EXP_ROOT, runs_dir(), metrics_dir(), PAPER_EXP_ROOT / "datasets", PAPER_EXP_ROOT / "tpa"):
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def apply_runtime_overrides(
    *,
    num_humans: int | None = None,
    product_order: int | None = None,
    perception_output_dir: str | None = None,
    perception_max_episodes: int | None = None,
    perception_enabled: bool | None = None,
    tpa_metrics_dir_path: str | None = None,
    max_episodes: int | None = None,
    save_images: bool | None = None,
) -> dict[str, Any]:
    """Mutate module-level configs. Call before gym.make / manager init."""
    from ..env_asset_cfg import cfg_human, cfg_material_product
    from ..env_asset_cfg.perception import cfg_perception

    applied: dict[str, Any] = {}

    if num_humans is not None:
        n = int(num_humans)
        if n < 1 or n > SCENE_HUMAN_SLOTS:
            raise ValueError(f"num_humans must be in 1..{SCENE_HUMAN_SLOTS}, got {n}")
        cfg_human.CfgHumanRegistrationInfos["NormalHuman"] = n
        applied["num_humans"] = n

    if product_order is not None:
        o = int(product_order)
        max_reg = int(cfg_material_product.CfgRegistrationInfos["ProductWaterPipe"])
        if o < 1 or o > max_reg:
            raise ValueError(f"product_order must be in 1..{max_reg}, got {o}")
        cfg_material_product.CfgProductOrder["ProductWaterPipe"] = o
        applied["product_order"] = o

    if perception_output_dir is not None:
        cfg_perception.CfgPerception["output_dir"] = str(perception_output_dir)
        applied["perception_output_dir"] = str(perception_output_dir)

    if perception_max_episodes is not None:
        cfg_perception.CfgPerception["max_episodes"] = int(perception_max_episodes)
        applied["perception_max_episodes"] = int(perception_max_episodes)

    if perception_enabled is not None:
        cfg_perception.CfgPerception["enabled"] = bool(perception_enabled)
        applied["perception_enabled"] = bool(perception_enabled)

    if save_images is not None:
        cfg_perception.CfgPerception["save_images"] = bool(save_images)
        applied["save_images"] = bool(save_images)

    # Stash for managers that read env vars / module attrs
    if tpa_metrics_dir_path is not None:
        os.environ["HC_TPA_METRICS_DIR"] = str(tpa_metrics_dir_path)
        applied["tpa_metrics_dir"] = str(tpa_metrics_dir_path)
    if max_episodes is not None:
        os.environ["HC_MAX_EPISODES"] = str(int(max_episodes))
        applied["max_episodes"] = int(max_episodes)

    # Keep training cfg dataset_dir in sync when collecting source
    if perception_output_dir is not None:
        cfg_perception.CfgPerceptionTraining["dataset_dir"] = str(perception_output_dir)

    print(f"[paper_exp] applied overrides: {applied}")
    return applied


def active_human_id_vocab() -> list[str]:
    from ..env_asset_cfg.cfg_human import CfgHumanRegistrationInfos

    n = int(CfgHumanRegistrationInfos["NormalHuman"])
    return [f"{i:02d}" for i in range(n)]
