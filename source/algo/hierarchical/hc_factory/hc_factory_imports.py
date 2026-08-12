# -*- coding: utf-8 -*-
"""Import hc_factory env modules (``source.isaaclab_tasks...`` layout)."""
from __future__ import annotations

import importlib
from types import ModuleType

_MODULE_PREFIXES = (
    "source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory",
    "isaaclab_tasks.direct.hc_factory",
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CfgProcessTaskGalleryInAll": ("env_asset_cfg.cfg_process_task_gallery", "CfgProcessTaskGalleryInAll"),
    "CfgProcessTaskGalleryDetailedClassified": (
        "env_asset_cfg.cfg_process_task_gallery",
        "CfgProcessTaskGalleryDetailedClassified",
    ),
    "HcVectorEnvCfg": ("env_asset_cfg.cfg_hc_env", "HcVectorEnvCfg"),
    "CfgProductProcess": ("env_asset_cfg.cfg_material_product", "CfgProductProcess"),
    "AlgoHierarchicalMasker": ("src.algo_hierarchical_masker", "AlgoHierarchicalMasker"),
    "find_free_gantry_index": ("src.task_progress_manager", "find_free_gantry_index"),
    "find_workstation_index_for_task": ("src.task_progress_manager", "find_workstation_index_for_task"),
    "staging_slot_index": ("src.task_progress_manager", "staging_slot_index"),
}

_cache: dict[str, object] = {}


def import_hc_module(relative: str) -> ModuleType:
    last_err: ModuleNotFoundError | None = None
    for prefix in _MODULE_PREFIXES:
        full_name = f"{prefix}.{relative}"
        try:
            return importlib.import_module(full_name)
        except ModuleNotFoundError as exc:
            last_err = exc
            continue
    raise ModuleNotFoundError(
        f"Cannot import hc_factory.{relative} (tried: {_MODULE_PREFIXES})"
    ) from last_err


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _cache:
        rel, attr = _LAZY_EXPORTS[name]
        _cache[name] = getattr(import_hc_module(rel), attr)
    return _cache[name]


__all__ = list(_LAZY_EXPORTS.keys()) + ["import_hc_module"]
