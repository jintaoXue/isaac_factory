"""Runtime disturbance config — set from CLI before env construction.

Usage (train.py)::

    --disturbance_dim machine|human|logistics|material|none
    --disturbance_intensity 1.0
    [--disturbance_human_count N]
    [--disturbance_agv_count N]
    [--disturbance_gantry_count N]
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DISTURBANCE_DIMS = ("none", "material", "human", "logistics", "machine")

# Snapshot of defaults restored when re-applying (multi-run in one process).
_DEFAULT_SNAPSHOT: dict[str, Any] | None = None

RuntimeDisturbanceCfg: dict[str, Any] = {
    "dim": "none",
    "intensity": 1.0,
    # Optional CLI overrides (None = use intensity-derived defaults).
    "human_count": None,
    "agv_count": None,
    "gantry_count": None,
    # Derived / applied values (filled by apply_disturbance_to_cfgs).
    "applied": {},
    # L1 noise / multipliers read by managers at runtime.
    "machine_process_noise_std": 0.0,
    "machine_success_rate": 1.0,
    "human_subtask_noise_std": 2.0,  # matches current SubtaskTimeNoiseStdSteps default
    "human_time_scale": 1.0,
    "gantry_animation_noise_std": 2.0,
    "gantry_time_scale": 1.0,
    # Material: fraction of raw submaterials skipped at reset (shortage).
    "material_shortage_frac": 0.0,
    # L2 event schedule (logic steps). end < 0 means disabled.
    "event_start_step": -1,
    "event_duration_steps": 0,
    "event_target": None,  # e.g. machine type name or human idx
}


def configure_disturbance_from_cli(
    dim: str = "none",
    intensity: float = 1.0,
    human_count: int | None = None,
    agv_count: int | None = None,
    gantry_count: int | None = None,
) -> dict[str, Any]:
    """Fill RuntimeDisturbanceCfg from CLI; call before gym.make / apply."""
    dim = (dim or "none").lower().strip()
    if dim not in DISTURBANCE_DIMS:
        raise ValueError(f"disturbance_dim must be one of {DISTURBANCE_DIMS}, got {dim!r}")
    intensity = max(0.0, float(intensity))

    RuntimeDisturbanceCfg["dim"] = dim
    RuntimeDisturbanceCfg["intensity"] = intensity
    RuntimeDisturbanceCfg["human_count"] = human_count
    RuntimeDisturbanceCfg["agv_count"] = agv_count
    RuntimeDisturbanceCfg["gantry_count"] = gantry_count
    RuntimeDisturbanceCfg["applied"] = {}

    # Reset derived fields to safe baseline, then specialize by dim.
    RuntimeDisturbanceCfg["machine_process_noise_std"] = 0.0
    RuntimeDisturbanceCfg["machine_success_rate"] = 1.0
    RuntimeDisturbanceCfg["human_subtask_noise_std"] = 2.0
    RuntimeDisturbanceCfg["human_time_scale"] = 1.0
    RuntimeDisturbanceCfg["gantry_animation_noise_std"] = 2.0
    RuntimeDisturbanceCfg["gantry_time_scale"] = 1.0
    RuntimeDisturbanceCfg["material_shortage_frac"] = 0.0
    RuntimeDisturbanceCfg["event_start_step"] = -1
    RuntimeDisturbanceCfg["event_duration_steps"] = 0
    RuntimeDisturbanceCfg["event_target"] = None

    if dim == "none" or intensity <= 0.0:
        return RuntimeDisturbanceCfg

    if dim == "machine":
        # Process-time noise around nominal; occasional failure / rework.
        RuntimeDisturbanceCfg["machine_process_noise_std"] = 5.0 * intensity
        RuntimeDisturbanceCfg["machine_success_rate"] = max(0.7, 1.0 - 0.08 * intensity)
        RuntimeDisturbanceCfg["event_start_step"] = 800
        RuntimeDisturbanceCfg["event_duration_steps"] = int(120 * intensity)
        RuntimeDisturbanceCfg["event_target"] = "num02_rollerbedCNCPipeIntersectionCuttingMachine"
    elif dim == "human":
        RuntimeDisturbanceCfg["human_subtask_noise_std"] = 2.0 + 8.0 * intensity
        RuntimeDisturbanceCfg["human_time_scale"] = 1.0 + 0.35 * intensity
        RuntimeDisturbanceCfg["event_start_step"] = 600
        RuntimeDisturbanceCfg["event_duration_steps"] = int(150 * intensity)
        RuntimeDisturbanceCfg["event_target"] = "human_0"
    elif dim == "logistics":
        RuntimeDisturbanceCfg["gantry_animation_noise_std"] = 2.0 + 6.0 * intensity
        RuntimeDisturbanceCfg["gantry_time_scale"] = 1.0 + 0.4 * intensity
        RuntimeDisturbanceCfg["event_start_step"] = 700
        RuntimeDisturbanceCfg["event_duration_steps"] = int(100 * intensity)
        RuntimeDisturbanceCfg["event_target"] = "gantry_0"
    elif dim == "material":
        RuntimeDisturbanceCfg["material_shortage_frac"] = min(0.6, 0.25 * intensity)
        RuntimeDisturbanceCfg["machine_success_rate"] = max(0.75, 1.0 - 0.1 * intensity)
        RuntimeDisturbanceCfg["event_start_step"] = -1

    return RuntimeDisturbanceCfg


def _ensure_default_snapshot() -> None:
    global _DEFAULT_SNAPSHOT
    if _DEFAULT_SNAPSHOT is not None:
        return
    from .cfg_human import CfgHumanRegistrationInfos
    from .cfg_robot import CfgRobotRegistrationInfos
    from .cfg_machine import CfgMachine
    from . import cfg_process_subtask_gallery as subtask_mod

    _DEFAULT_SNAPSHOT = {
        "human": deepcopy(CfgHumanRegistrationInfos),
        "robot": deepcopy(CfgRobotRegistrationInfos),
        "active_gantry_indices": list(CfgMachine["num07_gantry_group"]["active_gantry_indices"]),
        "gantry_animation_time": CfgMachine["num07_gantry_group"]["registration_infos"][
            "num07_gantry_group"
        ]["animation_time"],
        "gantry_animation_time_noise_std": CfgMachine["num07_gantry_group"]["registration_infos"][
            "num07_gantry_group"
        ].get("animation_time_noise_std", 2.0),
        "subtask_noise_std": float(subtask_mod.SubtaskTimeNoiseStdSteps),
        "machine_animation_times": {
            mtype: {
                part: info.get("animation_time")
                for part, info in cfg.get("registration_infos", {}).items()
            }
            for mtype, cfg in CfgMachine.items()
            if mtype != "num07_gantry_group"
        },
        "machine_animation_noise": {
            mtype: {
                part: info.get("animation_time_noise_std", 0.0)
                for part, info in cfg.get("registration_infos", {}).items()
            }
            for mtype, cfg in CfgMachine.items()
            if mtype != "num07_gantry_group"
        },
    }


def apply_disturbance_to_cfgs() -> dict[str, Any]:
    """Mutate global registration / timing cfgs. Must run before managers construct."""
    _ensure_default_snapshot()
    assert _DEFAULT_SNAPSHOT is not None

    from .cfg_human import CfgHumanRegistrationInfos
    from .cfg_robot import CfgRobotRegistrationInfos
    from .cfg_machine import CfgMachine
    from . import cfg_process_subtask_gallery as subtask_mod

    # Restore defaults first (idempotent re-apply).
    CfgHumanRegistrationInfos.clear()
    CfgHumanRegistrationInfos.update(deepcopy(_DEFAULT_SNAPSHOT["human"]))
    CfgRobotRegistrationInfos.clear()
    CfgRobotRegistrationInfos.update(deepcopy(_DEFAULT_SNAPSHOT["robot"]))
    CfgMachine["num07_gantry_group"]["active_gantry_indices"].clear()
    CfgMachine["num07_gantry_group"]["active_gantry_indices"].extend(
        _DEFAULT_SNAPSHOT["active_gantry_indices"]
    )
    gantry_info = CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]
    gantry_info["animation_time"] = _DEFAULT_SNAPSHOT["gantry_animation_time"]
    gantry_info["animation_time_noise_std"] = _DEFAULT_SNAPSHOT["gantry_animation_time_noise_std"]
    subtask_mod.SubtaskTimeNoiseStdSteps = _DEFAULT_SNAPSHOT["subtask_noise_std"]
    for mtype, parts in _DEFAULT_SNAPSHOT["machine_animation_times"].items():
        for part, t in parts.items():
            if part in CfgMachine[mtype]["registration_infos"]:
                CfgMachine[mtype]["registration_infos"][part]["animation_time"] = t
                CfgMachine[mtype]["registration_infos"][part]["animation_time_noise_std"] = (
                    _DEFAULT_SNAPSHOT["machine_animation_noise"][mtype].get(part, 0.0)
                )

    dim = RuntimeDisturbanceCfg["dim"]
    intensity = float(RuntimeDisturbanceCfg["intensity"])
    applied: dict[str, Any] = {"dim": dim, "intensity": intensity}

    if dim == "none" or intensity <= 0.0:
        RuntimeDisturbanceCfg["applied"] = applied
        return applied

    # Always push L1 noise knobs into modules that managers read.
    subtask_mod.SubtaskTimeNoiseStdSteps = float(RuntimeDisturbanceCfg["human_subtask_noise_std"])
    applied["human_subtask_noise_std"] = subtask_mod.SubtaskTimeNoiseStdSteps
    applied["human_time_scale"] = RuntimeDisturbanceCfg["human_time_scale"]
    applied["machine_process_noise_std"] = RuntimeDisturbanceCfg["machine_process_noise_std"]
    applied["machine_success_rate"] = RuntimeDisturbanceCfg["machine_success_rate"]
    applied["material_shortage_frac"] = RuntimeDisturbanceCfg["material_shortage_frac"]

    if dim == "human":
        default_n = int(_DEFAULT_SNAPSHOT["human"].get("NormalHuman", 5))
        n = RuntimeDisturbanceCfg["human_count"]
        if n is None:
            n = max(1, int(round(default_n - 2 * intensity)))
        n = max(1, min(int(n), default_n))
        CfgHumanRegistrationInfos["NormalHuman"] = n
        applied["human_count"] = n

    elif dim == "logistics":
        default_agv = int(_DEFAULT_SNAPSHOT["robot"].get("AGV", 2))
        n_agv = RuntimeDisturbanceCfg["agv_count"]
        if n_agv is None:
            n_agv = max(1, int(round(default_agv - intensity)))
        n_agv = max(1, min(int(n_agv), default_agv))
        CfgRobotRegistrationInfos["AGV"] = n_agv
        applied["agv_count"] = n_agv

        default_gantry = list(_DEFAULT_SNAPSHOT["active_gantry_indices"])
        n_g = RuntimeDisturbanceCfg["gantry_count"]
        if n_g is None:
            n_g = 1 if intensity >= 0.75 else len(default_gantry)
        n_g = max(1, min(int(n_g), 4))
        gantry_indices = CfgMachine["num07_gantry_group"]["active_gantry_indices"]
        gantry_indices.clear()
        gantry_indices.extend(range(n_g))
        applied["gantry_count"] = n_g
        applied["active_gantry_indices"] = list(gantry_indices)

        scale = float(RuntimeDisturbanceCfg["gantry_time_scale"])
        gantry_info["animation_time"] = max(
            1, int(round(_DEFAULT_SNAPSHOT["gantry_animation_time"] * scale))
        )
        gantry_info["animation_time_noise_std"] = float(
            RuntimeDisturbanceCfg["gantry_animation_noise_std"]
        )
        applied["gantry_animation_time"] = gantry_info["animation_time"]
        applied["gantry_animation_noise_std"] = gantry_info["animation_time_noise_std"]

    elif dim == "machine":
        std = float(RuntimeDisturbanceCfg["machine_process_noise_std"])
        for mtype, cfg in CfgMachine.items():
            if mtype == "num07_gantry_group":
                continue
            for part, info in cfg.get("registration_infos", {}).items():
                if "animation_time" in info:
                    info["animation_time_noise_std"] = std
        applied["machine_animation_time_noise_std"] = std

    elif dim == "material":
        applied["material_shortage_frac"] = RuntimeDisturbanceCfg["material_shortage_frac"]
        applied["machine_success_rate"] = RuntimeDisturbanceCfg["machine_success_rate"]

    applied["event_start_step"] = RuntimeDisturbanceCfg["event_start_step"]
    applied["event_duration_steps"] = RuntimeDisturbanceCfg["event_duration_steps"]
    applied["event_target"] = RuntimeDisturbanceCfg["event_target"]
    RuntimeDisturbanceCfg["applied"] = applied

    print(f"[Disturbance] dim={dim} intensity={intensity} applied={applied}")
    return applied
