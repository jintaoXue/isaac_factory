"""Freeze / env-state JSON dumps under ``output/debug_env_state``.

Auto-dump when an ongoing-task fingerprint is unchanged for ``FREEZE_STEPS``.
Manual: ``touch output/debug_env_state/REQUEST_DUMP``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

DEBUG_DIR = Path(__file__).resolve().parent.parent / "output" / "debug_env_state"
REQUEST_DUMP_FLAG = DEBUG_DIR / "REQUEST_DUMP"
FREEZE_STEPS = 400

_DROP_TOP = frozenset({"articulations", "rigid_prims", "camera", "perception"})
_DROP_LEAF = frozenset({"object", "generated_route", "rgb"})

_freeze_fp: str | None = None
_freeze_count = 0
_last_dump_fp: str | None = None


def to_jsonable(obj: Any, *, max_list: int = 64, max_tensor: int = 64) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        out: dict[str, Any] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "numel": int(t.numel()),
        }
        if t.numel() <= max_tensor:
            out["values"] = t.tolist()
        else:
            flat = t.reshape(-1)[:32]
            out["values_head"] = flat.tolist()
        return out
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v, max_list=max_list, max_tensor=max_tensor)
            for k, v in obj.items()
            if k not in _DROP_LEAF
        }
    if isinstance(obj, (list, tuple)):
        seq = list(obj)
        if len(seq) > max_list:
            return {
                "len": len(seq),
                "head": [to_jsonable(x, max_list=max_list, max_tensor=max_tensor) for x in seq[:8]],
            }
        return [to_jsonable(x, max_list=max_list, max_tensor=max_tensor) for x in seq]
    return {"_type": type(obj).__name__, "_repr": repr(obj)[:200]}


def ongoing_fingerprint(env: dict) -> str:
    progress = env.get("progress") or {}
    records = progress.get("ongoing_task_records") or {}
    parts: list[str] = []
    for pid in sorted(records.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        tr = records[pid]
        sd = tr.get("subtasks_dict") or {}
        parts.append(
            f"{pid}:{tr.get('task')}:{sd.get('ongoing_index')}:{sd.get('ongoing')}:{sd.get('finished')}:"
            f"g={tr.get('chosen_gantry_index')}:z={tr.get('preferred_gantry_zone')}:"
            f"out={sd.get('outbound_attached')}/{sd.get('outbound_mode')}"
        )
    return "|".join(parts) or "empty"


def _slim_agent(state: dict | None) -> dict | None:
    if not isinstance(state, dict):
        return state
    keep = (
        "state",
        "ongoing_task_record_index",
        "current_area_id",
        "target_area_id",
        "route_index",
        "route_length",
        "subtask_time_step",
        "subtask_time_target",
    )
    return {k: to_jsonable(state.get(k)) for k in keep}


def _task_record_summary(tr: dict) -> dict:
    sd = tr.get("subtasks_dict") or {}
    return {
        "task": tr.get("task"),
        "task_type": tr.get("task_type"),
        "product": tr.get("product"),
        "product_index": tr.get("product_index"),
        "human": tr.get("human"),
        "robot": tr.get("robot"),
        "robot_index": tr.get("robot_index"),
        "chosen_gantry_index": tr.get("chosen_gantry_index"),
        "preferred_gantry_zone": tr.get("preferred_gantry_zone"),
        "goal_gantry_zone": tr.get("goal_gantry_zone"),
        "chosen_machine_workstation": tr.get("chosen_machine_workstation"),
        "task_start_time_step": tr.get("task_start_time_step"),
        "subtasks": {
            "ongoing": sd.get("ongoing"),
            "ongoing_index": sd.get("ongoing_index"),
            "finished": sd.get("finished"),
            "num_subtasks": sd.get("num_subtasks"),
            "material_start_area": sd.get("material_start_area"),
            "material_goal_area": sd.get("material_goal_area"),
            "outbound_attached": sd.get("outbound_attached"),
            "outbound_mode": sd.get("outbound_mode"),
            "start_gantry_parking": (sd.get("start_area_ids") or {}).get("gantry_parking_areas_ids"),
            "start_robot_parking": (sd.get("start_area_ids") or {}).get("robot_parking_areas_ids"),
            "goal_gantry_parking": (sd.get("goal_area_ids") or {}).get("gantry_parking_areas_ids")
            if isinstance(sd.get("goal_area_ids"), dict)
            else None,
            "goal_robot_parking": (sd.get("goal_area_ids") or {}).get("robot_parking_areas_ids")
            if isinstance(sd.get("goal_area_ids"), dict)
            else None,
        },
    }


def _gantry_animation_dump(gantry_machine) -> dict | None:
    if gantry_machine is None:
        return None
    anim = getattr(gantry_machine, "animation_num07_gantry_group", None)
    if anim is None:
        return None
    active = list(getattr(gantry_machine, "ACTIVE_GANTRY_INDICES", []))
    try:
        return {
            "path_length": list(anim.path_length),
            "distance_traveled": list(anim.distance_traveled),
            "speed": list(anim.speed),
            "done_attr": list(anim.done) if isinstance(anim.done, list) else anim.done,
            "is_done_method": [bool(anim.is_done(i)) for i in active],
            "is_yield_move": list(anim.is_yield_move),
            "move_loaded": list(anim.move_loaded),
            "base_speed": anim.base_speed,
            "move_dt": anim.move_dt,
            "yielding": list(getattr(gantry_machine, "_yielding", [])),
            "yield_target_x": list(getattr(gantry_machine, "_yield_target_x", [])),
            "safe_x_gap": float(getattr(gantry_machine, "safe_x_gap", -1)),
        }
    except Exception as exc:  # noqa: BLE001 — debug dump must not crash sim
        return {"error": repr(exc)}


def _gantry_world_xy(gantry_machine, env: dict) -> dict | None:
    if gantry_machine is None:
        return None
    try:
        jp = env["articulations"]["num07_gantry_group"]["joint_position"]
        active = list(gantry_machine.ACTIVE_GANTRY_INDICES)
        return {
            str(i): {
                "x": gantry_machine._joint_to_world_x(jp, i),
                "y": gantry_machine._joint_to_world_y(jp, i),
            }
            for i in active
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def build_freeze_dump(env: dict, gantry_machine=None, *, reason: str = "manual") -> dict:
    progress = env.get("progress") or {}
    records = progress.get("ongoing_task_records") or {}
    gstate = (env.get("machine") or {}).get("num07_gantry_group") or {}

    gidx = None
    for tr in records.values():
        if tr.get("chosen_gantry_index") is not None:
            gidx = int(tr["chosen_gantry_index"])
            break

    diagnosis: dict[str, Any] = {"reason": reason, "fingerprint": ongoing_fingerprint(env)}
    if gantry_machine is not None and gidx is not None:
        anim = gantry_machine.animation_num07_gantry_group
        xy_missing = gstate.get("target_area_xy", [None] * 4)[gidx] is None
        joints_missing = gstate.get("target_joints_position", [None] * 4)[gidx] is None
        yielding = bool(gantry_machine._yielding[gidx])
        anim_done = bool(anim.is_done(gidx))
        plen = float(anim.path_length[gidx])
        dtr = float(anim.distance_traveled[gidx])
        ratio = None if plen <= 1e-8 else dtr / plen
        if xy_missing:
            likely = "waiting_xy"
        elif joints_missing and yielding:
            likely = "blocked_by_yield_before_set_target"
        elif joints_missing:
            likely = "never_started_move"
        elif plen > 1e-8 and dtr < 1e-6 and not anim_done:
            likely = "move_blocked_safe_gap_or_yield"
        elif anim_done and not (
            (records and list(records.values())[0].get("subtasks_dict", {}).get("finished", [True, True])[1])
        ):
            likely = "done_flag_desync"
        else:
            likely = "move_in_progress_or_other"
        diagnosis.update(
            {
                "chosen_gantry_index": gidx,
                "xy_missing": xy_missing,
                "joints_missing": joints_missing,
                "is_yielding": yielding,
                "anim_done": anim_done,
                "progress_ratio": ratio,
                "path_length": plen,
                "distance_traveled": dtr,
                "likely": likely,
            }
        )

    # Heuristic when no gantry assigned yet (finding_free_gantry / wait AGV).
    for pid, tr in records.items():
        sd = tr.get("subtasks_dict") or {}
        ongoing = sd.get("ongoing") or []
        finished = sd.get("finished") or []
        if len(ongoing) > 1 and ongoing[1] == "finding_free_gantry" and not finished[1]:
            diagnosis["likely"] = "waiting_preferred_gantry_zone"
            diagnosis["preferred_gantry_zone"] = tr.get("preferred_gantry_zone")
            diagnosis["gantry_states"] = gstate.get("state")
        if (
            tr.get("task_type") == "processing"
            and sd.get("goal_area_ids") is not None
            and not sd.get("outbound_attached")
        ):
            diagnosis["likely"] = "waiting_agv_for_cross_outbound"
            diagnosis["goal_area"] = sd.get("material_goal_area")

    # Idle WIP: producing but nothing runnable / nothing ongoing.
    n_producing = len(progress.get("producing") or [])
    if not records and n_producing > 0:
        diagnosis["likely"] = "idle_wip_dispatch_deadlock"
        diagnosis["n_producing"] = n_producing

    materials_brief: dict[str, Any] = {}
    for name, mat in (env.get("material") or {}).items():
        if not isinstance(mat, dict):
            continue
        kv = mat.get("key_variables") or {}
        subs = mat.get("submaterials") or {}
        materials_brief[name] = {
            "idx": kv.get("idx"),
            "finished_task": mat.get("finished_task"),
            "ongoing_task_record_index": mat.get("ongoing_task_record_index"),
            "storage": {
                sk: (sv or {}).get("storage_name") if isinstance(sv, dict) else sv
                for sk, sv in subs.items()
            },
        }

    masks = env.get("agent_action_mask") or {}
    def _mask_nonzero(t) -> list:
        if t is None or not isinstance(t, torch.Tensor):
            return []
        return [int(i) for i in t.detach().cpu().nonzero(as_tuple=False).view(-1).tolist()]

    mask_brief = {
        "human_task": _mask_nonzero((masks.get("human") or {}).get("task_availability_mask")),
        "robot_task": _mask_nonzero((masks.get("robot") or {}).get("task_availability_mask")),
        "machine_task": _mask_nonzero((masks.get("machine") or {}).get("task_availability_mask")),
        "B": _mask_nonzero(masks.get("agent_B_product_selector")),
    }
    c_mask = masks.get("agent_C_process_task_planner")
    if isinstance(c_mask, torch.Tensor) and c_mask.dim() == 2:
        mask_brief["C_rows_nonzero"] = {
            str(i): [int(j) for j in c_mask[i].detach().cpu().nonzero(as_tuple=False).view(-1).tolist()]
            for i in range(min(c_mask.shape[0], n_producing + 1))
            if int(c_mask[i].sum().item()) > 1  # more than just "none"
        }

    humans = {k: _slim_agent(v) for k, v in (env.get("human") or {}).items()}
    robots = {k: _slim_agent(v) for k, v in (env.get("robot") or {}).items()}

    return {
        "meta": {
            "reason": reason,
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_step": int(env.get("time_step", -1)),
            "episode_num": int(env.get("episode_num", -1)),
        },
        "diagnosis": diagnosis,
        "progress": {
            "producing": progress.get("producing"),
            "producing_indexs": progress.get("producing_indexs"),
            "finished": progress.get("finished"),
            "not_started": progress.get("not_started"),
            "next_product": progress.get("next_product"),
            "next_product_index": progress.get("next_product_index"),
        },
        "task_records": {str(k): _task_record_summary(v) for k, v in records.items()},
        "materials": materials_brief,
        "masks": mask_brief,
        "human": humans,
        "robot": robots,
        "gantry_state": {
            "state": gstate.get("state"),
            "ongoing_task_record_index": gstate.get("ongoing_task_record_index"),
            "target_area_id": gstate.get("target_area_id"),
            "target_area_xy": to_jsonable(gstate.get("target_area_xy")),
            "target_joints_position_is_none": [
                v is None for v in (gstate.get("target_joints_position") or [])
            ],
            "target_joints_position": to_jsonable(gstate.get("target_joints_position")),
        },
        "animation": _gantry_animation_dump(gantry_machine),
        "world_xy": _gantry_world_xy(gantry_machine, env),
        "machine_states_brief": {
            name: to_jsonable(m.get("state"))
            for name, m in (env.get("machine") or {}).items()
            if name != "num07_gantry_group"
        },
    }


def dump_freeze_json(
    env: dict,
    gantry_machine=None,
    *,
    reason: str = "manual",
    filename: str | None = None,
) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(env.get("time_step", -1))
    ep = int(env.get("episode_num", -1))
    name = filename or f"freeze_ep{ep}_t{ts}_{reason}.json"
    path = DEBUG_DIR / name
    payload = build_freeze_dump(env, gantry_machine, reason=reason)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also refresh a stable latest pointer.
    latest = DEBUG_DIR / "freeze_latest.json"
    latest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[debug_env_dump] wrote {path} likely={payload.get('diagnosis', {}).get('likely')}")
    return path


def maybe_dump_freeze(env: dict, gantry_machine=None, *, freeze_steps: int = FREEZE_STEPS) -> Path | None:
    """Call once per env step. Returns dump path when a dump was written."""
    global _freeze_fp, _freeze_count, _last_dump_fp

    if REQUEST_DUMP_FLAG.exists():
        try:
            REQUEST_DUMP_FLAG.unlink()
        except OSError:
            pass
        return dump_freeze_json(env, gantry_machine, reason="request")

    fp = ongoing_fingerprint(env)
    records = (env.get("progress") or {}).get("ongoing_task_records") or {}
    n_producing = len((env.get("progress") or {}).get("producing") or [])

    # Empty ongoing but WIP still open → also treat as freeze.
    if not records:
        idle_fp = f"idle_wip:{n_producing}"
        if n_producing <= 0:
            _freeze_fp = idle_fp
            _freeze_count = 0
            return None
        if idle_fp == _freeze_fp:
            _freeze_count += 1
        else:
            _freeze_fp = idle_fp
            _freeze_count = 0
            return None
        if _freeze_count < freeze_steps or idle_fp == _last_dump_fp:
            return None
        _last_dump_fp = idle_fp
        return dump_freeze_json(env, gantry_machine, reason=f"idle{freeze_steps}")

    if fp == _freeze_fp:
        _freeze_count += 1
    else:
        _freeze_fp = fp
        _freeze_count = 0
        return None

    if _freeze_count < freeze_steps:
        return None
    if fp == _last_dump_fp:
        return None
    _last_dump_fp = fp
    return dump_freeze_json(env, gantry_machine, reason=f"freeze{freeze_steps}")
