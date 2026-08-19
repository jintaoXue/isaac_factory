"""Serialize / restore a single env's logic + pose state.

Manager objects keep ``self.state`` as a *reference* into ``env_state_action_dict``.
Restore must rebind those pointers or the next step still mutates the old dict.

``generated_route`` is dropped on capture and left empty on restore; RouteManager
rebuilds it from area_id. Gantry PoseAnimation clocks are reset (not perfectly
replayed); joint poses are written via ``apply_data_to_sim``.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import torch

from .debug_env_dump import ongoing_fingerprint

_LOGIC = ("progress", "human", "robot", "machine", "material", "storage")
_STRIP_LEAF = frozenset({"object", "generated_route"})


def n_finished(env: dict) -> int:
    fin = (env.get("progress") or {}).get("finished") or {}
    if not isinstance(fin, dict):
        return 0
    total = 0
    for v in fin.values():
        total += len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else int(v or 0)
    return total


def wip_cap(progress: dict | None, default: int = 10) -> int:
    """Runtime producing cap (curriculum); encoder dim stays at default=10."""
    if not progress:
        return default
    cap = progress.get("stage_wip_cap")
    return int(cap) if cap is not None else default


def _clone(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items() if k not in _STRIP_LEAF}
    if isinstance(obj, (list, tuple)):
        return [_clone(x) for x in obj]
    return copy.deepcopy(obj)


def progress_key(env: dict) -> str:
    """Decision-equivalent WIP id. Ignores pose / time_step / route_index."""
    progress = env.get("progress") or {}
    slots: list[str] = []
    records = progress.get("ongoing_task_records") or {}
    for pid in sorted(records.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        tr = records[pid] or {}
        sd = tr.get("subtasks_dict") or {}
        finished = tuple(bool(x) for x in (sd.get("finished") or []))
        slots.append(f"{tr.get('product')}:{tr.get('task')}:{sd.get('ongoing_index')}:{finished}")
    not_started = progress.get("not_started") or {}
    leftover = sorted((str(k), int(v or 0)) for k, v in not_started.items()) if isinstance(not_started, dict) else []

    def _busy(group: dict | None) -> list[str]:
        names = []
        for name, ent in sorted((group or {}).items()):
            if isinstance(ent, dict) and ent.get("state", "free") != "free":
                names.append(str(name))
        return names

    machines = []
    for name, ent in sorted((env.get("machine") or {}).items()):
        if not isinstance(ent, dict):
            continue
        states = ent.get("state")
        if isinstance(states, list) and any(s not in ("free", "invalid", None) for s in states):
            machines.append(str(name))

    payload = {
        "nfin": n_finished(env),
        "slots": slots,
        "left": leftover,
        "next": progress.get("next_product"),
        "h": _busy(env.get("human")),
        "r": _busy(env.get("robot")),
        "m": machines,
        "fp": ongoing_fingerprint(env),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def capture(env: dict) -> dict:
    """CPU snapshot suitable for pickle. No prim handles, no routes."""
    art = {}
    for name, data in (env.get("articulations") or {}).items():
        if not isinstance(data, dict):
            continue
        jp = data.get("joint_position")
        art[name] = {"joint_position": _clone(jp)}
    prims = {}
    for name, data in (env.get("rigid_prims") or {}).items():
        if not isinstance(data, dict):
            continue
        prims[name] = {
            "position": _clone(data.get("position")),
            "orientation": _clone(data.get("orientation")),
        }
    return {
        "time_step": int(env.get("time_step", 0) or 0),
        "episode_num": int(env.get("episode_num", 0) or 0),
        "n_finished": n_finished(env),
        "progress_key": progress_key(env),
        **{k: _clone(env.get(k) or {}) for k in _LOGIC},
        "articulations": art,
        "rigid_prims": prims,
    }


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    return obj


def _clear_routes(group: dict | None) -> None:
    for ent in (group or {}).values():
        if isinstance(ent, dict):
            ent["generated_route"] = []
            ent["route_index"] = 0
            ent["route_length"] = 0


def _rebind_group(objects: list, env_group: dict, name_fn) -> None:
    for obj in objects:
        name = name_fn(obj)
        if name in env_group:
            obj.state = env_group[name]


def rebind_managers(single_env) -> None:
    env = single_env.env_state_action_dict
    device = single_env.cuda_device

    _rebind_group(
        single_env.human_manager.human_list,
        env.get("human") or {},
        lambda h: f"num_{h.idx:02d}_{h.type_name}",
    )
    _rebind_group(
        single_env.robot_manager.robot_list,
        env.get("robot") or {},
        lambda r: f"num_{r.idx:02d}_{r.type_name}",
    )
    _rebind_group(
        single_env.product_material_manager.material_batch_list,
        env.get("material") or {},
        lambda m: f"num_{m.idx:02d}_{m.type_name}",
    )
    _rebind_group(
        single_env.storage_manager.storage_list,
        env.get("storage") or {},
        lambda s: f"{s.class_name}_{s.idx:02d}",
    )
    for machine in list(single_env.machine_manager.iter_machines()) + list(
        single_env.machine_manager.iter_logistic_machines()
    ):
        name = machine.type_name
        if name in (env.get("machine") or {}):
            machine.state = env["machine"][name]

    for human in single_env.human_manager.human_list:
        if hasattr(human, "_animation"):
            human._animation.reset()

    # PoseAnimation / GantryGroupAnimation clocks cannot be replayed; snap joints instead.
    for machine in list(single_env.machine_manager.iter_machines()) + list(
        single_env.machine_manager.iter_logistic_machines()
    ):
        infos = getattr(machine, "registration_infos", None) or {}
        for obj_name in infos:
            anim = getattr(machine, f"animation_{obj_name}", None)
            if anim is None:
                continue
            jp = (env.get("articulations") or {}).get(obj_name, {}).get("joint_position")
            if isinstance(jp, torch.Tensor) and hasattr(anim, "initialize"):
                pose = jp.to(device)
                anim.initialize(pose, pose)

    art = env.get("articulations") or {}
    for name, data in art.items():
        if not isinstance(data, dict):
            continue
        obj = None
        for machine in list(single_env.machine_manager.iter_machines()) + list(
            single_env.machine_manager.iter_logistic_machines()
        ):
            cand = getattr(machine, name, None)
            if cand is not None:
                obj = cand
                break
        if obj is not None:
            data["object"] = obj

    prims = env.get("rigid_prims") or {}
    for human in single_env.human_manager.human_list:
        key = f"num_{human.idx:02d}_{human.type_name}"
        if key in prims:
            prims[key]["object"] = human.prim
    for robot in single_env.robot_manager.robot_list:
        key = f"num_{robot.idx:02d}_{robot.type_name}"
        if key in prims:
            prims[key]["object"] = robot.prim
    for mat in single_env.product_material_manager.material_batch_list:
        for obj_name in (mat.meta_registeration_info or {}):
            prim = getattr(mat, obj_name, None)
            if prim is None:
                continue
            # materials register as rigid_prims under various names; keep existing object if present
            for key, data in prims.items():
                if data.get("object") is None and key.endswith(obj_name):
                    data["object"] = prim


def restore(single_env, ckpt: dict) -> dict:
    """Write ``ckpt`` into ``single_env`` and sync sim + masks. Returns the live dict."""
    env = single_env.env_state_action_dict
    device = single_env.cuda_device
    live = _to_device(copy.deepcopy(ckpt), device)

    env["time_step"] = int(live.get("time_step", 0) or 0)
    for key in _LOGIC:
        env[key] = live.get(key) or {}

    _clear_routes(env.get("human"))
    _clear_routes(env.get("robot"))

    for name, snap in (live.get("articulations") or {}).items():
        slot = env.setdefault("articulations", {}).setdefault(name, {})
        if snap.get("joint_position") is not None:
            slot["joint_position"] = snap["joint_position"]
    for name, snap in (live.get("rigid_prims") or {}).items():
        slot = env.setdefault("rigid_prims", {}).setdefault(name, {})
        if snap.get("position") is not None:
            slot["position"] = snap["position"]
        if snap.get("orientation") is not None:
            slot["orientation"] = snap["orientation"]

    rebind_managers(single_env)
    single_env.apply_data_to_sim()
    single_env.algo_hierarchical_masker.generate_agents_mask(env)
    return env


def soft_cosine(pre_a: dict, pre_b: dict) -> float:
    """Cosine on flattened preprocess tensors, skipping time / route progress."""
    skip = ("time_norm", "route_progress", "subtask_time_counter")

    def _flat(pre: dict, acc: list[torch.Tensor]) -> None:
        for k, v in pre.items():
            if k in skip:
                continue
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                acc.append(v.detach().float().reshape(-1).cpu())
            elif isinstance(v, dict):
                _flat(v, acc)

    a, b = [], []
    _flat(pre_a, a)
    _flat(pre_b, b)
    if not a or not b:
        return 0.0
    va, vb = torch.cat(a), torch.cat(b)
    n = min(va.numel(), vb.numel())
    if n <= 0:
        return 0.0
    va, vb = va[:n], vb[:n]
    denom = float(va.norm() * vb.norm())
    if denom <= 1e-8:
        return 0.0
    return float(torch.dot(va, vb) / denom)
