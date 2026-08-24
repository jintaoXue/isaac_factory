"""将 ``env_state_action_dict`` 压成定长 nested tensor，供 replay buffer ``stack``。

约定
----
- ``ongoing_task_records`` → 槽位表，上限 ``max_ongoing``（= parallel_producing_limit）
- ``subtasks`` / 物料状态序列 → pad 到 ``max_subtasks``
- ``generated_route`` → 只保留起点、终点、``route_progress``（不存整条路径）
- human / robot / material 按 ``idx`` 入槽；machine / storage 按名字排序入槽
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# 离线样例（``python -m`` / 冒烟用），与 dump 无关
_SAMPLE_PY = (
    Path(__file__).resolve().parent.parent / "env_asset_cfg" / "cfg_env_state_action_sample.py"
).resolve()


# =============================================================================
# 定长上限（对齐 HcVectorEnvCfg；gallery 最长 subtask=9 → pad 16）
# =============================================================================


@dataclass(frozen=True)
class BufferPreprocessCfg:
    max_ongoing: int = 10  # = HcVectorEnvCfg.single_env_parallel_producing_limit
    max_human: int = 6  # HcVectorEnvCfg.human_number_upper_bound
    max_robot: int = 4  # HcVectorEnvCfg.robot_upper_bound
    max_material: int = 16  # = HcVectorEnvCfg.material_batch_upper_bound
    max_machine: int = 16
    max_storage: int = 32
    max_workstations: int = 4  # gantry 最多 4 工位
    max_subtasks: int = 16
    max_product_types: int = 4
    num_agents: int = 4  # subtask 行: human / gantry / machine / robot
    time_norm: float = 10000.0


CFG = BufferPreprocessCfg()


# =============================================================================
# 字符串 → id（pad=0, unk=1）
# =============================================================================


class Vocab:
    def __init__(self, names: list[str]):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for n in names:
            if n and n not in self.stoi:
                self.stoi[n] = len(self.stoi)

    def __call__(self, s: str | None) -> int:
        if not s:
            return 0
        return self.stoi.get(str(s), 1)


TASK = Vocab(
    [
        "none",
        "logistic_for_pipe_cutting",
        "pipe_cutting",
        "logistic_for_pipe_grooving",
        "pipe_grooving",
        "logistic_for_batch_spot_welding",
        "batch_spot_welding",
        "logistic_for_arc_welding_root",
        "arc_welding_root",
        "logistic_for_MIG_welding_surface",
        "MIG_welding_surface",
        "logistic_for_paint_rust_proof",
        "paint_rust_proof",
    ]
)
SUBTASK = Vocab(
    [
        "go_to_material",
        "material_on_gantry",
        "control_gantry",
        "material_on_robot",
        "go_to_goal_area",
        "material_on_goal_area",
        "go_to_processing_machine",
        "control_machine",
        "wait",
        "done",
        "none",
        "carry_to_robot",
        "go_to_goal_robot",
        "carry_to_goal_area",
        "move_to_goal_area",
        "finding_free_gantry",
        "finding_free_robot",
        "process",
    ]
)
MAT_STATE = Vocab(
    ["on_start_area", "on_gantry", "on_robot", "on_goal_area", "on_machine", "disappear"]
)
TASK_TYPE = Vocab(["logistic", "processing", "none"])
PRODUCT = Vocab(["ProductWaterPipe"])
ENTITY_STATE = Vocab(["free", "partial", "full", "empty", "waiting_processing_task"])
SUBMAT = Vocab(
    [
        "product_00_pipe_raw",
        "product_00_pipe",
        "product_00_flange",
        "product_00_elbow",
        "product_00_semi",
        "product_00_maded",
    ]
)
MACHINE = Vocab(
    [
        "num00_rotaryPipeAutomaticWeldingMachine",
        "num01_weldingRobot",
        "num02_rollerbedCNCPipeIntersectionCuttingMachine",
        "num03_laserCuttingMachine",
        "num04_groovingMachineLarge",
        "num05_groovingMachineSmall",
        "num06_highPressureFoamingMachine",
        "num07_gantry_group",
        "num08_workbench",
    ]
)


# =============================================================================
# 小工具
# =============================================================================


def _f(x: float | int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(float(x), dtype=torch.float32, device=device)


def _i(x: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(int(x), dtype=torch.int64, device=device)


def _z(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device | None) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _as_int(v: Any, default: int = -1) -> int:
    """None / 空 tensor → default；其余转 int。"""
    if v is None:
        return default
    if isinstance(v, torch.Tensor):
        return default if v.numel() == 0 else int(v.item())
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _count(v: Any) -> float:
    """list 用 len；标量数量直接用（样例里 not_started 可能是 int）。"""
    if v is None:
        return 0.0
    if isinstance(v, (list, tuple, set, dict)):
        return float(len(v))
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _entity_idx(key: str, ent: dict | None = None) -> int | None:
    """优先 key_variables.idx，否则从名字里抠数字（num_00_ / num00_）。"""
    if ent and isinstance(ent.get("key_variables"), dict):
        idx = ent["key_variables"].get("idx")
        if idx is not None:
            return int(idx)
    m = re.search(r"(?:num_?|_)(\d+)", key)
    return int(m.group(1)) if m else None


def _state_id(state: str | None) -> int:
    """free 等直接查表；working_X / materialReadyFor_X 用 task id 偏移编码。"""
    if not state:
        return 0
    s = str(state)
    if s in ENTITY_STATE.stoi:
        return ENTITY_STATE(s)
    if s.startswith("working_"):
        return 1000 + TASK(s[len("working_") :])
    if s.startswith("materialReadyFor_"):
        return 2000 + TASK(s[len("materialReadyFor_") :])
    return ENTITY_STATE(s)


def _wp6(wp: dict | None) -> list[float]:
    """航点 → [x, y, qw, qx, qy, qz]。"""
    if not isinstance(wp, dict):
        return [0.0] * 6
    x, y = float(wp.get("x") or 0.0), float(wp.get("y") or 0.0)
    ori = wp.get("orientation")
    if isinstance(ori, torch.Tensor):
        o = ori.detach().float().cpu().flatten().tolist()
    elif isinstance(ori, (list, tuple)):
        o = [float(v) for v in ori]
    else:
        o = []
    o = (o + [0.0] * 4)[:4]
    return [x, y, *o]


def _route(ent: dict) -> tuple[list[float], list[float], float, float, float]:
    """整条 generated_route → (start6, end6, progress, length, has_route)。"""
    route = ent.get("generated_route") or []
    length = int(ent.get("route_length") or 0) or (len(route) if isinstance(route, list) else 0)
    index = int(ent.get("route_index") or 0)
    has = 1.0 if isinstance(route, list) and length > 0 and route else 0.0
    start = _wp6(route[0] if has else None)
    end = _wp6(route[-1] if has else None)
    prog = min(1.0, max(0.0, float(index) / float(max(length, 1)))) if has else 0.0
    return start, end, prog, float(length), has


def _area3(area_ids: dict | None) -> list[int]:
    """start/goal_area_ids → [human, robot, gantry] 各取 list 首元素。"""
    if not isinstance(area_ids, dict):
        return [-1, -1, -1]

    def first(k: str) -> int:
        xs = area_ids.get(k) or []
        return int(xs[0]) if isinstance(xs, (list, tuple)) and xs else -1

    return [
        first("human_working_areas_ids"),
        first("robot_parking_areas_ids"),
        first("gantry_parking_areas_ids"),
    ]


def _clone_tensors(obj: Any, device: torch.device | None) -> Any:
    """递归克隆 action / mask 里已有的定长 tensor。"""
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        return t.cpu() if device is None else t.to(device)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = _clone_tensors(v, device)
            if pv is not None:
                out[str(k)] = pv
        return out
    if isinstance(obj, bool):
        return _i(int(obj), device)
    if isinstance(obj, (int, float)):
        return _f(obj, device)
    return None


# =============================================================================
# 各字段编码器
# =============================================================================


def _encode_ongoing_tasks(
    records: dict, t: int, cfg: BufferPreprocessCfg, device: torch.device | None
) -> dict[str, torch.Tensor]:
    """变长 dict → 定长槽；含 pad 后的 subtask_seq / mat_state_seq。"""
    m, s, a = cfg.max_ongoing, cfg.max_subtasks, cfg.num_agents

    # 标量 / id 槽
    mask = _z((m,), torch.float32, device)
    product_index = _z((m,), torch.int64, device) - 1
    task_id = _z((m,), torch.int64, device)
    task_type_id = _z((m,), torch.int64, device)
    product_id = _z((m,), torch.int64, device)
    human_slot = _z((m,), torch.int64, device) - 1
    robot_slot = _z((m,), torch.int64, device) - 1
    machine_id = _z((m,), torch.int64, device)
    logistic_machine_id = _z((m,), torch.int64, device)
    workstation_i = _z((m,), torch.int64, device) - 1
    gantry_i = _z((m,), torch.int64, device) - 1
    task_done = _z((m,), torch.float32, device)
    is_final = _z((m,), torch.float32, device)
    age_norm = _z((m,), torch.float32, device)
    ongoing_sub_ids = _z((m, a), torch.int64, device)
    ongoing_index = _z((m,), torch.float32, device)
    finished_flags = _z((m, a), torch.float32, device)
    num_subtasks_n = _z((m,), torch.float32, device)
    logistic_submat_id = _z((m,), torch.int64, device)
    start_area_ids = _z((m, 3), torch.int64, device) - 1
    goal_area_ids = _z((m, 3), torch.int64, device) - 1
    next_task_id = _z((m,), torch.int64, device)
    next_logistic_id = _z((m,), torch.int64, device)
    next_machine_id = _z((m,), torch.int64, device)
    next_workstation_i = _z((m,), torch.int64, device) - 1
    # 整段 plan
    subtask_seq = _z((m, s, a), torch.int64, device)
    subtask_mask = _z((m, s), torch.float32, device)
    mat_state_seq = _z((m, s), torch.int64, device)

    items = []
    for k, rec in (records or {}).items():
        if isinstance(rec, dict):
            items.append((_as_int(rec.get("product_index"), _as_int(k)), rec))
    items.sort(key=lambda x: x[0])

    for slot, (pid, rec) in enumerate(items[:m]):
        mask[slot] = 1.0
        product_index[slot] = pid
        task_id[slot] = TASK(rec.get("task"))
        task_type_id[slot] = TASK_TYPE(rec.get("task_type"))
        product_id[slot] = PRODUCT(rec.get("product"))
        human_slot[slot] = _as_int(rec.get("human_index"))
        robot_slot[slot] = _as_int(rec.get("robot_index"))
        machine_id[slot] = MACHINE(rec.get("target_machine"))
        logistic_machine_id[slot] = MACHINE(rec.get("logistic_machine"))
        workstation_i[slot] = _as_int(rec.get("chosen_workstation_index"))
        gantry_i[slot] = _as_int(rec.get("chosen_gantry_index"))
        task_done[slot] = float(bool(rec.get("task_done")))
        is_final[slot] = float(bool(rec.get("is_final_task")))
        age_norm[slot] = (t - _as_int(rec.get("task_start_time_step"), t)) / cfg.time_norm
        logistic_submat_id[slot] = SUBMAT(rec.get("logistic_submaterial"))
        next_task_id[slot] = TASK(rec.get("next_processing_task"))
        next_logistic_id[slot] = TASK(rec.get("next_logistic_task"))
        next_machine_id[slot] = MACHINE(rec.get("next_target_machine"))
        next_workstation_i[slot] = _as_int(rec.get("next_chosen_workstation_index"))

        sd = rec.get("subtasks_dict")
        if not isinstance(sd, dict):
            continue

        # 当前步（4 角色）
        ongoing = sd.get("ongoing") or []
        for j in range(min(a, len(ongoing))):
            ongoing_sub_ids[slot, j] = SUBTASK(ongoing[j])
        finished = sd.get("finished") or []
        for j in range(min(a, len(finished))):
            finished_flags[slot, j] = float(bool(finished[j]))
        nsub = _as_int(sd.get("num_subtasks"), len(sd.get("subtasks") or []))
        ongoing_index[slot] = _as_int(sd.get("ongoing_index"), 0) / float(max(s, 1))
        num_subtasks_n[slot] = nsub / float(max(s, 1))
        start_area_ids[slot] = torch.tensor(_area3(sd.get("start_area_ids")), dtype=torch.int64, device=device)
        goal_area_ids[slot] = torch.tensor(_area3(sd.get("goal_area_ids")), dtype=torch.int64, device=device)

        # pad 整段 subtask plan
        for ti, row in enumerate((sd.get("subtasks") or [])[:s]):
            subtask_mask[slot, ti] = 1.0
            if isinstance(row, (list, tuple)):
                for j in range(min(a, len(row))):
                    subtask_seq[slot, ti, j] = SUBTASK(row[j])

        # 当前 logistic 物料的状态序列（无则取 dict 第一条）
        mats = sd.get("material_states_in_subtasks") or {}
        key = rec.get("logistic_submaterial")
        seq = mats.get(key) if isinstance(mats, dict) and key in mats else None
        if seq is None and isinstance(mats, dict) and mats:
            seq = next(iter(mats.values()))
        if isinstance(seq, (list, tuple)):
            for ti, st in enumerate(seq[:s]):
                mat_state_seq[slot, ti] = MAT_STATE(st)

    return {
        "mask": mask,
        "product_index": product_index,
        "task_id": task_id,
        "task_type_id": task_type_id,
        "product_id": product_id,
        "human_slot": human_slot,
        "robot_slot": robot_slot,
        "machine_id": machine_id,
        "logistic_machine_id": logistic_machine_id,
        "workstation_i": workstation_i,
        "gantry_i": gantry_i,
        "task_done": task_done,
        "is_final": is_final,
        "age_norm": age_norm,
        "ongoing_sub_ids": ongoing_sub_ids,
        "ongoing_index": ongoing_index,
        "finished_flags": finished_flags,
        "num_subtasks_n": num_subtasks_n,
        "logistic_submat_id": logistic_submat_id,
        "start_area_ids": start_area_ids,
        "goal_area_ids": goal_area_ids,
        "next_task_id": next_task_id,
        "next_logistic_id": next_logistic_id,
        "next_machine_id": next_machine_id,
        "next_workstation_i": next_workstation_i,
        "subtask_seq": subtask_seq,
        "subtask_mask": subtask_mask,
        "mat_state_seq": mat_state_seq,
    }


def _encode_movers(
    group: dict, max_n: int, device: torch.device | None, *, with_subtask_time: bool = False
) -> dict[str, torch.Tensor]:
    """human / robot：按 idx 入槽；route → start/end/progress。"""
    mask = _z((max_n,), torch.float32, device)
    state_id = _z((max_n,), torch.int64, device)
    task_rec = _z((max_n,), torch.int64, device) - 1
    cur_area = _z((max_n,), torch.int64, device) - 1
    tgt_area = _z((max_n,), torch.int64, device) - 1
    route_start = _z((max_n, 6), torch.float32, device)
    route_end = _z((max_n, 6), torch.float32, device)
    route_progress = _z((max_n,), torch.float32, device)
    route_length = _z((max_n,), torch.float32, device)
    has_route = _z((max_n,), torch.float32, device)
    detour = _z((max_n,), torch.float32, device)
    yield_ = _z((max_n,), torch.float32, device)
    sub_t = _z((max_n,), torch.float32, device) if with_subtask_time else None
    fatigue = _z((max_n,), torch.float32, device) if with_subtask_time else None
    efficiency = _z((max_n,), torch.float32, device) if with_subtask_time else None

    for key, ent in (group or {}).items():
        if not isinstance(ent, dict):
            continue
        idx = _entity_idx(str(key), ent)
        if idx is None or not (0 <= idx < max_n):
            continue
        mask[idx] = 1.0
        state_id[idx] = _state_id(ent.get("state"))
        task_rec[idx] = _as_int(ent.get("ongoing_task_record_index"))
        cur_area[idx] = _as_int(ent.get("current_area_id"))
        tgt_area[idx] = _as_int(ent.get("target_area_id"))
        start, end, prog, length, has = _route(ent)
        route_start[idx] = torch.tensor(start, dtype=torch.float32, device=device)
        route_end[idx] = torch.tensor(end, dtype=torch.float32, device=device)
        route_progress[idx], route_length[idx], has_route[idx] = prog, length, has
        detour[idx] = float(bool(ent.get("detour_active")))
        yield_[idx] = float(bool(ent.get("yield_active")))
        if sub_t is not None:
            sub_t[idx] = float(_as_int(ent.get("subtask_time_counter"), 0))
            # human-factors state (added for Hier4TPA human fatigue / efficiency)
            fatigue[idx] = float(ent.get("fatigue", 0.0) or 0.0)
            efficiency[idx] = float(ent.get("efficiency", 1.0) or 1.0)

    out = {
        "mask": mask,
        "state_id": state_id,
        "ongoing_task_record_index": task_rec,
        "current_area_id": cur_area,
        "target_area_id": tgt_area,
        "route_start": route_start,
        "route_end": route_end,
        "route_progress": route_progress,
        "route_length": route_length,
        "has_route": has_route,
        "detour_active": detour,
        "yield_active": yield_,
    }
    if sub_t is not None:
        out["subtask_time_counter"] = sub_t
        out["fatigue"] = fatigue
        out["efficiency"] = efficiency
    return out


def _encode_machines(group: dict, cfg: BufferPreprocessCfg, device: torch.device | None) -> dict[str, torch.Tensor]:
    """machine：名字排序入槽；工位维 pad 到 max_workstations。"""
    n, w = cfg.max_machine, cfg.max_workstations
    mask = _z((n,), torch.float32, device)
    machine_id = _z((n,), torch.int64, device)
    ws_mask = _z((n, w), torch.float32, device)
    state_id = _z((n, w), torch.int64, device)
    proc_time = _z((n, w), torch.float32, device)
    task_rec = _z((n, w), torch.int64, device) - 1

    for slot, key in enumerate(sorted(group or {})[:n]):
        ent = group[key]
        if not isinstance(ent, dict):
            continue
        mask[slot] = 1.0
        machine_id[slot] = MACHINE(key)
        states, times, idxs = ent.get("state") or [], ent.get("processing_time_step") or [], ent.get(
            "ongoing_task_record_index"
        ) or []
        for j in range(min(w, max(len(states), len(times), len(idxs)))):
            ws_mask[slot, j] = 1.0
            if j < len(states):
                state_id[slot, j] = _state_id(states[j])
            if j < len(times):
                proc_time[slot, j] = float(_as_int(times[j], 0))
            if j < len(idxs):
                task_rec[slot, j] = _as_int(idxs[j])

    return {
        "mask": mask,
        "machine_id": machine_id,
        "workstation_mask": ws_mask,
        "state_id": state_id,
        "processing_time_step": proc_time,
        "ongoing_task_record_index": task_rec,
    }


def _encode_materials(group: dict, cfg: BufferPreprocessCfg, device: torch.device | None) -> dict[str, torch.Tensor]:
    n = cfg.max_material
    mask = _z((n,), torch.float32, device)
    product_id = _z((n,), torch.int64, device)
    finished_task_id = _z((n,), torch.int64, device)
    task_rec = _z((n,), torch.int64, device) - 1

    for key, ent in (group or {}).items():
        if not isinstance(ent, dict):
            continue
        idx = _entity_idx(str(key), ent)
        if idx is None or not (0 <= idx < n):
            continue
        mask[idx] = 1.0
        kv = ent.get("key_variables") if isinstance(ent.get("key_variables"), dict) else {}
        product_id[idx] = PRODUCT(kv.get("type_name"))
        finished_task_id[idx] = TASK(ent.get("finished_task"))
        task_rec[idx] = _as_int(ent.get("ongoing_task_record_index"))

    return {
        "mask": mask,
        "product_id": product_id,
        "finished_task_id": finished_task_id,
        "ongoing_task_record_index": task_rec,
    }


def _encode_storage(group: dict, cfg: BufferPreprocessCfg, device: torch.device | None) -> dict[str, torch.Tensor]:
    n, b = cfg.max_storage, cfg.max_material
    mask = _z((n,), torch.float32, device)
    state_id = _z((n,), torch.int64, device)
    num_material = _z((n,), torch.float32, device)
    material_type_id = _z((n,), torch.int64, device)
    idx_list = _z((n, b), torch.int64, device) - 1
    idx_mask = _z((n, b), torch.float32, device)

    for slot, key in enumerate(sorted(group or {})[:n]):
        ent = group[key]
        if not isinstance(ent, dict):
            continue
        mask[slot] = 1.0
        state_id[slot] = _state_id(ent.get("state"))
        num_material[slot] = float(_as_int(ent.get("num_material"), 0))
        material_type_id[slot] = SUBMAT(ent.get("material_type"))
        for j, v in enumerate((ent.get("material_idx_list") or [])[:b]):
            idx_list[slot, j] = _as_int(v)
            idx_mask[slot, j] = 1.0

    return {
        "mask": mask,
        "state_id": state_id,
        "num_material": num_material,
        "material_type_id": material_type_id,
        "material_idx_list": idx_list,
        "material_idx_mask": idx_mask,
    }


def _encode_progress(
    progress: dict, t: int, cfg: BufferPreprocessCfg, device: torch.device | None
) -> dict:
    finished = progress.get("finished") or {}
    not_started = progress.get("not_started") or {}
    product_order = progress.get("product_order") or {}
    producing = progress.get("producing") or []
    records = progress.get("ongoing_task_records") or {}

    # [producing_n, finished_n, not_started_n, ongoing_n, has_next, done, t_norm]
    scalars = torch.tensor(
        [
            _count(producing),
            sum(_count(v) for v in finished.values()) if isinstance(finished, dict) else 0.0,
            sum(_count(v) for v in not_started.values()) if isinstance(not_started, dict) else 0.0,
            float(len(records)),
            float(progress.get("next_product") is not None),
            float(bool(progress.get("production_done"))),
            float(t) / cfg.time_norm,
        ],
        dtype=torch.float32,
        device=device,
    )

    p = cfg.max_product_types
    type_mask = _z((p,), torch.float32, device)
    type_ids = _z((p,), torch.int64, device)
    order_qty = _z((p,), torch.float32, device)
    not_started_n = _z((p,), torch.float32, device)
    finished_n = _z((p,), torch.float32, device)
    for slot, name in enumerate(
        sorted(set(list(product_order) + list(not_started) + list(finished)))[:p]
    ):
        type_mask[slot] = 1.0
        type_ids[slot] = PRODUCT(name)
        order_qty[slot] = float(product_order.get(name, 0) or 0)
        not_started_n[slot] = _count(not_started.get(name))
        finished_n[slot] = _count(finished.get(name))

    prod_idx = _z((cfg.max_ongoing,), torch.int64, device) - 1
    prod_mask = _z((cfg.max_ongoing,), torch.float32, device)
    for i, v in enumerate(list(progress.get("producing_indexs") or [])[: cfg.max_ongoing]):
        prod_idx[i] = _as_int(v)
        prod_mask[i] = 1.0

    return {
        "scalars": scalars,
        "next_product_id": _i(PRODUCT(progress.get("next_product")), device),
        "next_product_index": _i(_as_int(progress.get("next_product_index")), device),
        "product_type_mask": type_mask,
        "product_type_ids": type_ids,
        "product_order_qty": order_qty,
        "not_started_n": not_started_n,
        "finished_n": finished_n,
        "producing_indexs": prod_idx,
        "producing_indexs_mask": prod_mask,
        "ongoing_tasks": _encode_ongoing_tasks(records, t, cfg, device),
    }


# =============================================================================
# 对外 API
# =============================================================================


def _encode_rl(rl: dict, device: torch.device | None) -> dict[str, torch.Tensor]:
    """``env['rl']`` → 定长标量（reward/done/truncated/success）。"""
    return {
        "reward": _f(float(rl.get("reward", 0.0) or 0.0), device),
        "done": _f(float(bool(rl.get("done"))), device),
        "truncated": _f(float(bool(rl.get("truncated"))), device),
        "success": _f(float(bool(rl.get("success"))), device),
    }


def preprocess_for_buffer(
    env: dict,
    device: torch.device | None = None,
    cfg: BufferPreprocessCfg | None = None,
) -> dict:
    """单步 env dict → 定长 nested CPU/device tensors（可直接 ``stack_batch``）。"""
    cfg = cfg or CFG
    t = int(env.get("time_step", 0) or 0)
    out: dict[str, Any] = {
        "time_step": _f(t, device),
        "episode_num": _f(int(env.get("episode_num", 0) or 0), device),
        "progress": _encode_progress(env.get("progress") or {}, t, cfg, device),
        "human": _encode_movers(env.get("human") or {}, cfg.max_human, device, with_subtask_time=True),
        "robot": _encode_movers(env.get("robot") or {}, cfg.max_robot, device),
        "machine": _encode_machines(env.get("machine") or {}, cfg, device),
        "material": _encode_materials(env.get("material") or {}, cfg, device),
        "storage": _encode_storage(env.get("storage") or {}, cfg, device),
        "rl": _encode_rl(env.get("rl") or {}, device),
    }
    if "agent_action_mask" in env:
        out["agent_action_mask"] = _clone_tensors(env["agent_action_mask"], device)
    if "action" in env:
        out["action"] = _clone_tensors(env["action"], device)
    return out


def stack_batch(batch: list[dict]) -> dict:
    """list[preprocess 结果] → 最前维 batch。要求各样本 leaf shape 一致。"""
    if not batch:
        return {}
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch if k in b]
        if not vals:
            continue
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(v0, dict):
            out[k] = stack_batch(vals)
        else:
            out[k] = vals
    return out


# =============================================================================
# 冒烟：python data.py
# =============================================================================

if __name__ == "__main__":
    import importlib.util

    assert _SAMPLE_PY.is_file(), f"missing sample {_SAMPLE_PY}"
    spec = importlib.util.spec_from_file_location("_s", _SAMPLE_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    pre = preprocess_for_buffer(mod.EnvStateActionSample)
    b = stack_batch([pre, pre])
    ot = pre["progress"]["ongoing_tasks"]
    h = pre["human"]
    print("batch time_step", tuple(b["time_step"].shape))
    print("ongoing mask", ot["mask"].tolist(), "subtasks", int(ot["subtask_mask"].sum().item()))
    print("human0 progress", float(h["route_progress"][0]), "start_xy", h["route_start"][0, :2].tolist())
