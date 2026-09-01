"""Disturbance injector: L2 runtime events + helpers used by managers."""

from __future__ import annotations

import random
from typing import Any

from ..env_asset_cfg.cfg_disturbance import (
    RuntimeDisturbanceCfg,
    active_dims,
    episode_l2_schedule,
    episode_qc_holds,
    QC_HOLD_TASKS,
)


class DisturbanceInjector:
    """Per-env runtime disturbance (machine DOWN / human absent / gantry down).

    L0/L1 config mutations are applied once via ``apply_disturbance_to_cfgs`` before
    env construction. L2 queues are resampled each episode (target / start /
    duration), then activated one at a time when the target is idle.
    """

    def __init__(self, env_id: int, collector: Any | None = None):
        self.env_id = env_id
        self.collector = collector
        self._active = False
        self._pending = False
        self._event_done = False
        self._saved_state: dict[str, Any] | None = None
        self._event_id = 0
        self._activated_at = -1
        self._duration = 0
        self._logged_resource_id = ""
        self._queue: list[dict[str, Any]] = []
        self._q_i = 0
        self._pending_since = -1
        self._current_target: str | None = None
        self._current_dim: str | None = None
        self._current_max_units: int | None = None

    def reset(self, env: dict) -> None:
        self._restore_if_needed(env)
        self._active = False
        self._pending = False
        self._event_done = False
        self._saved_state = None
        self._event_id = 0
        self._activated_at = -1
        self._duration = 0
        self._logged_resource_id = ""
        self._q_i = 0
        self._pending_since = -1
        self._current_target = None
        self._current_dim = None
        self._current_max_units = None
        self._queue = self._load_schedule()

        dims = active_dims()
        if dims == ["none"]:
            return
        dim = str(RuntimeDisturbanceCfg.get("dim") or "+".join(dims))

        if self._queue:
            ep = getattr(self.collector, "episode_id", "?")
            brief = [(e.get("start"), e.get("duration"), e.get("target")) for e in self._queue]
            print(f"[Disturbance] episode={ep} env={self.env_id} L2 n={len(self._queue)} {brief}")
        qc = RuntimeDisturbanceCfg.get("qc_holds") or []
        if qc:
            qbrief = [(e.get("start"), e.get("hold_steps"), e.get("target")) for e in qc]
            print(f"[Disturbance] episode={getattr(self.collector, 'episode_id', '?')} QC n={len(qc)} {qbrief}")

        # Episode-level config disturbance record (L0/L1 + this episode's L2 queue).
        if self.collector is not None:
            applied = dict(RuntimeDisturbanceCfg.get("applied") or {})
            applied["event_schedule"] = [dict(e) for e in self._queue]
            applied["qc_holds"] = list(RuntimeDisturbanceCfg.get("qc_holds") or [])
            applied["event_schedule_mode"] = RuntimeDisturbanceCfg.get(
                "event_schedule_mode", "resample_per_episode"
            )
            applied["episode_id"] = getattr(self.collector, "episode_id", "")
            self.collector.log_disturbance(
                {
                    "disturbance_id": f"{dim}_cfg",
                    "disturbance_type": f"{dim}_config",
                    "target_resource_id": "episode",
                    "target_resource_type": dim,
                    "start_time_step": 0,
                    "end_time_step": "",
                    "intensity": RuntimeDisturbanceCfg.get("intensity", 1.0),
                    "parameter_before": "nominal",
                    "parameter_after": str(applied),
                    "notes": f"primary_dimension={dim} n_l2={len(self._queue)}",
                }
            )
            self._log_qc_windows()

    def _load_schedule(self) -> list[dict[str, Any]]:
        dims = active_dims()
        dim = str(RuntimeDisturbanceCfg.get("dim") or "+".join(dims))
        RuntimeDisturbanceCfg["qc_holds"] = []
        if dims == ["none"]:
            return []
        mode = str(RuntimeDisturbanceCfg.get("event_schedule_mode") or "resample_per_episode")
        raw = RuntimeDisturbanceCfg.get("event_schedule") or []
        if mode == "fixed" and raw:
            return [dict(item) for item in raw if int(item.get("start", -1)) >= 0]
        if mode == "none":
            return []
        from .bottleneck_data import BottleneckRunContext

        episode_id = int(getattr(self.collector, "episode_id", 0) or 0)
        seed = int(getattr(BottleneckRunContext, "seed", 0) or 0)
        intensity = float(RuntimeDisturbanceCfg.get("intensity", 1.0) or 1.0)
        if "machine" in dims:
            RuntimeDisturbanceCfg["qc_holds"] = episode_qc_holds(
                intensity, seed, int(self.env_id), episode_id
            )
        return episode_l2_schedule(dim, intensity, seed, int(self.env_id), episode_id)

    def step(self, env: dict) -> None:
        dim = str(RuntimeDisturbanceCfg.get("dim", "none"))
        if dim == "none" or not self._queue:
            return

        t = int(env.get("time_step", 0))

        if self._active and self._activated_at >= 0 and t >= self._activated_at + self._duration:
            end = self._activated_at + self._duration
            rid = self._logged_resource_id or self._feature_resource_id(self._current_target)
            log_dim = self._current_dim or dim
            self._restore_if_needed(env)
            self._log_event(env, log_dim, rid, self._activated_at, end, starting=False)
            self._active = False
            self._saved_state = None
            self._activated_at = -1
            self._logged_resource_id = ""
            self._q_i += 1
            self._pending = False
            self._pending_since = -1

        if self._active:
            return
        if self._q_i >= len(self._queue):
            self._event_done = True
            return

        ev = self._queue[self._q_i]
        ev_dim = str(ev.get("dim") or dim)
        ev_target = ev.get("target")
        ev_start = int(ev.get("start", -1))
        ev_dur = int(ev.get("duration", 0))
        if ev_start < 0 or ev_dur <= 0 or t < ev_start:
            return

        if not self._pending:
            self._pending = True
            self._pending_since = t
            self._duration = ev_dur
            self._current_target = ev_target
            self._current_dim = ev_dim
            max_units = ev.get("max_units")
            self._current_max_units = int(max_units) if max_units not in (None, "") else None

        wait_limit = max(ev_dur * 2, 500)
        if self._pending_since >= 0 and t - self._pending_since > wait_limit:
            self._pending = False
            self._pending_since = -1
            self._q_i += 1
            return

        ok = self._activate(env, ev_dim, ev_target)
        if ok:
            self._active = True
            self._pending = False
            self._activated_at = t
            self._event_id += 1
            end = t + self._duration
            self._logged_resource_id = self._feature_resource_id(ev_target)
            self._log_event(env, ev_dim, self._logged_resource_id, t, end, starting=True)

    def _feature_resource_id(self, fallback: str | None) -> str:
        """IDs used by resource_event_log / window_feature_table (workstation-level)."""
        s = self._saved_state or {}
        kind = s.get("kind")
        if kind == "machine":
            return f"{s['machine']}_ws{s['ws']}"
        if kind == "gantry":
            return f"gantry_{s['idx']}"
        if kind == "agv":
            idx = s.get("idx")
            if idx is not None:
                return f"robot_{idx}"
        if kind == "human":
            idx = s.get("idx")
            if idx is not None:
                return f"human_{idx}"
        if kind == "material":
            mtype = s.get("material_type") or fallback or ""
            return f"material_{mtype}" if mtype else (fallback or "")
        return fallback or ""

    def _log_event(
        self,
        env: dict,
        dim: str,
        resource_id: str,
        start: int,
        end: int,
        *,
        starting: bool,
    ) -> None:
        if self.collector is None:
            return
        self.collector.log_disturbance(
            {
                "disturbance_id": f"{dim}_event_{self._event_id}",
                "disturbance_type": {
                    "machine": "machine_failure",
                    "human": "human_unavailable",
                    "logistics": "transport_delay",
                    "material": "material_shortage",
                }.get(dim, dim),
                "target_resource_id": resource_id,
                "target_resource_type": dim,
                "start_time_step": start,
                "end_time_step": "" if starting else end,
                "intensity": RuntimeDisturbanceCfg.get("intensity", 1.0),
                "parameter_before": "nominal" if starting else "disturbed",
                "parameter_after": "disturbed" if starting else "restored",
                "notes": f"event {'start' if starting else 'end'} at step {env.get('time_step')}",
            }
        )

    def _log_qc_windows(self) -> None:
        if self.collector is None:
            return
        for i, h in enumerate(RuntimeDisturbanceCfg.get("qc_holds") or [], start=1):
            start = int(h.get("start", 0))
            end = start + int(h.get("duration", 0))
            target = str(h.get("target") or "")
            self.collector.log_disturbance(
                {
                    "disturbance_id": f"qc_hold_{i}",
                    "disturbance_type": "quality_hold",
                    "target_resource_id": f"{target}_ws0" if target else "episode",
                    "target_resource_type": "machine",
                    "start_time_step": start,
                    "end_time_step": end,
                    "intensity": RuntimeDisturbanceCfg.get("intensity", 1.0),
                    "parameter_before": "nominal",
                    "parameter_after": f"hold_steps={h.get('hold_steps')}",
                    "notes": f"qc window {start}-{end} hold={h.get('hold_steps')} target={target}",
                }
            )

    def _activate(self, env: dict, dim: str, target: str | None) -> bool:
        if dim == "machine":
            return self._activate_machine_down(env, target)
        if dim == "human":
            return self._activate_human_absent(env, target)
        if dim == "logistics":
            if str(target or "").startswith("agv_") or str(target or "").startswith("robot_"):
                return self._activate_agv_down(env, target)
            return self._activate_gantry_down(env, target)
        if dim == "material":
            return self._activate_material_shortage(env, target)
        return False

    def _activate_machine_down(self, env: dict, target: str | None) -> bool:
        """Only mark a truly idle workstation DOWN; never interrupt an ongoing task."""
        machine_name = target or "num02_rollerbedCNCPipeIntersectionCuttingMachine"
        machines = env.get("machine", {})
        if machine_name not in machines:
            return False
        m = machines[machine_name]
        state = m["state"]
        ongoing = m.get("ongoing_task_record_index", [None] * len(state))
        for i, s in enumerate(state):
            if s == "free" and (i >= len(ongoing) or ongoing[i] is None):
                self._saved_state = {
                    "kind": "machine",
                    "machine": machine_name,
                    "ws": i,
                    "prev": s,
                }
                state[i] = "invalid"
                return True
        return False

    def _activate_human_absent(self, env: dict, target: str | None) -> bool:
        humans = env.get("human", {})
        if not humans:
            return False
        idx = 0
        if target and target.startswith("human_"):
            try:
                idx = int(target.split("_", 1)[1])
            except ValueError:
                idx = 0
        preferred = f"num_{idx:02d}_NormalHuman"
        candidates = []
        if preferred in humans:
            candidates.append(preferred)
        candidates.extend(k for k in humans if k != preferred)

        for key in candidates:
            h = humans[key]
            if h.get("state") != "free" or h.get("ongoing_task_record_index") is not None:
                continue
            self._saved_state = {
                "kind": "human",
                "key": key,
                "idx": h.get("key_variables", {}).get("idx", idx),
                "prev_state": h.get("state"),
            }
            # Non-free + no task → idle animation, excluded from allocator mask.
            h["state"] = "working_disturbance_absent"
            return True
        return False

    def _activate_gantry_down(self, env: dict, target: str | None) -> bool:
        """Only disable an idle active gantry."""
        gantry = env.get("machine", {}).get("num07_gantry_group")
        if gantry is None:
            return False
        state = gantry["state"]
        ongoing = gantry.get("ongoing_task_record_index", [None] * len(state))
        preferred = 0
        if target and target.startswith("gantry_"):
            try:
                preferred = int(target.split("_", 1)[1])
            except ValueError:
                preferred = 0

        from ..env_asset_cfg.cfg_machine import CfgMachine

        active = list(CfgMachine["num07_gantry_group"].get("active_gantry_indices", range(len(state))))
        order = [preferred] + [i for i in active if i != preferred]
        for idx in order:
            if idx < 0 or idx >= len(state):
                continue
            if state[idx] != "free":
                continue
            if idx < len(ongoing) and ongoing[idx] is not None:
                continue
            self._saved_state = {
                "kind": "gantry",
                "idx": idx,
                "prev": state[idx],
            }
            state[idx] = "invalid"
            return True
        return False

    def _activate_agv_down(self, env: dict, target: str | None) -> bool:
        """Only disable an idle AGV (same idea as human leave)."""
        robots = env.get("robot") or {}
        if not robots:
            return False
        idx = 0
        raw = str(target or "")
        if raw.startswith("agv_") or raw.startswith("robot_"):
            try:
                idx = int(raw.split("_", 1)[1])
            except ValueError:
                idx = 0
        preferred = f"num_{idx:02d}_AGV"
        candidates = []
        if preferred in robots:
            candidates.append(preferred)
        candidates.extend(k for k in robots if k != preferred)
        for key in candidates:
            r = robots[key]
            if r.get("state") != "free" or r.get("ongoing_task_record_index") is not None:
                continue
            kv = r.get("key_variables") or {}
            self._saved_state = {
                "kind": "agv",
                "key": key,
                "idx": kv.get("idx", idx),
                "prev_state": r.get("state"),
            }
            r["state"] = "working_disturbance_absent"
            return True
        return False

    def _activate_material_shortage(self, env: dict, target: str | None) -> bool:
        """Hide idle warehouse flange or elbow; restore later.

        L1 hides one kit SKU once pipes are heading to kitting. The workbench
        stays ``materialReadyFor_batch_spot_welding`` until restore (weld
        starves, grooving cannot discharge). L2 pulses the other kit SKU.
        Never yank parts already in process.
        """
        from .material import _release_storage_slot

        material_type = target or "product_00_flange"
        if material_type.startswith("material_"):
            material_type = material_type[len("material_") :]
        if material_type == "kit":
            types = ["product_00_pipe_raw", "product_00_flange", "product_00_elbow"]
        else:
            types = [material_type]

        candidates: list[dict[str, Any]] = []
        for mat_key, ms in (env.get("material") or {}).items():
            kv = ms.get("key_variables") or {}
            idx = kv.get("idx")
            if idx is None:
                try:
                    idx = int(str(mat_key).split("_")[1])
                except (IndexError, ValueError):
                    continue
            for sku in types:
                sub = (ms.get("submaterials") or {}).get(sku)
                if not sub:
                    continue
                loc = sub.get("storage_name")
                if not loc or "Storage_" not in str(loc):
                    continue
                candidates.append(
                    {
                        "mat_key": mat_key,
                        "idx": int(idx),
                        "material_type": sku,
                        "prim_key": f"num_{int(idx):02d}_{sku}",
                        "loc": loc,
                    }
                )
        if not candidates:
            return False
        random.shuffle(candidates)
        limit = self._current_max_units
        if limit is not None:
            candidates = candidates[: max(0, int(limit))]
        if not candidates:
            return False

        hidden: list[dict[str, Any]] = []
        for item in candidates:
            storage = (env.get("storage") or {}).get(item["loc"])
            ms = (env.get("material") or {}).get(item["mat_key"])
            sub = (ms.get("submaterials") or {}).get(item["material_type"]) if ms else None
            if sub is None:
                continue
            if storage is not None:
                _release_storage_slot(storage, item["idx"])
            sub["storage_name"] = "disappear"
            rp = (env.get("rigid_prims") or {}).get(item["prim_key"])
            if rp is not None and rp.get("position") is not None:
                rp["position"][0][2] = -100
            hidden.append(
                {
                    "mat_key": item["mat_key"],
                    "idx": item["idx"],
                    "material_type": item["material_type"],
                    "prim_key": item["prim_key"],
                }
            )
        if not hidden:
            return False
        self._saved_state = {
            "kind": "material",
            "material_type": material_type,
            "hidden": hidden,
        }
        return True

    def _restore_if_needed(self, env: dict) -> None:
        if not self._saved_state:
            return
        kind = self._saved_state["kind"]
        if kind == "machine":
            m = env.get("machine", {}).get(self._saved_state["machine"])
            if m is not None:
                # Only restore if still in our DOWN marker (avoid clobbering a new assignment).
                ws = self._saved_state["ws"]
                if m["state"][ws] == "invalid":
                    m["state"][ws] = "free"
        elif kind == "human":
            h = env.get("human", {}).get(self._saved_state["key"])
            if h is not None and h.get("ongoing_task_record_index") is None:
                if h.get("state") == "working_disturbance_absent":
                    h["state"] = "free"
        elif kind == "gantry":
            g = env.get("machine", {}).get("num07_gantry_group")
            if g is not None:
                idx = self._saved_state["idx"]
                if g["state"][idx] == "invalid":
                    g["state"][idx] = "free"
        elif kind == "agv":
            r = env.get("robot", {}).get(self._saved_state["key"])
            if r is not None and r.get("ongoing_task_record_index") is None:
                if r.get("state") == "working_disturbance_absent":
                    r["state"] = "free"
        elif kind == "material":
            self._restore_material_shortage(env)
        self._saved_state = None

    def _restore_material_shortage(self, env: dict) -> None:
        """Put hidden SKU back into free warehouse slots. Skip pieces already reset."""
        from .material import find_free_storage, reserve_storage_slot, storage_slot_pose

        storages = env.get("storage") or {}
        materials = env.get("material") or {}
        for item in self._saved_state.get("hidden") or []:
            sku = item.get("material_type") or self._saved_state.get("material_type")
            if not sku or sku == "kit":
                continue
            ms = materials.get(item["mat_key"])
            if not ms:
                continue
            sub = (ms.get("submaterials") or {}).get(sku)
            if not sub or sub.get("storage_name") != "disappear":
                continue
            try:
                storage_name = find_free_storage(storages, sku)
            except ValueError:
                continue
            storage = storages[storage_name]
            slot_idx = reserve_storage_slot(storage, sku, item["idx"])
            sub["storage_name"] = storage_name
            prim_key = item.get("prim_key") or f"num_{item['idx']:02d}_{sku}"
            rp = (env.get("rigid_prims") or {}).get(prim_key)
            if rp is None:
                continue
            try:
                pos, ori = storage_slot_pose(storage, slot_idx)
            except (IndexError, ValueError):
                continue
            rp["position"] = pos
            rp["orientation"] = ori


def should_skip_material_placement(batch_idx: int, material_type: str) -> bool:
    """Kept for call sites. Shortage is timed hide+restore, never a permanent skip."""
    del batch_idx, material_type
    return False


def sample_machine_process_time(
    base: float,
    per_machine_std: float = 0.0,
    time_step: int | None = None,
) -> int:
    from .utils import sample_noisy_steps

    std = float(RuntimeDisturbanceCfg.get("machine_process_noise_std", 0.0) or 0.0)
    std = max(std, float(per_machine_std or 0.0))
    wear = float(RuntimeDisturbanceCfg.get("tool_wear_per_1k_steps", 0.0) or 0.0)
    extra = wear * (int(time_step or 0) / 1000.0)
    return sample_noisy_steps(base + extra, std)


def qc_hold_extra_steps(machine_type: str, task: str, time_step: int) -> int:
    """Extra process steps after a weld/kitting cycle if a QC window is active."""
    if task not in QC_HOLD_TASKS:
        return 0
    t = int(time_step)
    for h in RuntimeDisturbanceCfg.get("qc_holds") or []:
        start = int(h.get("start", -1))
        dur = int(h.get("duration", 0))
        if start < 0 or dur <= 0 or t < start or t >= start + dur:
            continue
        if h.get("target") not in (None, machine_type):
            continue
        return max(0, int(h.get("hold_steps") or 0))
    return 0


def machine_process_succeeded() -> bool:
    p = float(RuntimeDisturbanceCfg.get("machine_success_rate", 1.0) or 1.0)
    if p >= 1.0:
        return True
    return random.random() < p


def sample_human_subtask_time(
    base: float,
    noise_std: float,
    human_idx: int | None = None,
) -> int:
    from .utils import sample_noisy_steps

    scale = float(RuntimeDisturbanceCfg.get("human_time_scale", 1.0) or 1.0)
    skills = RuntimeDisturbanceCfg.get("human_skill_scales") or []
    if human_idx is not None and 0 <= int(human_idx) < len(skills):
        scale *= float(skills[int(human_idx)])
    std = float(RuntimeDisturbanceCfg.get("human_subtask_noise_std", noise_std) or noise_std)
    return sample_noisy_steps(base * scale, std)
