"""Disturbance injector: L2 runtime events + helpers used by managers."""

from __future__ import annotations

import random
from typing import Any

from ..env_asset_cfg.cfg_disturbance import (
    RuntimeDisturbanceCfg,
    sample_episode_event_schedule,
)


class DisturbanceInjector:
    """Per-env runtime disturbance (machine DOWN / human absent / gantry down).

    L0/L1 config mutations are applied once via ``apply_disturbance_to_cfgs`` before
    env construction. This class only handles mid-episode events and logging hooks.
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
        self._planned_start = -1
        self._actual_target: str | None = None
        self._rng = random.Random()

    def reset(self, env: dict) -> None:
        self._restore_if_needed(env)
        self._active = False
        self._pending = False
        self._event_done = False
        self._saved_state = None
        self._event_id = 0
        self._activated_at = -1
        self._actual_target = None

        episode_id = int(env.get("episode_num", 0) or 0)
        self._planned_start, self._duration, self._rng = sample_episode_event_schedule(
            self.env_id, episode_id
        )

        dim = RuntimeDisturbanceCfg.get("dim", "none")
        if dim == "none":
            return

        # Episode-level config disturbance record (L0/L1).
        if self.collector is not None:
            applied = RuntimeDisturbanceCfg.get("applied") or {}
            self.collector.log_disturbance(
                {
                    "disturbance_id": f"{dim}_cfg",
                    "event_phase": "CONFIG",
                    "disturbance_type": f"{dim}_config",
                    "target_resource_id": "episode",
                    "target_resource_type": dim,
                    "start_time_step": 0,
                    "end_time_step": "",
                    "planned_start_time_step": self._planned_start,
                    "actual_start_time_step": "",
                    "actual_end_time_step": "",
                    "planned_duration_steps": self._duration,
                    "actual_target_resource_id": "episode",
                    "intensity": RuntimeDisturbanceCfg.get("intensity", 1.0),
                    "parameter_before": "nominal",
                    "parameter_after": str(applied),
                    "notes": f"primary_dimension={dim}",
                }
            )

    def step(self, env: dict) -> None:
        dim = RuntimeDisturbanceCfg.get("dim", "none")
        if (
            dim == "none"
            or self._planned_start < 0
            or self._duration <= 0
            or self._event_done
        ):
            return

        t = int(env.get("time_step", 0))
        target = RuntimeDisturbanceCfg.get("event_target")

        if env.get("progress", {}).get("production_done", False):
            if self._active:
                self._finish_active_event(env, dim, t)
            return

        # Arm pending window once: do not force-fail a busy resource; do not re-arm after done.
        if t >= self._planned_start and not self._active and not self._pending:
            self._pending = True

        if self._pending and not self._active:
            actual_target = self._activate(env, dim, target)
            if actual_target:
                self._active = True
                self._pending = False
                self._activated_at = t
                self._event_id += 1
                self._actual_target = actual_target
                end = t + self._duration
                self._log_event(env, dim, t, end, starting=True)

        if self._active and self._activated_at >= 0 and t >= self._activated_at + self._duration:
            self._finish_active_event(env, dim, self._activated_at + self._duration)

    def _finish_active_event(self, env: dict, dim: str, end: int) -> None:
        start = self._activated_at
        self._restore_if_needed(env)
        self._log_event(env, dim, start, end, starting=False)
        self._active = False
        self._saved_state = None
        self._activated_at = -1
        self._event_done = True

    def _log_event(
        self,
        env: dict,
        dim: str,
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
                "event_phase": "START" if starting else "END",
                "disturbance_type": {
                    "machine": "machine_failure",
                    "human": "human_unavailable",
                    "logistics": "transport_delay",
                    "material": "material_shortage",
                }.get(dim, dim),
                "target_resource_id": self._actual_target or "",
                "target_resource_type": dim,
                "start_time_step": start,
                "end_time_step": "" if starting else end,
                "planned_start_time_step": self._planned_start,
                "actual_start_time_step": start,
                "actual_end_time_step": "" if starting else end,
                "planned_duration_steps": self._duration,
                "actual_target_resource_id": self._actual_target or "",
                "intensity": RuntimeDisturbanceCfg.get("intensity", 1.0),
                "parameter_before": "nominal" if starting else "disturbed",
                "parameter_after": "disturbed" if starting else "restored",
                "notes": f"event {'start' if starting else 'end'} at step {env.get('time_step')}",
            }
        )

    def _activate(self, env: dict, dim: str, target: str | None) -> str | None:
        if dim == "machine":
            return self._activate_machine_down(env, target)
        if dim == "human":
            return self._activate_human_absent(env, target)
        if dim == "logistics":
            return self._activate_gantry_down(env, target)
        if dim == "material":
            return self._activate_material_hold(env)
        return None

    def _activate_machine_down(self, env: dict, target: str | None) -> str | None:
        """Only mark a truly idle workstation DOWN; never interrupt an ongoing task."""
        from ..env_asset_cfg.cfg_machine import CfgMachine

        machines = env.get("machine", {})
        candidates = [
            name
            for name in machines
            if name != "num07_gantry_group"
            and CfgMachine.get(name, {}).get("corresponding_process_task") != ["none"]
        ]
        self._rng.shuffle(candidates)
        if target in candidates:
            candidates.remove(target)
            candidates.insert(0, target)
        for machine_name in candidates:
            m = machines[machine_name]
            state = m["state"]
            ongoing = m.get("ongoing_task_record_index", [None] * len(state))
            workstation_indices = list(range(len(state)))
            self._rng.shuffle(workstation_indices)
            for i in workstation_indices:
                if state[i] == "free" and (i >= len(ongoing) or ongoing[i] is None):
                    self._saved_state = {
                        "kind": "machine",
                        "machine": machine_name,
                        "ws": i,
                        "prev": state[i],
                    }
                    state[i] = "invalid"
                    return f"{machine_name}_ws{i}"
        return None

    def _activate_human_absent(self, env: dict, target: str | None) -> str | None:
        humans = env.get("human", {})
        if not humans:
            return None
        idx = 0
        if target and target.startswith("human_"):
            try:
                idx = int(target.split("_", 1)[1])
            except ValueError:
                idx = 0
        preferred = f"num_{idx:02d}_NormalHuman"
        candidates: list[str] = []
        if preferred in humans:
            candidates.append(preferred)
        candidates.extend(k for k in humans if k != preferred)
        self._rng.shuffle(candidates)

        for key in candidates:
            h = humans[key]
            if h.get("state") != "free" or h.get("ongoing_task_record_index") is not None:
                continue
            self._saved_state = {
                "kind": "human",
                "key": key,
                "prev_state": h.get("state"),
            }
            # Non-free + no task → idle animation, excluded from allocator mask.
            h["state"] = "working_disturbance_absent"
            human_idx = int(h.get("key_variables", {}).get("idx", idx))
            return f"human_{human_idx}"
        return None

    def _activate_gantry_down(self, env: dict, target: str | None) -> str | None:
        """Only disable an idle active gantry."""
        gantry = env.get("machine", {}).get("num07_gantry_group")
        if gantry is None:
            return None
        state = gantry["state"]
        ongoing = gantry.get("ongoing_task_record_index", [None] * len(state))
        preferred: int | None = None
        if target and target.startswith("gantry_"):
            try:
                preferred = int(target.split("_", 1)[1])
            except ValueError:
                preferred = None

        from ..env_asset_cfg.cfg_machine import CfgMachine

        active = list(CfgMachine["num07_gantry_group"].get("active_gantry_indices", range(len(state))))
        self._rng.shuffle(active)
        order = [preferred] + [i for i in active if i != preferred] if preferred in active else active
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
            return f"gantry_{idx}"
        return None

    def _activate_material_hold(self, env: dict) -> str | None:
        candidates = []
        progress = env.get("progress", {})
        finished_jobs = {
            int(job_id)
            for job_ids in progress.get("finished", {}).values()
            for job_id in job_ids
        }
        producing = {int(job_id) for job_id in progress.get("producing_indexs", [])}
        for material_key, material_state in env.get("material", {}).items():
            batch_idx = int(material_state.get("key_variables", {}).get("idx", -1))
            if batch_idx in finished_jobs:
                continue
            if batch_idx not in producing:
                continue
            if material_state.get("ongoing_task_record_index") is not None:
                continue
            if material_state.get("disturbance_material_hold"):
                continue
            submaterials = material_state.get("submaterials") or {}
            available = [
                name
                for name, info in submaterials.items()
                if name != "product_00_pipe"
                and info.get("storage_name") not in (None, "disappear")
            ]
            if available:
                candidates.append((material_key, material_state, available))
        if not candidates:
            return None

        material_key, material_state, available = self._rng.choice(candidates)
        material_type = self._rng.choice(available)
        material_state["disturbance_material_hold"] = material_type
        self._saved_state = {
            "kind": "material",
            "material_key": material_key,
            "material_type": material_type,
        }
        batch_idx = int(material_state["key_variables"]["idx"])
        return f"material_{batch_idx}_{material_type}"

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
        elif kind == "material":
            material = env.get("material", {}).get(self._saved_state["material_key"])
            if material is not None:
                material.pop("disturbance_material_hold", None)
        self._saved_state = None


def should_skip_material_placement(batch_idx: int, material_type: str) -> bool:
    """Material shortage: skip placing some raw parts at reset."""
    frac = float(RuntimeDisturbanceCfg.get("material_shortage_frac", 0.0) or 0.0)
    if frac <= 0.0:
        return False
    # Prefer starving flange/elbow (kitting) rather than pipe, so cutting can still start.
    if material_type == "product_00_pipe":
        return False
    # Deterministic-ish per (batch, type) using RNG; caller should have seeded.
    return random.random() < frac


def sample_machine_process_time(base: float, per_machine_std: float = 0.0) -> int:
    from .utils import sample_noisy_steps

    std = float(RuntimeDisturbanceCfg.get("machine_process_noise_std", 0.0) or 0.0)
    std = max(std, float(per_machine_std or 0.0))
    return sample_noisy_steps(base, std)


def machine_process_succeeded() -> bool:
    p = float(RuntimeDisturbanceCfg.get("machine_success_rate", 1.0) or 1.0)
    if p >= 1.0:
        return True
    return random.random() < p


def sample_human_subtask_time(base: float, noise_std: float) -> int:
    from .utils import sample_noisy_steps

    scale = float(RuntimeDisturbanceCfg.get("human_time_scale", 1.0) or 1.0)
    std = float(RuntimeDisturbanceCfg.get("human_subtask_noise_std", noise_std) or noise_std)
    return sample_noisy_steps(base * scale, std)
