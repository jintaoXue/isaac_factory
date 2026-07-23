"""Disturbance injector: L2 runtime events + helpers used by managers."""

from __future__ import annotations

import random
from typing import Any

from ..env_asset_cfg.cfg_disturbance import RuntimeDisturbanceCfg


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

    def reset(self, env: dict) -> None:
        self._restore_if_needed(env)
        self._active = False
        self._pending = False
        self._event_done = False
        self._saved_state = None
        self._event_id = 0
        self._activated_at = -1
        self._duration = 0

        dim = RuntimeDisturbanceCfg.get("dim", "none")
        if dim == "none":
            return

        # Episode-level config disturbance record (L0/L1).
        if self.collector is not None:
            applied = RuntimeDisturbanceCfg.get("applied") or {}
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
                    "notes": f"primary_dimension={dim}",
                }
            )

    def step(self, env: dict) -> None:
        dim = RuntimeDisturbanceCfg.get("dim", "none")
        start = int(RuntimeDisturbanceCfg.get("event_start_step", -1))
        duration = int(RuntimeDisturbanceCfg.get("event_duration_steps", 0))
        if dim == "none" or start < 0 or duration <= 0 or self._event_done:
            return

        t = int(env.get("time_step", 0))
        target = RuntimeDisturbanceCfg.get("event_target")

        # Arm pending window once: do not force-fail a busy resource; do not re-arm after done.
        if t >= start and not self._active and not self._pending:
            self._pending = True
            self._duration = duration

        if self._pending and not self._active:
            ok = self._activate(env, dim, target)
            if ok:
                self._active = True
                self._pending = False
                self._activated_at = t
                self._event_id += 1
                end = t + self._duration
                self._log_event(env, dim, target, t, end, starting=True)

        if self._active and self._activated_at >= 0 and t >= self._activated_at + self._duration:
            end = self._activated_at + self._duration
            self._restore_if_needed(env)
            self._log_event(env, dim, target, self._activated_at, end, starting=False)
            self._active = False
            self._saved_state = None
            self._activated_at = -1
            self._event_done = True  # one L2 event per episode

    def _log_event(
        self,
        env: dict,
        dim: str,
        target: str | None,
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
                "target_resource_id": target or "",
                "target_resource_type": dim,
                "start_time_step": start,
                "end_time_step": "" if starting else end,
                "intensity": RuntimeDisturbanceCfg.get("intensity", 1.0),
                "parameter_before": "nominal" if starting else "disturbed",
                "parameter_after": "disturbed" if starting else "restored",
                "notes": f"event {'start' if starting else 'end'} at step {env.get('time_step')}",
            }
        )

    def _activate(self, env: dict, dim: str, target: str | None) -> bool:
        if dim == "machine":
            return self._activate_machine_down(env, target)
        if dim == "human":
            return self._activate_human_absent(env, target)
        if dim == "logistics":
            return self._activate_gantry_down(env, target)
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
