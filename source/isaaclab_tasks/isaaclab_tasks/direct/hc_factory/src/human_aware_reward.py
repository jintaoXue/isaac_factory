"""Bounded human-aware reward; no simulator imports or changes to feasibility."""
from __future__ import annotations

import math


class HumanAwareReward:
    def __init__(self, config, skill_fn, efficiency_fn):
        self.enabled = bool(config.get("human_aware_reward", False))
        self.monitor = self.enabled or bool(config.get("human_reward_metrics", False))
        self.skill = skill_fn
        self.efficiency = efficiency_fn
        self.mismatch_coef = float(config.get("human_mismatch_coef", 0.05))
        self.overwork_coef = float(config.get("human_overwork_coef", 0.01))
        self.recovery_coef = float(config.get("human_recovery_coef", 0.0))
        self.threshold = float(config.get("human_fatigue_threshold", 0.8))
        self.cap = float(config.get("human_shaping_cap", 0.04))
        for value in (self.mismatch_coef, self.overwork_coef, self.recovery_coef, self.cap):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Human reward coefficients/cap must be finite and nonnegative")
        if not 0 <= self.threshold < 1:
            raise ValueError("human_fatigue_threshold must be in [0, 1)")
        self.reset()

    def reset(self):
        self.before = {}
        self.assignments = []

    def begin_step(self, state):
        self.reset()
        if not self.monitor:
            return
        records = state.get("progress", {}).get("ongoing_task_records", {})
        for name, human in state.get("human", {}).items():
            record = records.get(human.get("ongoing_task_record_index")) or {}
            ongoing = record.get("subtasks_dict", {}).get("ongoing", [])
            subtask = ongoing[0] if ongoing else None
            active = bool(subtask and subtask not in ("wait", "done", "none"))
            self.before[name] = (float(human.get("fatigue", 0.0)), active)

    def on_assignment(self, state, record):
        """Call only after a dispatch succeeds, before marking its human busy.

        Current human masks allow every free human. Recompute availability after
        each dispatch so a worker reserved earlier in the same tick is excluded.
        Task-level skill (subtask=None) is a speed proxy, not exact task duration.
        """
        if not self.monitor:
            return
        chosen = record.get("human")
        if chosen is None:
            return
        candidates = {}
        for i, (name, human) in enumerate(state.get("human", {}).items()):
            if human.get("state") != "free":
                continue
            skill = self.skill(i, record.get("task"), None)
            speed = self.efficiency(float(human.get("fatigue", 0.0))) * skill
            candidates[name] = (skill, speed)
        if chosen not in candidates:
            raise ValueError("Human reward received a non-free assigned worker")
        skill, speed = candidates[chosen]
        best = max(v[1] for v in candidates.values())
        gap = max(0.0, min(1.0, 1.0 - speed / max(best, 1e-8)))
        self.assignments.append((skill, speed, gap))

    def finish_step(self, state):
        if not self.monitor:
            return {}, {}
        humans = state.get("human", {})
        overwork = recovery = 0.0
        for name, (before, active) in self.before.items():
            after = float(humans.get(name, {}).get("fatigue", before))
            if active:
                overwork += (max(0.0, after - self.threshold) / (1.0 - self.threshold)) ** 2
            elif before >= self.threshold:
                recovery += max(0.0, before - after)
        n = max(1, len(self.before))
        parts = {
            "human_mismatch": -self.mismatch_coef * sum(a[2] for a in self.assignments),
            "human_overwork": -self.overwork_coef * overwork / n,
            "human_recovery": self.recovery_coef * recovery / n,
        }
        magnitude = sum(abs(v) for v in parts.values())
        scale = min(1.0, self.cap / magnitude) if magnitude else 1.0
        parts = {k: v * scale if self.enabled else 0.0 for k, v in parts.items()}
        stats = {
            "enabled": float(self.enabled),
            "assignments": len(self.assignments),
            "assigned_skill_sum": sum(a[0] for a in self.assignments),
            "assigned_speed_sum": sum(a[1] for a in self.assignments),
            "mismatch_count": sum(a[2] > 0.05 for a in self.assignments),
            "gap_sum": sum(a[2] for a in self.assignments),
            "cap_hit": float(self.enabled and magnitude > self.cap),
        }
        # Consume dispatch events exactly once; reset/restore never retains them.
        self.reset()
        return parts, stats
