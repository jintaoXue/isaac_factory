"""Incremental-ΔN curriculum. Encoder dim and WIP cap stay 10; order stays 16.

Each stage is a *segment*: start from ``start_nfin`` (catalog warmstart), finish
``delta_N`` more products within ``T_budget = delta_N * per_T_max``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

T_MAX_ANCHOR = 25000
N_ANCHOR = 16
WIP_CAP = 10  # same as HcVectorEnvCfg.single_env_parallel_producing_limit

# (stage, start_nfin, delta_N) — target_nfin = start + delta; T_budget = delta * per_T_max
STAGES: tuple[tuple[int, int, int], ...] = (
    (0, 0, 2),
    (1, 2, 2),
    (2, 4, 4),
    (3, 8, 4),
    (4, 12, 4),
)


def per_t_max(anchor: int = T_MAX_ANCHOR) -> int:
    return max(1, int(round(anchor / N_ANCHOR)))


def t_max_for(n_products: int, anchor: int = T_MAX_ANCHOR) -> int:
    """Legacy full-horizon helper (explore catalog / 25)."""
    return max(1, int(round(anchor * n_products / N_ANCHOR)))


@dataclass
class StageSpec:
    stage: int
    start_nfin: int
    delta_n: int
    target_nfin: int
    n_products: int  # full order size (always 16)
    wip_cap: int
    t_max: int  # segment budget = delta_n * per_T_max
    per_t_max: int


def spec_for(stage: int, anchor: int = T_MAX_ANCHOR) -> StageSpec:
    stage = max(0, min(int(stage), len(STAGES) - 1))
    s, start, delta = STAGES[stage]
    pt = per_t_max(anchor)
    return StageSpec(
        stage=s,
        start_nfin=start,
        delta_n=delta,
        target_nfin=min(N_ANCHOR, start + delta),
        n_products=N_ANCHOR,
        wip_cap=WIP_CAP,
        t_max=max(1, delta * pt),
        per_t_max=pt,
    )


class CurriculumScheduler:
    def __init__(
        self,
        *,
        enabled: bool = False,
        start_stage: int = 0,
        t_max_anchor: int = T_MAX_ANCHOR,
        window: int = 20,
        success_th: float = 0.7,
        stagnation_th: float = 0.2,
        makespan_th: float = 1.0,
        product_type: str = "ProductWaterPipe",
    ) -> None:
        self.enabled = bool(enabled)
        self.anchor = int(t_max_anchor)
        self.window = int(window)
        self.success_th = float(success_th)
        self.stagnation_th = float(stagnation_th)
        self.makespan_th = float(makespan_th)
        self.product_type = product_type
        self.stage = int(start_stage) if enabled else len(STAGES) - 1
        self._success: deque[float] = deque(maxlen=self.window)
        self._stag: deque[float] = deque(maxlen=self.window)
        self._nm: deque[float] = deque(maxlen=self.window)

    @property
    def spec(self) -> StageSpec:
        return spec_for(self.stage if self.enabled else len(STAGES) - 1, self.anchor)

    def apply(self, single_env, *, overlay_existing: bool = False) -> StageSpec:
        """Write segment target / T_budget onto a live env. Full order stays 16; WIP cap 10."""
        spec = self.spec
        env = single_env.env_state_action_dict
        progress = env.setdefault("progress", {})
        if not self.enabled:
            progress.setdefault("stage_wip_cap", spec.wip_cap)
            return spec
        tm = single_env.task_manager
        tm.max_episodic_steps = spec.t_max
        progress["stage_wip_cap"] = spec.wip_cap
        progress["curriculum_stage"] = spec.stage
        progress["segment_start_nfin"] = spec.start_nfin
        progress["segment_delta_n"] = spec.delta_n
        progress["segment_target_nfin"] = spec.target_nfin
        progress["product_order"] = {self.product_type: spec.n_products}
        if not overlay_existing:
            progress["not_started"] = {self.product_type: spec.n_products}
        else:
            n_fin = 0
            finished = progress.get("finished") or {}
            for v in finished.values():
                n_fin += len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else int(v or 0)
            n_prod = len(progress.get("producing") or [])
            leftover = max(0, spec.n_products - n_fin - n_prod)
            progress["not_started"] = {self.product_type: leftover}
        return spec

    def observe_episode(self, *, success: bool, stagnation: bool, ep_len: int) -> bool:
        """Record one finished episode. Returns True if stage advanced."""
        if not self.enabled:
            return False
        spec = self.spec
        self._success.append(1.0 if success else 0.0)
        self._stag.append(1.0 if stagnation else 0.0)
        self._nm.append(ep_len / max(1, spec.t_max))
        if len(self._success) < self.window:
            return False
        if self.stage >= len(STAGES) - 1:
            return False
        ok = (
            sum(self._success) / len(self._success) >= self.success_th
            and sum(self._stag) / len(self._stag) < self.stagnation_th
            and sum(self._nm) / len(self._nm) < self.makespan_th
        )
        if not ok:
            return False
        self.stage += 1
        self._success.clear()
        self._stag.clear()
        self._nm.clear()
        return True
