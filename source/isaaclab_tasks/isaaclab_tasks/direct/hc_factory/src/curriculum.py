"""Reverse curriculum with fixed training target=10; eval/explore use 16.

Each stage is a segment:
    warmstart from ``start_nfin`` finished products,
    aim for ``target_nfin`` (= ``N_TRAIN_TARGET``),
    so the required progress is ``delta_n = target - start``,
    with ``T_budget = delta_n * per_t_max``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# Full-order anchor (N=16 budget). With human fatigue, ~4× pre-fatigue horizon.
# N=10 budget = round(anchor/16)*10  (64000 → 40000).
T_MAX_ANCHOR = 64000
N_FULL_ORDER = 16  # eval / explore catalog full horizon
N_TRAIN_TARGET = 10  # curriculum training cap
N_ANCHOR = N_FULL_ORDER  # legacy alias for per-product time scaling
PER_T_MAX = 4000  # round(T_MAX_ANCHOR / N_FULL_ORDER); use per_t_max() in code
WIP_CAP = 10  # same as HcVectorEnvCfg.single_env_parallel_producing_limit

# (stage, start_nfin, delta_n) with fixed target_nfin=N_TRAIN_TARGET.
STAGES: tuple[tuple[int, int, int], ...] = (
    (0, 8, 2),
    (1, 6, 4),
    (2, 4, 6),
    (3, 2, 8),
    (4, 0, 10),
)


def per_t_max(anchor: int = T_MAX_ANCHOR) -> int:
    return max(1, int(round(int(anchor) / N_FULL_ORDER)))


def t_max_for(n_products: int, anchor: int = T_MAX_ANCHOR) -> int:
    """Full-horizon helper (explore catalog / eval)."""
    return max(1, int(n_products) * per_t_max(anchor))


def _clear_segment_fields(progress: dict) -> None:
    progress.pop("segment_target_nfin", None)
    progress.pop("segment_start_nfin", None)
    progress.pop("segment_delta_n", None)


def apply_train_order(
    single_env,
    *,
    n_products: int = N_TRAIN_TARGET,
    product_type: str = "ProductWaterPipe",
    anchor: int = T_MAX_ANCHOR,
) -> int:
    """Set training order size and matching T_max on a live env."""
    t_max = t_max_for(n_products, anchor)
    single_env.task_manager.max_episodic_steps = t_max
    progress = single_env.env_state_action_dict.setdefault("progress", {})
    progress["product_order"] = {product_type: int(n_products)}
    progress["not_started"] = {product_type: int(n_products)}
    _clear_segment_fields(progress)
    return t_max


def apply_eval_order(
    single_env,
    *,
    product_type: str = "ProductWaterPipe",
    anchor: int = T_MAX_ANCHOR,
) -> int:
    """Set full-order eval (N_FULL_ORDER) and T_max on a live env."""
    t_max = t_max_for(N_FULL_ORDER, anchor)
    single_env.task_manager.max_episodic_steps = t_max
    progress = single_env.env_state_action_dict.setdefault("progress", {})
    progress["product_order"] = {product_type: N_FULL_ORDER}
    progress["not_started"] = {product_type: N_FULL_ORDER}
    _clear_segment_fields(progress)
    return t_max


@dataclass
class StageSpec:
    stage: int
    start_nfin: int
    delta_n: int
    target_nfin: int
    n_products: int  # training order size (N_TRAIN_TARGET)
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
        target_nfin=N_TRAIN_TARGET,
        n_products=N_TRAIN_TARGET,
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
        """Write segment target / T_budget onto a live env. Training order = N_TRAIN_TARGET."""
        spec = self.spec
        env = single_env.env_state_action_dict
        progress = env.setdefault("progress", {})
        if not self.enabled:
            apply_train_order(
                single_env,
                n_products=N_TRAIN_TARGET,
                product_type=self.product_type,
                anchor=self.anchor,
            )
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
