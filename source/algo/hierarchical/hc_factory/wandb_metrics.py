"""Shared wandb groups for rule / hier / flat / eval.

Namespaces:
  - MetricPeak: live shop-floor / concurrency state
  - MetricCore: episode-level business KPIs (training)
  - MetricHuman: human-factors (fatigue EMA / episode stats)
  - MetricTest: episode-level KPIs for --test / eval only
  - MetricTrain: training-process health
  - MetricLoss: optimization losses
  - Curriculum: segment metadata

Local mirror (always on when experiment_dir is set):
  - ``metrics.jsonl``: one JSON object per ``log_metrics`` call
  - ``metrics_summary.json``: last payload + row count at close
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import wandb

from .hc_factory_imports import import_hc_module

_curr = import_hc_module("src.curriculum")


def _jsonable(value: Any) -> Any:
    """Best-effort convert tensor / numpy scalars for JSON."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class LocalMetricsWriter:
    """Append-only local mirror of wandb.log payloads under experiment_dir."""

    def __init__(self, experiment_dir: str):
        self.dir = str(experiment_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "metrics.jsonl")
        self.summary_path = os.path.join(self.dir, "metrics_summary.json")
        self._n = 0
        self._last: dict[str, Any] = {}
        # Fresh file per run (experiment_dir is unique by timestamp).
        with open(self.path, "w", encoding="utf-8"):
            pass
        print(f"[Metrics] local jsonl → {self.path}")

    def log(self, payload: dict[str, Any]) -> None:
        row = {str(k): _jsonable(v) for k, v in payload.items()}
        self._last = row
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._n += 1

    def close(self) -> None:
        summary = {"n_rows": self._n, "last": self._last}
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Metrics] wrote {self._n} rows → {self.path}; summary → {self.summary_path}")


def log_metrics(
    payload: dict[str, Any],
    *,
    local: LocalMetricsWriter | None = None,
    use_wandb: bool = False,
) -> None:
    """Write metrics to local jsonl always (if local set); optionally mirror to wandb."""
    if local is not None:
        local.log(payload)
    if use_wandb and wandb.run is not None:
        wandb.log(payload)

_CORE_KEYS = (
    "01_normalized_makespan",
    "02_mean_per_product_span",
    "03_success_rate",
    "04_normalized_return",
    "05_makespan",
    "06_n_finished",
    "07_finished_abs",
    "08_ep_return",
    "09_mean_makespan",
    "10_mean_makespan_success",
)

# Registered HeterogeneousHuman count (cfg_human.CfgHumanRegistrationInfos).
_DEFAULT_NUM_HUMAN = 5
_DEFAULT_FATIGUE_EMA_ALPHA = 0.02


def define_shared_metrics(
    *, rl: bool = False, curriculum: bool = False, test: bool = False, catalog: bool = False
) -> None:
    wandb.define_metric("Train/step")
    wandb.define_metric("MetricCore/episode")
    wandb.define_metric("MetricFullorderCore/episode")
    wandb.define_metric("MetricTest/episode")
    wandb.define_metric("MetricHuman/episode")

    for key in _CORE_KEYS:
        wandb.define_metric(f"MetricCore/{key}", step_metric="MetricCore/episode")
        wandb.define_metric(f"MetricFullorderCore/{key}", step_metric="MetricFullorderCore/episode")
        if test:
            wandb.define_metric(f"MetricTest/{key}", step_metric="MetricTest/episode")
    if test:
        wandb.define_metric("Eval/seed")
        wandb.define_metric("Eval/seed_episode_idx")

    for key in (
        "01_producing",
        "02_ongoing_product",
        "03_peak_producing",
        "04_peak_ongoing_product",
        "05_peak_ongoing_human",
        "06_peak_ongoing_robot",
    ):
        wandb.define_metric(f"MetricPeak/{key}", step_metric="Train/step")
        wandb.define_metric(f"MetricFullorderPeak/{key}", step_metric="Train/step")

    # Per-worker rolling fatigue + aggregates (Train/step); episode KPIs use MetricHuman/episode.
    for i in range(_DEFAULT_NUM_HUMAN):
        wandb.define_metric(f"MetricHuman/{i:02d}_fatigue_ema", step_metric="Train/step")
    for key in ("mean_fatigue_ema", "max_fatigue_ema", "ep_mean_fatigue", "ep_end_mean_fatigue"):
        step = "MetricHuman/episode" if key.startswith("ep_") else "Train/step"
        wandb.define_metric(f"MetricHuman/{key}", step_metric=step)

    for key in (
        "01_epsilon",
        "02_steps_per_min",
        "03_wall_time_min",
        "04_wall_time_hour",
        "05_buffer_A",
        "06_buffer_B",
        "07_buffer_C",
        "08_buffer_D_human",
        "09_buffer_D_robot",
        "10_buffer_flat",
    ):
        wandb.define_metric(f"MetricTrain/{key}", step_metric="Train/step")

    if rl:
        for key in (
            "01_critic_mean",
            "02_critic_A",
            "03_critic_B",
            "04_critic_C",
            "05_critic_D_human",
            "06_critic_D_robot",
            "07_critic_flat",
        ):
            wandb.define_metric(f"MetricLoss/{key}", step_metric="Train/step")
        wandb.define_metric("Stagnation/resets_per_episode", step_metric="MetricCore/episode")

    if curriculum:
        for key in ("01_stage", "02_target_nfin", "03_start_nfin", "04_delta_n", "05_t_budget"):
            wandb.define_metric(f"Curriculum/{key}", step_metric="MetricCore/episode")

    if catalog:
        wandb.define_metric("MetricCatalog/episode")
        for key in ("01_unique_keys", "02_joined_cumulative", "03_not_joined_cumulative"):
            wandb.define_metric(f"MetricCatalog/{key}", step_metric="Train/step")
        for key in (
            "01_unique_keys",
            "02_new_keys",
            "03_joined",
            "04_not_joined",
            "05_new_keys_since_run",
            "06_nfin_buckets_covered",
            "07_joined_cumulative",
            "08_not_joined_cumulative",
            "09_join_fraction",
            "10_join_fraction_cumulative",
        ):
            wandb.define_metric(f"MetricCatalog/{key}", step_metric="MetricCatalog/episode")


def _human_idx_from_entity(key: str, ent: dict) -> int | None:
    kv = ent.get("key_variables")
    if isinstance(kv, dict) and kv.get("idx") is not None:
        try:
            return int(kv["idx"])
        except (TypeError, ValueError):
            pass
    m = re.search(r"num_(\d+)_", str(key))
    return int(m.group(1)) if m else None


class HumanFatigueTracker:
    """Per-env human fatigue EMA + episode accumulators for MetricHuman/*."""

    def __init__(
        self,
        num_humans: int = _DEFAULT_NUM_HUMAN,
        ema_alpha: float = _DEFAULT_FATIGUE_EMA_ALPHA,
    ):
        self.num_humans = int(num_humans)
        self.alpha = float(ema_alpha)
        self.ema = [0.0] * self.num_humans
        self._ema_ready = [False] * self.num_humans
        self._ep_sum = 0.0
        self._ep_count = 0

    def update(self, env_dict: dict) -> None:
        """Ingest one env frame; skip empty / out-of-range human slots."""
        humans = env_dict.get("human") or {}
        if not isinstance(humans, dict):
            return
        for key, ent in humans.items():
            if not isinstance(ent, dict):
                continue
            idx = _human_idx_from_entity(str(key), ent)
            if idx is None or not (0 <= idx < self.num_humans):
                continue
            f = float(ent.get("fatigue", 0.0) or 0.0)
            f = min(1.0, max(0.0, f))
            if not self._ema_ready[idx]:
                self.ema[idx] = f
                self._ema_ready[idx] = True
            else:
                a = self.alpha
                self.ema[idx] = (1.0 - a) * self.ema[idx] + a * f
            self._ep_sum += f
            self._ep_count += 1

    def step_payload(self) -> dict[str, Any]:
        """Train/step curves: per-worker EMA + mean/max over registered workers."""
        payload: dict[str, Any] = {}
        ready = [self.ema[i] for i in range(self.num_humans) if self._ema_ready[i]]
        for i in range(self.num_humans):
            if self._ema_ready[i]:
                payload[f"MetricHuman/{i:02d}_fatigue_ema"] = float(self.ema[i])
        if ready:
            payload["MetricHuman/mean_fatigue_ema"] = float(sum(ready) / len(ready))
            payload["MetricHuman/max_fatigue_ema"] = float(max(ready))
        return payload

    def episode_payload(self, *, episode: int) -> dict[str, Any]:
        """Episode-end stats (aligned with MetricCore/episode via MetricHuman/episode)."""
        payload: dict[str, Any] = {"MetricHuman/episode": int(episode)}
        if self._ep_count > 0:
            payload["MetricHuman/ep_mean_fatigue"] = float(self._ep_sum / self._ep_count)
        ready = [self.ema[i] for i in range(self.num_humans) if self._ema_ready[i]]
        if ready:
            # End-of-episode proxy: last EMA (env may already have reset next_obs).
            payload["MetricHuman/ep_end_mean_fatigue"] = float(sum(ready) / len(ready))
        return payload

    def reset_episode(self) -> None:
        """Clear episode accumulators and EMA so the next episode starts clean."""
        self.ema = [0.0] * self.num_humans
        self._ema_ready = [False] * self.num_humans
        self._ep_sum = 0.0
        self._ep_count = 0


class HumanFatigueMonitor:
    """Multi-env wrapper; step logs use env 0 (matches existing Peak logging)."""

    def __init__(
        self,
        num_envs: int = 1,
        num_humans: int = _DEFAULT_NUM_HUMAN,
        ema_alpha: float = _DEFAULT_FATIGUE_EMA_ALPHA,
    ):
        n = max(1, int(num_envs))
        self.trackers = [
            HumanFatigueTracker(num_humans=num_humans, ema_alpha=ema_alpha) for _ in range(n)
        ]

    def ensure(self, env_id: int) -> HumanFatigueTracker:
        while env_id >= len(self.trackers):
            self.trackers.append(
                HumanFatigueTracker(
                    num_humans=self.trackers[0].num_humans if self.trackers else _DEFAULT_NUM_HUMAN,
                    ema_alpha=self.trackers[0].alpha if self.trackers else _DEFAULT_FATIGUE_EMA_ALPHA,
                )
            )
        return self.trackers[env_id]

    def update(self, env_id: int, env_dict: dict) -> None:
        self.ensure(env_id).update(env_dict)

    def step_payload(self, env_id: int = 0) -> dict[str, Any]:
        return self.ensure(env_id).step_payload()

    def on_episode_done(self, env_id: int, *, episode: int) -> dict[str, Any]:
        tr = self.ensure(env_id)
        payload = tr.episode_payload(episode=episode)
        tr.reset_episode()
        return payload


def axis_payload(env_step: int, wall_sec: float, steps_per_min: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Train/step": int(env_step),
        "MetricTrain/03_wall_time_min": float(wall_sec) / 60.0,
        "MetricTrain/04_wall_time_hour": float(wall_sec) / 3600.0,
    }
    if steps_per_min is not None:
        payload["MetricTrain/02_steps_per_min"] = float(steps_per_min)
    return payload


def episode_metrics(
    *,
    episode: int,
    success: bool,
    truncated: bool,
    makespan: int,
    n_finished: int,
    finished_abs: int | None = None,
    ep_return: float,
    t_budget: int,
    success_rate: float | None = None,
    mean_makespan: float | None = None,
    mean_makespan_success: float | None = None,
    prefix: str = "MetricCore",
) -> dict[str, Any]:
    """Episode-level business KPIs. ``n_finished`` is work done in this episode/segment."""
    t_budget = max(1, int(t_budget))
    n_finished = max(0, int(n_finished))
    makespan = int(makespan)
    payload: dict[str, Any] = {
        f"{prefix}/episode": int(episode),
        f"{prefix}/01_normalized_makespan": float(makespan) / float(t_budget),
        f"{prefix}/02_mean_per_product_span": float(makespan) / float(max(1, n_finished)),
        f"{prefix}/04_normalized_return": float(ep_return) / float(max(1, makespan)),
        f"{prefix}/05_makespan": makespan,
        f"{prefix}/06_n_finished": n_finished,
        f"{prefix}/08_ep_return": float(ep_return),
    }
    if finished_abs is not None:
        payload[f"{prefix}/07_finished_abs"] = int(finished_abs)
    if success_rate is not None:
        payload[f"{prefix}/03_success_rate"] = float(success_rate)
    if mean_makespan is not None:
        payload[f"{prefix}/09_mean_makespan"] = float(mean_makespan)
    if mean_makespan_success is not None:
        payload[f"{prefix}/10_mean_makespan_success"] = float(mean_makespan_success)
    return payload


def fullorder_core_metrics(base_payload: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in base_payload.items():
        if key.startswith("MetricCore/"):
            payload[key.replace("MetricCore/", "MetricFullorderCore/", 1)] = value
    return payload


def shop_metrics(
    *,
    producing: int | None = None,
    ongoing: int | None = None,
    peak_producing: int | None = None,
    peak_ongoing: int | None = None,
    peak_ongoing_human: int | None = None,
    peak_ongoing_robot: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    mapping = {
        "01_producing": producing,
        "02_ongoing_product": ongoing,
        "03_peak_producing": peak_producing,
        "04_peak_ongoing_product": peak_ongoing,
        "05_peak_ongoing_human": peak_ongoing_human,
        "06_peak_ongoing_robot": peak_ongoing_robot,
    }
    for key, value in mapping.items():
        if value is not None:
            payload[f"MetricPeak/{key}"] = int(value)
    return payload


def fullorder_peak_metrics(base_payload: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in base_payload.items():
        if key.startswith("MetricPeak/"):
            payload[key.replace("MetricPeak/", "MetricFullorderPeak/", 1)] = value
    return payload


def train_metrics(
    *,
    epsilon: float | None = None,
    buffer_A: int | None = None,
    buffer_B: int | None = None,
    buffer_C: int | None = None,
    buffer_D_human: int | None = None,
    buffer_D_robot: int | None = None,
    buffer_flat: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if epsilon is not None:
        payload["MetricTrain/01_epsilon"] = float(epsilon)
    mapping = {
        "05_buffer_A": buffer_A,
        "06_buffer_B": buffer_B,
        "07_buffer_C": buffer_C,
        "08_buffer_D_human": buffer_D_human,
        "09_buffer_D_robot": buffer_D_robot,
        "10_buffer_flat": buffer_flat,
    }
    for key, value in mapping.items():
        if value is not None:
            payload[f"MetricTrain/{key}"] = int(value)
    return payload


def _eval_n_finished(row) -> int:
    n_fin = int(getattr(row, "n_finished", 0) or 0)
    if n_fin <= 0 and bool(getattr(row, "success", False)):
        return int(_curr.N_FULL_ORDER)
    return n_fin


def build_eval_episode_payload(
    episode: int,
    row,
    *,
    t_budget: int,
    makespans: list[int],
    success_ms: list[int],
    n_success: int,
) -> dict[str, Any]:
    """One eval episode row under MetricCore/* (same namespace as train)."""
    n_fin = _eval_n_finished(row)
    mean_ms = sum(makespans) / len(makespans)
    mean_ok = sum(success_ms) / len(success_ms) if success_ms else None
    payload = episode_metrics(
        episode=episode,
        success=bool(row.success),
        truncated=bool(row.truncated),
        makespan=int(row.makespan),
        n_finished=n_fin,
        finished_abs=n_fin,
        ep_return=float(row.ep_return),
        t_budget=t_budget,
        success_rate=float(n_success) / float(episode),
        mean_makespan=mean_ms,
        mean_makespan_success=mean_ok,
        prefix="MetricCore",
    )
    seed = getattr(row, "seed", None)
    ep_idx = getattr(row, "episode_idx", None)
    if seed is not None:
        payload["Eval/seed"] = int(seed)
    if ep_idx is not None:
        payload["Eval/seed_episode_idx"] = int(ep_idx)
    return payload


def log_eval_progress(
    *,
    eval_step: int,
    ep_len: int,
    n_finished: int,
    t_budget: int,
    seed: int,
    episode_idx: int,
    local: LocalMetricsWriter | None = None,
    use_wandb: bool = True,
) -> None:
    """Legacy heartbeat; prefer EvalMetricsTracker step logs (Train/step axis)."""
    del eval_step, seed, episode_idx
    t_budget = max(1, int(t_budget))
    payload = {
        "Train/step": int(ep_len),
        "MetricCore/06_n_finished": int(n_finished),
        "MetricCore/01_normalized_makespan": float(ep_len) / float(t_budget),
    }
    log_metrics(payload, local=local, use_wandb=use_wandb)


def log_eval_episode_row(
    episode: int,
    row,
    *,
    t_budget: int,
    makespans: list[int],
    success_ms: list[int],
    n_success: int,
    local: LocalMetricsWriter | None = None,
    use_wandb: bool = True,
) -> dict[str, Any]:
    payload = build_eval_episode_payload(
        episode,
        row,
        t_budget=t_budget,
        makespans=makespans,
        success_ms=success_ms,
        n_success=n_success,
    )
    log_metrics(payload, local=local, use_wandb=use_wandb)
    return payload


def log_eval_episodes(
    results: list,
    *,
    t_budget: int,
    algo_name: str,
    local: LocalMetricsWriter | None = None,
    use_wandb: bool = True,
) -> None:
    """Log eval episodes under MetricCore/* (batch; prefer EvalStream for live runs)."""
    if use_wandb and wandb.run is None and local is None:
        return
    makespans: list[int] = []
    success_ms: list[int] = []
    n_success = 0
    for i, row in enumerate(results, start=1):
        n_fin = _eval_n_finished(row)
        if row.success:
            n_success += 1
            success_ms.append(int(row.makespan))
        makespans.append(int(row.makespan))
        log_eval_episode_row(
            i,
            row,
            t_budget=t_budget,
            makespans=makespans,
            success_ms=success_ms,
            n_success=n_success,
            local=local,
            use_wandb=use_wandb,
        )
    dest = []
    if local is not None:
        dest.append(local.path)
    if use_wandb and wandb.run is not None:
        dest.append("wandb")
    print(f"[Eval:{algo_name}] logged {len(results)} episodes → {', '.join(dest) or 'nowhere'}")
