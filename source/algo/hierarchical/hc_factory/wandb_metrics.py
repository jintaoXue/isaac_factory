"""Shared wandb groups for rule / hier / flat / eval.

Namespaces:
  - MetricPeak: live shop-floor / concurrency state
  - MetricCore: episode-level business KPIs
  - MetricTrain: training-process health
  - MetricLoss: optimization losses
  - Curriculum: segment metadata
"""

from __future__ import annotations

from typing import Any

import wandb


def define_shared_metrics(*, rl: bool = False, curriculum: bool = False) -> None:
    wandb.define_metric("Train/step")
    wandb.define_metric("MetricCore/episode")
    wandb.define_metric("MetricFullorderCore/episode")

    for key in (
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
    ):
        wandb.define_metric(f"MetricCore/{key}", step_metric="MetricCore/episode")
        wandb.define_metric(f"MetricFullorderCore/{key}", step_metric="MetricFullorderCore/episode")

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
) -> dict[str, Any]:
    """Episode-level business KPIs. ``n_finished`` is work done in this episode/segment."""
    t_budget = max(1, int(t_budget))
    n_finished = max(0, int(n_finished))
    makespan = int(makespan)
    payload: dict[str, Any] = {
        "MetricCore/episode": int(episode),
        "MetricCore/01_normalized_makespan": float(makespan) / float(t_budget),
        "MetricCore/02_mean_per_product_span": float(makespan) / float(max(1, n_finished)),
        "MetricCore/04_normalized_return": float(ep_return) / float(max(1, makespan)),
        "MetricCore/05_makespan": makespan,
        "MetricCore/06_n_finished": n_finished,
        "MetricCore/08_ep_return": float(ep_return),
    }
    if finished_abs is not None:
        payload["MetricCore/07_finished_abs"] = int(finished_abs)
    if success_rate is not None:
        payload["MetricCore/03_success_rate"] = float(success_rate)
    if mean_makespan is not None:
        payload["MetricCore/09_mean_makespan"] = float(mean_makespan)
    if mean_makespan_success is not None:
        payload["MetricCore/10_mean_makespan_success"] = float(mean_makespan_success)
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


def log_eval_episodes(
    results: list,
    *,
    t_budget: int,
    algo_name: str,
) -> None:
    """Log eval episodes with the same MetricCore keys as training."""
    if wandb.run is None:
        return
    makespans: list[int] = []
    success_ms: list[int] = []
    for i, row in enumerate(results, start=1):
        n_fin = int(getattr(row, "n_finished", 0) or 0)
        if n_fin <= 0 and bool(row.success):
            n_fin = 16
        if row.success:
            success_ms.append(int(row.makespan))
        makespans.append(int(row.makespan))
        mean_ms = sum(makespans) / len(makespans)
        mean_ok = sum(success_ms) / len(success_ms) if success_ms else None
        payload = episode_metrics(
            episode=i,
            success=bool(row.success),
            truncated=bool(row.truncated),
            makespan=int(row.makespan),
            n_finished=n_fin,
            finished_abs=n_fin,
            ep_return=float(row.ep_return),
            t_budget=t_budget,
            mean_makespan=mean_ms,
            mean_makespan_success=mean_ok,
        )
        payload["Train/step"] = i
        payload.update(fullorder_core_metrics(payload))
        wandb.log(payload)
    print(f"[Eval:{algo_name}] wandb logged {len(results)} episodes (MetricCore)")
