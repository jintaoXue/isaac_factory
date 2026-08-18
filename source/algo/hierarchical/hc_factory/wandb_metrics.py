"""Shared wandb groups for rule / hier / flat / eval.

Metric1 — absolute episode outcome (same keys overlay across algos)
Metric2 — relative efficiency (per product / vs T_budget)
Metric3 — live shop-floor + throughput
Train   — step clock + RL internals (epsilon, buffer, reward parts)
"""

from __future__ import annotations

from typing import Any

import wandb


def define_shared_metrics(*, rl: bool = False, curriculum: bool = False) -> None:
    wandb.define_metric("Train/step")
    wandb.define_metric("Metric1/episode")

    for key in (
        "success",
        "truncated",
        "makespan",
        "n_finished",
        "ep_return",
    ):
        wandb.define_metric(f"Metric1/{key}", step_metric="Metric1/episode")

    for key in (
        "per_makespan",
        "normalized_makespan",
        "mean_makespan",
        "mean_makespan_success",
    ):
        wandb.define_metric(f"Metric2/{key}", step_metric="Metric1/episode")

    for key in (
        "finished",
        "producing",
        "ongoing",
        "peak_producing",
        "peak_ongoing",
        "peak_ongoing_human",
        "peak_ongoing_robot",
        "ep_t",
        "ep_reward",
        "n_dispatch",
        "wall_time_sec",
        "wall_time_min",
        "steps_per_min",
    ):
        wandb.define_metric(f"Metric3/{key}", step_metric="Train/step")

    wandb.define_metric("Train/wall_time_sec", step_metric="Train/step")
    wandb.define_metric("Train/wall_time_min", step_metric="Train/step")
    wandb.define_metric("Train/steps_per_min", step_metric="Train/step")
    wandb.define_metric("Train/ep_reward0", step_metric="Train/step")
    wandb.define_metric("Train/step_reward0", step_metric="Train/step")
    wandb.define_metric("Train/r_step0", step_metric="Train/step")
    wandb.define_metric("Train/r_finish0", step_metric="Train/step")
    wandb.define_metric("Train/r_task0", step_metric="Train/step")
    wandb.define_metric("Train/r_success0", step_metric="Train/step")

    if rl:
        wandb.define_metric("Train/epsilon", step_metric="Train/step")
        for name in ("A", "B", "C", "D_human", "D_robot", "flat"):
            wandb.define_metric(f"Train/buffer_{name}", step_metric="Train/step")
            wandb.define_metric(f"Loss/critic_{name}", step_metric="Train/step")
        wandb.define_metric("Loss/critic_mean", step_metric="Train/step")
        wandb.define_metric("Train/joint_valid0", step_metric="Train/step")
        wandb.define_metric("Stagnation/resets_per_episode", step_metric="Metric1/episode")

    if curriculum:
        for key in ("stage", "delta_N", "start_nfin", "target_nfin"):
            wandb.define_metric(f"Curriculum/{key}", step_metric="Metric1/episode")


def axis_payload(env_step: int, wall_sec: float, steps_per_min: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Train/step": int(env_step),
        "Train/wall_time_sec": float(wall_sec),
        "Train/wall_time_min": float(wall_sec) / 60.0,
        "Metric3/wall_time_sec": float(wall_sec),
        "Metric3/wall_time_min": float(wall_sec) / 60.0,
    }
    if steps_per_min is not None:
        payload["Train/steps_per_min"] = float(steps_per_min)
        payload["Metric3/steps_per_min"] = float(steps_per_min)
    return payload


def episode_metrics(
    *,
    episode: int,
    success: bool,
    truncated: bool,
    makespan: int,
    n_finished: int,
    ep_return: float,
    t_budget: int,
    mean_makespan: float | None = None,
    mean_makespan_success: float | None = None,
) -> dict[str, Any]:
    """Absolute + relative KPIs. ``n_finished`` is work done *this episode* (delta)."""
    t_budget = max(1, int(t_budget))
    n_finished = max(0, int(n_finished))
    makespan = int(makespan)
    payload: dict[str, Any] = {
        "Metric1/episode": int(episode),
        "Metric1/success": float(success),
        "Metric1/truncated": float(truncated),
        "Metric1/makespan": makespan,
        "Metric1/n_finished": n_finished,
        "Metric1/ep_return": float(ep_return),
        "Metric2/normalized_makespan": float(makespan) / float(t_budget),
    }
    if n_finished > 0:
        payload["Metric2/per_makespan"] = float(makespan) / float(n_finished)
    if mean_makespan is not None:
        payload["Metric2/mean_makespan"] = float(mean_makespan)
    if mean_makespan_success is not None:
        payload["Metric2/mean_makespan_success"] = float(mean_makespan_success)
    return payload


def shop_metrics(
    *,
    finished: int | None = None,
    producing: int | None = None,
    ongoing: int | None = None,
    peak_producing: int | None = None,
    peak_ongoing: int | None = None,
    peak_ongoing_human: int | None = None,
    peak_ongoing_robot: int | None = None,
    ep_t: int | None = None,
    ep_reward: float | None = None,
    n_dispatch: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    mapping = {
        "finished": finished,
        "producing": producing,
        "ongoing": ongoing,
        "peak_producing": peak_producing,
        "peak_ongoing": peak_ongoing,
        "peak_ongoing_human": peak_ongoing_human,
        "peak_ongoing_robot": peak_ongoing_robot,
        "ep_t": ep_t,
        "n_dispatch": n_dispatch,
    }
    for key, value in mapping.items():
        if value is not None:
            payload[f"Metric3/{key}"] = int(value)
    if ep_reward is not None:
        payload["Metric3/ep_reward"] = float(ep_reward)
    return payload


def reward_parts_metrics(rl0: dict | None, *, ep_reward0: float | None = None) -> dict[str, Any]:
    rl0 = rl0 or {}
    parts = rl0.get("reward_parts") or {}
    payload: dict[str, Any] = {
        "Train/step_reward0": float(rl0.get("reward", 0.0) or 0.0),
        "Train/r_step0": float(parts.get("step", 0.0) or 0.0),
        "Train/r_finish0": float(parts.get("finish", 0.0) or 0.0),
        "Train/r_task0": float(parts.get("task", 0.0) or 0.0),
        "Train/r_success0": float(parts.get("success", 0.0) or 0.0),
    }
    if ep_reward0 is not None:
        payload["Train/ep_reward0"] = float(ep_reward0)
    return payload


def log_eval_episodes(
    results: list,
    *,
    t_budget: int,
    algo_name: str,
) -> None:
    """Log eval episodes with the same Metric1/Metric2 keys as training."""
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
            ep_return=float(row.ep_return),
            t_budget=t_budget,
            mean_makespan=mean_ms,
            mean_makespan_success=mean_ok,
        )
        payload["Train/step"] = i
        wandb.log(payload)
    print(f"[Eval:{algo_name}] wandb logged {len(results)} episodes (Metric1/Metric2)")
