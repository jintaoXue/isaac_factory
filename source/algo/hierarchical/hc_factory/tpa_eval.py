"""Shared evaluation utilities for HcFactory TPA agents (rule / hier / flat)."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .hier_utils import compute_team_reward, read_rl_done


@dataclass
class EpisodeResult:
    seed: int
    episode_idx: int
    makespan: int
    success: bool
    truncated: bool
    ep_return: float


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    mean = float(sum(values) / len(values))
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return {"mean": mean, "std": float(math.sqrt(var)), "n": len(values)}


def summarize_episodes(results: list[EpisodeResult]) -> dict:
    makespans = [float(r.makespan) for r in results]
    successes = [1.0 if r.success else 0.0 for r in results]
    truncations = [1.0 if r.truncated else 0.0 for r in results]
    returns = [float(r.ep_return) for r in results]
    success_ms = [float(r.makespan) for r in results if r.success]

    return {
        "num_episodes": len(results),
        "makespan": _mean_std(makespans),
        "makespan_success_only": _mean_std(success_ms),
        "success_rate": _mean_std(successes),
        "truncation_rate": _mean_std(truncations),
        "ep_return": _mean_std(returns),
    }


def save_eval_results(output_dir: str, payload: dict, prefix: str = "eval") -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload.get("summary", {}), f, indent=2, ensure_ascii=False)
    return json_path, summary_path


def print_eval_summary(summary: dict, algo_name: str) -> None:
    ms = summary.get("makespan", {})
    ms_ok = summary.get("makespan_success_only", {})
    succ = summary.get("success_rate", {})
    trunc = summary.get("truncation_rate", {})
    print(f"\n[Eval:{algo_name}] episodes={summary.get('num_episodes', 0)}")
    if ms.get("mean") is not None:
        print(f"  Makespan:     {ms['mean']:.1f} ± {ms['std']:.1f}")
    if ms_ok.get("mean") is not None:
        print(f"  Makespan(ok): {ms_ok['mean']:.1f} ± {ms_ok['std']:.1f}")
    if succ.get("mean") is not None:
        print(f"  Success:      {succ['mean'] * 100:.1f}% ± {succ['std'] * 100:.1f}%")
    if trunc.get("mean") is not None:
        print(f"  Truncation:   {trunc['mean'] * 100:.1f}% ± {trunc['std'] * 100:.1f}%")


def run_eval_episodes(
    vec_env,
    act_fn,
    *,
    seeds: list[int],
    episodes_per_seed: int,
    max_episodic_steps: int,
    epsilon: float = 0.0,
    env_id: int = 0,
    on_reset=None,
) -> list[EpisodeResult]:
    """Roll out evaluation episodes on ``env_id`` of the vec env."""
    results: list[EpisodeResult] = []
    for seed in seeds:
        set_global_seed(seed)
        obs: list[dict] = vec_env.reset()
        if on_reset is not None:
            on_reset()
        ep_counter = 0
        while ep_counter < episodes_per_seed:
            ep_return = 0.0
            ep_len = 0
            while ep_len < max_episodic_steps:
                actions, actions_extra = act_fn(obs, epsilon)
                for i, action in enumerate(actions):
                    obs[i]["action"] = action
                next_obs = vec_env.step(actions, actions_extra)
                ep_return += compute_team_reward(obs[env_id], next_obs[env_id])
                done, truncated, success = read_rl_done(next_obs[env_id])
                ep_len += 1
                obs = next_obs
                if done:
                    results.append(
                        EpisodeResult(
                            seed=seed,
                            episode_idx=ep_counter,
                            makespan=ep_len,
                            success=success,
                            truncated=truncated,
                            ep_return=ep_return,
                        )
                    )
                    ep_counter += 1
                    break
            else:
                results.append(
                    EpisodeResult(
                        seed=seed,
                        episode_idx=ep_counter,
                        makespan=ep_len,
                        success=False,
                        truncated=True,
                        ep_return=ep_return,
                    )
                )
                ep_counter += 1
    return results


def build_eval_payload(
    *,
    algo_name: str,
    results: list[EpisodeResult],
    seeds: list[int],
    episodes_per_seed: int,
    epsilon: float,
    checkpoint: str | None,
    extra: dict | None = None,
) -> dict:
    summary = summarize_episodes(results)
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        seed_rows = [r for r in results if r.seed == seed]
        per_seed[str(seed)] = summarize_episodes(seed_rows)
    payload = {
        "algo": algo_name,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "epsilon": epsilon,
        "checkpoint": checkpoint,
        "summary": summary,
        "per_seed": per_seed,
        "episodes": [asdict(r) for r in results],
    }
    if extra:
        payload.update(extra)
    return payload
