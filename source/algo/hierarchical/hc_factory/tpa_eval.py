"""Shared evaluation utilities for HcFactory TPA agents (rule / hier / flat)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from .hier_utils import compute_team_reward, count_busy_agents, crossed_interval, read_rl_done, steps_per_min
from .hc_factory_imports import import_hc_module

_curr = import_hc_module("src.curriculum")


def _count_finished(env_dict: dict) -> int:
    fin = env_dict.get("progress", {}).get("finished", {})
    if not isinstance(fin, dict):
        return 0
    return sum(
        len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else int(v or 0)
        for v in fin.values()
    )


def _mean_per_product_span_str(ep_t: int, *, episode_n_finished: int, finished: int) -> str:
    n_done = int(episode_n_finished)
    if n_done <= 0:
        n_done = int(finished)
    if n_done <= 0:
        return "n/a"
    return f"{float(ep_t + 1) / float(n_done):.1f}"


def _eprint(msg: str) -> None:
    """Eval progress goes to stderr so wandb stdout redirect does not hide it."""
    print(msg, file=sys.stderr, flush=True)


@dataclass
class EpisodeResult:
    seed: int
    episode_idx: int
    makespan: int
    success: bool
    truncated: bool
    ep_return: float
    n_finished: int = 0


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
    _eprint(f"\n[Eval:{algo_name}] episodes={summary.get('num_episodes', 0)}")
    if ms.get("mean") is not None:
        _eprint(f"  Makespan:     {ms['mean']:.1f} ± {ms['std']:.1f}")
    if ms_ok.get("mean") is not None:
        _eprint(f"  Makespan(ok): {ms_ok['mean']:.1f} ± {ms_ok['std']:.1f}")
    if succ.get("mean") is not None:
        _eprint(f"  Success:      {succ['mean'] * 100:.1f}% ± {succ['std'] * 100:.1f}%")
    if trunc.get("mean") is not None:
        _eprint(f"  Truncation:   {trunc['mean'] * 100:.1f}% ± {trunc['std'] * 100:.1f}%")


def _eval_n_finished(row: EpisodeResult) -> int:
    n_fin = int(row.n_finished or 0)
    if n_fin <= 0 and bool(row.success):
        return int(_curr.N_FULL_ORDER)
    return n_fin


class EvalMetricsTracker:
    """Train-aligned MetricCore/Peak/Human logging during eval rollouts."""

    def __init__(
        self,
        *,
        t_budget: int,
        algo_name: str,
        n_target: int,
        log_interval: int,
        local=None,
        use_wandb: bool = False,
        num_envs: int = 1,
    ) -> None:
        from .wandb_metrics import HumanFatigueMonitor, axis_payload, fullorder_core_metrics, fullorder_peak_metrics, log_metrics, shop_metrics

        self.t_budget = int(t_budget)
        self.algo_name = algo_name
        self.n_target = int(n_target)
        self.log_interval = max(1, int(log_interval))
        self.local = local
        self.use_wandb = bool(use_wandb)
        self.num_envs = max(1, int(num_envs))
        self._axis_payload = axis_payload
        self._shop_metrics = shop_metrics
        self._fullorder_core_metrics = fullorder_core_metrics
        self._fullorder_peak_metrics = fullorder_peak_metrics
        self._log_metrics = log_metrics
        self._fatigue = HumanFatigueMonitor(num_envs=self.num_envs)
        self.global_step = 0
        self._t0 = time.time()
        self._last_logged_step = 0
        self.peak_producing = 0
        self.peak_ongoing = 0
        self.peak_ongoing_human = 0
        self.peak_ongoing_robot = 0
        self.makespan_all: list[int] = []
        self.makespan_success: list[int] = []
        self.success_hist: list[float] = []
        self.episodes_done = 0

    def _wall_sec(self) -> float:
        return float(time.time() - self._t0)

    def _update_peaks(self, env_id: int, env_dict: dict) -> tuple[int, int, int, int]:
        progress = env_dict.get("progress") or {}
        n_producing = len(progress.get("producing") or [])
        n_ongoing = len(progress.get("ongoing_task_records") or {})
        n_human = count_busy_agents(env_dict.get("human"))
        n_robot = count_busy_agents(env_dict.get("robot"))
        self.peak_producing = max(self.peak_producing, n_producing)
        self.peak_ongoing = max(self.peak_ongoing, n_ongoing)
        self.peak_ongoing_human = max(self.peak_ongoing_human, n_human)
        self.peak_ongoing_robot = max(self.peak_ongoing_robot, n_robot)
        return n_producing, n_ongoing, n_human, n_robot

    def on_env_step(
        self,
        env_id: int,
        env_dict: dict,
        *,
        seed: int,
        episode_idx: int,
        ep_len: int,
    ) -> None:
        self.global_step += 1
        self._fatigue.update(env_id, env_dict)
        self._update_peaks(env_id, env_dict)
        if ep_len == 1:
            _eprint(
                f"[Eval:{self.algo_name}] rollout seed={seed} ep_idx={episode_idx} "
                f"target={self.n_target} T_max={self.t_budget}"
            )
        if crossed_interval(self._last_logged_step, self.global_step, self.log_interval):
            self._last_logged_step = self.global_step
            self._log_step(env_dict, seed=seed, episode_idx=episode_idx)

    def _log_step(self, env_dict: dict, *, seed: int, episode_idx: int) -> None:
        progress = env_dict.get("progress") or {}
        t0 = int(env_dict.get("time_step", 0) or 0)
        finished = _count_finished(env_dict)
        remain = max(0, self.n_target - finished)
        nmk = float(t0 + 1) / float(max(1, self.t_budget))
        mpps_str = _mean_per_product_span_str(
            t0,
            episode_n_finished=finished,
            finished=finished,
        )
        wall = self._wall_sec()
        spm = steps_per_min(self.global_step, wall, self.num_envs)
        spm_str = f"{spm:.1f}" if spm is not None else "n/a"
        _eprint(
            f"[Eval:{self.algo_name}] step={self.global_step} seed={seed} ep_idx={episode_idx} "
            f"ep_t={t0} target={self.n_target} finished={finished} remain={remain} "
            f"nmk={nmk:.3f} mpps={mpps_str} steps/min={spm_str}"
        )
        peak_payload = self._shop_metrics(
            producing=len(progress.get("producing") or []),
            ongoing=len(progress.get("ongoing_task_records") or {}),
            peak_producing=self.peak_producing,
            peak_ongoing=self.peak_ongoing,
            peak_ongoing_human=self.peak_ongoing_human,
            peak_ongoing_robot=self.peak_ongoing_robot,
        )
        payload = self._axis_payload(self.global_step, wall, spm)
        payload.update(peak_payload)
        payload.update(self._fullorder_peak_metrics(peak_payload))
        payload.update(self._fatigue.step_payload(0))
        self._log_metrics(payload, local=self.local, use_wandb=self.use_wandb)

    def finish_episode(self, result: EpisodeResult) -> dict:
        from .wandb_metrics import episode_metrics

        self.episodes_done += 1
        n_fin = _eval_n_finished(result)
        finished_abs = n_fin
        if result.success:
            self.makespan_success.append(int(result.makespan))
            self.makespan_all.append(int(result.makespan))
        elif result.truncated:
            self.makespan_all.append(int(result.makespan))
        self.success_hist.append(float(result.success))

        mean_ms = sum(self.makespan_all) / len(self.makespan_all) if self.makespan_all else None
        mean_ms_ok = (
            sum(self.makespan_success) / len(self.makespan_success) if self.makespan_success else None
        )
        success_rate = sum(self.success_hist) / len(self.success_hist) if self.success_hist else None
        success_rate_str = f"{success_rate:.2f}" if success_rate is not None else "n/a"
        nmk = float(result.makespan) / float(max(1, self.t_budget))
        mpps_str = _mean_per_product_span_str(
            int(result.makespan) - 1,
            episode_n_finished=n_fin,
            finished=n_fin,
        )
        _eprint(
            f"[Eval:{self.algo_name}] EP_DONE ep={self.episodes_done} seed={result.seed} "
            f"idx={result.episode_idx} len={result.makespan} finished={finished_abs} "
            f"success={result.success} success_rate={success_rate_str} nmk={nmk:.3f} mpps={mpps_str}"
        )

        wall = self._wall_sec()
        spm = steps_per_min(self.global_step, wall, self.num_envs)
        human_ep = self._fatigue.on_episode_done(0, episode=self.episodes_done)
        core_payload = episode_metrics(
            episode=self.episodes_done,
            success=bool(result.success),
            truncated=bool(result.truncated),
            makespan=int(result.makespan),
            n_finished=n_fin,
            finished_abs=finished_abs,
            ep_return=float(result.ep_return),
            t_budget=self.t_budget,
            success_rate=success_rate,
            mean_makespan=mean_ms,
            mean_makespan_success=mean_ms_ok,
            prefix="MetricCore",
        )
        peak_payload = self._shop_metrics(
            peak_producing=self.peak_producing,
            peak_ongoing=self.peak_ongoing,
            peak_ongoing_human=self.peak_ongoing_human,
            peak_ongoing_robot=self.peak_ongoing_robot,
        )
        payload = self._axis_payload(self.global_step, wall, spm)
        payload.update(core_payload)
        payload.update(peak_payload)
        payload.update(human_ep)
        payload.update(self._fullorder_core_metrics(core_payload))
        payload.update(self._fullorder_peak_metrics(peak_payload))
        payload["Eval/seed"] = int(result.seed)
        payload["Eval/seed_episode_idx"] = int(result.episode_idx)
        return payload


class EvalStream:
    """Live eval sink: episodes.jsonl + train-aligned metrics + terminal progress."""

    def __init__(
        self,
        output_dir: str,
        *,
        t_budget: int,
        algo_name: str,
        total_episodes: int,
        n_target: int,
        log_interval: int = 100,
        local=None,
        use_wandb: bool = False,
        num_envs: int = 1,
    ) -> None:
        self.output_dir = output_dir
        self.t_budget = int(t_budget)
        self.algo_name = algo_name
        self.total_episodes = int(total_episodes)
        self.tracker = EvalMetricsTracker(
            t_budget=t_budget,
            algo_name=algo_name,
            n_target=n_target,
            log_interval=log_interval,
            local=local,
            use_wandb=use_wandb,
            num_envs=num_envs,
        )
        self.local = local
        self.use_wandb = use_wandb
        self.results: list[EpisodeResult] = []
        os.makedirs(output_dir, exist_ok=True)
        self.episodes_path = os.path.join(output_dir, "episodes.jsonl")
        self.partial_path = os.path.join(output_dir, "eval_summary_partial.json")
        open(self.episodes_path, "w", encoding="utf-8").close()
        _eprint(
            f"[Eval:{algo_name}] live stream → {self.episodes_path} "
            f"(total={self.total_episodes}, T_budget={self.t_budget}, log_interval={log_interval})"
        )

    def _flush_partial_summary(self) -> None:
        summary = summarize_episodes(self.results)
        with open(self.partial_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_episodes_done": len(self.results),
                    "num_episodes_total": self.total_episodes,
                    "summary": summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def on_env_step(
        self,
        env_id: int,
        env_dict: dict,
        *,
        seed: int,
        episode_idx: int,
        ep_len: int,
    ) -> None:
        self.tracker.on_env_step(
            env_id,
            env_dict,
            seed=seed,
            episode_idx=episode_idx,
            ep_len=ep_len,
        )

    def on_episode_done(self, result: EpisodeResult) -> None:
        from .wandb_metrics import log_metrics

        self.results.append(result)
        with open(self.episodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        payload = self.tracker.finish_episode(result)
        log_metrics(payload, local=self.local, use_wandb=self.use_wandb)
        self._flush_partial_summary()
        ms_mean = sum(int(r.makespan) for r in self.results) / len(self.results)
        sr = sum(1 for r in self.results if r.success) / len(self.results)
        _eprint(
            f"[Eval:{self.algo_name}] summary ep {len(self.results)}/{self.total_episodes} "
            f"running_mean={ms_mean:.0f} sr={sr:.2f}"
        )


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
    stream: EvalStream | None = None,
    on_episode_done: Callable[[EpisodeResult], None] | None = None,
    progress_log_interval: int | None = None,
) -> list[EpisodeResult]:
    """Roll out evaluation episodes on ``env_id`` of the vec env."""
    del progress_log_interval  # legacy; use stream.log_interval via EvalStream ctor
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
                n_finished = max(_count_finished(obs[env_id]), _count_finished(next_obs[env_id]))
                if stream is not None:
                    stream.on_env_step(
                        env_id,
                        next_obs[env_id],
                        seed=seed,
                        episode_idx=ep_counter,
                        ep_len=ep_len,
                    )
                obs = next_obs
                if done:
                    row = EpisodeResult(
                        seed=seed,
                        episode_idx=ep_counter,
                        makespan=ep_len,
                        success=success,
                        truncated=truncated,
                        ep_return=ep_return,
                        n_finished=n_finished,
                    )
                    results.append(row)
                    if stream is not None:
                        stream.on_episode_done(row)
                    elif on_episode_done is not None:
                        on_episode_done(row)
                    ep_counter += 1
                    break
            else:
                row = EpisodeResult(
                    seed=seed,
                    episode_idx=ep_counter,
                    makespan=ep_len,
                    success=False,
                    truncated=True,
                    ep_return=ep_return,
                    n_finished=_count_finished(obs[env_id]),
                )
                results.append(row)
                if stream is not None:
                    stream.on_episode_done(row)
                elif on_episode_done is not None:
                    on_episode_done(row)
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
