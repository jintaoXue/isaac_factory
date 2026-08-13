# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
from collections import deque

import wandb
from rl_games.common import vecenv

from .agent_A_product_sequencer import ProductSequencingAgent
from .agent_B_product_priority import ProductPriorityAgent
from .agent_C_process_task_planner import ProcessTaskPlanningAgent
from .agent_D_human_robot_allocator import HumanRobotMachineAllocationAgent
from .hierarchical_dispatch import build_rule_based_action
from .hier_utils import compute_team_reward, read_rl_done


def _count_finished(env_dict: dict) -> int:
    fin = env_dict.get("progress", {}).get("finished", {})
    if not isinstance(fin, dict):
        return 0
    return sum(
        len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else int(v or 0)
        for v in fin.values()
    )


class RuleBasedHierarchical():
    def __init__(self, base_name, params):
        config = params['config']
        self.config = config
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info : dict = self.vec_env.get_env_info()

        self.cuda_device = self.env_info["cuda_device"]
        self.max_parallel_cd_dispatch = int(config.get("max_parallel_cd_dispatch", 1))
        self.agent_A = ProductSequencingAgent(self.cuda_device)
        self.agent_B = ProductPriorityAgent(self.cuda_device)
        self.agent_C = ProcessTaskPlanningAgent(self.cuda_device)
        self.agent_D = HumanRobotMachineAllocationAgent(self.cuda_device)

        self.train_dir = config.get("train_dir", "runs")
        self.experiment_dir = os.path.join(self.train_dir, config.get("full_experiment_name", "rule_based"))
        self.max_episodic_steps = int(config.get("max_episodic_steps", 45000))
        self.max_sim_episodes = config.get("max_sim_episodes")
        if self.max_sim_episodes is not None:
            self.max_sim_episodes = int(self.max_sim_episodes)
        self.log_interval = int(config.get("log_interval", 100))
        self.makespan_window = int(config.get("makespan_window", 50))
        self.makespan_all: deque[int] = deque(maxlen=self.makespan_window)
        self.makespan_success: deque[int] = deque(maxlen=self.makespan_window)
        self.global_step = 0
        self.episodes_done = 0
        self.use_wandb = bool(config.get("wandb_activate", False))
        self._train_t0 = None
        self.peak_producing = 0
        self.peak_ongoing = 0
        self._ep_peak_producing = [0 for _ in range(max(1, self.num_actors))]
        self._ep_peak_ongoing = [0 for _ in range(max(1, self.num_actors))]

        if self.use_wandb:
            self.init_wandb_logger()

        print(
            f"[Rule] max_parallel_cd_dispatch={self.max_parallel_cd_dispatch} "
            f"max_sim_episodes={self.max_sim_episodes} "
            f"log_interval={self.log_interval} max_episodic_steps={self.max_episodic_steps} "
            f"wandb={self.use_wandb}"
        )

    def init_wandb_logger(self) -> None:
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/wall_time_sec", step_metric="Train/step")
        wandb.define_metric("Train/wall_time_min", step_metric="Train/step")
        wandb.define_metric("Train/ep_reward0", step_metric="Train/step")
        wandb.define_metric("Train/finished0", step_metric="Train/step")
        wandb.define_metric("Train/episode0", step_metric="Train/step")
        wandb.define_metric("Train/ep_t0", step_metric="Train/step")
        wandb.define_metric("Train/n_dispatch0", step_metric="Train/step")
        wandb.define_metric("Train/producing0", step_metric="Train/step")
        wandb.define_metric("Train/ongoing0", step_metric="Train/step")
        wandb.define_metric("Train/peak_producing", step_metric="Train/step")
        wandb.define_metric("Train/peak_ongoing", step_metric="Train/step")
        wandb.define_metric("Metrics/MeanMakespan", step_metric="Train/step")
        wandb.define_metric("Metrics/MeanMakespan_success", step_metric="Train/step")

        # wall-clock as alternate x-axis for episode metrics
        wandb.define_metric("Metrics/wall_time_sec")
        wandb.define_metric("Metrics/step_episode", step_metric="Train/step")
        for key in (
            "Metrics/EpRet",
            "Metrics/EpLen",
            "Metrics/EpMakespan",
            "Metrics/EpSuccess",
            "Metrics/EpTruncated",
            "Metrics/EpNDispatch",
            "Metrics/EpWallTimeSec",
            "Metrics/EpMaxProducing",
            "Metrics/EpMaxOngoing",
        ):
            wandb.define_metric(key, step_metric="Metrics/wall_time_sec")

    def _wall_time_sec(self) -> float:
        if self._train_t0 is None:
            return 0.0
        return float(time.time() - self._train_t0)

    def _update_concurrency_peaks(self, env_id: int, env_dict: dict) -> tuple[int, int]:
        progress = env_dict.get("progress") or {}
        n_producing = len(progress.get("producing") or [])
        n_ongoing = len(progress.get("ongoing_task_records") or {})
        if env_id >= len(self._ep_peak_producing):
            self._ep_peak_producing.extend([0] * (env_id + 1 - len(self._ep_peak_producing)))
            self._ep_peak_ongoing.extend([0] * (env_id + 1 - len(self._ep_peak_ongoing)))
        self._ep_peak_producing[env_id] = max(self._ep_peak_producing[env_id], n_producing)
        self._ep_peak_ongoing[env_id] = max(self._ep_peak_ongoing[env_id], n_ongoing)
        self.peak_producing = max(self.peak_producing, n_producing)
        self.peak_ongoing = max(self.peak_ongoing, n_ongoing)
        return n_producing, n_ongoing

    def act(self, obs, epsilon: float | None = None):
        del epsilon
        actions : list[dict] = []
        actions_extra : list[dict] = []
        for env_state in obs:
            action = build_rule_based_action(
                env_state,
                self.cuda_device,
                max_parallel_cd_dispatch=self.max_parallel_cd_dispatch,
                agent_a=self.agent_A,
                agent_b=self.agent_B,
                agent_c=self.agent_C,
                agent_d=self.agent_D,
            )
            actions.append(action)
            actions_extra.append({})
        return actions, actions_extra

    def test(self) -> dict:
        from .tpa_eval import (
            build_eval_payload,
            print_eval_summary,
            run_eval_episodes,
            save_eval_results,
        )

        seeds = list(self.config.get("test_seeds") or [int(self.config.get("seed", 42))])
        episodes_per_seed = int(self.config.get("test_times", 1))
        output_dir = os.path.join(self.experiment_dir, "eval")

        results = run_eval_episodes(
            self.vec_env,
            lambda o, eps: self.act(o),
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_episodic_steps=self.max_episodic_steps,
            epsilon=0.0,
        )
        payload = build_eval_payload(
            algo_name="rule_based",
            results=results,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            epsilon=0.0,
            checkpoint=None,
        )
        json_path, summary_path = save_eval_results(output_dir, payload)
        print_eval_summary(payload["summary"], "rule_based")
        print(f"[Rule] eval saved: {json_path}\n[Rule] summary: {summary_path}")
        return payload

    def train(self):
        if self.config.get("test"):
            return self.test()

        obs: list[dict] = self.vec_env.reset()
        n_envs = len(obs)
        episode_reward = [0.0 for _ in range(n_envs)]
        episode_len = [0 for _ in range(n_envs)]
        episode_n_dispatch = [0 for _ in range(n_envs)]
        self._ep_peak_producing = [0 for _ in range(n_envs)]
        self._ep_peak_ongoing = [0 for _ in range(n_envs)]
        self._train_t0 = time.time()
        stop = False

        print(
            f"[Rule] train start n_envs={n_envs} "
            f"K={self.max_parallel_cd_dispatch} max_eps={self.max_sim_episodes}"
        )

        while not stop:
            actions, actions_extra = self.act(obs)
            next_obs = self.vec_env.step(actions, actions_extra)
            wall = self._wall_time_sec()

            for env_id in range(n_envs):
                n_producing, n_ongoing = self._update_concurrency_peaks(env_id, next_obs[env_id])
                del n_producing, n_ongoing
                reward = compute_team_reward(obs[env_id], next_obs[env_id])
                done, truncated, success = read_rl_done(next_obs[env_id])
                n_disp = len(actions[env_id].get("dispatch_list") or [])
                episode_reward[env_id] += reward
                episode_len[env_id] += 1
                episode_n_dispatch[env_id] += n_disp

                if done:
                    self.episodes_done += 1
                    completed_ep = int(next_obs[env_id].get("episode_num", 0) or 0)
                    ep_len = episode_len[env_id]
                    ep_max_prod = self._ep_peak_producing[env_id]
                    ep_max_ong = self._ep_peak_ongoing[env_id]
                    if success:
                        self.makespan_success.append(ep_len)
                        self.makespan_all.append(ep_len)
                    elif truncated:
                        self.makespan_all.append(ep_len)

                    mean_ms = (
                        sum(self.makespan_all) / len(self.makespan_all) if self.makespan_all else None
                    )
                    mean_ms_ok = (
                        sum(self.makespan_success) / len(self.makespan_success)
                        if self.makespan_success
                        else None
                    )
                    mean_ms_str = f"{mean_ms:.1f}" if mean_ms is not None else "n/a"
                    mean_ok_str = f"{mean_ms_ok:.1f}" if mean_ms_ok is not None else "n/a"
                    print(
                        f"[Rule] EP_DONE env={env_id} episode={completed_ep} "
                        f"done_count={self.episodes_done} "
                        f"len={ep_len} reward={episode_reward[env_id]:.2f} "
                        f"success={int(success)} truncated={int(truncated)} "
                        f"n_dispatch={episode_n_dispatch[env_id]} "
                        f"ep_max_prod={ep_max_prod} ep_max_ong={ep_max_ong} "
                        f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing} "
                        f"finished={_count_finished(next_obs[env_id])} "
                        f"wall={wall/60.0:.2f}min mean_ms={mean_ms_str} mean_ms_ok={mean_ok_str}"
                    )
                    if self.use_wandb:
                        payload = {
                            "Train/step": self.global_step,
                            "Train/wall_time_sec": wall,
                            "Train/wall_time_min": wall / 60.0,
                            "Train/peak_producing": self.peak_producing,
                            "Train/peak_ongoing": self.peak_ongoing,
                            "Metrics/wall_time_sec": wall,
                            "Metrics/step_episode": self.episodes_done,
                            "Metrics/EpRet": episode_reward[env_id],
                            "Metrics/EpLen": ep_len,
                            "Metrics/EpMakespan": ep_len,
                            "Metrics/EpSuccess": float(success),
                            "Metrics/EpTruncated": float(truncated),
                            "Metrics/EpNDispatch": episode_n_dispatch[env_id],
                            "Metrics/EpWallTimeSec": wall,
                            "Metrics/EpMaxProducing": ep_max_prod,
                            "Metrics/EpMaxOngoing": ep_max_ong,
                        }
                        if mean_ms is not None:
                            payload["Metrics/MeanMakespan"] = mean_ms
                        if mean_ms_ok is not None:
                            payload["Metrics/MeanMakespan_success"] = mean_ms_ok
                        wandb.log(payload)

                    episode_reward[env_id] = 0.0
                    episode_len[env_id] = 0
                    episode_n_dispatch[env_id] = 0
                    self._ep_peak_producing[env_id] = 0
                    self._ep_peak_ongoing[env_id] = 0

                    if (
                        self.max_sim_episodes is not None
                        and self.episodes_done >= self.max_sim_episodes
                    ):
                        stop = True

            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                env0 = next_obs[0]
                progress = env0.get("progress") or {}
                rl0 = env0.get("rl") or {}
                parts = rl0.get("reward_parts") or {}
                ep0 = int(env0.get("episode_num", 0) or 0)
                t0 = int(env0.get("time_step", 0) or 0)
                finished = _count_finished(env0)
                producing = progress.get("producing") or []
                not_started = progress.get("not_started") or {}
                n_not_started = sum(int(v or 0) for v in not_started.values()) if isinstance(not_started, dict) else 0
                ongoing = progress.get("ongoing_task_records") or {}
                next_p = progress.get("next_product")
                mean_ms = (
                    sum(self.makespan_all) / len(self.makespan_all) if self.makespan_all else None
                )
                mean_ms_str = f"{mean_ms:.1f}" if mean_ms is not None else "n/a"
                wall = self._wall_time_sec()
                print(
                    f"[Rule] step={self.global_step} episode={ep0} ep_t={t0} "
                    f"ep_reward0={episode_reward[0]:.2f} finished={finished} "
                    f"producing={len(producing)} ongoing={len(ongoing)} "
                    f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing} "
                    f"not_started={n_not_started} next={next_p} "
                    f"r={float(rl0.get('reward', 0.0) or 0.0):.3f} "
                    f"(step={float(parts.get('step', 0.0) or 0.0):.3f} "
                    f"finish={float(parts.get('finish', 0.0) or 0.0):.3f} "
                    f"task={float(parts.get('task', 0.0) or 0.0):.3f} "
                    f"success={float(parts.get('success', 0.0) or 0.0):.3f}) "
                    f"wall={wall/60.0:.2f}min mean_ms={mean_ms_str} n_envs={n_envs}"
                )
                if self.use_wandb:
                    payload = {
                        "Train/step": self.global_step,
                        "Train/wall_time_sec": wall,
                        "Train/wall_time_min": wall / 60.0,
                        "Train/ep_reward0": episode_reward[0],
                        "Train/finished0": finished,
                        "Train/episode0": ep0,
                        "Train/ep_t0": t0,
                        "Train/n_dispatch0": len(actions[0].get("dispatch_list") or []),
                        "Train/producing0": len(producing),
                        "Train/ongoing0": len(ongoing),
                        "Train/peak_producing": self.peak_producing,
                        "Train/peak_ongoing": self.peak_ongoing,
                    }
                    if mean_ms is not None:
                        payload["Metrics/MeanMakespan"] = mean_ms
                    wandb.log(payload)

            obs = next_obs

        wall = self._wall_time_sec()
        print(
            f"[Rule] train finished episodes_done={self.episodes_done} "
            f"steps={self.global_step} wall={wall/60.0:.2f}min "
            f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing}"
        )
        if self.use_wandb:
            wandb.log(
                {
                    "Train/step": self.global_step,
                    "Train/wall_time_sec": wall,
                    "Train/wall_time_min": wall / 60.0,
                    "Train/peak_producing": self.peak_producing,
                    "Train/peak_ongoing": self.peak_ongoing,
                    "Metrics/wall_time_sec": wall,
                }
            )
            wandb.finish()
