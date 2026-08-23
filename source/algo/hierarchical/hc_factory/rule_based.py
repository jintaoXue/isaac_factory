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
from .hc_factory_imports import import_hc_module
from .hier_utils import compute_team_reward, count_busy_agents, crossed_interval, env_steps, read_rl_done, steps_per_min
from .horizon_hooks import hc_env_list
from .wandb_metrics import (
    axis_payload,
    define_shared_metrics,
    episode_metrics,
    fullorder_core_metrics,
    fullorder_peak_metrics,
    log_eval_episodes,
    shop_metrics,
    train_metrics,
)

_curr = import_hc_module("src.curriculum")


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
        self.t_max_anchor = int(config.get("t_max_anchor", _curr.T_MAX_ANCHOR))
        self.product_type = str(config.get("product_type", "ProductWaterPipe"))
        if config.get("test"):
            self.train_n_products = _curr.N_FULL_ORDER
            self.max_episodic_steps = int(
                config.get("max_episodic_steps", _curr.t_max_for(_curr.N_FULL_ORDER, self.t_max_anchor))
            )
        else:
            self.train_n_products = int(config.get("train_n_products", _curr.N_TRAIN_TARGET))
            self.max_episodic_steps = _curr.t_max_for(self.train_n_products, self.t_max_anchor)
        self._hc_env_list = None
        self.max_sim_episodes = config.get("max_sim_episodes")
        if self.max_sim_episodes is not None:
            self.max_sim_episodes = int(self.max_sim_episodes)
        self.log_interval = int(config.get("log_interval", 100))
        self.makespan_window = int(config.get("makespan_window", 50))
        self.makespan_all: deque[int] = deque(maxlen=self.makespan_window)
        self.makespan_success: deque[int] = deque(maxlen=self.makespan_window)
        self.success_hist: deque[float] = deque(maxlen=self.makespan_window)
        self.global_step = 0
        self.episodes_done = 0
        self.use_wandb = bool(config.get("wandb_activate", False))
        self._train_t0 = None
        self.peak_producing = 0
        self.peak_ongoing = 0
        self.peak_ongoing_human = 0
        self.peak_ongoing_robot = 0
        self._ep_peak_producing = [0 for _ in range(max(1, self.num_actors))]
        self._ep_peak_ongoing = [0 for _ in range(max(1, self.num_actors))]
        self._ep_peak_ongoing_human = [0 for _ in range(max(1, self.num_actors))]
        self._ep_peak_ongoing_robot = [0 for _ in range(max(1, self.num_actors))]

        print(
            f"[Rule] max_parallel_cd_dispatch={self.max_parallel_cd_dispatch} "
            f"max_sim_episodes={self.max_sim_episodes} "
            f"log_interval={self.log_interval} N={self.train_n_products} "
            f"T_max={self.max_episodic_steps} wandb={self.use_wandb}"
        )

    def _env_list(self):
        if self._hc_env_list is None:
            self._hc_env_list = hc_env_list(self.vec_env)
        return self._hc_env_list

    def _apply_train_order(self, env_id: int) -> None:
        _curr.apply_train_order(
            self._env_list()[env_id],
            n_products=self.train_n_products,
            product_type=self.product_type,
            anchor=self.t_max_anchor,
        )

    def _apply_eval_order_all(self) -> int:
        t_max = _curr.t_max_for(_curr.N_FULL_ORDER, self.t_max_anchor)
        for env in self._env_list():
            t_max = _curr.apply_eval_order(env, product_type=self.product_type, anchor=self.t_max_anchor)
        self.max_episodic_steps = t_max
        return t_max

    def _bind_train_horizon(self) -> None:
        self._hc_env_list = hc_env_list(self.vec_env)
        for env_id in range(len(self._hc_env_list)):
            self._apply_train_order(env_id)

    def init_wandb_logger(self) -> None:
        define_shared_metrics(
            rl=False,
            curriculum=False,
            test=bool(self.config.get("test")),
        )

    def _wall_time_sec(self) -> float:
        if self._train_t0 is None:
            return 0.0
        return float(time.time() - self._train_t0)

    def _update_concurrency_peaks(self, env_id: int, env_dict: dict) -> tuple[int, int, int, int]:
        progress = env_dict.get("progress") or {}
        n_producing = len(progress.get("producing") or [])
        n_ongoing = len(progress.get("ongoing_task_records") or {})
        n_human = count_busy_agents(env_dict.get("human"))
        n_robot = count_busy_agents(env_dict.get("robot"))
        if env_id >= len(self._ep_peak_producing):
            pad = env_id + 1 - len(self._ep_peak_producing)
            self._ep_peak_producing.extend([0] * pad)
            self._ep_peak_ongoing.extend([0] * pad)
            self._ep_peak_ongoing_human.extend([0] * pad)
            self._ep_peak_ongoing_robot.extend([0] * pad)
        self._ep_peak_producing[env_id] = max(self._ep_peak_producing[env_id], n_producing)
        self._ep_peak_ongoing[env_id] = max(self._ep_peak_ongoing[env_id], n_ongoing)
        self._ep_peak_ongoing_human[env_id] = max(self._ep_peak_ongoing_human[env_id], n_human)
        self._ep_peak_ongoing_robot[env_id] = max(self._ep_peak_ongoing_robot[env_id], n_robot)
        self.peak_producing = max(self.peak_producing, n_producing)
        self.peak_ongoing = max(self.peak_ongoing, n_ongoing)
        self.peak_ongoing_human = max(self.peak_ongoing_human, n_human)
        self.peak_ongoing_robot = max(self.peak_ongoing_robot, n_robot)
        return n_producing, n_ongoing, n_human, n_robot

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

        self.vec_env.reset()
        self._apply_eval_order_all()

        results = run_eval_episodes(
            self.vec_env,
            lambda o, eps: self.act(o),
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_episodic_steps=self.max_episodic_steps,
            epsilon=0.0,
            on_reset=lambda: self._apply_eval_order_all(),
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
        if self.use_wandb:
            self.init_wandb_logger()
            log_eval_episodes(results, t_budget=self.max_episodic_steps, algo_name="rule_based")
        print(f"[Rule] eval saved: {json_path}\n[Rule] summary: {summary_path}")
        return payload

    def train(self):
        if self.config.get("test"):
            return self.test()

        if self.use_wandb:
            self.init_wandb_logger()

        obs: list[dict] = self.vec_env.reset()
        self._bind_train_horizon()
        n_envs = len(obs)
        episode_reward = [0.0 for _ in range(n_envs)]
        episode_len = [0 for _ in range(n_envs)]
        episode_n_dispatch = [0 for _ in range(n_envs)]
        # Completed products count *within the current episode*.
        # Important: env may reset internally when done, so rely on per-step n_product_finished.
        episode_n_finished = [0 for _ in range(n_envs)]
        self._ep_peak_producing = [0 for _ in range(n_envs)]
        self._ep_peak_ongoing = [0 for _ in range(n_envs)]
        self._ep_peak_ongoing_human = [0 for _ in range(n_envs)]
        self._ep_peak_ongoing_robot = [0 for _ in range(n_envs)]
        self._train_t0 = time.time()
        stop = False
        last_logged_env_steps = 0

        print(
            f"[Rule] train start n_envs={n_envs} N={self.train_n_products} "
            f"T_max={self.max_episodic_steps} K={self.max_parallel_cd_dispatch} "
            f"max_eps={self.max_sim_episodes}"
        )

        while not stop:
            actions, actions_extra = self.act(obs)
            next_obs = self.vec_env.step(actions, actions_extra)
            wall = self._wall_time_sec()

            for env_id in range(n_envs):
                n_producing, n_ongoing, n_human, n_robot = self._update_concurrency_peaks(env_id, next_obs[env_id])
                del n_producing, n_ongoing, n_human, n_robot
                reward = compute_team_reward(obs[env_id], next_obs[env_id])
                done, truncated, success = read_rl_done(next_obs[env_id])
                n_disp = len(actions[env_id].get("dispatch_list") or [])
                episode_reward[env_id] += reward
                episode_len[env_id] += 1
                episode_n_dispatch[env_id] += n_disp
                rl_step = next_obs[env_id].get("rl") or {}
                episode_n_finished[env_id] += int(rl_step.get("n_product_finished", 0) or 0)

                if done:
                    self.episodes_done += 1
                    completed_ep = int(next_obs[env_id].get("episode_num", 0) or 0)
                    ep_len = episode_len[env_id]
                    ep_max_prod = self._ep_peak_producing[env_id]
                    ep_max_ong = self._ep_peak_ongoing[env_id]
                    ep_max_human = self._ep_peak_ongoing_human[env_id]
                    ep_max_robot = self._ep_peak_ongoing_robot[env_id]
                    if success:
                        self.makespan_success.append(ep_len)
                        self.makespan_all.append(ep_len)
                    elif truncated:
                        self.makespan_all.append(ep_len)
                    self.success_hist.append(float(success))

                    mean_ms = (
                        sum(self.makespan_all) / len(self.makespan_all) if self.makespan_all else None
                    )
                    mean_ms_ok = (
                        sum(self.makespan_success) / len(self.makespan_success)
                        if self.makespan_success
                        else None
                    )
                    success_rate = (
                        sum(self.success_hist) / len(self.success_hist) if self.success_hist else None
                    )
                    mean_ms_str = f"{mean_ms:.1f}" if mean_ms is not None else "n/a"
                    mean_ok_str = f"{mean_ms_ok:.1f}" if mean_ms_ok is not None else "n/a"
                    spm = steps_per_min(self.global_step, wall, self.num_actors)
                    spm_str = f"{spm:.1f}" if spm is not None else "n/a"
                    print(
                        f"[Rule] EP_DONE env={env_id} episode={completed_ep} "
                        f"done_count={self.episodes_done} "
                        f"len={ep_len} reward={episode_reward[env_id]:.2f} "
                        f"success={int(success)} truncated={int(truncated)} "
                        f"n_dispatch={episode_n_dispatch[env_id]} "
                        f"ep_max_prod={ep_max_prod} ep_max_ong={ep_max_ong} "
                        f"ep_max_human={ep_max_human} ep_max_robot={ep_max_robot} "
                        f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing} "
                        f"peak_human={self.peak_ongoing_human} peak_robot={self.peak_ongoing_robot} "
                        f"finished={_count_finished(next_obs[env_id])} ep_done_finished={episode_n_finished[env_id]} "
                        f"steps/min={spm_str} mean_ms={mean_ms_str} mean_ms_ok={mean_ok_str}"
                    )
                    if self.use_wandb:
                        n_fin = int(episode_n_finished[env_id])
                        payload = axis_payload(env_steps(self.global_step, self.num_actors), wall, spm)
                        core_payload = episode_metrics(
                            episode=self.episodes_done,
                            success=success,
                            truncated=truncated,
                            makespan=ep_len,
                            n_finished=n_fin,
                            finished_abs=n_fin,
                            ep_return=episode_reward[env_id],
                            t_budget=self.max_episodic_steps,
                            success_rate=success_rate,
                            mean_makespan=mean_ms,
                            mean_makespan_success=mean_ms_ok,
                        )
                        peak_payload = shop_metrics(
                            peak_producing=self.peak_producing,
                            peak_ongoing=self.peak_ongoing,
                            peak_ongoing_human=self.peak_ongoing_human,
                            peak_ongoing_robot=self.peak_ongoing_robot,
                        )
                        payload.update(core_payload)
                        # Rule train is always a complete order (N=train_n_products), not a curriculum segment.
                        payload.update(fullorder_core_metrics(core_payload))
                        payload.update(peak_payload)
                        payload.update(fullorder_peak_metrics(peak_payload))
                        wandb.log(payload)

                    episode_reward[env_id] = 0.0
                    episode_len[env_id] = 0
                    episode_n_dispatch[env_id] = 0
                    episode_n_finished[env_id] = 0
                    self._ep_peak_producing[env_id] = 0
                    self._ep_peak_ongoing[env_id] = 0
                    self._ep_peak_ongoing_human[env_id] = 0
                    self._ep_peak_ongoing_robot[env_id] = 0
                    self._apply_train_order(env_id)

                    if (
                        self.max_sim_episodes is not None
                        and self.episodes_done >= self.max_sim_episodes
                    ):
                        stop = True

            self.global_step += 1
            env_step = env_steps(self.global_step, self.num_actors)

            if crossed_interval(last_logged_env_steps, env_step, self.log_interval):
                last_logged_env_steps = env_step
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
                spm = steps_per_min(self.global_step, wall, self.num_actors)
                spm_str = f"{spm:.1f}" if spm is not None else "n/a"
                print(
                    f"[Rule] step={env_steps(self.global_step, self.num_actors)} episode={ep0} ep_t={t0} "
                    f"ep_reward0={episode_reward[0]:.2f} finished={finished} ep_done_finished={episode_n_finished[0]} "
                    f"producing={len(producing)} ongoing={len(ongoing)} "
                    f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing} "
                    f"peak_human={self.peak_ongoing_human} peak_robot={self.peak_ongoing_robot} "
                    f"not_started={n_not_started} next={next_p} "
                    f"r={float(rl0.get('reward', 0.0) or 0.0):.3f} "
                    f"(step={float(parts.get('step', 0.0) or 0.0):.3f} "
                    f"finish={float(parts.get('finish', 0.0) or 0.0):.3f} "
                    f"task={float(parts.get('task', 0.0) or 0.0):.3f} "
                    f"success={float(parts.get('success', 0.0) or 0.0):.3f}) "
                    f"steps/min={spm_str} mean_ms={mean_ms_str} n_envs={n_envs}"
                )
                if self.use_wandb:
                    payload = axis_payload(env_steps(self.global_step, self.num_actors), wall, spm)
                    peak_payload = shop_metrics(
                        producing=len(producing),
                        ongoing=len(ongoing),
                        peak_producing=self.peak_producing,
                        peak_ongoing=self.peak_ongoing,
                        peak_ongoing_human=self.peak_ongoing_human,
                        peak_ongoing_robot=self.peak_ongoing_robot,
                    )
                    payload.update(peak_payload)
                    payload.update(fullorder_peak_metrics(peak_payload))
                    payload.update(train_metrics())
                    wandb.log(payload)

            obs = next_obs

        wall = self._wall_time_sec()
        spm = steps_per_min(self.global_step, wall, self.num_actors)
        spm_str = f"{spm:.1f}" if spm is not None else "n/a"
        print(
            f"[Rule] train finished episodes_done={self.episodes_done} "
            f"steps={env_steps(self.global_step, self.num_actors)} steps/min={spm_str} "
            f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing} "
            f"peak_human={self.peak_ongoing_human} peak_robot={self.peak_ongoing_robot}"
        )
        if self.use_wandb:
            finish_payload = axis_payload(env_steps(self.global_step, self.num_actors), wall, spm)
            peak_payload = shop_metrics(
                peak_producing=self.peak_producing,
                peak_ongoing=self.peak_ongoing,
                peak_ongoing_human=self.peak_ongoing_human,
                peak_ongoing_robot=self.peak_ongoing_robot,
            )
            finish_payload.update(peak_payload)
            finish_payload.update(fullorder_peak_metrics(peak_payload))
            wandb.log(finish_payload)
            wandb.finish()
