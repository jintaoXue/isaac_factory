# -*- coding: utf-8 -*-
"""FlatTPA: one-stage joint masked DQN baseline for TPA.

Selects a single joint index over legal (A,B,C,H,R) tuples, decodes to the same env
action dict as hierarchical A→B→C→D. Joint validity uses the same mask source.
"""
from __future__ import division

import os
import time
from collections import deque

import torch
import wandb
from rl_games.common import vecenv

from .flat_action_space import FlatJointActionSpace
from .flat_obs import FlatObsEncoder
from .flat_rl_agent import RLFlatAgent
from .hier_utils import compute_team_reward, crossed_interval, env_steps, read_rl_done, steps_per_min
from .wandb_metrics import (
    axis_payload,
    define_shared_metrics,
    episode_metrics,
    fullorder_core_metrics,
    fullorder_peak_metrics,
    shop_metrics,
    train_metrics,
)


def _clear_rl(env_dict: dict) -> None:
    env_dict["rl"] = {
        "reward": 0.0,
        "done": False,
        "truncated": False,
        "success": False,
        "reward_parts": {
            "step": 0.0,
            "finish": 0.0,
            "task": 0.0,
            "success": 0.0,
        },
    }


def _count_finished(env_dict: dict) -> int:
    fin = env_dict.get("progress", {}).get("finished", {})
    if not isinstance(fin, dict):
        return 0
    return sum(
        len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else int(v or 0)
        for v in fin.values()
    )


class FlatTPA:
    """One-stage masked DQN over joint A×B×C×H×R action space."""

    def __init__(self, base_name, params):
        config = params["config"]
        self.config = config
        self.env_config = config.get("env_config", {})
        self.num_actors = int(config.get("num_actors", 3))
        self.env_name = config["env_name"]
        print("Env name:", self.env_name)

        self.env_info = config.get("env_info")
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = None

        if self.vec_env is not None:
            n = getattr(self.vec_env, "num_envs", None)
            if n is None and hasattr(self.vec_env, "env_list"):
                n = len(self.vec_env.env_list)
            if n is None and isinstance(self.env_info, dict):
                n = self.env_info.get("num_envs")
            if n is not None and int(n) != self.num_actors:
                print(
                    f"[Flat] warn: num_actors={self.num_actors} != env.num_envs={n}; "
                    f"using env.num_envs"
                )
                self.num_actors = int(n)

        self.cuda_device = self.env_info["cuda_device"]
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay_steps = config.get("epsilon_decay_steps", 100000)
        self.learn_interval = config.get("learn_interval", 1)
        self.save_interval = config.get("save_interval", 1000)
        self.log_interval = config.get("log_interval", 100)
        self.global_step = 0
        self.max_episodic_steps = int(config.get("max_episodic_steps", 16000))

        dqn_kwargs = {
            "hidden_dim": config.get("hidden_dim", 128),
            "lr": config.get("learning_rate", 1e-4),
            "gamma": config.get("gamma", 0.99),
            "buffer_capacity": config.get("replay_buffer_size", 50000),
            "batch_size": config.get("batch_size", 64),
            "target_update_interval": config.get("target_update_interval", 500),
        }

        parallel_limit = config.get("parallel_producing_limit", 10)
        self.obs_encoder = FlatObsEncoder(
            self.cuda_device,
            parallel_producing_limit=parallel_limit,
            state_dim=int(config.get("state_dim", 256)),
        )
        self.action_space = FlatJointActionSpace(self.cuda_device)
        self.agent = RLFlatAgent(self.obs_encoder, self.action_space, self.cuda_device, **dqn_kwargs)

        self.train_dir = config.get("train_dir", "runs")
        self.experiment_dir = os.path.join(self.train_dir, config["full_experiment_name"])
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        os.makedirs(self.nn_dir, exist_ok=True)
        print("Flat experiment dir:", self.experiment_dir, "num_actors:", self.num_actors)

        self.use_wandb = config.get("wandb_activate", False)
        self._train_t0 = None
        self._loss_window = deque(maxlen=200)
        self.makespan_success = deque(maxlen=int(config.get("makespan_window", 50)))
        self.makespan_all = deque(maxlen=int(config.get("makespan_window", 50)))
        self.success_hist = deque(maxlen=int(config.get("makespan_window", 50)))
        self.episodes_done = 0

    def init_wandb_logger(self) -> None:
        define_shared_metrics(
            rl=True,
            curriculum=False,
            test=bool(self.config.get("test")),
        )

    def _wall_time_sec(self) -> float:
        if self._train_t0 is None:
            return 0.0
        return float(time.time() - self._train_t0)

    def get_epsilon(self) -> float:
        ratio = min(1.0, self.global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def restore(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path

    def _maybe_load_checkpoint(self, env_state_action_dict: dict) -> None:
        ckpt = getattr(self, "_checkpoint_path", None) or self.config.get("load_dir")
        if not ckpt and not self.config.get("load_name"):
            return
        from .tpa_checkpoint import load_flat_checkpoint

        self.act_one_env(env_state_action_dict, epsilon=0.0)
        load_flat_checkpoint(self, self.config, getattr(self, "_checkpoint_path", None))

    def test(self) -> dict:
        assert self.vec_env is not None, "vec_env required for test()"
        seeds = list(self.config.get("test_seeds") or [int(self.config.get("seed", 42))])
        episodes_per_seed = int(self.config.get("test_times", 1))
        eval_epsilon = float(self.config.get("test_epsilon", 0.0))
        output_dir = os.path.join(self.experiment_dir, "eval")

        obs: list[dict] = self.vec_env.reset()
        self._maybe_load_checkpoint(obs[0])

        from .tpa_eval import (
            EvalStream,
            build_eval_payload,
            print_eval_summary,
            run_eval_episodes,
            save_eval_results,
        )

        if self.use_wandb:
            self.init_wandb_logger()

        total_eps = len(seeds) * episodes_per_seed
        log_iv = int(self.config.get("test_progress_log_interval", self.log_interval))
        n_target = int(self.config.get("train_n_products", 16))
        stream = EvalStream(
            output_dir,
            t_budget=self.max_episodic_steps,
            algo_name="flat",
            total_episodes=total_eps,
            n_target=n_target,
            log_interval=log_iv,
            local=None,
            use_wandb=self.use_wandb,
            num_envs=self.num_actors,
        )

        results = run_eval_episodes(
            self.vec_env,
            lambda o, eps: self.act(o, epsilon=eps)[0:2],
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_episodic_steps=self.max_episodic_steps,
            epsilon=eval_epsilon,
            stream=stream,
        )
        payload = build_eval_payload(
            algo_name="flat",
            results=results,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            epsilon=eval_epsilon,
            checkpoint=getattr(self, "_checkpoint_path", None),
        )
        json_path, summary_path = save_eval_results(output_dir, payload)
        print_eval_summary(payload["summary"], "flat")
        print(f"[Flat] eval saved: {json_path}\n[Flat] summary: {summary_path}")
        return payload

    def act_one_env(
        self, env_state_action_dict: dict, epsilon: float, pre: dict | None = None
    ) -> tuple[dict, dict, int | None]:
        action, joint_idx = self.agent.act(env_state_action_dict, epsilon, pre=pre)
        return action, {}, joint_idx

    def act(
        self,
        obs: list[dict],
        epsilon: float | None = None,
        prev_pre_list: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict], list[int | None]]:
        eps = self.get_epsilon() if epsilon is None else epsilon
        actions: list[dict] = []
        actions_extra: list[dict] = []
        joint_indices: list[int | None] = []
        for i, env_state_action_dict in enumerate(obs):
            pre_i = prev_pre_list[i] if prev_pre_list is not None else None
            action, action_extra, joint_idx = self.act_one_env(env_state_action_dict, eps, pre=pre_i)
            actions.append(action)
            actions_extra.append(action_extra)
            joint_indices.append(joint_idx)
        return actions, actions_extra, joint_indices

    def observe_one_env(
        self,
        prev_obs: dict,
        joint_idx: int | None,
        reward: float,
        next_obs: dict,
        done: bool,
        should_learn: bool = False,
    ) -> float | None:
        loss = self.agent.observe_step(
            prev_obs, joint_idx, reward, next_obs, done, learn=should_learn
        )
        if loss is not None:
            self._loss_window.append(float(loss))
        return loss

    def save_checkpoint(self, step: int) -> None:
        enc_path = os.path.join(self.nn_dir, f"state_encoder_step_{step}.pth")
        torch.save(self.obs_encoder.state_dict(), enc_path)
        if self.agent.dqn is not None:
            self.agent.dqn.save(os.path.join(self.nn_dir, f"flat_joint_step_{step}.pth"))

    def train(self) -> None:
        if self.config.get("test"):
            return self.test()
        assert self.vec_env is not None, "vec_env required for train()"
        if self.use_wandb:
            self.init_wandb_logger()
        obs: list[dict] = self.vec_env.reset()
        self._train_t0 = time.time()
        episode_reward = [0.0 for _ in range(len(obs))]
        episode_len = [0 for _ in range(len(obs))]
        # Completed products count within the current episode.
        # Env may reset on done, so accumulate per-step n_product_finished instead of reading progress.finished.
        episode_n_finished = [0 for _ in range(len(obs))]
        prev_pre_list = [self.obs_encoder.preprocess(o) for o in obs]
        last_saved_env_steps = 0
        last_logged_env_steps = 0
        last_learned_env_steps = 0

        while True:
            epsilon = self.get_epsilon()
            actions, actions_extra, joint_indices = self.act(obs, prev_pre_list=prev_pre_list)
            for i, action in enumerate(actions):
                obs[i]["action"] = action

            next_obs = self.vec_env.step(actions, actions_extra)
            self.global_step += 1
            env_step = env_steps(self.global_step, self.num_actors)
            do_learn = crossed_interval(last_learned_env_steps, env_step, self.learn_interval)
            if do_learn:
                last_learned_env_steps = env_step

            for env_id in range(len(obs)):
                reward = compute_team_reward(obs[env_id], next_obs[env_id])
                done, truncated, success = read_rl_done(next_obs[env_id])
                episode_reward[env_id] += reward
                episode_len[env_id] += 1
                rl_step = next_obs[env_id].get("rl") or {}
                episode_n_finished[env_id] += int(rl_step.get("n_product_finished", 0) or 0)
                next_pre = self.obs_encoder.preprocess(next_obs[env_id])
                self.observe_one_env(
                    prev_pre_list[env_id],
                    joint_indices[env_id],
                    reward,
                    next_pre,
                    done=done,
                    should_learn=do_learn,
                )
                prev_pre_list[env_id] = next_pre
                if done:
                    self.episodes_done += 1
                    completed_ep = int(next_obs[env_id].get("episode_num", 0) or 0)
                    ep_len = episode_len[env_id]
                    n_fin = int(episode_n_finished[env_id])
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
                    if self.use_wandb:
                        wall = self._wall_time_sec()
                        payload = axis_payload(env_steps(self.global_step, self.num_actors), wall)
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
                        payload.update(core_payload)
                        payload.update(fullorder_core_metrics(core_payload))
                        wandb.log(payload)
                    episode_reward[env_id] = 0.0
                    episode_len[env_id] = 0
                    episode_n_finished[env_id] = 0
                    _clear_rl(next_obs[env_id])
                    del completed_ep

            obs = next_obs

            if crossed_interval(last_logged_env_steps, env_step, self.log_interval):
                last_logged_env_steps = env_step
                finished = _count_finished(next_obs[0])
                rl0 = next_obs[0].get("rl", {})
                ep0 = int(next_obs[0].get("episode_num", 0) or 0)
                t0 = int(next_obs[0].get("time_step", 0) or 0)
                mean_ms_str = (
                    f"{sum(self.makespan_all)/len(self.makespan_all):.1f}"
                    if self.makespan_all
                    else "n/a"
                )
                joint_valid = self.action_space.num_valid_actions(next_obs[0]) if self.action_space._ready else -1
                wall = self._wall_time_sec()
                spm = steps_per_min(self.global_step, wall, self.num_actors)
                print(
                    f"[Flat] step={env_steps(self.global_step, self.num_actors)} episode={ep0} ep_t={t0} eps={epsilon:.3f} "
                    f"ep_reward0={episode_reward[0]:.2f} finished={finished} ep_done_finished={episode_n_finished[0]} "
                    f"mean_ms={mean_ms_str} joint_valid={joint_valid} rl={rl0} n_envs={len(obs)}"
                )
                if self.use_wandb:
                    loss_mean = sum(self._loss_window) / len(self._loss_window) if self._loss_window else None
                    buf_len = len(self.agent.dqn.buffer) if self.agent.dqn is not None else 0
                    payload = axis_payload(env_steps(self.global_step, self.num_actors), wall, spm)
                    peak_payload = shop_metrics()
                    payload.update(peak_payload)
                    payload.update(fullorder_peak_metrics(peak_payload))
                    payload.update(train_metrics(epsilon=epsilon, buffer_flat=buf_len))
                    if loss_mean is not None:
                        payload["MetricLoss/07_critic_flat"] = loss_mean
                    wandb.log(payload)

            if crossed_interval(last_saved_env_steps, env_step, self.save_interval):
                self.save_checkpoint(env_step)
                last_saved_env_steps = env_step
                print(f"[Flat] checkpoint saved at step {env_step}")
