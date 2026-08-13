# -*- coding: utf-8 -*-
"""HierarchicalTPA: env orchestration for hierarchical A→B→C→D agents.

Online: raw ``env_state_action_dict`` ↔ env.
Learning: preprocess once per step (no full-env deepcopy), then Hierarchical agents.
``rl.reward/done`` are written by env ``TaskManager.update_rl_signals``.
"""
from __future__ import division

import os
import time
from collections import deque

import torch
import torch.optim as optim
import wandb
from rl_games.common import vecenv

from .hier_obs import HierObsEncoder
from .hier_rl_agents import (
    RLHumanRobotAllocatorAgent,
    RLProcessTaskPlanningAgent,
    RLProductSelectionAgent,
    RLProductSequencingAgent,
)
from .hier_utils import compute_team_reward, read_rl_done


def _clear_rl(env_dict: dict) -> None:
    """Drop stale terminal flags after TPA has consumed them (env already reset)."""
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


def _buffer_len(agent) -> int:
    dqn = getattr(agent, "dqn", None)
    if dqn is None or getattr(dqn, "buffer", None) is None:
        return 0
    return len(dqn.buffer)


class HierarchicalTPA:
    """Four-layer hierarchical decision stack (Masked DQN backend first; swappable later).

    Decision order (same as rule_based):
        A product sequencing → B product selection → C task planning → D allocation
    """

    def __init__(self, base_name, params):
        config = params["config"]
        self.config = config
        self.env_config = config.get("env_config", {})
        # Align with HcVectorEnvCfg.scene.num_envs (default 3)
        self.num_actors = int(config.get("num_actors", 3))
        self.env_name = config["env_name"]
        print("Env name:", self.env_name)

        self.env_info = config.get("env_info")
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = None

        # Prefer live env count if wrapper exposes it
        if self.vec_env is not None:
            n = getattr(self.vec_env, "num_envs", None)
            if n is None and hasattr(self.vec_env, "env_list"):
                n = len(self.vec_env.env_list)
            if n is None and isinstance(self.env_info, dict):
                n = self.env_info.get("num_envs")
            if n is not None and int(n) != self.num_actors:
                print(
                    f"[Hier] warn: num_actors={self.num_actors} != env.num_envs={n}; "
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
        self.max_episodic_steps = int(config.get("max_episodic_steps", 45000))

        dqn_kwargs = {
            "hidden_dim": config.get("hidden_dim", 128),
            "lr": config.get("learning_rate", 1e-4),
            "gamma": config.get("gamma", 0.99),
            "buffer_capacity": config.get("replay_buffer_size", 50000),
            "batch_size": config.get("batch_size", 64),
            "target_update_interval": config.get("target_update_interval", 500),
        }

        parallel_limit = config.get("parallel_producing_limit", 10)
        self.max_parallel_cd_dispatch = int(config.get("max_parallel_cd_dispatch", 1))
        self.max_sim_episodes = config.get("max_sim_episodes")
        if self.max_sim_episodes is not None:
            self.max_sim_episodes = int(self.max_sim_episodes)
        self.episodes_done = 0
        self._train_t0 = None
        self.peak_producing = 0
        self.peak_ongoing = 0
        self._ep_peak_producing: list[int] = []
        self._ep_peak_ongoing: list[int] = []
        self.obs_encoder = HierObsEncoder(
            self.cuda_device,
            parallel_producing_limit=parallel_limit,
            state_dim=int(config.get("state_dim", 256)),
        )
        encoder_lr = float(config.get("encoder_learning_rate", config.get("learning_rate", 1e-4)))
        self.encoder_optimizer = optim.Adam(self.obs_encoder.parameters(), lr=encoder_lr)

        self.agent_A = RLProductSequencingAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_B = RLProductSelectionAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_C = RLProcessTaskPlanningAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_D = RLHumanRobotAllocatorAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)

        self.train_dir = config.get("train_dir", "runs")
        self.experiment_dir = os.path.join(self.train_dir, config["full_experiment_name"])
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        os.makedirs(self.nn_dir, exist_ok=True)
        print("Hierarchical experiment dir:", self.experiment_dir, "num_actors:", self.num_actors)

        self.use_wandb = config.get("wandb_activate", False)
        # DQN has no actor loss; rolling TD (critic) losses + makespan windows
        self._loss_window = {
            "A": deque(maxlen=200),
            "B": deque(maxlen=200),
            "C": deque(maxlen=200),
            "D_human": deque(maxlen=200),
            "D_robot": deque(maxlen=200),
        }
        self.makespan_success = deque(maxlen=int(config.get("makespan_window", 50)))
        self.makespan_all = deque(maxlen=int(config.get("makespan_window", 50)))
        if self.use_wandb:
            self.init_wandb_logger()

    def init_wandb_logger(self):
        """Mirror rl_filter / rainbow metric namespace for live training charts."""
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/wall_time_sec", step_metric="Train/step")
        wandb.define_metric("Train/wall_time_min", step_metric="Train/step")
        wandb.define_metric("Train/epsilon", step_metric="Train/step")
        wandb.define_metric("Train/ep_reward0", step_metric="Train/step")
        wandb.define_metric("Train/finished0", step_metric="Train/step")
        wandb.define_metric("Train/episode0", step_metric="Train/step")
        wandb.define_metric("Train/ep_t0", step_metric="Train/step")
        wandb.define_metric("Train/step_reward0", step_metric="Train/step")
        wandb.define_metric("Train/r_step0", step_metric="Train/step")
        wandb.define_metric("Train/r_finish0", step_metric="Train/step")
        wandb.define_metric("Train/r_task0", step_metric="Train/step")
        wandb.define_metric("Train/r_success0", step_metric="Train/step")
        wandb.define_metric("Train/buffer_A", step_metric="Train/step")
        wandb.define_metric("Train/buffer_B", step_metric="Train/step")
        wandb.define_metric("Train/buffer_C", step_metric="Train/step")
        wandb.define_metric("Train/buffer_D_human", step_metric="Train/step")
        wandb.define_metric("Train/buffer_D_robot", step_metric="Train/step")
        wandb.define_metric("Train/producing0", step_metric="Train/step")
        wandb.define_metric("Train/ongoing0", step_metric="Train/step")
        wandb.define_metric("Train/peak_producing", step_metric="Train/step")
        wandb.define_metric("Train/peak_ongoing", step_metric="Train/step")
        # DQN TD loss ≈ critic loss (no separate actor head in current Hierarchical)
        for name in ("A", "B", "C", "D_human", "D_robot"):
            wandb.define_metric(f"Loss/critic_{name}", step_metric="Train/step")
        wandb.define_metric("Loss/critic_mean", step_metric="Train/step")
        wandb.define_metric("Metrics/MeanMakespan", step_metric="Train/step")
        wandb.define_metric("Metrics/MeanMakespan_success", step_metric="Train/step")

        wandb.define_metric("Metrics/wall_time_sec")
        wandb.define_metric("Metrics/step_episode", step_metric="Train/step")
        for key in (
            "Metrics/EpRet",
            "Metrics/EpLen",
            "Metrics/EpMakespan",
            "Metrics/EpSuccess",
            "Metrics/EpTruncated",
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
        while env_id >= len(self._ep_peak_producing):
            self._ep_peak_producing.append(0)
            self._ep_peak_ongoing.append(0)
        self._ep_peak_producing[env_id] = max(self._ep_peak_producing[env_id], n_producing)
        self._ep_peak_ongoing[env_id] = max(self._ep_peak_ongoing[env_id], n_ongoing)
        self.peak_producing = max(self.peak_producing, n_producing)
        self.peak_ongoing = max(self.peak_ongoing, n_ongoing)
        return n_producing, n_ongoing

    def get_epsilon(self) -> float:
        ratio = min(1.0, self.global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def restore(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path

    def _maybe_load_checkpoint(self, env_state_action_dict: dict) -> None:
        ckpt = getattr(self, "_checkpoint_path", None) or self.config.get("load_dir")
        if not ckpt and not self.config.get("load_name"):
            return
        from .tpa_checkpoint import load_hier_checkpoint

        self.act_one_env(env_state_action_dict, epsilon=0.0)
        load_hier_checkpoint(self, self.config, getattr(self, "_checkpoint_path", None))

    def test(self) -> dict:
        assert self.vec_env is not None, "vec_env required for test()"
        seeds = list(self.config.get("test_seeds") or [int(self.config.get("seed", 42))])
        episodes_per_seed = int(self.config.get("test_times", 1))
        eval_epsilon = float(self.config.get("test_epsilon", 0.0))
        output_dir = os.path.join(self.experiment_dir, "eval")

        obs: list[dict] = self.vec_env.reset()
        self._maybe_load_checkpoint(obs[0])

        from .tpa_eval import (
            build_eval_payload,
            print_eval_summary,
            run_eval_episodes,
            save_eval_results,
        )

        results = run_eval_episodes(
            self.vec_env,
            lambda o, eps: self.act(o, epsilon=eps)[0:2],
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_episodic_steps=self.max_episodic_steps,
            epsilon=eval_epsilon,
        )
        payload = build_eval_payload(
            algo_name="hier",
            results=results,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            epsilon=eval_epsilon,
            checkpoint=getattr(self, "_checkpoint_path", None),
        )
        json_path, summary_path = save_eval_results(output_dir, payload)
        print_eval_summary(payload["summary"], "hier")
        print(f"[Hier] eval saved: {json_path}\n[Hier] summary: {summary_path}")
        return payload

    def act_one_env(self, env_state_action_dict: dict, epsilon: float) -> tuple[dict, dict]:
        from .hierarchical_dispatch import build_hier_rl_action

        action = build_hier_rl_action(
            env_state_action_dict,
            self.cuda_device,
            self,
            epsilon,
            max_parallel_cd_dispatch=self.max_parallel_cd_dispatch,
        )
        return action, {}

    def act(self, obs: list[dict], epsilon: float | None = None) -> tuple[list[dict], list[dict]]:
        eps = self.get_epsilon() if epsilon is None else epsilon
        actions: list[dict] = []
        actions_extra: list[dict] = []
        for env_state_action_dict in obs:
            action, action_extra = self.act_one_env(env_state_action_dict, eps)
            actions.append(action)
            actions_extra.append(action_extra)
        return actions, actions_extra

    def _had_meaningful_decision(self, action: dict) -> bool:
        if action.get("dispatch_list"):
            return True
        product_sequencing = action.get("product_sequencing")
        return isinstance(product_sequencing, torch.Tensor) and product_sequencing.sum() > 0

    def _joint_learn(self, entries: list[tuple[str, torch.Tensor, object | None]]) -> dict[str, float]:
        """Sum TD losses, backprop through shared encoder + per-agent Q heads."""
        if not entries:
            return {}

        q_optimizers = []
        dqns = []
        for _name, _loss, dqn in entries:
            if dqn is None:
                continue
            dqn.optimizer.zero_grad()
            q_optimizers.append(dqn.optimizer)
            dqns.append(dqn)

        self.encoder_optimizer.zero_grad()
        total_loss = sum(loss for _name, loss, _dqn in entries)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), 1.0)
        self.encoder_optimizer.step()
        for optimizer in q_optimizers:
            optimizer.step()
        for dqn in dqns:
            dqn.register_train_step()

        return {name: float(loss.detach().item()) for name, loss, _dqn in entries}

    def observe_one_env(
        self,
        prev_obs: dict,
        action: dict,
        reward: float,
        next_obs: dict,
        done: bool,
        epsilon: float,
    ) -> dict[str, float]:
        """Store transitions and learn. Returns per-agent DQN (critic) losses."""
        losses: dict[str, float] = {}
        if self.global_step % self.learn_interval != 0:
            return losses
        if not self._had_meaningful_decision(action):
            return losses

        pending: list[tuple[str, torch.Tensor, object | None]] = []

        loss_a = self.agent_A.observe_step(
            prev_obs, action["product_sequencing"], reward, next_obs, done, epsilon
        )
        if loss_a is not None:
            pending.append(("A", loss_a, self.agent_A.dqn))

        loss_b = self.agent_B.observe_step(
            prev_obs,
            action["product_sequencing"],
            action["product_selection"],
            reward,
            next_obs,
            done,
            epsilon,
        )
        if loss_b is not None:
            pending.append(("B", loss_b, self.agent_B.dqn))

        loss_c = self.agent_C.observe_step(
            prev_obs,
            action["product_selection"],
            action["process_task_planning"],
            reward,
            next_obs,
            done,
            epsilon,
        )
        if loss_c is not None:
            pending.append(("C", loss_c, self.agent_C.dqn))

        loss_d_h, loss_d_r = self.agent_D.observe_step(
            prev_obs,
            action["process_task_planning"],
            action["human_robot_allocation"],
            reward,
            next_obs,
            done,
            epsilon,
        )
        if loss_d_h is not None:
            pending.append(("D_human", loss_d_h, self.agent_D.human_dqn))
        if loss_d_r is not None:
            pending.append(("D_robot", loss_d_r, self.agent_D.robot_dqn))

        losses = self._joint_learn(pending)
        for key, value in losses.items():
            self._loss_window[key].append(value)
        return losses

    def save_checkpoint(self, step: int) -> None:
        enc_path = os.path.join(self.nn_dir, f"state_encoder_step_{step}.pth")
        import torch

        torch.save(self.obs_encoder.state_dict(), enc_path)

        for agent, name in [
            (self.agent_A, "agent_A"),
            (self.agent_B, "agent_B"),
            (self.agent_C, "agent_C"),
        ]:
            if agent.dqn is not None:
                agent.dqn.save(os.path.join(self.nn_dir, f"{name}_step_{step}.pth"))
        if self.agent_D.human_dqn is not None:
            self.agent_D.human_dqn.save(os.path.join(self.nn_dir, f"agent_D_human_step_{step}.pth"))
        if self.agent_D.robot_dqn is not None:
            self.agent_D.robot_dqn.save(os.path.join(self.nn_dir, f"agent_D_robot_step_{step}.pth"))

    def train(self):
        if self.config.get("test"):
            return self.test()
        assert self.vec_env is not None, "vec_env required for train()"
        obs: list[dict] = self.vec_env.reset()
        episode_reward = [0.0 for _ in range(len(obs))]
        episode_len = [0 for _ in range(len(obs))]
        self._ep_peak_producing = [0 for _ in range(len(obs))]
        self._ep_peak_ongoing = [0 for _ in range(len(obs))]
        self._train_t0 = time.time()
        stop = False

        while not stop:
            epsilon = self.get_epsilon()
            # Fixed-shape tensor snapshot (no articulations / route lists deepcopy)
            prev_pre_list = [self.obs_encoder.preprocess(o) for o in obs]

            actions, actions_extra = self.act(obs)
            for i, a in enumerate(actions):
                obs[i]["action"] = a

            # Env writes rl and resets itself when done
            next_obs = self.vec_env.step(actions, actions_extra)

            for env_id in range(len(obs)):
                self._update_concurrency_peaks(env_id, next_obs[env_id])
                reward = compute_team_reward(obs[env_id], next_obs[env_id])
                done, truncated, success = read_rl_done(next_obs[env_id])
                episode_reward[env_id] += reward
                episode_len[env_id] += 1
                next_pre = self.obs_encoder.preprocess(next_obs[env_id])
                self.observe_one_env(
                    prev_pre_list[env_id],
                    actions[env_id],
                    reward,
                    next_pre,
                    done=done,
                    epsilon=epsilon,
                )
                if done:
                    self.episodes_done += 1
                    # env already reset: episode_num in next_obs is the *new* episode index
                    completed_ep = int(next_obs[env_id].get("episode_num", 0) or 0)
                    del completed_ep
                    ep_len = episode_len[env_id]
                    ep_max_prod = self._ep_peak_producing[env_id]
                    ep_max_ong = self._ep_peak_ongoing[env_id]
                    wall = self._wall_time_sec()
                    # makespan: success → actual length; truncated → count as horizon
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
                    self._ep_peak_producing[env_id] = 0
                    self._ep_peak_ongoing[env_id] = 0
                    _clear_rl(next_obs[env_id])
                    if (
                        self.max_sim_episodes is not None
                        and self.episodes_done >= self.max_sim_episodes
                    ):
                        stop = True

            obs = next_obs
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                finished = _count_finished(next_obs[0])
                rl0 = next_obs[0].get("rl", {})
                progress0 = next_obs[0].get("progress") or {}
                producing0 = len(progress0.get("producing") or [])
                ongoing0 = len(progress0.get("ongoing_task_records") or {})
                ep0 = int(next_obs[0].get("episode_num", 0) or 0)
                t0 = int(next_obs[0].get("time_step", 0) or 0)
                step_r0 = float((rl0 or {}).get("reward", 0.0) or 0.0)
                mean_ms_str = (
                    f"{sum(self.makespan_all)/len(self.makespan_all):.1f}"
                    if self.makespan_all
                    else "n/a"
                )
                print(
                    f"[Hier] step={self.global_step} episode={ep0} ep_t={t0} eps={epsilon:.3f} "
                    f"ep_reward0={episode_reward[0]:.2f} finished={finished} "
                    f"producing={producing0} ongoing={ongoing0} "
                    f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing} "
                    f"mean_ms={mean_ms_str} rl={rl0} n_envs={len(obs)}"
                )
                if self.use_wandb:
                    wall = self._wall_time_sec()
                    buf_d_h = (
                        len(self.agent_D.human_dqn.buffer)
                        if self.agent_D.human_dqn is not None
                        else 0
                    )
                    buf_d_r = (
                        len(self.agent_D.robot_dqn.buffer)
                        if self.agent_D.robot_dqn is not None
                        else 0
                    )
                    parts = (rl0 or {}).get("reward_parts") or {}
                    loss_payload = {}
                    critic_vals = []
                    for name, window in self._loss_window.items():
                        if window:
                            m = sum(window) / len(window)
                            loss_payload[f"Loss/critic_{name}"] = m
                            critic_vals.append(m)
                    if critic_vals:
                        loss_payload["Loss/critic_mean"] = sum(critic_vals) / len(critic_vals)
                    mean_ms = (
                        sum(self.makespan_all) / len(self.makespan_all) if self.makespan_all else None
                    )
                    mean_ms_ok = (
                        sum(self.makespan_success) / len(self.makespan_success)
                        if self.makespan_success
                        else None
                    )
                    payload = {
                        "Train/step": self.global_step,
                        "Train/wall_time_sec": wall,
                        "Train/wall_time_min": wall / 60.0,
                        "Train/epsilon": epsilon,
                        "Train/ep_reward0": episode_reward[0],
                        "Train/finished0": finished,
                        "Train/episode0": ep0,
                        "Train/ep_t0": t0,
                        "Train/step_reward0": step_r0,
                        "Train/r_step0": float(parts.get("step", 0.0) or 0.0),
                        "Train/r_finish0": float(parts.get("finish", 0.0) or 0.0),
                        "Train/r_task0": float(parts.get("task", 0.0) or 0.0),
                        "Train/r_success0": float(parts.get("success", 0.0) or 0.0),
                        "Train/buffer_A": _buffer_len(self.agent_A),
                        "Train/buffer_B": _buffer_len(self.agent_B),
                        "Train/buffer_C": _buffer_len(self.agent_C),
                        "Train/buffer_D_human": buf_d_h,
                        "Train/buffer_D_robot": buf_d_r,
                        "Train/producing0": producing0,
                        "Train/ongoing0": ongoing0,
                        "Train/peak_producing": self.peak_producing,
                        "Train/peak_ongoing": self.peak_ongoing,
                    }
                    payload.update(loss_payload)
                    if mean_ms is not None:
                        payload["Metrics/MeanMakespan"] = mean_ms
                    if mean_ms_ok is not None:
                        payload["Metrics/MeanMakespan_success"] = mean_ms_ok
                    wandb.log(payload)

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(self.global_step)
                print(f"[Hier] checkpoint saved at step {self.global_step}")

        wall = self._wall_time_sec()
        print(
            f"[Hier] train finished episodes_done={self.episodes_done} "
            f"steps={self.global_step} wall={wall:.1f}s ({wall/60.0:.2f} min) "
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
