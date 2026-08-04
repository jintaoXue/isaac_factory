# -*- coding: utf-8 -*-
"""MultiAgentTPA: env orchestration for hierarchical A→B→C→D agents.

Online: raw ``env_state_action_dict`` ↔ env.
Learning: preprocess once per step (no full-env deepcopy), then MARL agents.
``rl.reward/done`` are written by env ``TaskManager.update_rl_signals``.
"""
from __future__ import division

import os

import wandb
from rl_games.common import vecenv

from .marl_obs import MARLObsEncoder
from .marl_rl_agents import (
    RLHumanRobotAllocatorAgent,
    RLProcessTaskPlanningAgent,
    RLProductSelectionAgent,
    RLProductSequencingAgent,
)
from .marl_utils import compute_team_reward, read_rl_done


def _clear_rl(env_dict: dict) -> None:
    """Drop stale terminal flags after TPA has consumed them (env already reset)."""
    env_dict["rl"] = {
        "reward": 0.0,
        "done": False,
        "truncated": False,
        "success": False,
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


class MultiAgentTPA:
    """Four-layer hierarchical decision stack (MARL backend first; swappable later).

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
                    f"[MARL] warn: num_actors={self.num_actors} != env.num_envs={n}; "
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
        self.max_episode_steps = int(config.get("max_episode_steps", 4000))

        dqn_kwargs = {
            "hidden_dim": config.get("hidden_dim", 128),
            "lr": config.get("learning_rate", 1e-4),
            "gamma": config.get("gamma", 0.99),
            "buffer_capacity": config.get("replay_buffer_size", 50000),
            "batch_size": config.get("batch_size", 64),
            "target_update_interval": config.get("target_update_interval", 500),
        }

        parallel_limit = config.get("parallel_producing_limit", 5)
        self.obs_encoder = MARLObsEncoder(
            self.cuda_device,
            parallel_producing_limit=parallel_limit,
            state_dim=int(config.get("state_dim", 256)),
        )

        self.agent_A = RLProductSequencingAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_B = RLProductSelectionAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_C = RLProcessTaskPlanningAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_D = RLHumanRobotAllocatorAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)

        self.train_dir = config.get("train_dir", "runs")
        self.experiment_dir = os.path.join(self.train_dir, config["full_experiment_name"])
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        os.makedirs(self.nn_dir, exist_ok=True)
        print("MARL experiment dir:", self.experiment_dir, "num_actors:", self.num_actors)

        self.use_wandb = config.get("wandb_activate", False)
        if self.use_wandb:
            self.init_wandb_logger()

    def init_wandb_logger(self):
        """Mirror rl_filter / rainbow metric namespace for live training charts."""
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/epsilon", step_metric="Train/step")
        wandb.define_metric("Train/ep_reward0", step_metric="Train/step")
        wandb.define_metric("Train/finished0", step_metric="Train/step")
        wandb.define_metric("Train/episode0", step_metric="Train/step")
        wandb.define_metric("Train/ep_t0", step_metric="Train/step")
        wandb.define_metric("Train/step_reward0", step_metric="Train/step")
        wandb.define_metric("Train/buffer_A", step_metric="Train/step")
        wandb.define_metric("Train/buffer_B", step_metric="Train/step")
        wandb.define_metric("Train/buffer_C", step_metric="Train/step")
        wandb.define_metric("Train/buffer_D_human", step_metric="Train/step")
        wandb.define_metric("Train/buffer_D_robot", step_metric="Train/step")

        wandb.define_metric("Metrics/step_episode", step_metric="Train/step")
        wandb.define_metric("Metrics/EpRet", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpLen", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpSuccess", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpTruncated", step_metric="Metrics/step_episode")

    def get_epsilon(self) -> float:
        ratio = min(1.0, self.global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def act_one_env(self, env_state_action_dict: dict, epsilon: float) -> tuple[dict, dict]:
        """A→B→C→D on one raw env dict; returns action dict to write back."""
        product_sequencing = self.agent_A.act(env_state_action_dict, epsilon)
        product_selection = self.agent_B.act(env_state_action_dict, product_sequencing, epsilon)
        process_task_planning = self.agent_C.act(env_state_action_dict, product_selection, epsilon)
        human_robot_allocation = self.agent_D.act(
            env_state_action_dict, product_selection, process_task_planning, epsilon
        )
        action = {
            "product_sequencing": product_sequencing,
            "product_selection": product_selection,
            "process_task_planning": process_task_planning,
            "human_robot_allocation": human_robot_allocation,
        }
        return action, {}

    def act(self, obs: list[dict]) -> tuple[list[dict], list[dict]]:
        epsilon = self.get_epsilon()
        actions: list[dict] = []
        actions_extra: list[dict] = []
        for env_state_action_dict in obs:
            action, action_extra = self.act_one_env(env_state_action_dict, epsilon)
            actions.append(action)
            actions_extra.append(action_extra)
        return actions, actions_extra

    def observe_one_env(
        self,
        prev_obs: dict,
        action: dict,
        reward: float,
        next_obs: dict,
        done: bool,
        epsilon: float,
    ) -> None:
        """Store transitions and learn. ``prev_obs`` / ``next_obs`` may be preprocessed dicts."""
        if self.global_step % self.learn_interval != 0:
            return

        self.agent_A.observe_step(
            prev_obs, action["product_sequencing"], reward, next_obs, done, epsilon
        )
        self.agent_B.observe_step(
            prev_obs,
            action["product_sequencing"],
            action["product_selection"],
            reward,
            next_obs,
            done,
            epsilon,
        )
        self.agent_C.observe_step(
            prev_obs,
            action["product_selection"],
            action["process_task_planning"],
            reward,
            next_obs,
            done,
            epsilon,
        )
        self.agent_D.observe_step(
            prev_obs,
            action["process_task_planning"],
            action["human_robot_allocation"],
            reward,
            next_obs,
            done,
            epsilon,
        )

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
        assert self.vec_env is not None, "vec_env required for train()"
        obs: list[dict] = self.vec_env.reset()
        episode_reward = [0.0 for _ in range(len(obs))]
        episode_len = [0 for _ in range(len(obs))]

        while True:
            epsilon = self.get_epsilon()
            # Fixed-shape tensor snapshot (no articulations / route lists deepcopy)
            prev_pre_list = [self.obs_encoder.preprocess(o) for o in obs]

            actions, actions_extra = self.act(obs)
            for i, a in enumerate(actions):
                obs[i]["action"] = a

            # Env writes rl and resets itself when done
            next_obs = self.vec_env.step(actions, actions_extra)

            for env_id in range(len(obs)):
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
                    # env already reset: episode_num in next_obs is the *new* episode index
                    completed_ep = int(next_obs[env_id].get("episode_num", 0) or 0)
                    if self.use_wandb:
                        wandb.log(
                            {
                                "Train/step": self.global_step,
                                "Metrics/step_episode": completed_ep,
                                "Metrics/EpRet": episode_reward[env_id],
                                "Metrics/EpLen": episode_len[env_id],
                                "Metrics/EpSuccess": float(success),
                                "Metrics/EpTruncated": float(truncated),
                            }
                        )
                    episode_reward[env_id] = 0.0
                    episode_len[env_id] = 0
                    _clear_rl(next_obs[env_id])

            obs = next_obs
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                finished = _count_finished(next_obs[0])
                rl0 = next_obs[0].get("rl", {})
                ep0 = int(next_obs[0].get("episode_num", 0) or 0)
                t0 = int(next_obs[0].get("time_step", 0) or 0)
                step_r0 = float((rl0 or {}).get("reward", 0.0) or 0.0)
                print(
                    f"[MARL] step={self.global_step} episode={ep0} ep_t={t0} eps={epsilon:.3f} "
                    f"ep_reward0={episode_reward[0]:.2f} finished={finished} "
                    f"rl={rl0} n_envs={len(obs)}"
                )
                if self.use_wandb:
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
                    wandb.log(
                        {
                            "Train/step": self.global_step,
                            "Train/epsilon": epsilon,
                            "Train/ep_reward0": episode_reward[0],
                            "Train/finished0": finished,
                            "Train/episode0": ep0,
                            "Train/ep_t0": t0,
                            "Train/step_reward0": step_r0,
                            "Train/buffer_A": _buffer_len(self.agent_A),
                            "Train/buffer_B": _buffer_len(self.agent_B),
                            "Train/buffer_C": _buffer_len(self.agent_C),
                            "Train/buffer_D_human": buf_d_h,
                            "Train/buffer_D_robot": buf_d_r,
                        }
                    )

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(self.global_step)
                print(f"[MARL] checkpoint saved at step {self.global_step}")
