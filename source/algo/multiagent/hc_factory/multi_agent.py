# -*- coding: utf-8 -*-
from __future__ import division

import copy
import os

from rl_games.common import vecenv

from .marl_obs import MARLObsEncoder
from .marl_rl_agents import (
    RLHumanRobotAllocatorAgent,
    RLProcessTaskPlanningAgent,
    RLProductSelectionAgent,
    RLProductSequencingAgent,
)
from .marl_utils import compute_team_reward


class MARLMultiAgent:
    """Four-layer hierarchical MARL framework aligned with hc_factory action logic.

    Decision stack (sequential, same as rule_based):
        Agent A — product sequencing
        Agent B — product selection  (conditions on A)
        Agent C — process task planning (conditions on B)
        Agent D — human-robot allocation (conditions on C)

    Each agent is a masked DQN policy that respects env-provided action masks.
    All agents share a team reward based on production progress.
    """

    def __init__(self, base_name, params):
        config = params["config"]
        self.config = config
        self.env_config = config.get("env_config", {})
        self.num_actors = config.get("num_actors", 1)
        self.env_name = config["env_name"]
        print("Env name:", self.env_name)

        self.env_info = config.get("env_info")
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.cuda_device = self.env_info["cuda_device"]
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay_steps = config.get("epsilon_decay_steps", 100000)
        self.learn_interval = config.get("learn_interval", 1)
        self.save_interval = config.get("save_interval", 1000)
        self.log_interval = config.get("log_interval", 100)
        self.global_step = 0

        dqn_kwargs = {
            "hidden_dim": config.get("hidden_dim", 128),
            "lr": config.get("learning_rate", 1e-4),
            "gamma": config.get("gamma", 0.99),
            "buffer_capacity": config.get("replay_buffer_size", 50000),
            "batch_size": config.get("batch_size", 64),
            "target_update_interval": config.get("target_update_interval", 500),
        }

        parallel_limit = config.get("parallel_producing_limit", 5)
        self.obs_encoder = MARLObsEncoder(self.cuda_device, parallel_producing_limit=parallel_limit)

        self.agent_A = RLProductSequencingAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_B = RLProductSelectionAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_C = RLProcessTaskPlanningAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_D = RLHumanRobotAllocatorAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)

        self.train_dir = config.get("train_dir", "runs")
        self.experiment_dir = os.path.join(self.train_dir, config["full_experiment_name"])
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        os.makedirs(self.nn_dir, exist_ok=True)
        print("MARL experiment dir:", self.experiment_dir)

    def get_epsilon(self) -> float:
        ratio = min(1.0, self.global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def act_one_env(self, env_state_action_dict: dict, epsilon: float) -> tuple[dict, dict]:
        """Run A → B → C → D for a single environment instance."""
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
        """Store transitions and trigger learning for all four agents."""
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
        obs: list[dict] = self.vec_env.reset()
        episode_reward = 0.0

        while True:
            epsilon = self.get_epsilon()
            prev_obs_list = copy.deepcopy(obs)

            actions, actions_extra = self.act(obs)
            next_obs = self.vec_env.step(actions, actions_extra)

            for env_id in range(len(obs)):
                reward = compute_team_reward(prev_obs_list[env_id], next_obs[env_id])
                episode_reward += reward
                self.observe_one_env(
                    prev_obs_list[env_id],
                    actions[env_id],
                    reward,
                    next_obs[env_id],
                    done=False,
                    epsilon=epsilon,
                )

            obs = next_obs
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                finished = sum(
                    len(v) for v in next_obs[0]["progress"]["finished"].values()
                )
                print(
                    f"[MARL] step={self.global_step} eps={epsilon:.3f} "
                    f"ep_reward={episode_reward:.2f} finished={finished}"
                )

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(self.global_step)
                print(f"[MARL] checkpoint saved at step {self.global_step}")
