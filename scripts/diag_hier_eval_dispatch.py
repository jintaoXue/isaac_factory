#!/usr/bin/env python3
"""Smoke diagnostic: compare rule vs hier dispatch on first N env steps."""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="HRTPaHC-v1")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--load_dir", type=str, required=True)
parser.add_argument("--load_step", type=int, default=2450000)
parser.add_argument("--train_n_products", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from source.isaaclab_rl.isaaclab_rl.rl_games import (  # noqa: E402
    RlGamesGpuEnvHRTPA,
    RlGamesVecEnvWrapperHRTPA,
)
from source.algo.hierarchical.hc_factory.hierarchical_dispatch import build_rule_based_action  # noqa: E402
from source.algo.hierarchical.hc_factory.hierarchical_tpa import HierarchicalTPA  # noqa: E402
from source.algo.hierarchical.hc_factory.hc_factory_imports import import_hc_module  # noqa: E402

_curr = import_hc_module("src.curriculum")


def _summarize_action(action: dict) -> dict:
    a = action.get("product_sequencing")
    return {
        "A_sum": int(a.sum().item()) if a is not None else 0,
        "disp": len(action.get("dispatch_list") or []),
    }


def _mask_sums(obs: dict) -> dict:
    m = obs.get("agent_action_mask") or {}
    a = m.get("agent_A_product_sequencer")
    progress = obs.get("progress") or {}
    return {
        "A_mask": int(a.sum().item()) if a is not None else -1,
        "not_started": dict(progress.get("not_started") or {}),
        "next_product": progress.get("next_product"),
        "producing": len(progress.get("producing") or []),
    }


@hydra_task_config(args_cli.task, "hier")
def main(env_cfg, algo_cfg):
    rl_device = algo_cfg["params"]["config"].get("device", "cuda:0")
    clip_obs = algo_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = algo_cfg["params"]["env"].get("clip_actions", 1.0)
    if args_cli.headless:
        env_cfg.ui_window_class_type = None
    env_cfg.train_cfg = algo_cfg
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapperHRTPA(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "RlgWrapperHRTPA",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnvHRTPA(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu_HRTPA", {"vecenv_type": "RlgWrapperHRTPA", "env_creator": lambda **kwargs: env}
    )

    cfg = dict(algo_cfg["params"]["config"])
    cfg.update(
        {
            "num_actors": env.unwrapped.num_envs,
            "load_dir": args_cli.load_dir,
            "load_step": int(args_cli.load_step),
            "test": True,
            "train_n_products": int(args_cli.train_n_products),
            "train_dir": "logs/rl_games/HcFactory",
            "full_experiment_name": "diag_hier_eval",
            "wandb_activate": False,
        }
    )
    agent = HierarchicalTPA("diag", {"config": cfg})
    obs_list = agent.vec_env.reset()
    agent.horizon.bind(agent.vec_env, len(obs_list))
    agent.horizon.apply_order_eval(int(args_cli.train_n_products))
    agent._maybe_load_checkpoint(obs_list[0])

    obs = obs_list[0]
    print("[diag] after setup:", _mask_sums(obs))
    print("[diag] dqns ready:", agent.agent_A.dqn is not None, agent.agent_B.dqn is not None)

    for t in range(int(args_cli.steps)):
        rule = build_rule_based_action(obs, agent.cuda_device, max_parallel_cd_dispatch=1)
        hier, _ = agent.act_one_env(obs, epsilon=0.0)
        print(f"step={t:03d} masks={_mask_sums(obs)} rule={_summarize_action(rule)} hier={_summarize_action(hier)}")
        actions = [hier]
        for i, a in enumerate(actions):
            obs_list[i]["action"] = a
        obs_list = agent.vec_env.step(actions, [{}])
        obs = obs_list[0]

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
