# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import setproctitle

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
parser.add_argument("--test", action="store_true", default=False, help="load model and test.")
parser.add_argument("--test_times", type=int, default=None, help="test times for one setting.")
parser.add_argument("--test_all_settings", action="store_true", default=False, help="test all settings.")
parser.add_argument("--load_dir", type=str, default=None, help="dir to model checkpoint.")
parser.add_argument("--load_name", type=str, default=None, help="name of model checkpoint.")
parser.add_argument("--wandb_activate", action="store_true", default=None, help="Activate wandb logging.")
parser.add_argument("--wandb_project", type=str, default=None, help="name of wandb project.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--use_fatigue_mask", action="store_true", default=False, help="Use fatigue mask.")
parser.add_argument("--other_filters", action="store_true", default=False, help="Use other filters.")
parser.add_argument("--gantt_chart_data", action="store_true", default=False, help="Generate gantt chart data.")
parser.add_argument(
    "--ftg_thresh_phy",
    type=float,
    default=0.95,
    help="Override the physical fatigue threshold (0-1).",
)
parser.add_argument("--num_particles", type=int, default=500, help="Number of particles for the particle filter.")
parser.add_argument("--measure_noise_sigma", type=float, default=0.00005, help="Noise sigma for the measure noise.")
parser.add_argument(
    "--active_livestream",
    action="store_true",
    default=False,
    help="Activate livestreaming.",
)
parser.add_argument(
    "--livestream_public_ip",
    type=str,
    default=None,
    help="Public IP for Isaac Sim livestream (sets --/app/livestream/publicEndpointAddress). Use with --livestream 2.",
)
parser.add_argument(
    "--livestream_port",
    type=int,
    default=49100,
    help="Port for Isaac Sim livestream (sets --/app/livestream/port).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if getattr(args_cli, "active_livestream", False):
    args_cli.livestream = 2
    # Inject livestream public endpoint and port into extra_args (for isaac-sim.streaming.sh style options)
    if getattr(args_cli, "livestream_public_ip", None):
        os.environ["PUBLIC_IP"] = args_cli.livestream_public_ip
        port = getattr(args_cli, "livestream_port", 49100)
        extra_args = (
            f"--/app/livestream/publicEndpointAddress={args_cli.livestream_public_ip} "
            f"--/app/livestream/port={port}"
        )
        args_cli._livestream_args = (getattr(args_cli, "_livestream_args", None) or "").strip()
        if args_cli._livestream_args:
            args_cli._livestream_args += " " + extra_args
        else:
            args_cli._livestream_args = extra_args

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from source.isaaclab_rl.isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper, RlGamesGpuEnvHRTA, RlGamesVecEnvWrapperHRTA 
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from source.algo.safe_rl import ppolag_filter_dis, rl_filter, ppo_dis, dqn, cpo_filter, rl_filter_mlp, rl_filter_selfattn, rl_filter_no_noisy, rl_filter_no_dueling

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import wandb
# from source.isaaclab_tasks.isaaclab_tasks.direct import human_robot_task_allocation
# from source.isaaclab_tasks.isaaclab_tasks.direct.human_robot_task_allocation.rl_games_env import RlGamesGpuEnvHRTA
from source.isaaclab_tasks.isaaclab_tasks.direct.ergonomic_hrta.eg_hrta_env_cfg import HRTaskAllocEnvCfg
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.hc_env_cfg import HcEnvCfg

@hydra_task_config(args_cli.task, args_cli.algo)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    '''process name'''
    setproctitle.setproctitle("HcFactory")
    '''update args'''
    if args_cli.wandb_activate:
        agent_cfg["params"]["config"]['wandb_activate'] = args_cli.wandb_activate
    if args_cli.test:
        agent_cfg["params"]["config"]['test'] = args_cli.test
    if args_cli.test_times:
        agent_cfg["params"]["config"]['test_times'] = args_cli.test_times
    if args_cli.test_all_settings:
        agent_cfg["params"]["config"]['test_all_settings'] = args_cli.test_all_settings
    if args_cli.load_dir:
        agent_cfg["params"]["config"]['load_dir'] = args_cli.load_dir
    if args_cli.load_name:
        agent_cfg["params"]["config"]['load_name'] = args_cli.load_name
    if args_cli.wandb_project:
        agent_cfg["params"]["config"]['wandb_project'] = args_cli.wandb_project
    if args_cli.use_fatigue_mask:
        agent_cfg["params"]["config"]['use_fatigue_mask'] = args_cli.use_fatigue_mask
    if args_cli.other_filters:
        agent_cfg["params"]["config"]['other_filters'] = args_cli.other_filters
    if args_cli.gantt_chart_data:
        agent_cfg["params"]["config"]['gantt_chart_data'] = args_cli.gantt_chart_data
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["params"]["config"]["device"] = args_cli.device if args_cli.device is not None else agent_cfg["params"]["config"]["device"]
    agent_cfg["params"]["config"]["device_name"] = args_cli.device if args_cli.device is not None else agent_cfg["params"]["config"]["device_name"]
    env_cfg.cuda_device_str = args_cli.device if args_cli.device is not None else env_cfg.cuda_device_str
    if args_cli.ftg_thresh_phy is not None:
        env_cfg.ftg_thresh_phy = args_cli.ftg_thresh_phy
    if args_cli.num_particles is not None:
        env_cfg.num_particles = args_cli.num_particles
    if args_cli.measure_noise_sigma is not None:
        env_cfg.measure_noise_sigma = args_cli.measure_noise_sigma
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["time_str"] = time_str
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", time_str)
    
    if agent_cfg["params"]["config"]["test"]:
        if agent_cfg["params"]["config"]['env_rule_based_exploration']:
            log_dir = 'test_rule_'+ log_dir
        else:
            log_dir= 'test'+ '_'.join(agent_cfg["params"]["config"]['load_name'].split('_')[1:3]) + '_' + agent_cfg["params"]["config"]['load_dir'][-22:-3] + '_' + log_dir
    else:
        log_dir = agent_cfg["params"]["algo"]["name"] + '_' + log_dir
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env_cfg.train_cfg = agent_cfg
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapperHRTA(env, rl_device, clip_obs, clip_actions)
    # env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    
    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    vecenv.register(
        "RlgWrapperHRTA", lambda config_name, num_actors, **kwargs: RlGamesGpuEnvHRTA(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu_HRTA", {"vecenv_type": "RlgWrapperHRTA", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    # runner.algo_factory.register_builder('rainbow', lambda **kwargs: rainbow.RainbowAgent(**kwargs))
    runner.algo_factory.register_builder('rl_filter', lambda **kwargs: rl_filter.SafeRlFilterAgent(**kwargs))
    runner.algo_factory.register_builder('ppolag_filter_dis', lambda **kwargs: ppolag_filter_dis.SafeRlFilterAgentPPO(**kwargs))
    runner.algo_factory.register_builder('ppo_dis', lambda **kwargs: ppo_dis.SafeRlFilterAgentPPO(**kwargs))
    runner.algo_factory.register_builder('dqn', lambda **kwargs: dqn.DqnAgent(**kwargs))
    runner.algo_factory.register_builder('cpo_filter', lambda **kwargs: cpo_filter.SafeRlFilterAgentCPO(**kwargs))
    runner.algo_factory.register_builder('rl_filter_mlp', lambda **kwargs: rl_filter_mlp.SafeRlFilterAgentMLP(**kwargs))
    runner.algo_factory.register_builder('rl_filter_selfattn', lambda **kwargs: rl_filter_selfattn.SafeRlFilterAgentSelfAttention(**kwargs))
    runner.algo_factory.register_builder('rl_filter_no_noisy', lambda **kwargs: rl_filter_no_noisy.SafeRlFilterAgentNoNoisy(**kwargs))
    runner.algo_factory.register_builder('rl_filter_no_dueling', lambda **kwargs: rl_filter_no_dueling.SafeRlFilterAgentNoDueling(**kwargs))
    # runner.algo_factory.register_builder('rainbownoe', lambda **kwargs: rainbownoe.RainbownoeAgent(**kwargs))
    # runner.algo_factory.register_builder('rainbowepsilon', lambda **kwargs: rainbowepsilon.RainbowepsilonAgent(**kwargs))
    # runner.algo_factory.register_builder('epsilon_noisy', lambda **kwargs: epsilon_noisy.EpsilonNoisyAgent(**kwargs))
    # runner.algo_factory.register_builder('no_dueling', lambda **kwargs: no_dueling.NoduelAgent(**kwargs))
    # runner.algo_factory.register_builder('edqn', lambda **kwargs: edqn.RainbowepsilonAgent(**kwargs))

    runner.load(agent_cfg)
    # reset the agent and env
    runner.reset()
    if agent_cfg["params"]["config"]['wandb_activate']:
        if agent_cfg["params"]["config"]["test"]:
            fatigue_str = f"ftg_{args_cli.ftg_thresh_phy}"
            num_particles_str = f"parti_{args_cli.num_particles}"
            measure_noise_sigma_str = f"noise_{args_cli.measure_noise_sigma}"
            if agent_cfg["params"]["config"]['env_rule_based_exploration']:
                run_name = 'test_rule_'+ time_str
            else:
                load_name = agent_cfg["params"]["config"]['load_name'].split('_')[-1][:-4] + '_' + agent_cfg["params"]["config"]['load_dir'][-22:-3]
                run_name = f"test_{agent_cfg['params']['algo']['name']}_{load_name}" + '_' + fatigue_str + '_' + num_particles_str + '_' + measure_noise_sigma_str
        else:
            run_name = f"{agent_cfg['params']['algo']['name']}_{time_str}"

        wandb.init(
            project=agent_cfg["params"]["config"]['wandb_project'],
            group='',
            config=env_cfg.__dict__,
            sync_tensorboard=False,
            name=run_name,
            resume="allow",
        )

    # train the agent
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
