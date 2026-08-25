# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
from isaaclab.app import AppLauncher
import setproctitle

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=200, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
parser.add_argument("--test", action="store_true", default=False, help="Run evaluation (Makespan / Success / Truncation) instead of training.")
parser.add_argument("--test_times", type=int, default=None, help="Episodes per seed during --test.")
parser.add_argument("--test_seeds", type=str, default=None, help="Comma-separated seeds for --test, e.g. 42,43,44,45,46.")
parser.add_argument("--test_all_settings", action="store_true", default=False, help="test all settings.")
parser.add_argument("--load_dir", type=str, default=None, help="dir to model checkpoint.")
parser.add_argument("--load_name", type=str, default=None, help="name of model checkpoint.")
parser.add_argument("--wandb_activate", action="store_true", default=None, help="Activate wandb logging.")
parser.add_argument("--wandb_project", type=str, default=None, help="name of wandb project.")
parser.add_argument("--wandb_name", type=str, default=None, help="Optional wandb run name override.")
parser.add_argument(
    "--max_parallel_cd_dispatch",
    type=int,
    default=None,
    help="Max C/D dispatches per step (1=single-product, >1=multi-product).",
)
parser.add_argument(
    "--explore",
    action="store_true",
    default=False,
    help="Masked-random catalog collection (hier, epsilon=1, no DQN backward).",
)
parser.add_argument(
    "--warmstart",
    type=str,
    default=None,
    help="Path to an env checkpoint pkl (Tier-A).",
)
parser.add_argument(
    "--curriculum",
    action="store_true",
    default=False,
    help="Enable product-count curriculum (reverse stages, training target=10).",
)
parser.add_argument(
    "--max_sim_episodes",
    type=int,
    default=None,
    help="Max environment simulation episodes (stop after this many completed rounds; rule_based / hier).",
)
parser.add_argument(
    "--train_n_products",
    type=int,
    default=None,
    help="Rule train order size (10 or 16; default from yaml).",
)
parser.add_argument(
    "--explore_n_products",
    type=int,
    default=None,
    help="Explore / random baseline order size (10 or 16; default 16).",
)
parser.add_argument(
    "--no_explore_save_catalog",
    action="store_true",
    default=False,
    help="Explore mode: do not write env_checkpoints/random_explore catalog pkls.",
)
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


def _set_hc_process_title(algo: str | None = None, device: str | None = None) -> None:
    """nvidia-smi / ps display: HcFactory-<algo>-xjt cuda:N"""
    algo_tag = algo or args_cli.algo or "unknown"
    device_tag = device or args_cli.device or "cuda:0"
    setproctitle.setproctitle(f"HcFactory-{algo_tag}-xjt {device_tag}")


_set_hc_process_title()


def _has_registered_cameras() -> bool:
    """Load cfg_camera without importing isaaclab_tasks (isaacsim not ready yet)."""
    import importlib.util

    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/perception/cfg_camera.py",
    )
    spec = importlib.util.spec_from_file_location("_hc_cfg_camera", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.has_registered_cameras()


_enable_cameras = bool(
    args_cli.video or (getattr(args_cli, "enable_cameras", False) and _has_registered_cameras())
)
if _enable_cameras:
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
import pickle
import random
from datetime import datetime

from source.isaaclab_rl.isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper, RlGamesGpuEnvHRTPA, RlGamesVecEnvWrapperHRTPA 
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from source.algo.hierarchical.hc_factory import flat_tpa
from source.algo.hierarchical.hc_factory import hierarchical_tpa
from source.algo.hierarchical.hc_factory import rule_based

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml


def dump_pickle(filename: str, data) -> None:
    """Save config snapshot; Isaac Lab 2.3+ no longer exports dump_pickle."""
    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import wandb

from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.env_asset_cfg.cfg_hc_env import HcVectorEnvCfg
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.hc_render import HcVideoRecorder

@hydra_task_config(args_cli.task, args_cli.algo)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, algo_cfg: dict):

    '''process name: HcFactory-<algo>-xjt cuda:N (visible in nvidia-smi)'''
    _set_hc_process_title(args_cli.algo, args_cli.device)
    '''update args'''
    if args_cli.wandb_activate:
        algo_cfg["params"]["config"]['wandb_activate'] = args_cli.wandb_activate
    if args_cli.test:
        algo_cfg["params"]["config"]['test'] = args_cli.test
    if args_cli.test_times:
        algo_cfg["params"]["config"]['test_times'] = args_cli.test_times
    if args_cli.test_seeds:
        algo_cfg["params"]["config"]["test_seeds"] = [
            int(s.strip()) for s in args_cli.test_seeds.split(",") if s.strip()
        ]
    if args_cli.test_all_settings:
        algo_cfg["params"]["config"]['test_all_settings'] = args_cli.test_all_settings
    if args_cli.load_dir:
        algo_cfg["params"]["config"]['load_dir'] = args_cli.load_dir
    if args_cli.load_name:
        algo_cfg["params"]["config"]['load_name'] = args_cli.load_name
    if args_cli.wandb_project:
        algo_cfg["params"]["config"]['wandb_project'] = args_cli.wandb_project
    if args_cli.wandb_name:
        algo_cfg["params"]["config"]["wandb_name"] = args_cli.wandb_name
    if args_cli.max_parallel_cd_dispatch is not None:
        algo_cfg["params"]["config"]["max_parallel_cd_dispatch"] = int(args_cli.max_parallel_cd_dispatch)
    if args_cli.max_sim_episodes is not None:
        algo_cfg["params"]["config"]["max_sim_episodes"] = int(args_cli.max_sim_episodes)
    if getattr(args_cli, "explore", False):
        algo_cfg["params"]["config"]["explore"] = True
        algo_cfg["params"]["config"]["explore_catalog"] = True
    if getattr(args_cli, "train_n_products", None) is not None:
        algo_cfg["params"]["config"]["train_n_products"] = int(args_cli.train_n_products)
    if getattr(args_cli, "explore_n_products", None) is not None:
        algo_cfg["params"]["config"]["explore_n_products"] = int(args_cli.explore_n_products)
    if getattr(args_cli, "no_explore_save_catalog", False):
        algo_cfg["params"]["config"]["explore_save_catalog"] = False
    warmstart = (getattr(args_cli, "warmstart", None) or "").strip() or os.environ.get("HC_WARMSTART", "").strip()
    if warmstart:
        algo_cfg["params"]["config"]["warmstart"] = warmstart
    if getattr(args_cli, "curriculum", False):
        algo_cfg["params"]["config"]["curriculum"] = True
    if args_cli.use_fatigue_mask:
        algo_cfg["params"]["config"]['use_fatigue_mask'] = args_cli.use_fatigue_mask
    if args_cli.other_filters:
        algo_cfg["params"]["config"]['other_filters'] = args_cli.other_filters
    if args_cli.gantt_chart_data:
        algo_cfg["params"]["config"]['gantt_chart_data'] = args_cli.gantt_chart_data
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    algo_cfg["params"]["config"]["device"] = args_cli.device if args_cli.device is not None else algo_cfg["params"]["config"]["device"]
    algo_cfg["params"]["config"]["device_name"] = args_cli.device if args_cli.device is not None else algo_cfg["params"]["config"]["device_name"]
    env_cfg.cuda_device_str = args_cli.device if args_cli.device is not None else env_cfg.cuda_device_str
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    algo_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else algo_cfg["params"]["seed"]
    # Legacy RL algos read max_epochs; rule/hier/flat use max_sim_episodes instead.
    if args_cli.max_iterations is not None:
        algo_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        algo_cfg["params"]["load_checkpoint"] = True
        algo_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {algo_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        algo_cfg["params"]["seed"] += app_launcher.global_rank
        algo_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        algo_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        algo_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        _set_hc_process_title(args_cli.algo, f"cuda:{app_launcher.local_rank}")
    else:
        _set_hc_process_title(
            args_cli.algo,
            algo_cfg["params"]["config"].get("device") or args_cli.device,
        )

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = algo_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", algo_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo_cfg["params"]["config"]["time_str"] = time_str
    log_dir = algo_cfg["params"]["config"].get("full_experiment_name", time_str)
    
    if algo_cfg["params"]["config"]["test"]:
        if algo_cfg["params"]["config"]['env_rule_based_exploration']:
            log_dir = 'test_rule_'+ log_dir
        elif algo_cfg["params"]["config"]['load_name']:
            log_dir= 'test'+ '_'.join(algo_cfg["params"]["config"]['load_name'].split('_')[1:3]) + '_' + algo_cfg["params"]["config"]['load_dir'][-22:-3] + '_' + log_dir
        else:
            log_dir = 'test_' + algo_cfg["params"]["algo"]["name"] + '_' + log_dir
    else:
        log_dir = algo_cfg["params"]["algo"]["name"] + '_' + log_dir
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    algo_cfg["params"]["config"]["train_dir"] = log_root_path
    algo_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), algo_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), algo_cfg)

    # read configurations about the agent-training
    rl_device = algo_cfg["params"]["config"]["device"]
    clip_obs = algo_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = algo_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env_cfg.train_cfg = algo_cfg
    if args_cli.headless:
        env_cfg.ui_window_class_type = None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapperHRTPA(env, rl_device, clip_obs, clip_actions)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
        }
        print("[INFO] Recording videos during training (HcVideoRecorder).")
        print_dict(video_kwargs, nesting=4)
        env = HcVideoRecorder(env, **video_kwargs)
    # env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    
    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    vecenv.register(
        "RlgWrapperHRTPA", lambda config_name, num_actors, **kwargs: RlGamesGpuEnvHRTPA(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu_HRTPA", {"vecenv_type": "RlgWrapperHRTPA", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    algo_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    # runner.algo_factory.register_builder('rl_filter', lambda **kwargs: rl_filter.SafeRlFilterAgent(**kwargs))
    runner.algo_factory.register_builder('rule_based', lambda **kwargs: rule_based.RuleBasedHierarchical(**kwargs))
    runner.algo_factory.register_builder('hier', lambda **kwargs: hierarchical_tpa.HierarchicalTPA(**kwargs))
    runner.algo_factory.register_builder('flat', lambda **kwargs: flat_tpa.FlatTPA(**kwargs))

    runner.load(algo_cfg)
    # reset the agent and env
    runner.reset()
    if algo_cfg["params"]["config"]['wandb_activate']:
        if algo_cfg["params"]["config"]["test"]:
            fatigue_str = f"ftg_{args_cli.ftg_thresh_phy}"
            num_particles_str = f"parti_{args_cli.num_particles}"
            measure_noise_sigma_str = f"noise_{args_cli.measure_noise_sigma}"
            if algo_cfg["params"]["config"]['env_rule_based_exploration']:
                run_name = 'test_rule_'+ time_str
            else:
                load_name = algo_cfg["params"]["config"]['load_name'].split('_')[-1][:-4] + '_' + algo_cfg["params"]["config"]['load_dir'][-22:-3]
                run_name = f"test_{algo_cfg['params']['algo']['name']}_{load_name}" + '_' + fatigue_str + '_' + num_particles_str + '_' + measure_noise_sigma_str
        else:
            run_name = f"{algo_cfg['params']['algo']['name']}_{time_str}"
        if algo_cfg["params"]["config"].get("wandb_name"):
            run_name = str(algo_cfg["params"]["config"]["wandb_name"])

        wandb_cfg = dict(env_cfg.__dict__)
        wandb_cfg["algo"] = algo_cfg["params"]["algo"]["name"]
        wandb_cfg["max_parallel_cd_dispatch"] = algo_cfg["params"]["config"].get(
            "max_parallel_cd_dispatch"
        )
        wandb_cfg["max_sim_episodes"] = algo_cfg["params"]["config"].get("max_sim_episodes")

        # WANDB_MODE / HC_WANDB_MODE: online | offline | disabled
        # offline = no network during train (avoids ReadTimeout); sync later with `wandb sync`.
        wandb_mode = (
            os.environ.get("HC_WANDB_MODE")
            or os.environ.get("WANDB_MODE")
            or "online"
        ).strip().lower()
        if wandb_mode not in ("online", "offline", "disabled", "shared"):
            wandb_mode = "online"
        init_kwargs = dict(
            project=algo_cfg["params"]["config"]['wandb_project'],
            group='',
            config=wandb_cfg,
            sync_tensorboard=False,
            name=run_name,
            resume="allow",
            mode=wandb_mode,
        )
        # Per-run identity (does not change machine-global wandb login / ~/.netrc):
        #   WANDB_API_KEY / HC_WANDB_API_KEY  → which account
        #   WANDB_ENTITY / HC_WANDB_ENTITY    → which team/user owns the project
        entity = (
            os.environ.get("HC_WANDB_ENTITY")
            or os.environ.get("WANDB_ENTITY")
            or ""
        ).strip()
        if entity:
            init_kwargs["entity"] = entity
        api_key = (
            os.environ.get("HC_WANDB_API_KEY")
            or os.environ.get("WANDB_API_KEY")
            or ""
        ).strip()
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        try:
            init_kwargs["settings"] = wandb.Settings(
                init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "120")),
            )
        except Exception:
            pass
        entity_str = init_kwargs.get("entity") or "(default login)"
        print(
            f"[wandb] init mode={wandb_mode} entity={entity_str} "
            f"project={init_kwargs['project']} name={run_name}"
        )
        wandb.init(**init_kwargs)

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
