"""Engine-free factory environment and Hydra bridge for train.py."""
from pathlib import Path
import random

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import numpy as np
import torch

from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.env_asset_cfg.cfg_logic_env import HcLogicEnvCfg
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.hc_single_env_base import HcSingleEnvBase
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.src.route import RouteManagerVectorEnv


class HcLogicVectorEnv:
    """Same single-env managers and tick order; no scene or physics context."""
    def __init__(self, cfg):
        self.cfg = self.cfg_vector_env = cfg
        self.cuda_device = torch.device(cfg.cuda_device_str)
        self.num_envs = int(cfg.scene.num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.route_manager = RouteManagerVectorEnv(cuda_device=self.cuda_device)
        self.env_list = [HcSingleEnvBase(i, self.route_manager, self.cuda_device) for i in range(self.num_envs)]
        for env in self.env_list:
            env.task_manager.configure_human_reward(cfg.train_cfg['params']['config'])
        print(f"[Simulation] backend=logic; envs={self.num_envs}; physics=OFF; render=OFF; one tick per step")

    @property
    def unwrapped(self):
        return self

    def reset(self, num_worker=None, num_robot=None, evaluate=False):
        return [env.reset_env() for env in self.env_list]

    def step(self, action=None, action_extra=None):
        for i, env in enumerate(self.env_list):
            env.step_env_logic(action[i] if action is not None else None,
                               action_extra[i] if action_extra is not None else None)
        return [env.env_state_action_dict for env in self.env_list]

    def get_env_info(self):
        return {'cuda_device': self.cuda_device, 'num_envs': self.num_envs}

    def get_number_of_agents(self):
        return 1

    def close(self):
        pass


def hydra_task_config(task, algo):
    if task != 'HRTPaHC-v1' or algo not in ('hier', 'flat', 'rule_based'):
        raise ValueError('Logic backend supports --task HRTPaHC-v1 --algo hier|flat|rule_based')
    cfg_dir = Path(__file__).parent / 'isaaclab_tasks/isaaclab_tasks/direct/hc_factory/algo_cfg'
    ConfigStore.instance().store(name='hc_logic', node={
        'env': HcLogicEnvCfg().to_dict(), 'agent': OmegaConf.load(cfg_dir / f'{algo}.yaml')})

    def decorate(fn):
        @hydra.main(version_base='1.3', config_path=str(cfg_dir), config_name='hc_logic')
        def wrapped(cfg):
            fn(HcLogicEnvCfg(**OmegaConf.to_container(cfg.env, resolve=True)),
               OmegaConf.to_container(cfg.agent, resolve=True))
        return wrapped
    return decorate


def dump_yaml(path, value):
    path = Path(path if str(path).endswith('.yaml') else str(path)+'.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(value.to_dict() if hasattr(value, 'to_dict') else value), path)


def retrieve_file_path(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(resolved)
