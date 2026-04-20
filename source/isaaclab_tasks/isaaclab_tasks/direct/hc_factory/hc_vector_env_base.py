    # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
# from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path, set_prim_visibility
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import RigidPrim, Articulation
from isaacsim.core.api.world import World


from .env_asset_cfg.cfg_hc_env import HcVectorEnvCfg
from abc import abstractmethod
import numpy as np
import torch


class HcVectorEnvBase(DirectRLEnv):
    cfg_vector_env: HcVectorEnvCfg
    def __init__(self, cfg: HcVectorEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_vector_env= cfg
        self.cuda_device = torch.device(self.cfg_vector_env.cuda_device_str)
        super().__init__(cfg, render_mode, **kwargs)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.env_list = []
    
    def setup_one_env(self, env_id: int):
        pass

    def _setup_scene(self):
        assert self.num_envs == 2, "Temporary testing num_envs == 2"
        assert self.cfg_vector_env._valid_train_cfg()

        for i in range(self.num_envs):
            sub_env_path = f"/World/envs/env_{i}"
            # the usd file already has a ground plane
            add_reference_to_stage(usd_path = self.cfg_vector_env.asset_path, prim_path = sub_env_path + "/obj")
        # for debug, visualize only prims 
        # stage_utils.print_stage_prim_paths()
        '''test settings'''
        #TODO:全库固定随机种子，训练和test要分开
        self._test = self.cfg_vector_env.train_cfg['params']['config']['test']
        if self._test:
            # np.random.seed(self.cfg_vector_env.train_cfg['params']['seed'])
            np.random.seed(1)
            self.set_up_test_setting(self.cfg_vector_env.train_cfg['params']['config'])
        self.train_env_len_settings = self.cfg_vector_env.train_env_len_setting
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def reset(self, num_worker=None, num_robot=None, evaluate=False):
        """Resets the task and applies default zero actions to recompute observations and states."""
        # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(f"[{now}] Running RL reset")

        self.reset_buf[0] = 1
        self.reset_step(num_worker=num_worker, num_robot=num_robot, evaluate=evaluate)
        actions = torch.zeros((self.num_envs, 1), device=self.cuda_device)
        obs_dict, _, _, _, _ = self.step(actions)

        return obs_dict
    