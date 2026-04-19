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

import torch
# from abc import abstractmethod
# import numpy as np
# from .cfgs.hc_env_cfg import PoseAnimation
from .src.machine import MachineManager
from .src.material import ProductMaterialManager
from .src.human import HumanManager
from .src.robot import RobotManager
from .src.storage import StorageManager
from .src.route import RouteManagerVectorEnv
from .env_asset_cfg.cfg_hc_env import single_env_state_action_dict_template, HcVectorEnvCfg



class HcSingleEnvBase():
    def __init__(self, env_id: int, cfg_vector_env: HcVectorEnvCfg):
        self.env_id : int = env_id
        self.env_id_str : str = f"env_{env_id}"
        self.cfg_vector_env : HcVectorEnvCfg = cfg_vector_env
        self.cuda_device = torch.device(self.cfg_vector_env.cuda_device_str)
        self.cfg_vector_env._valid_train_cfg()
        self.env_rule_based_exploration = self.cfg_vector_env.train_cfg['params']['config']['env_rule_based_exploration']
        self.reward_buf = torch.zeros(1, dtype=torch.float32, device=self.sim.device)
        self.env_state_action_dict = single_env_state_action_dict_template
        self.setup_env()
        self.reset_env()
    
    def setup_env(self):
        self._set_up_machine()
        self._set_up_material()
        self._set_up_human()
        self._set_up_robot()
        self._set_up_storage()
        self._set_up_route()

    def reset_env(self):
        for m in self.iter_managers():
            m.reset(self.env_state_action_dict)

    def step_env(self, action: dict):
        for m in self.iter_managers():
            m.step(self.env_state_action_dict)

    def iter_managers(self):
        return (
            self.machine_manager,
            self.product_material_manager,
            self.human_manager,
            self.robot_manager,
            self.storage_manager,
            self.route_manager,
        )

    def _set_up_machine(self):
        self.machine_manager = MachineManager(env_id=self.env_id)
        
    def _set_up_material(self):
        self.product_material_manager = ProductMaterialManager(env_id=self.env_id)

    def _set_up_human(self):
        self.human_manager = HumanManager(env_id=self.env_id)

    def _set_up_robot(self):
        self.robot_manager = RobotManager(env_id=self.env_id)

    def _set_up_storage(self):
        self.storage_manager = StorageManager(env_id=self.env_id)
    
    def _set_up_route(self):
        self.route_manager = RouteManagerVectorEnv(env_id=self.env_id)


