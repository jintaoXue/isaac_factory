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
import copy
# from abc import abstractmethod
# import numpy as np
# from .cfgs.hc_env_cfg import PoseAnimation
from abc import abstractmethod
from .src.machine import MachineManager
from .src.material import ProductMaterialManager
from .src.human import HumanManager
from .src.robot import RobotManager
from .src.storage import StorageManager
from .src.route import RouteManagerVectorEnv
from .env_asset_cfg.cfg_hc_env import single_env_state_action_dict_template, HcVectorEnvCfg



class HcSingleEnvBase():
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id : int = env_id
        self.env_id_str : str = f"env_{env_id}"
        self.cuda_device = cuda_device
        self.reward_buf = torch.zeros(1, dtype=torch.float32, device=self.cuda_device)
        # 每个 env 持有独立的 state dict，避免多 env 共享引用导致状态串扰
        self.env_state_action_dict = copy.deepcopy(single_env_state_action_dict_template)
        self.register_env_assets()
    
    def register_env_assets(self):
        self.machine_manager = MachineManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.product_material_manager = ProductMaterialManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.human_manager = HumanManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.robot_manager = RobotManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.storage_manager = StorageManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.route_manager = RouteManagerVectorEnv(env_id=self.env_id, cuda_device=self.cuda_device)

    def iter_managers(self):
        return (
            self.machine_manager,
            self.product_material_manager,
            self.human_manager,
            self.robot_manager,
            self.storage_manager,
            self.route_manager,
        )

    def reset_env(self):
        for m in self.iter_managers():
            m.reset(self.env_state_action_dict)
        return self.env_state_action_dict

    def apply_data_to_sim(self) -> None:
        #articulations
        articulations : dict = self.env_state_action_dict["articulations"]
        for name, data in articulations.items():
            articulation : Articulation = data["articulation"]
            articulation.set_joint_positions(data["joint_positions"])
            # Currently, joint velocities are set to zero.
            joint_velocities = torch.zeros_like(data["joint_positions"], device=self.cuda_device)
            articulation.set_joint_velocities(joint_velocities)
        #rigid prims
        rigid_prims : dict = self.env_state_action_dict["rigid_prims"]
        for name, data in rigid_prims.items():
            rigid_prim : RigidPrim = data["rigid_prim"]
            rigid_prim.set_local_poses(positions=data["positions"], orientations=data["orientations"])
            rigid_prim.set_velocities(torch.zeros((1,6), device=self.cuda_device))


    def step_env_logic(self, action: dict | None = None, action_extra: dict | None = None) -> None:
        for m in self.iter_managers():
            m.step(self.env_state_action_dict)
        return
