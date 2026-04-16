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


from .env_asset_cfg.hc_env_cfg import HcVectorEnvCfg
# from abc import abstractmethod
# import numpy as np
# from .cfgs.hc_env_cfg import PoseAnimation
from .env_asset_cfg.cfg_material_product import CfgProductProcess, CfgProductionOrder
from .env_asset_cfg.cfg_machine import CfgMachines
from src.machine import num01_rotaryPipeAutomaticWeldingMachine, num02_weldingRobot, \
    num03_rollerbedCNCPipeIntersectionCuttingMachine, num04_laserCuttingMachine, num05_groovingMachineLarge, \
    num06_groovingMachineSmall, num07_highPressureFoamingMachine, num08_gantry_group, num09_workbench
from .src.material import ProductMaterialManager
import torch



class HcSingleEnvBase():
    def __init__(self, env_id: int, cfg_vector_env: HcVectorEnvCfg):
        self.env_id : int = env_id
        self.env_id_str : str = f"env_{env_id}"
        self.cfg_vector_env : HcVectorEnvCfg = cfg_vector_env
        self.cfg_machines : CfgMachines = CfgMachines
        self.cfg_products_process = CfgProductProcess
        self.cfg_production_order = CfgProductionOrder
        self.cuda_device = torch.device(self.cfg_vector_env.cuda_device_str)
        self.cfg_vector_env._valid_train_cfg()
        self.env_rule_based_exploration = self.cfg_vector_env.train_cfg['params']['config']['env_rule_based_exploration']
        self.reward_buf = torch.zeros(1, dtype=torch.float32, device=self.sim.device)
        self.setup_one_env()
        
    def setup_one_env(self):

        self._set_up_machine()
        self._set_up_material()
        self._set_up_human()
        self._set_up_robot()

    def _set_up_machine(self):

        self.num01_rotaryPipeAutomaticWeldingMachine = num01_rotaryPipeAutomaticWeldingMachine(env_id=self.env_id)
        self.num02_weldingRobot = num02_weldingRobot(env_id=self.env_id)
        self.num03_rollerbedCNCPipeIntersectionCuttingMachine = num03_rollerbedCNCPipeIntersectionCuttingMachine(env_id=self.env_id)
        self.num04_laserCuttingMachine = num04_laserCuttingMachine(env_id=self.env_id)
        self.num05_groovingMachineLarge = num05_groovingMachineLarge(env_id=self.env_id)
        self.num06_groovingMachineSmall = num06_groovingMachineSmall(env_id=self.env_id)
        self.num07_highPressureFoamingMachine = num07_highPressureFoamingMachine(env_id=self.env_id)
        self.num08_gantry_group = num08_gantry_group(env_id=self.env_id)
        self.num09_workbench = num09_workbench(env_id=self.env_id)
        
    
    def _set_up_material(self):
        self.product_material_manager = ProductMaterialManager(cfg_product_process=self.cfg_products_process, cfg_production_order=self.cfg_production_order, env_id=self.env_id)


    def _set_up_human(self):
        self.human_manager = HumanManager(cfg_human=self.cfg_human, env_id=self.env_id)

    def _set_up_robot(self):
        self.robot_manager = RobotManager(cfg_robot=self.cfg_robot, env_id=self.env_id)