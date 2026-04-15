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


from .cfgs.hc_env_cfg import HcVectorEnvCfg
# from abc import abstractmethod
# import numpy as np
# from .cfgs.hc_env_cfg import PoseAnimation
from .cfgs.cfg_material_product import CfgProductProcess, CfgProductionOrder
from .cfgs.cfg_machine import CfgMachines

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

    def _set_up_machine(self):

        combined = self.cfg_machines.get("registeration_infos_combined")

        self.num02_weldingRobot_part02_robot_arm_and_base = None
        self.animation_num02_weldingRobot_part02_robot_arm_and_base: PoseAnimation = None
        self.num02_weldingRobot_part04_mobile_base_for_material = None
        self.animation_num02_weldingRobot_part04_mobile_base_for_material: PoseAnimation = None

        self.num03_rollerbedCNCPipeIntersectionCuttingMachine_part01_station = None
        self.animation_num03_rollerbedCNCPipeIntersectionCuttingMachine_part01_station: PoseAnimation = None
        self.num03_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine = None
        self.animation_num03_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine: PoseAnimation = None

        self.num04_laserCuttingMachine = None
        self.animation_num04_laserCuttingMachine: PoseAnimation = None

        self.num05_groovingMachineLarge_part01_large_fixed_base = None
        self.animation_num05_groovingMachineLarge_part01_large_fixed_base: PoseAnimation = None
        self.num05_groovingMachineLarge_part02_large_mobile_base = None
        self.animation_num05_groovingMachineLarge_part02_large_mobile_base: PoseAnimation = None

        self.num06_groovingMachineSmall_part01_small_fixed_base = None
        self.animation_num06_groovingMachineSmall_part01_small_fixed_base: PoseAnimation = None
        self.num06_groovingMachineSmall_part02_small_mobile_handle = None
        self.animation_num06_groovingMachineSmall_part02_small_mobile_handle: PoseAnimation = None

        self.num07_highPressureFoamingMachine = None
        self.animation_num07_highPressureFoamingMachine: PoseAnimation = None

        self.num08_gantry_group = None
        self.animation_num08_gantry_group: PoseAnimation = None

        self.num09_workbench = None
        self.animation_num09_workbench: PoseAnimation = None

        # 再根据配置创建 Articulation
        for obj_name, info in combined.items():
            articulation = Articulation(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id),
                name=obj_name,
                reset_xform_properties=bool(info.get("reset_xform_properties", False)),
            )
            setattr(self, obj_name, articulation)

            setattr(self, f"animation_{obj_name}", PoseAnimation(
                start_pose=info["joint_positions_reset"],
                end_pose=info["joint_positions_reset"],
                time=info["animation_time"],
            ))
    
    def _set_up_material(self):
        order = self.cfg_production_order["registeration_infos"]

        self.material_prims = []
        self.materials_list: list[dict] = []

        for product, n_prod in order.items(): #product is the product name, n_prod is the number of products

            cfg = self.cfg_products_process.get(product)
            need = cfg.get("related_materials", {})
            metas = cfg.get("meta_registeration_info", {})
            for meta_key, meta in metas.items():

                per = need.get(meta_key)
                prim_tpl = meta.get("prim_paths_expr")
                name_tpl = meta.get("name", meta_key)
                if not isinstance(prim_tpl, str):
                    continue

                for k in range(n_prod * per):
                    s = str(k)
                    prim = prim_tpl.format(i=self.env_id)
                    name = s.format(idx=idx)
                    self.material_prims.append(RigidPrim(prim_paths_expr=prim, name=name, reset_xform_properties=False))
                    self.materials_list.append({"product": product, "meta_key": meta_key, "name": name, "prim_paths_expr": prim})