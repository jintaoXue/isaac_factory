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
from .hc_single_env import HcSingleEnv


class HcVectorEnvBase(DirectRLEnv):
    cfg_vector_env: HcVectorEnvCfg
    def __init__(self, cfg: HcVectorEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_vector_env= cfg
        self.cuda_device = torch.device(self.cfg_vector_env.cuda_device_str)
        self.env_list : list[type[HcSingleEnv]] = []
        super().__init__(cfg, render_mode, **kwargs)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)

    def _setup_scene(self):
        # assert self.num_envs == 2, "Temporary testing num_envs == 2"
        assert self.cfg_vector_env._valid_train_cfg()

        for i in range(self.num_envs):
            sub_env_path = f"/World/envs/env_{i}"
            # the usd file already has a ground plane
            add_reference_to_stage(usd_path = self.cfg_vector_env.asset_path, prim_path = sub_env_path + "/obj")
            self.setup_one_env(env_id=i)
        # for debug, visualize only prims 
        # stage_utils.print_stage_prim_paths()
        # # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def setup_one_env(self, env_id: int):
        single_env = HcSingleEnv(env_id=env_id, cuda_device=self.cuda_device)
        self.env_list.append(single_env)

    def reset(self, num_worker=None, num_robot=None, evaluate=False):
        """Resets the task and applies default zero actions to recompute observations and states."""
        # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(f"[{now}] Running RL reset")
        for env in self.env_list:
            env_state_action_dict = env.reset_env()
        return

    def step(self, action: dict | None = None, action_extra: dict | None = None) -> None:
        self.step_env_logic(action, action_extra)
        self.step_env_physics()
        return

    def step_env_logic(self, action: dict | None = None, action_extra: dict | None = None) -> None:
        for single_env in self.env_list:
            single_env.step_env_logic(action, action_extra)

    def step_env_physics(self) -> None:
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # simulate
            if self._sim_step_counter % self.cfg.sim_step_interval == 0:
                self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            # self.scene.update(dt=self.physics_dt)
    
    def apply_data_to_sim(self) -> None:
        for single_env in self.env_list:
            single_env.apply_data_to_sim()
    