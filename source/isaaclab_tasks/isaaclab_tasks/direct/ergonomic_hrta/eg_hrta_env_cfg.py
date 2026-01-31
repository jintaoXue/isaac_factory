# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
# from collections.abc import Sequence

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

# import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
# from isaaclab.utils.math import sample_uniform
# from .....isaaclab_assets.isaaclab_assets.robots import production_assets
import os
from .....isaaclab.isaaclab.envs.common import ViewerCfg

#high_level_task
high_level_task_dic =  {-1:'none', 0: 'hoop_preparing', 1:'bending_tube_preparing', 2:'hoop_loading_inner', 3:'bending_tube_loading_inner', 4:'hoop_loading_outer', 
                5:'bending_tube_loading_outer', 6:'cutting_cube', 7:'collect_product', 8:'placing_product'}
high_level_task_rev_dic = {v: k for k, v in high_level_task_dic.items()}

@configclass
class HRTaViewerCfg(ViewerCfg):
    #view cutting machine
    # # self.camera_position = [-30, 0.99, 3.8]
    # # self.camera_target = [0, 0.99, 3]
    
    # # view welding station left
    # # self.camera_position = [-5, 5, 25]
    # # self.camera_target = [-20, 5, 3]
    # # view welding station right
    # self.camera_position = [-45, 5, 20]
    # self.camera_target = [-28, 5, 3]

    # # view of all
    # self.camera_position = [-18, 14, 50]
    # self.camera_target = [-18, 5, 3]
    # eye: tuple[float, float, float] = (-18, 14, 50)
    # """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    # lookat: tuple[float, float, float] = (-18, 5, 3)
    # view of grippers
    eye: tuple[float, float, float] = (-18, 14, 50)
    lookat: tuple[float, float, float] = (-18, 5, 3)

@configclass
class HRTaskAllocEnvCfg(DirectRLEnvCfg):

    # env
    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    sim_step_interval = 5000
    # viewer
    viewer: HRTaViewerCfg = HRTaViewerCfg()
    #dynamic env len settings, for human 1-3 x robot 1-3, <= 1500
    # train_env_len_setting = [[4000, 4000, 4000], [1800, 1800, 1800], [1500, 1500, 1500]]
    train_env_len_setting = [[3500, 2000, 2000], [1800, 1500, 1500], [1800, 1400, 1400]]
    #max_episode_length = max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation) = 25/(1/120 * 2) = 1500 steps
    episode_length_s = 80.0 
    action_space = 10
    #The real state/observation_space is complicated, settiing 2 is only for initializing gym Env
    observation_space = 2
    state_space = 2    
    #asset path, include machine, human, robot
    asset_path = os.path.expanduser("~") + "/work/Dataset/3D_model/all.usd"
    occupancy_map_path = os.path.expanduser("~") + "/work/Dataset/3D_model/occupancy_map.png"
    route_character_file_path = os.path.expanduser("~") + "/work/Dataset/3D_model/routes_character.pkl"
    route_agv_file_path = os.path.expanduser("~") + "/work/Dataset/3D_model/routes_agv.pkl"
    n_max_product = 5
    n_max_human = 3
    n_max_robot = 3
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    # cuda decive
    cuda_device_str = "cuda:0"
    #train_cfg will be update when running train.py
    train_cfg = None
    cutting_machine_oper_len = 25
    welding_once_time = 25
    human_loading_time = 8
    human_putting_time = 5
    machine_time_random = 3
    human_time_random = 2
    # machine_time_random = 0
    # human_time_random = 0
    # box_capacity = 4
    # box_capacity_hoop = 4
    # box_capacity_bending_tube = 2
    # box_capacity_product = 1
    #fatigue
    ftg_thresh_phy = 0.95
    ftg_thresh_psy = 0.8
    hyper_param_time = 0.3
    # if not use fatigue mask, set False
    use_partial_filter = True
    measure_noise_mu = 0.0
    measure_noise_sigma = 0.00005
    num_particles = 500

    def _valid_train_cfg(self):
        #update train_cfg when running train.py
        return self.train_cfg != None

class BoxCapacity:
    hoop = 2
    bending_tube = 2
    product = 1