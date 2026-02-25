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
class HcViewerCfg(ViewerCfg):
    #num02_weldingRobot
    eye: tuple[float, float, float] = (23.5, 12, 15)
    lookat: tuple[float, float, float] = (23.5, 17, 0.5)

@configclass
class HcEnvCfg(DirectRLEnvCfg):

    # env
    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    sim_step_interval = 1
    # viewer
    viewer: HcViewerCfg = HcViewerCfg()
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
    asset_path = os.path.expanduser("~") + "/work/Dataset/HC_data/final_for_isaac/HC_import.usd"
    # asset_path = os.path.expanduser("~") + "/work/Dataset/3D_model/all.usd"
    occupancy_map_path = os.path.expanduser("~") + "/work/Dataset/3D_model/occupancy_map.png"
    route_character_file_path = os.path.expanduser("~") + "/work/Dataset/3D_model/routes_character.pkl"
    route_agv_file_path = os.path.expanduser("~") + "/work/Dataset/3D_model/routes_agv.pkl"
    n_max_product = 0
    n_max_human = 0
    n_max_robot = 0
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

########### key_articulation_pos_dic ###########
########### key_articulation_pos_dic ###########
########### key_articulation_pos_dic ###########

joint_pos_dic_num02_weldingRobot_part02_robot_arm_and_base = {
    #joint 1: track_platform, joint 2: arm01_base, joint 3: arm02_base, joint 4: arm03_base, joint 5: arm04_base, joint 6: welding_torch
    "working_pose": [3.2, -1.5, -0.3, 0.1, 0.2, 0.0],
    "moving_pose_time": 100,  
}

class MovingPose:
    def __init__(self, start_pose: torch.Tensor, end_pose: torch.Tensor, time: int):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.time = time
        self.step_time = 0

    def get_next_pose(self):
        self.step_time += 1
        return self.start_pose + (self.end_pose - self.start_pose) * self.step_time / self.time

    def is_done(self):
        return self.step_time >= self.time