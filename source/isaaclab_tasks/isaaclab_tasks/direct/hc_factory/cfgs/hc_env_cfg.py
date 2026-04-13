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
# from .......isaaclab.isaaclab.envs.common import ViewerCfg
from isaaclab.envs.common import ViewerCfg


@configclass
class HcViewerCfg(ViewerCfg):
    #num02_weldingRobot
    # eye: tuple[float, float, float] = (23.5, 12, 15)
    # lookat: tuple[float, float, float] = (23.5, 17, 0.5)
    
    #num01
    # eye: tuple[float, float, float] = (43.5, 12, 25)
    # lookat: tuple[float, float, float] = (43.5, 17, 0.5)

    # #num05
    # eye: tuple[float, float, float] = (0, 12, 25)
    # lookat: tuple[float, float, float] = (0, 17, 0.5)

    #num03
    eye: tuple[float, float, float] = (10, 12, 25)
    lookat: tuple[float, float, float] = (10, 17, 0.5)

    #half factory
    eye: tuple[float, float, float] = (23.5, 2, 40)
    lookat: tuple[float, float, float] = (23.5, 10, 0.5)

    #num08 gantry
    eye: tuple[float, float, float] = (0, 10, 100)
    lookat: tuple[float, float, float] = (0, 2, 0.5)


# @configclass
# class HcSingleEnvCfg():




@configclass
class HcVectorEnvCfg(DirectRLEnvCfg):

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3, env_spacing=100.0, replicate_physics=True)
    # cuda decive
    cuda_device_str = "cuda:0"
    #train_cfg will be update when running train.py
    train_cfg = None


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

class PoseAnimation:
    def __init__(self, start_pose: torch.Tensor, end_pose: torch.Tensor, time: int):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.time = time
        self.step_time = 0

    def get_next_pose(self):
        self.step_time += 1
        return self.start_pose + (self.end_pose - self.start_pose) * self.step_time / self.time

    def is_done(self):
        dis = torch.norm(self.start_pose - self.end_pose)
        return dis < 0.01 or self.step_time >= self.time 




