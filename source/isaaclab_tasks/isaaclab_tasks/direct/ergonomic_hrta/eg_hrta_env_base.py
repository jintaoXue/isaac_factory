    # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .eg_hrta_env_cfg import HRTaskAllocEnvCfg

# from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path, set_prim_visibility
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import RigidPrim, Articulation
from isaacsim.core.api.world import World

from .eg_hrta_task_manager import Materials, TaskManager
from .eg_hrta_map_route import MapRoute
from abc import abstractmethod
import numpy as np

class HRTaskAllocEnvBase(DirectRLEnv):
    cfg: HRTaskAllocEnvCfg

    def __init__(self, cfg: HRTaskAllocEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.env_rule_based_exploration = cfg.train_cfg['params']['config']['env_rule_based_exploration']
        
    def _setup_scene(self):
        assert self.scene.num_envs == 1, "Temporary only support num_envs == 1"
        assert self.cfg._valid_train_cfg()
        self.cuda_device = torch.device(self.cfg.cuda_device_str)
        for i in range(self.scene.num_envs):
            sub_env_path = f"/World/envs/env_{i}"
            # the usd file already has a ground plane
            add_reference_to_stage(usd_path = self.cfg.asset_path, prim_path = sub_env_path + "/obj")
            raw_ground_path = sub_env_path + "/obj" + "/GroundPlane"
            # ground_prim = self.scene.stage.GetPrimAtPath(raw_ground_path)
            # color_attr = ground_prim.GetAttribute("inputs:diffuse_color_constant")
            # color_attr.Set(Gf.Vec3f(0.6, 0.13, 0.13))
            # set_prim_visibility(prim=ground_prim, visible=False)
            # if get_prim_at_path(raw_ground_path):
            #     delete_prim(raw_ground_path)
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # for debug, visualize only prims 
        # stage_utils.print_stage_prim_paths()
        '''test settings'''
        self._test = self.cfg.train_cfg['params']['config']['test']
        if self._test:
            # np.random.seed(self.cfg.train_cfg['params']['seed'])
            np.random.seed(1)
            self.set_up_test_setting(self.cfg.train_cfg['params']['config'])
        self.train_env_len_settings = self.cfg.train_env_len_setting
        cube_list, hoop_list, bending_tube_list, upper_tube_list, product_list = [],[],[],[],[]
        for i in range(self.cfg.n_max_product):
            cube, hoop, bending_tube, upper_tube, product = self.set_up_material(num=i)
            cube_list.append(cube)
            hoop_list.append(hoop)
            bending_tube_list.append(bending_tube)
            upper_tube_list.append(upper_tube)
            product_list.append(product)
        #materials states, machine state
        self._set_up_machine()
        self.materials : Materials = Materials(cube_list=cube_list, hoop_list=hoop_list, bending_tube_list=bending_tube_list, upper_tube_list=upper_tube_list, product_list = product_list)
        '''for humans workers (characters), robots (agv+boxs) and task manager'''
        character_list =self.set_up_human(num=self.cfg.n_max_human)
        agv_list, box_list = self.set_up_robot(num=self.cfg.n_max_robot)
        self.task_manager : TaskManager = TaskManager(character_list, agv_list, box_list, self.cuda_device, self.cfg, self.cfg.train_cfg['params']['config'])
        map_route = MapRoute(self.cfg)
        self.task_manager.characters.routes_dic, self.task_manager.agvs.routes_dic = map_route.load_pre_def_routes()
        self.task_manager.boxs.routes_dic = self.task_manager.agvs.routes_dic

        # # clone and replicate
        # self.scene.clone_environments(copy_from_source=False)
        
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
    
    def reset_step(self, num_worker=None, num_robot=None, evaluate=False):
        env_id = 0
        if self.reset_buf[env_id] == 1:
            self.reset_buf[env_id] = 0
            self.episode_length_buf[env_id] = 0
            self.reset_step_helper(acti_num_char=num_worker, acti_num_robot=num_robot, evaluate=evaluate)
        return


    def reset_step_helper(self, acti_num_char=None, acti_num_robot=None, evaluate=False) -> None:
        # acti_num_char = 2
        # acti_num_robot = 1
        #TODO
        if self._test:
            if self._test_all_settings:
                self.test_all_idx += 1
                if self.test_all_idx in range(0, len(self.test_settings_list)):
                    acti_num_char, acti_num_robot = self.test_settings_list[self.test_all_idx]
                # self.test_all_idx += 1 
            self.task_manager.reset(acti_num_char, acti_num_robot)
            self.dynamic_episode_len = self.test_env_max_length
        else:
            # assert acti_num_char is None, "wrong training setting"
            self.task_manager.reset(acti_num_char, acti_num_robot)
            self.dynamic_episode_len = self.train_env_len_settings[self.task_manager.characters.acti_num_charc-1][self.task_manager.agvs.acti_num_agv-1]
        self.evaluate = evaluate
        self.have_over_work = False
        self.reset_worker_random_time()
        self.reset_machine_random_time()
        self.materials.reset()
        self.reset_machine_state()
        self.update_task_mask()
        self.task_manager.obs = self.get_observations()
        self.scene.write_data_to_sim()
        self.sim.forward()
        if not self._test:
            self.dynamic_episode_len = self.train_env_len_settings[self.task_manager.characters.acti_num_charc-1][self.task_manager.agvs.acti_num_agv-1]
        return 
    
    def done_update(self):
        """Assign environments for reset if successful or failed."""
        task_finished = self.materials.done()
        is_last_step = self.episode_length_buf[0] >= self.dynamic_episode_len - 1
        #TODO for debug
        # is_last_step = False
        # If max episode length has been reached
        if is_last_step or task_finished :
            self.reset_buf[0] = 1
            '''gantt chart'''
            if self._test and self.gantt_chart_data:
                self.save_gantt_chart() 
            '''end'''
            name = 'traning ' if not self.evaluate else 'evaluate '
            if self.env_rule_based_exploration:
                name = 'rule_based' + name
            name += " worker:{}, agv&box:{}, env_len:{}, max_env_len:{}, finished:{}, over_work:{}".format(self.task_manager.characters.acti_num_charc, 
                                                    self.task_manager.agvs.acti_num_agv, self.episode_length_buf[0], self.dynamic_episode_len, task_finished, self.have_over_work)
            self.extras['print_info'] = name
            # print(name+" worker:{}, agv&box:{}, env_len:{}, max_env_len:{}, finished:{}, over_work:{}".format(self.task_manager.characters.acti_num_charc, 
            #                                         self.task_manager.agvs.acti_num_agv, self.episode_length_buf[0], self.dynamic_episode_len, task_finished, self.have_over_work))
        else:
            pass
    
    # def update_ergonomic

    def update_task_mask(self):
        # if self.episode_length_buf[0] >= 700:
        #     a = 1
        self.task_mask = self.get_task_mask()
        if self.cfg.train_cfg['params']['config']['use_fatigue_mask']:
            self.fatigue_mask = self.get_fatigue_mask()
            self.task_mask = self.task_mask * self.fatigue_mask
        self.available_task_dic = self.get_task_mask_dic(self.task_mask)

    def get_rule_based_action(self):
        _task_mask = self.task_mask
        if _task_mask[1:].any():
            _task_mask[0] = 0
        return (_task_mask.argmax(0)).unsqueeze(0)
        
        
    def caculate_metric_action(self, actions):
        self.reward_action = None
        task_id = actions[0] - 1
        task = self.task_manager.task_dic[task_id.item()]
        if task not in self.available_task_dic.keys():
            self.reward_action = -0.1
        elif task == 'none':
            if self.task_mask[1:].count_nonzero() > 0:
                self.reward_action = -0.1
            else:
                self.reward_action = 0.
        else:
            self.reward_action = 0.1

    def calculate_metrics(self):
        task_finished = self.materials.done()
        is_last_step = self.episode_length_buf[0] >= self.dynamic_episode_len - 1
        """Compute reward at current timestep."""
        reward_time = (self.episode_length_buf[0] - self.pre_progress_step)*-0.002
        progress = self.materials.progress()
        if is_last_step: 
            if task_finished:
                rew_task = 0.6
            else:
                rew_task = -1.5 + self.materials.progress()
        else:
            if task_finished:
                rew_task = 1
            else:
                rew_task = (progress - self.materials.pre_progress)*0.2

        self.reward_buf[0] = self.reward_action + reward_time + rew_task
        self.pre_progress_step = self.episode_length_buf[0].clone()
        self.materials.pre_progress = progress
        self.extras['progress'] = progress
        self.extras['rew_action'] = self.reward_action
        self.extras['env_length'] = self.episode_length_buf[0].clone().cpu().item()
        self.extras['max_env_len'] = self.dynamic_episode_len
        self.extras['time_step'] = f"{self.episode_length_buf[0].cpu()}"
        self.extras['num_worker'] = self.task_manager.characters.acti_num_charc
        self.extras['num_robot'] = self.task_manager.agvs.acti_num_agv
        self.extras['human_move'] = self.task_manager.characters.get_sum_movement()
        self.extras['agv_move'] = self.task_manager.agvs.get_sum_movement()
        self.extras['cost_value'] = self.compute_cost_value()
        if self._test:
            self.extras['worker_initial_pose'] = self.task_manager.ini_worker_pose
            self.extras['robot_initial_pose'] = self.task_manager.ini_agv_pose
            self.extras['box_initial_pose'] = self.task_manager.ini_box_pose
        # self.reward_test_list.append(self.reward_buf[0].clone())
        return
    
    def compute_cost_value(self):
        cost = 0.0
        scale = 0.005
        if self.task_manager.characters.have_overwork():
            cost += 0.1
        cost += self.task_manager.characters.compute_fatigue_cost()*scale
        return cost
        

    def get_fatigue_data(self):
        self.extras['overwork'] = self.task_manager.characters.have_overwork()
        self.extras['overwork_phy_values'] = []
        if self.extras['overwork']:
            self.have_over_work = True
            self.extras['overwork_phy_values'] = self.task_manager.characters.get_overwork_phy_values()
        if len(self.task_manager.fatigue_data_list)>0:
            self.extras['fatigue_data'] = self.task_manager.fatigue_data_list
            self.task_manager.fatigue_data_list = []
        elif 'fatigue_data' in self.extras:
            del self.extras['fatigue_data']

    def get_task_mask(self):

        task_mask = torch.zeros(len(self.task_manager.task_dic))
        worker, agv, box = self.check_task_lacking_entity()
        have_wab = worker and agv and box
        have_w = worker
        have_ab = agv and box
        if have_wab and self.state_depot_hoop == 0 and 'hoop_preparing' not in self.task_manager.task_in_dic.keys() and self.materials.hoop_states.count(0) > 0:
            task_mask[1] = 1
        if have_wab and self.state_depot_bending_tube == 0 and 'bending_tube_preparing' not in self.task_manager.task_in_dic.keys() and self.materials.bending_tube_states.count(0) > 0:
            task_mask[2] = 1
        if have_w and self.station_state_inner_left == 0 and 'hoop_loading_inner' not in self.task_manager.task_in_dic.keys() and self.materials.hoop_states.count(2)>0: #loading
            task_mask[3] = 1
        if have_w and self.station_state_inner_right == 0 and 'bending_tube_loading_inner' not in self.task_manager.task_in_dic.keys() and self.materials.bending_tube_states.count(2)>0: 
            task_mask[4] = 1
        if have_w and self.station_state_outer_left == 0 and 'hoop_loading_outer' not in self.task_manager.task_in_dic.keys() and self.materials.hoop_states.count(2)>0: #loading
            task_mask[5] = 1
        if have_w and self.station_state_outer_right == 0 and 'bending_tube_loading_outer' not in self.task_manager.task_in_dic.keys() and self.materials.bending_tube_states.count(2)>0: 
            task_mask[6] = 1
        if have_w and self.cutting_machine_state == 1 and 'cutting_cube' not in self.task_manager.task_in_dic.keys(): #cuttting cube
            task_mask[7] = 1
        #when material 处于准备好的状态，要么边线库有，要么正在被加工。如果没有准备好，那么不可能执行collect product的任务
        #English: only when material is ready for propcessing (at depot, loaded, processing, processed), the collect product mission is activate 
        if self.materials.have_collecting_product_req() and have_ab and (self.materials.produce_product_req() == True) and 'collect_product' not in self.task_manager.task_in_dic.keys():
            task_mask[8] = 1
        if have_w and 'collect_product' in self.task_manager.task_in_dic.keys() and self.task_manager.boxs.product_collecting_idx >=0 and \
                len(self.task_manager.boxs.product_idx_list[self.task_manager.boxs.product_collecting_idx])>0 and \
                'placing_product' not in self.task_manager.task_in_dic.keys() and self.gripper_inner_task not in range (4, 8):
            task_mask[9] = 1
            if self.task_manager.boxs.acti_num_box > 1 and len(self.task_manager.boxs.product_idx_list[self.task_manager.boxs.product_collecting_idx])<self.task_manager.boxs.capacity.product and self.materials.have_collecting_product_req():
                task_mask[9] = 0
        # if self.task_manager.characters.acti_num_charc == 1:
        if True:
            #fix bug
            is_last_hoop = sum([_state<=2 and _state >= -1 for _state in self.materials.hoop_states]) == 1
            #make sure the last one hoop is loaded in the same station including the cube waiting for welding
            if is_last_hoop and (task_mask[3] or task_mask[5]):
                if self.materials.outer_cube_processing_index == -1 or self.materials.cube_states[self.materials.outer_cube_processing_index] in range(9, 14):
                    task_mask[5] = 0
                if self.materials.inner_cube_processing_index == -1 or self.materials.cube_states[self.materials.inner_cube_processing_index] in range(9, 14):
                    task_mask[3] = 0
            
            is_last_bending_tube = sum([_state<=2 and _state >= -1 for _state in self.materials.bending_tube_states]) == 1
            #make sure the last one bending tube is loaded in the same station including the cube waiting for welding
            if is_last_bending_tube and (task_mask[4] or task_mask[6]):
                if self.materials.outer_cube_processing_index == -1 or self.materials.cube_states[self.materials.outer_cube_processing_index] in range(10, 14):
                    task_mask[6] = 0
                if self.materials.inner_cube_processing_index == -1 or self.materials.cube_states[self.materials.inner_cube_processing_index] in range(10, 14):
                    task_mask[4] = 0

        task_mask[0] = 1
        return task_mask

    def get_fatigue_mask(self):
        # fatigue_mask = torch.zeros(len(self.task_manager.task_dic), device=self.cuda_device)
        _masks = self.task_manager.characters.fatigue_task_masks
        fatigue_mask = _masks[0]
        for i in range(1, self.task_manager.characters.acti_num_charc):
            fatigue_mask =  fatigue_mask | _masks[i]
    
        return fatigue_mask
    
    def get_task_mask_dic(self, task_mask):
        available_task_dic = {}
        if task_mask[0] == 1:
            available_task_dic['none'] = -1
        if task_mask[1] == 1:
            available_task_dic['hoop_preparing'] = 0
        if task_mask[2] == 1:
            available_task_dic['bending_tube_preparing'] = 1
        if task_mask[3] == 1:
            available_task_dic['hoop_loading_inner'] = 2
        if task_mask[4] == 1: 
            available_task_dic['bending_tube_loading_inner'] = 3
        if task_mask[5] == 1:
            available_task_dic['hoop_loading_outer'] = 4
        if task_mask[6] == 1: 
            available_task_dic['bending_tube_loading_outer'] = 5
        if task_mask[7] == 1:
            available_task_dic['cutting_cube'] = 6
        if task_mask[8] == 1:
            available_task_dic['collect_product'] = 7
        if task_mask[9] == 1:
            available_task_dic['placing_product'] = 8
        return available_task_dic

    def check_task_lacking_entity(self):
        worker = [a*b for a,b in zip(self.task_manager.characters.states,self.task_manager.characters.tasks)].count(0)
        agv = [a*b for a,b in zip(self.task_manager.agvs.states,self.task_manager.agvs.tasks)].count(0)
        box = [a*b for a,b in zip(self.task_manager.boxs.states,self.task_manager.boxs.tasks)].count(0)
        return worker>0, agv>0, box>0

    def set_up_test_setting(self, train_sub_cfg):
        self.test_env_max_length = train_sub_cfg['test_env_max_length']
        self._test_all_settings = train_sub_cfg['test_all_settings']
        if self._test_all_settings:
            self.test_all_idx = -1
            self.test_settings_list = []
            for w in range(train_sub_cfg["max_num_worker"]):
                for r in range(train_sub_cfg["max_num_robot"]):
                    for i in range(train_sub_cfg['test_times']):  
                        self.test_settings_list.append((w+1,r+1))
        '''test one setting, the task_manager class will handle'''
        '''gantt chart'''
        self.gantt_chart_data = train_sub_cfg['gantt_chart_data']
        # if self.gantt_chart_data:
        #     self.actions_list = []
        #     self.time_frames = []
        #     self.gantt_charc = []
        #     self.gantt_agv = []
    
    def reset_worker_random_time(self):
        self.temp_random_time = np.random.uniform(0,self.cfg.human_time_random)
    
    def reset_machine_random_time(self):
        self.machine_random_time = np.random.uniform(0,self.cfg.machine_time_random)
    
    def reset_machine_state(self):
        # conveyor
        #0 free 1 working
        self.convey_state = 0
        #cutting machine
        #to do 
        self.cutting_state_dic = {0:"free", 1:"work", 2:"reseting"}
        self.cutting_machine_state = 0
        self.c_machine_oper_time = 0
        self.c_machine_oper_len = self.cfg.cutting_machine_oper_len
        #gripper
        speed = 0.6
        self.operator_gripper = torch.tensor([speed]*10, device=self.cuda_device)
        self.gripper_inner_task_dic = {0: "reset", 1:"pick_cut", 2:"place_cut_to_inner_station", 3:"place_cut_to_outer_station", 
                                    4:"pick_product_from_inner", 5:"pick_product_from_outer", 6:"place_product_from_inner", 7:"place_product_from_outer"}
        self.gripper_inner_task = 0
        self.gripper_inner_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_inner_state = 0

        self.gripper_outer_task_dic = {0: "reset", 1:"pick_upper_tube_for_inner_station", 2:"pick_upper_tube_for_outer_station", 3:"place_upper_tube_to_inner_station", 4:"place_upper_tube_to_outer_station"}
        self.gripper_outer_task = 0
        self.gripper_outer_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_outer_state = 0

        #welder 
        # self.max_speed_welder = 0.1
        self.welder_inner_oper_time = 0
        self.welder_outer_oper_time = 0
        self.welding_once_time = self.cfg.welding_once_time
        self.operator_welder = torch.tensor([0.4], device=self.cuda_device)
        self.welder_task_dic = {0: "reset", 1:"weld_left", 2:"weld_right", 3:"weld_middle",}
        self.welder_state_dic = {0: "free_empty", 1: "moving_left", 2:"welding_left", 3:"welded_left", 4:"moving_right",
                                 5:"welding_right", 6:"rotate_and_welding", 7:"welded_right", 8:"welding_middle" , 9:"welded_upper"}
        self.welder_inner_task = 0
        self.welder_inner_state = 0
        self.welder_outer_task = 0
        self.welder_outer_state = 0
        
        #station
        # self.welder_inner_oper_time = 10
        self.operator_station = torch.tensor([0.3, 0.3, 0.3, 0.3], device=self.cuda_device)
        self.station_task_left_dic = {0: "reset", 1:"weld"}
        self.station_state_left_dic = {0: "reset_empty", 1:"loading", 2:"rotating", 3:"waiting", 4:"welding", 5:"welded", 6:"finished", -1:"resetting"}
        self.station_task_inner_left = 0
        self.station_task_outer_left = 0
        self.station_state_inner_left = -1
        self.station_state_outer_left = -1

        self.station_middle_task_dic = {0: "reset", 1:"weld_left", 2:"weld_middle", 3:"weld_right"}
        self.station_state_middle_dic = {-1:"resetting", 0: "reset_empty", 1:"placing", 2:"placed", 3:"moving_left", 4:"welding_left", 
                                         5:"welded_left", 6:"welding_right", 7:"welded_right", 8:"welding_upper", 9:"welded_upper"}
        self.station_state_inner_middle = 0
        self.station_state_outer_middle = 0
        self.station_task_inner_middle = 0
        self.station_task_outer_middle = 0
        
        self.station_right_task_dic = {0: "reset", 1:"weld"}
        self.station_state_right_dic = {0: "reset_empty", 1:"placing", 2:"placed", 3:"moving", 4:"welding_right", -1:"resetting"}
        self.station_state_inner_right = 0
        self.station_state_outer_right = 0
        self.station_task_outer_right = 0
        self.station_task_inner_right = 0
        
        self.process_groups_dict = {}
        self.proc_groups_inner_list = []
        self.proc_groups_outer_list = []
        '''side table state'''
        self.depot_state_dic = {0: "empty", 1:"placing", 2: "placed"}
        # self.table_capacity = 4
        self.depot_hoop_set = set()
        self.depot_bending_tube_set = set()
        self.state_depot_hoop = 0
        self.state_depot_bending_tube = 0
        self.depot_product_set = set()
        '''progress step'''
        self.pre_progress_step = 0
        # self.available_task_dic = {'none': -1}

        return

    def _set_up_machine(self):
        self.obj_belt_0 = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_0/Belt",
            name="ConveyorBelt_A09_0_0/Belt",
            track_contact_forces=True,
        )
        self.obj_0_1 = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/obj_0_1",
            name="obj_0_1",
            track_contact_forces=True,
        )
        self.obj_belt_1 = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_2/Belt",
            name="ConveyorBelt_A09_0_2/Belt",
            track_contact_forces=True,
        )
        self.obj_part_10 = Articulation(
            prim_paths_expr="/World/envs/.*/obj/part10", name="obj_part_10", reset_xform_properties=False
        )
        self.obj_part_7 = Articulation(
            prim_paths_expr="/World/envs/.*/obj/part7", name="obj_part_7", reset_xform_properties=False
        )
        self.obj_part_7_manipulator = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part7/manipulator2/robotiq_arg2f_base_link", name="obj_part_7_manipulator", reset_xform_properties=False
        )
        self.obj_part_9_manipulator = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part9/manipulator2/robotiq_arg2f_base_link", name="obj_part_9_manipulator", reset_xform_properties=False
        )
        self.obj_11_station_0 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0", name="obj_11_station_0", reset_xform_properties=False
        )
        self.obj_11_station_0_revolution = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/revolution", name="Station0/revolution", reset_xform_properties=False
        )
        self.obj_11_station_1_revolution = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/revolution", name="Station1/revolution", reset_xform_properties=False
        )
        self.obj_11_station_0_middle = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/middle_left", name="Station0/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_1_middle = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/middle_left", name="Station1/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_0_right = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/right", name="Station0/right", reset_xform_properties=False
        )
        self.obj_11_station_1_right = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/right", name="Station1/right", reset_xform_properties=False
        )
        self.obj_11_station_1 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1", name="obj_11_station_1", reset_xform_properties=False
        )
        self.obj_11_welding_0 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding0", name="obj_11_welding_0", reset_xform_properties=False
        )
        self.obj_11_welding_1 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding1", name="obj_11_welding_1", reset_xform_properties=False
        )
        self.obj_2_loader_0 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader0", name="obj_2_loader_0", reset_xform_properties=False
        )
        self.obj_2_loader_1 =  Articulation(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader1", name="obj_2_loader_1", reset_xform_properties=False
        )

    def set_up_material(self, num):
        if num > 0:
            _str = ("{}".format(num)).zfill(2)
        else:
            _str = "0"
        materials_cube = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_"+_str,
            name="cube_"+_str,
            track_contact_forces=True,
        )
        materials_hoop = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_"+_str,
            name="hoop_"+_str,
            track_contact_forces=True,
        )
        materials_bending_tube = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_"+_str,
            name="bending_tube_"+_str,
            track_contact_forces=True,
        )
        materials_upper_tube = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_"+_str,
            name="upper_tube_"+_str,
            track_contact_forces=True,
        )
        product = RigidPrim(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_"+_str,
            name="product_"+_str,
            track_contact_forces=True,
        )
        return materials_cube, materials_hoop, materials_bending_tube, materials_upper_tube, product

    def set_up_human(self, num):

        character_list = []
        for i in range(1, num+1):

            _str = ("{}".format(i)).zfill(2)
            character = RigidPrim(
                prim_paths_expr="/World/envs/.*/obj/Characters/male_adult_construction_"+_str,
                name="character_{}".format(i+1),
                track_contact_forces=True,
            )
            character_list.append(character)

        return character_list 
    
    def set_up_robot(self, num):

        box_list = []
        robot_list = []
        for i in range(1, num+1):
            box = RigidPrim(
                prim_paths_expr="/World/envs/.*/obj/AGVs/box_0{}".format(i),
                name="box_{}".format(i),
                track_contact_forces=True,
            )
            box_list.append(box)

            agv = Articulation(
                prim_paths_expr="/World/envs/.*/obj/AGVs/agv_0{}".format(i),
                name="agv_{}".format(i),
                reset_xform_properties=False,
            )
            robot_list.append(agv)
        return robot_list, box_list 

    def save_gantt_chart(self):
        # plt.show()
        import os, pickle
    
        gant_path = os.getcwd() + '/figs/gantt/gantt_data_D3QN.pkl'
        dic = {}   
        dic['initial'] = [self.task_manager.ini_worker_pose ,self.task_manager.ini_agv_pose, self.task_manager.ini_box_pose]
        dic['worker'] = []
        dic['worker_tasks_dic'] = set(self.task_manager.characters.fatigue_list[0].task_human_subtasks_dic.keys())
        dic['agv'] = []
        dic['agv_tasks_dic'] = self.task_manager.agvs.task_range
        dic['agv_tasks_dic'].add('none')
        for i in range(self.task_manager.characters.acti_num_charc):
            #[('free', 'free', 'none', self.phy_fatigue, self.psy_fatigue, self.time_step)] #state, subtask, task, time_step
            dic['worker'].append(self.task_manager.characters.fatigue_list[i].subtask_level_f_history)
        for i in range(self.task_manager.agvs.acti_num_agv):
            dic['agv'].append(self.task_manager.agvs.task_level_history[i])
        with open(gant_path, 'wb') as f:
            pickle.dump(dic, f)
        return
    
    @abstractmethod
    def get_observations(self) -> dict:
        return
    