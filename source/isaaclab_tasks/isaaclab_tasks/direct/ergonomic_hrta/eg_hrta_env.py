# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt pick task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
PYTHON_PATH omniisaacgymenvs/scripts/rlgames_train.py task=FactoryTaskNutBoltPick
"""
import torch
from typing import Tuple
from .eg_hrta_env_base import HRTaskAllocEnvBase
from .eg_hrta_map_route import world_pose_to_navigation_pose
from isaacsim.core.prims import RigidPrim
import omni.physx.scripts.utils as physxUtils
from pxr import Gf, Sdf, Usd, UsdPhysics, UsdGeom, PhysxSchema
from omni.usd import get_world_transform_matrix

from ...utils import quaternion  
import numpy as np
import torch.nn.functional as Fun

MAX_FLOAT = 3.40282347e38
# import numpy as np

class HRTaskAllocEnv(HRTaskAllocEnvBase):
            
    def step(self, action: torch.Tensor | None, action_extra = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""
        # process actions debug TODO
        if self.env_rule_based_exploration:
            action = self.get_rule_based_action() if self.episode_length_buf[0] > 0 else action
        self.post_task_manager_step(action, action_extra)
        self._pre_physics_step(action)
        ###TODO only support single env training
        # action_mask
        while True:
            self.reset_step()
            self.episode_length_buf[:] += 1
            self.post_material_step()
            self.post_conveyor_belt_step()
            self.post_cutting_machine_step()
            self.post_grippers_step()
            self.post_weld_station_step()
            self.post_welder_step()
            self.done_update()
            self.update_task_mask()

            # check if we need to do rendering within the physics loop
            # note: checked here once to avoid multiple checks within the loop
            is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
            # perform physics stepping
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # self.scene.write_data_to_sim()
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

            if (self.task_mask[1:].count_nonzero() == 0 and self.reset_buf[0] == 0):
                # self.get_rule_based_action()
                self.post_task_manager_step(actions=torch.zeros([1], dtype=torch.int32))
            else:
                self.calculate_metrics()
                obs = self.get_observations()
                self.task_manager.obs = obs
                self.get_fatigue_data()
                break

        return obs, self.reward_buf, self.reset_buf, self.extras, action

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        task_id = actions[0] - 1
        task = self.task_manager.task_dic[task_id.item()]
        self.extras['action_info'] = task
        self.caculate_metric_action(actions)
        return actions

    def post_task_manager_step(self, actions, action_extra=None):
        if action_extra is not None and 'cost_mask' in action_extra.keys():
            self.task_manager.characters.cost_mask_from_net = action_extra['cost_mask']
        else:
            self.task_manager.characters.cost_mask_from_net = None
        #TODO only support single action, not actions
        task_id = -1 #default as none
        assert actions is not None, "action is None"
        #the action is contorlled by rl-based agent rather than the inner rule-based agent
        ###one hot vector to num classes
        # task_id = torch.argmax(actions[0], dim=0) - 1
        task_id = actions[0] - 1
        task = self.task_manager.task_dic[task_id.item()]
        # '''gantt chart'''
        # if self._test and self.gantt_chart_data:
        #     self.actions_list.append(task)
        #     self.time_frames.append(self.episode_length_buf[0].cpu().item())
        # '''end'''
        if task not in self.available_task_dic.keys(): #to ensure the safety of action 
            task = 'none'
        if task == 'none':
            pass
        else:
            if task == 'hoop_preparing' and self.materials.hoop_states.count(0) > 0:
                if self.task_manager.assign_task(task = 'hoop_preparing'):
                    self.state_depot_hoop = 1
            elif task == 'bending_tube_preparing' and self.materials.bending_tube_states.count(0) > 0:
                if self.task_manager.assign_task(task = 'bending_tube_preparing'):
                    self.state_depot_bending_tube = 1
            elif task == 'placing_product':
                self.task_manager.task_clearing(task='collect_product')
                if self.task_manager.assign_task(task='placing_product'):
                    self.task_manager.boxs.product_collecting_idx = -1 #collecting box
            elif self.task_manager.assign_task(task):
                if task == 'hoop_loading_inner':
                    hoop_index = self.materials.find_next_raw_hoop_index()
                    self.materials.hoop_states[hoop_index] = 3
                    self.materials.inner_hoop_processing_index = hoop_index
                elif task == 'hoop_loading_outer':
                    hoop_index = self.materials.find_next_raw_hoop_index()
                    self.materials.hoop_states[hoop_index] = 3
                    self.materials.outer_hoop_processing_index = hoop_index
                elif task == 'bending_tube_loading_inner':
                    _index = self.materials.find_next_raw_bending_tube_index()
                    self.materials.bending_tube_states[_index] = 3
                    self.materials.inner_bending_tube_loading_index = _index
                elif task == 'bending_tube_loading_outer':
                    _index = self.materials.find_next_raw_bending_tube_index()
                    self.materials.bending_tube_states[_index] = 3
                    self.materials.outer_bending_tube_loading_index = _index

        # '''gantt chart'''
        # if self._test and self.generate_gantt_chart:
        #     def convert(sub_dic, low2high_dic, task_ids):
        #         res = []
        #         low2high_dic['free'] = 'free'
        #         for id in task_ids:
        #             res.append([self.episode_length_buf[0].cpu().item(), sub_dic[id], low2high_dic[sub_dic[id]]])
        #         return res
        #     self.gantt_charc.append(convert(self.task_manager.characters.sub_task_character_dic, self.task_manager.characters.low2high_level_task_dic, self.task_manager.characters.tasks))
        #     self.gantt_agv.append(convert(self.task_manager.agvs.sub_task_dic, self.task_manager.agvs.low2high_level_task_dic, self.task_manager.agvs.tasks))
        #     '''end'''
        self.task_manager.step()
        for charac_idx in range(0, self.task_manager.characters.acti_num_charc):
            self.post_character_step(charac_idx)
        for agv_idx in range(0, self.task_manager.agvs.acti_num_agv):
            self.post_agv_step(agv_idx)
        for box_idx in range(0, self.task_manager.boxs.acti_num_box):
            self.post_trans_box_step(box_idx)

    def post_material_step(self):
        #part of materials state decision is in consideration
        capacity = self.task_manager.boxs.capacity
        if len(self.depot_hoop_set) > 0:
            for idx in self.depot_hoop_set:
                self.materials.hoop_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))
                self.materials.hoop_list[idx].set_world_poses(self.materials.position_depot_hoop[idx%capacity.hoop].to(self.cuda_device))
        elif self.state_depot_hoop == 2: #placed
            self.state_depot_hoop = 0
        
        if len(self.depot_bending_tube_set) > 0:
            for idx in self.depot_bending_tube_set:
                self.materials.bending_tube_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))
                self.materials.bending_tube_list[idx].set_world_poses(self.materials.position_depot_bending_tube[idx%capacity.bending_tube].to(self.cuda_device), self.materials.orientation_depot_bending_tube.to(self.cuda_device))
        elif self.state_depot_bending_tube == 2: #placed
            self.state_depot_bending_tube = 0

        for idx in self.depot_product_set:
            self.materials.product_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            self.materials.product_list[idx].set_world_poses(self.materials.position_depot_product[idx%5].to(self.cuda_device), self.materials.orientation_depot_product.to(self.cuda_device))

        #raw material
        for idx, state in zip(range(0, len(self.materials.hoop_states)), self.materials.hoop_states):
            if state == 0: #raw state
                self.materials.hoop_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))
                # self.materials.hoop_list[idx].set_world_poses(self.materials.position_raw_hoop[idx].to(self.cuda_device), self.materials.orientation_raw_hoop.to(self.cuda_device))
                
        for idx, state in zip(range(0, len(self.materials.bending_tube_states)), self.materials.bending_tube_states):
            if state == 0: #raw state
                self.materials.bending_tube_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))
                # self.materials.bending_tube_list[idx].set_world_poses(self.materials.position_raw_bending_tube[idx].to(self.cuda_device), self.materials.orientation_raw_bending_tube.to(self.cuda_device))
        
        for idx, state in zip(range(0, len(self.materials.cube_states)), self.materials.cube_states):
            if state == 0 or state == 1:
                position, orientaion = self.materials.cube_list[idx].get_world_poses()
                # self.materials.cube_list[idx].set_world_poses(position, orientaion)
                self.materials.cube_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))
    
    def post_character_step(self, idx):
        charac : RigidPrim = self.task_manager.characters.list[idx]
        state = self.task_manager.characters.states[idx]
        task = self.task_manager.characters.tasks[idx]
        high_level_task = self.task_manager.characters.low2high_level_task_mapping(task)
        self.task_manager.characters.step_fatigue(idx, state, task, high_level_task)
        _, corresp_agv_idx, corresp_box_idx = self.task_manager.corresp_charac_agv_box_idx(high_level_task) 
        current_pose = charac.get_world_poses()
        target_position = None
        target_orientation = None
        if state == 0: #"worker is free" 
            if task > 0:
                self.task_manager.characters.states[idx] = 1
            target_position, target_orientation = current_pose
        elif state == 1: #worker is approaching 
            reaching_flag = False
            if len(self.task_manager.characters.x_paths[idx]) == 0:
                target_position, target_orientation = current_pose
                s = world_pose_to_navigation_pose(current_pose)
                if task == 1:
                    g = self.task_manager.characters.picking_pose_hoop
                elif task == 2:
                    g = self.task_manager.characters.picking_pose_bending_tube
                elif task == 3:
                    g = self.task_manager.characters.picking_pose_table_hoop
                elif task == 4:
                    g = self.task_manager.characters.picking_pose_table_bending_tube
                elif task == 5 or task == 7: #hoop_loading_inner or outer
                    g = self.task_manager.characters.loading_pose_hoop
                elif task == 6 or task == 8: #bending_tube_loading_inner or outer
                    g = self.task_manager.characters.loading_pose_bending_tube
                elif task == 9:
                    g = self.task_manager.characters.cutting_cube_pose
                elif task == 10:
                    g = self.task_manager.characters.placing_product_pose
                if np.linalg.norm(np.array(s[:2]) - np.array(g[:2])) < 0.1:
                    reaching_flag = True
                else:
                    s_str = self.task_manager.find_closest_pose(pose_dic=self.task_manager.characters.poses_dic, ego_pose=s, in_dis=3)
                    g_str = self.task_manager.characters.sub_task_character_dic[task]
                    # self.task_manager.characters.x_paths[idx], self.task_manager.characters.y_paths[idx], self.task_manager.characters.yaws[idx] = self.path_planner(s.copy(), g.copy())
                    self.task_manager.characters.x_paths[idx], self.task_manager.characters.y_paths[idx], self.task_manager.characters.yaws[idx] = self.task_manager.characters.routes_dic[s_str][g_str]
            else:
                target_position, target_orientation, reaching_flag = self.task_manager.characters.step_next_pose(charac_idx = idx)
                target_position, target_orientation = torch.tensor(np.expand_dims(target_position, axis=0), device=self.cuda_device, dtype=torch.float32), torch.tensor(np.expand_dims(target_orientation, axis=0), device=self.cuda_device, dtype=torch.float32)
                if reaching_flag: 
                    self.task_manager.characters.reset_path(idx)

            if reaching_flag:
                if task in range(1, 5) or task == 10:
                    self.task_manager.characters.states[idx] = 2
                elif task in range(5, 9): #loading
                    self.task_manager.characters.states[idx] = 5
                elif task == 9: #cutting machine
                    self.task_manager.characters.states[idx] = 6
                
        elif state == 2: #worker is waiting agv
            if corresp_box_idx >= 0 and self.task_manager.agvs.states[corresp_agv_idx] == 3:
                if task in range(1, 3):
                    self.task_manager.characters.states[idx] = 3 
                elif task in range(3, 5) or task == 10:
                    self.task_manager.characters.states[idx] = 4
            target_position, target_orientation = charac.get_world_poses()
        elif state == 3: #putting in box 
            target_position, target_orientation = current_pose
            available_material = True
            can_load = True
            assert corresp_box_idx >= 0, "corresp_box_idx"
            assert corresp_agv_idx >= 0, "corresp_agv_idx"
            #check available materials first
            if task == 1:
                try: self.materials.hoop_states.index(0)
                except: available_material = False
                
                if self.task_manager.boxs.counts[corresp_box_idx] >= self.task_manager.boxs.capacity.hoop or available_material == False:
                    self.task_manager.characters.states[idx] = 1
                    self.task_manager.characters.tasks[idx] = 3 #put_hoop_on_table
                    self.task_manager.agvs.tasks[corresp_agv_idx] = 3
                    self.task_manager.agvs.states[corresp_agv_idx] = 2
                    can_load = False
            elif task == 2:
                try: self.materials.bending_tube_states.index(0)
                except: available_material = False
                if self.task_manager.boxs.counts[corresp_box_idx] >= self.task_manager.boxs.capacity.bending_tube or available_material == False:
                    self.task_manager.characters.states[idx] = 1
                    self.task_manager.characters.tasks[idx] = 4 #put_bending_tube_on_table
                    self.task_manager.agvs.tasks[corresp_agv_idx] = 4
                    self.task_manager.agvs.states[corresp_agv_idx] = 2
                    can_load = False
            if can_load and self.task_manager.characters.loading_operation_time_steps[idx] > self.task_manager.characters.PUTTING_TIME + self.temp_random_time:
                self.reset_worker_random_time()
                self.task_manager.characters.loading_operation_time_steps[idx] = 0.
                self.task_manager.boxs.counts[corresp_box_idx] += 1
                if task == 1:
                    hoop_idx = self.materials.hoop_states.index(0)
                    self.materials.hoop_states[hoop_idx] = 1
                    self.task_manager.boxs.hoop_idx_list[corresp_box_idx].append(hoop_idx)
                elif task == 2:
                    bending_tube_idx = self.materials.bending_tube_states.index(0)
                    self.materials.bending_tube_states[bending_tube_idx] = 1
                    self.task_manager.boxs.bending_tube_idx_sets[corresp_box_idx].add(bending_tube_idx)
            else:
                self.task_manager.characters.loading_operation_time_steps[idx] += self.task_manager.characters.step_processing(idx)
        elif state == 4: #putting materails
            target_position, target_orientation = current_pose
            if self.task_manager.boxs.counts[corresp_box_idx] == 0: #finished 
                if task == 3:
                    self.state_depot_hoop = 2 #placed
                    self.task_manager.task_clearing(task='hoop_preparing')
                elif task == 4:
                    self.state_depot_bending_tube = 2 #placed
                    self.task_manager.task_clearing(task='bending_tube_preparing')
                elif task == 10: 
                    self.task_manager.task_clearing(task='placing_product')
            elif self.task_manager.characters.loading_operation_time_steps[idx] > self.task_manager.characters.PUTTING_TIME + self.temp_random_time:
                self.reset_worker_random_time()
                self.task_manager.characters.loading_operation_time_steps[idx] = 0.
                self.task_manager.boxs.counts[corresp_box_idx] -= 1
                if task == 3:
                    try:
                        hoop_idx = self.task_manager.boxs.hoop_idx_list[corresp_box_idx].pop()
                        self.materials.hoop_states[hoop_idx] = 2
                        self.depot_hoop_set.add(hoop_idx)
                    except:
                        pass
                elif task == 4:
                    try:
                        bending_tube_idx = self.task_manager.boxs.bending_tube_idx_sets[corresp_box_idx].pop()
                        self.materials.bending_tube_states[bending_tube_idx] = 2
                        self.depot_bending_tube_set.add(bending_tube_idx)
                    except:
                        pass
                elif task == 10:
                    product_index = self.task_manager.boxs.product_idx_list[corresp_box_idx].pop()
                    self.materials.product_states[product_index] = 2
                    self.depot_product_set.add(product_index)
            else:
                self.task_manager.characters.loading_operation_time_steps[idx] += self.task_manager.characters.step_processing(idx)
        
        elif state == 5: #loading 
            target_position, target_orientation = current_pose
            if self.task_manager.characters.loading_operation_time_steps[idx] > self.task_manager.characters.LOADING_TIME + self.temp_random_time:
                self.reset_worker_random_time()
                self.task_manager.characters.loading_operation_time_steps[idx] = 0.
                if task == 5: 
                    self.depot_hoop_set.remove(self.materials.inner_hoop_processing_index)
                    self.station_state_inner_left = 2
                    self.materials.hoop_states[self.materials.inner_hoop_processing_index] = 5
                    self.task_manager.task_clearing(task='hoop_loading_inner')
                elif task == 6:
                    self.depot_bending_tube_set.remove(self.materials.inner_bending_tube_loading_index)  
                    self.station_state_inner_right = 2    
                    self.materials.bending_tube_states[self.materials.inner_bending_tube_loading_index] = 5
                    self.task_manager.task_clearing(task='bending_tube_loading_inner')            
                elif task == 7:
                    self.depot_hoop_set.remove(self.materials.outer_hoop_processing_index)  
                    self.station_state_outer_left = 2       
                    self.materials.hoop_states[self.materials.outer_hoop_processing_index] = 5           
                    self.task_manager.task_clearing(task='hoop_loading_outer')
                elif task == 8:
                    self.depot_bending_tube_set.remove(self.materials.outer_bending_tube_loading_index)
                    self.station_state_outer_right = 2
                    self.materials.bending_tube_states[self.materials.outer_bending_tube_loading_index] = 5
                    self.task_manager.task_clearing(task='bending_tube_loading_outer')
            else:
                self.task_manager.characters.loading_operation_time_steps[idx] += self.task_manager.characters.step_processing(idx)
        elif state == 6: #cutting machine
            target_position, target_orientation = current_pose
            self.c_machine_oper_time += self.task_manager.characters.step_processing(idx)
            if self.cutting_machine_state == 2: #is resetting
                self.task_manager.task_clearing(task='cutting_cube')
            
        charac.set_world_poses(positions=target_position, orientations=target_orientation)
        charac.set_velocities(torch.zeros((1,6), device=self.cuda_device))    

        return
    
    def post_agv_step(self, idx):
        agv : RigidPrim = self.task_manager.agvs.list[idx]
        state = self.task_manager.agvs.states[idx]
        task = self.task_manager.agvs.tasks[idx]
        high_level_task = self.task_manager.agvs.low2high_level_task_mapping(task)
        _, _, corresp_box_idx = self.task_manager.corresp_charac_agv_box_idx(high_level_task)
        if self._test and self.gantt_chart_data:
            self.task_manager.agvs.step_gantt_chart(idx, state, task, high_level_task)
        # corresp_charac_idx = self.task_manager.agvs.corresp_charac_idxs[idx] 
        # corresp_box_idx = self.task_manager.agvs.corresp_box_idxs[idx] 
        target_position = None
        target_orientation = None
        current_pose = agv.get_world_poses()
        if state == 0: #"free"
            target_position, target_orientation = current_pose
            if task > 0:
                self.task_manager.agvs.states[idx] = 1
        elif state == 1: #moving to box
            if corresp_box_idx < 0:
                target_position, target_orientation = current_pose
            else:
                reaching_flag = False
                if len(self.task_manager.agvs.x_paths[idx]) == 0:
                    target_position, target_orientation = current_pose
                    box_pose = self.task_manager.boxs.list[corresp_box_idx].get_world_poses()
                    box_position = box_pose[0]
                    dis = torch.norm(box_position[0][:2] - current_pose[0][0][:2])
                    if dis < 0.1:
                        reaching_flag = True
                    else:
                        s, g = world_pose_to_navigation_pose(current_pose), world_pose_to_navigation_pose(box_pose)
                        '''no spatial information'''
                        s_str = self.task_manager.find_closest_pose(pose_dic=self.task_manager.agvs.poses_dic, ego_pose=s, in_dis=15)
                        g_str = self.task_manager.find_closest_pose(pose_dic=self.task_manager.agvs.poses_dic, ego_pose=g, in_dis=15)
                        if s_str == g_str:
                            reaching_flag = True
                        else:
                            self.task_manager.agvs.x_paths[idx], self.task_manager.agvs.y_paths[idx], self.task_manager.agvs.yaws[idx] = self.task_manager.agvs.routes_dic[s_str][g_str]
                        # self.task_manager.agvs.x_paths[idx], self.task_manager.agvs.y_paths[idx], self.task_manager.agvs.yaws[idx] = self.path_planner(s.copy(), g.copy())
                else:
                    target_position, target_orientation, reaching_flag = self.task_manager.agvs.step_next_pose(agv_idx = idx)
                    target_position, target_orientation = torch.tensor(np.expand_dims(target_position, axis=0), device=self.cuda_device, dtype=torch.float32), torch.tensor(np.expand_dims(target_orientation, axis=0), device=self.cuda_device, dtype=torch.float32)
                    if reaching_flag:
                        self.task_manager.agvs.reset_path(idx)
                if reaching_flag:
                    self.task_manager.agvs.states[idx] = 2 #carrying_box       
                    self.task_manager.boxs.tasks[corresp_box_idx] = 2 #moving_with_box
                    
        elif state == 2: #carrying_box
            reaching_flag = False
            if len(self.task_manager.agvs.x_paths[idx]) == 0:
                target_position, target_orientation = current_pose
                s = world_pose_to_navigation_pose(current_pose)
                if task == 1:
                    g = self.task_manager.agvs.picking_pose_hoop  
                elif task == 2:
                    g = self.task_manager.agvs.picking_pose_bending_tube
                elif task == 3:
                    g = self.task_manager.agvs.picking_pose_table_hoop
                elif task == 4:
                    g = self.task_manager.agvs.picking_pose_table_bending_tube
                elif task == 5:
                    g = self.task_manager.agvs.collecting_product_pose
                elif task == 6:
                    g = self.task_manager.agvs.placing_product_pose
                if np.linalg.norm(np.array(s[:2]) - np.array(g[:2])) < 0.1:
                    reaching_flag = True
                else:
                    s_str = self.task_manager.find_closest_pose(pose_dic=self.task_manager.agvs.poses_dic, ego_pose=s, in_dis=3)
                    g_str = self.task_manager.agvs.sub_task_dic[task]
                    self.task_manager.agvs.x_paths[idx], self.task_manager.agvs.y_paths[idx], self.task_manager.agvs.yaws[idx] = self.task_manager.agvs.routes_dic[s_str][g_str]
                    # self.task_manager.agvs.x_paths[idx], self.task_manager.agvs.y_paths[idx], self.task_manager.agvs.yaws[idx] = self.path_planner(s.copy(), g.copy())
                # self.task_manager.agvs.path_idxs[idx] = 0
            else:
                target_position, target_orientation, reaching_flag = self.task_manager.agvs.step_next_pose(agv_idx = idx)
                target_position, target_orientation = torch.tensor(np.expand_dims(target_position, axis=0), device=self.cuda_device, dtype=torch.float32), torch.tensor(np.expand_dims(target_orientation, axis=0), device=self.cuda_device, dtype=torch.float32)
                if reaching_flag:
                    self.task_manager.agvs.reset_path(idx)
            if reaching_flag:
                if task == 5: #reset agvs state
                    self.task_manager.agvs.reset_idx(idx)
                    assert corresp_box_idx>=0, "error"
                    self.task_manager.boxs.states[corresp_box_idx] = 1 #waiting
                    self.task_manager.boxs.tasks[corresp_box_idx] = 3 #collect products
                    self.task_manager.task_in_dic[high_level_task]['agv_idx'] = -2 #task collecting product dont need agv later 
                else:
                    self.task_manager.agvs.states[idx] = 3
                
        elif state == 3: #finished carrying box to the target position and waiting 
            target_position, target_orientation = current_pose
        #to avoid the object fall underground
        target_position[:,-1] = 0.1
        agv.set_world_poses(positions=target_position, orientations=target_orientation)  
        agv.set_velocities(torch.zeros((1,6), device=self.cuda_device))  

        return 
    
    def post_trans_box_step(self, idx):
        box : RigidPrim = self.task_manager.boxs.list[idx]
        state = self.task_manager.boxs.states[idx]
        task = self.task_manager.boxs.tasks[idx]
        high_level_task = self.task_manager.boxs.high_level_tasks[idx]
        _, corresp_agv_idx, _ = self.task_manager.corresp_charac_agv_box_idx(high_level_task)
        target_position = None
        target_orientation = None
        current_pose = box.get_world_poses()
        hoop_idx_list = self.task_manager.boxs.hoop_idx_list[idx]
        bending_tube_idx_set = self.task_manager.boxs.bending_tube_idx_sets[idx]
        product_idx_list = self.task_manager.boxs.product_idx_list[idx]
        # capacity
        if state == 0: #free
            target_position, target_orientation = current_pose
            if task == 1:
                self.task_manager.boxs.states[idx] = 1
            elif task == 2 :
                self.task_manager.boxs.states[idx] = 2
        elif state == 1: #wating
            target_position, target_orientation = current_pose
            if corresp_agv_idx >= 0 and task == 2: #moving_with_box
                self.task_manager.boxs.states[idx] = 2
            elif task == 3: #collect_product
                '''collecting product and waiting for task manager to end the collecting task'''
        elif state == 2: #moving
            #todo
            assert corresp_agv_idx>=0, "error"
            target_position, target_orientation = self.task_manager.agvs.list[corresp_agv_idx].get_world_poses()
        #to avoid the object fall underground
        target_position[0][-1] = 0
        box.set_world_poses(positions=target_position, orientations=target_orientation)
        box.set_velocities(torch.zeros((1,6), device=self.cuda_device))  
        for idx in hoop_idx_list:
            offset=self.materials.in_box_offsets[idx%self.task_manager.boxs.capacity.hoop].to(self.cuda_device)
            self.materials.hoop_list[idx].set_world_poses(positions=target_position+offset)
            self.materials.hoop_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))  
        for idx in bending_tube_idx_set:
            offset=self.materials.in_box_offsets[idx%self.task_manager.boxs.capacity.bending_tube].to(self.cuda_device)
            self.materials.bending_tube_list[idx].set_world_poses(positions=target_position+offset)
            self.materials.bending_tube_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))  
        for idx in product_idx_list:
            offset=self.materials.in_box_offsets[idx%self.task_manager.boxs.capacity.product].to(self.cuda_device)
            self.materials.product_list[idx].set_world_poses(positions=target_position+offset)
            self.materials.product_list[idx].set_velocities(torch.zeros((1,6), device=self.cuda_device))  
        return

    def post_conveyor_belt_step(self):
        '''material long cube'''
        #first check the state
        if  self.convey_state == 0:
            #conveyor is free, we can process it
            # raw_cube_index, upper_tube_index, hoop_index, bending_tube_index = self.post_next_group_to_be_processed_step()
            raw_cube_index = self.materials.find_next_raw_cube_index()
            if raw_cube_index >=0 :
                #todo check convey startup
                self.materials.cube_convey_index = raw_cube_index
                self.convey_state = 1
                self.materials.cube_states[raw_cube_index] = 2
        elif self.convey_state ==1:
            #conveyor is waiting for the cube 
            if self.put_cube_on_conveyor(self.materials.cube_convey_index):
                self.convey_state = 2
                self.materials.cube_list[self.materials.cube_convey_index].set_world_poses(positions=torch.tensor([[0,   0.5,   1.5]], device=self.cuda_device), orientations=torch.tensor([[ 1.0, 0, 0.0,  0]], device=self.cuda_device))
                self.materials.cube_list[self.materials.cube_convey_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        elif self.convey_state == 2:
            #start conveying, the threhold means the cube arrived cutting area
            threhold = -21.5
            obj_index = self.materials.cube_convey_index
            obj :RigidPrim = self.materials.cube_list[obj_index]
            obj_state = self.materials.cube_states[obj_index]
            obj_world_pose = obj.get_world_poses()
            #check the threhold to know whether the cude arrived at cutting position
            if obj_state == 2:
                if obj_world_pose[0][0][0] <= threhold:
                    #conveyed to the cutting machine
                    self.materials.cube_states[obj_index] = 3
                    self.materials.cube_cut_index = obj_index
                else:
                    #keep conveying the cube
                    obj_world_pose[0][0][0] -=  0.4
                obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 1.0, 0, 0.0,  0]], device=self.cuda_device))
                obj.set_velocities(torch.zeros((1,6), device=self.cuda_device))
                return
            elif obj_state in range(3,6):
                #3:"conveyed", 4:"cutting", 5:"cut_done",
                # obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device=self.cuda_device))
                # obj.set_velocities(torch.zeros((1,6), device=self.cuda_device))
                obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 1.0, 0, 0.0,  0]], device=self.cuda_device))
                obj.set_velocities(torch.zeros((1,6), device=self.cuda_device))
                return
            elif obj_state == 6:
                self.convey_state = 0
                self.materials.cube_convey_index = -1
        return        

    def put_cube_on_conveyor(self, cude_index) -> bool:
        #todo 
        return True

    def post_cutting_machine_step(self):
        dof_pos_10 = None
        dof_vel_10 = torch.tensor([[0., 0]], device=self.cuda_device)
        initial_pose = torch.tensor([[-5., 0]], device=self.cuda_device)
        end_pose = torch.tensor([[-5, 0.35]], device=self.cuda_device)
        cube_cut_index = self.materials.cube_cut_index
        if self.cutting_machine_state == 0:
            '''reseted laser cutter'''
            # delta_position = torch.tensor([[0., 0]], device=self.cuda_device)
            dof_pos_10 = initial_pose
            if cube_cut_index>=0:
                self.cutting_machine_state = 1
        elif self.cutting_machine_state == 1:
            '''cutting cube'''
            if self.c_machine_oper_time < self.c_machine_oper_len + self.temp_random_time:
                # self.c_machine_oper_time += 1 by human worker
                dof_pos_10 = (end_pose - initial_pose)*self.c_machine_oper_time/(self.c_machine_oper_len + self.temp_random_time) + initial_pose
                self.materials.cube_states[cube_cut_index] = 4
            elif self.c_machine_oper_time >= self.c_machine_oper_len + self.temp_random_time:
                self.reset_worker_random_time()
                self.c_machine_oper_time = 0
                self.cutting_machine_state = 2
                dof_pos_10 = end_pose
                #sending picking flag to gripper
        elif self.cutting_machine_state == 2:
            '''reseting machine'''
            if self.c_machine_oper_time < self.c_machine_oper_len + self.temp_random_time:
                self.c_machine_oper_time += 1
                dof_pos_10 = (initial_pose - end_pose)*self.c_machine_oper_time/(self.c_machine_oper_len + self.temp_random_time) + end_pose
            elif self.c_machine_oper_time >= self.c_machine_oper_len + self.temp_random_time:
                self.reset_machine_random_time()
                self.c_machine_oper_time = 0
                self.cutting_machine_state = 0
                dof_pos_10 = initial_pose
                #for inner gripper start picking the cut cube
                self.materials.cube_states[cube_cut_index] = 5
                self.materials.pick_up_place_cube_index = cube_cut_index
                self.materials.cube_cut_index = -1
        self.obj_part_10.set_joint_positions(dof_pos_10[0])
        self.obj_part_10.set_joint_velocities(dof_vel_10[0])   

    def post_grippers_step(self):
        next_pos_inner = None
        next_pos_outer = None
        next_gripper_pose = torch.zeros(size=(20,), device=self.cuda_device)
        dof_vel = torch.zeros(size=(1,20), device=self.cuda_device)

        gripper_pose = self.obj_part_7.get_joint_positions(clone=False)
        gripper_pose_inner = torch.index_select(gripper_pose, 1, torch.tensor([1,3,5,7,12,13,14,15,18,19], device='cuda'))
        gripper_pose_outer = torch.index_select(gripper_pose, 1, torch.tensor([0,2,4,6,8,9,10,11,16,17], device='cuda'))

        next_pos_inner = self.post_inner_gripper_step(next_pos_inner, gripper_pose_inner)
        next_pos_outer = self.post_outer_gripper_step(next_pos_outer, gripper_pose_outer, gripper_pose_inner)
        next_gripper_pose = self.merge_two_grippers_pose(next_gripper_pose, next_pos_inner, next_pos_outer)

        self.obj_part_7.set_joint_positions(next_gripper_pose)
        self.obj_part_7.set_joint_velocities(dof_vel[0])
        self.obj_part_7.set_joint_efforts(dof_vel[0])
        return 
    
    def post_inner_gripper_step(self, next_pos_inner, gripper_pose_inner):
        pick_up_place_cube_index = self.materials.pick_up_place_cube_index
        inner_initial_pose = torch.zeros(size=(1,10), device=self.cuda_device)
        positions, orientations = self.obj_part_9_manipulator.get_world_poses()
        translate_tensor = torch.tensor([[0,   0,   0]], device=self.cuda_device) #default as cube
        if self.gripper_inner_state == 0:
            #gripper is free and empty todo
            self.gripper_inner_task = 0
            # stations_are_full = self.station_state_inner_middle and self.station_state_outer_middle #only state == 0 means free, -1 and >= 0 means full
            if self.station_state_inner_middle == 9 and 'collect_product' in self.task_manager.task_in_dic and self.task_manager.boxs.product_collecting_idx >= 0 \
                and self.task_manager.boxs.states[self.task_manager.boxs.product_collecting_idx] == 1 \
                and self.task_manager.boxs.counts[self.task_manager.boxs.product_collecting_idx]<self.task_manager.boxs.capacity.product: #welded product
                self.gripper_inner_task = 4
                self.gripper_inner_state = 1
            elif self.station_state_outer_middle == 9 and  'collect_product' in self.task_manager.task_in_dic and self.task_manager.boxs.product_collecting_idx >= 0 \
                and self.task_manager.boxs.states[self.task_manager.boxs.product_collecting_idx] == 1 \
                and self.task_manager.boxs.counts[self.task_manager.boxs.product_collecting_idx]<self.task_manager.boxs.capacity.product:
                self.gripper_inner_task = 5
                self.gripper_inner_state = 1
            elif (pick_up_place_cube_index>=0): #pick cut cube by cutting machine
                if self.station_state_inner_middle == 0 or (self.station_state_outer_middle == 0 and self.gripper_outer_state == 0): #station is waiting
                    self.gripper_inner_task = 1
                    self.gripper_inner_state = 1
            
            next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], inner_initial_pose[0], 'reset')
        elif self.gripper_inner_state == 1:
            #gripper is picking
            if self.gripper_inner_task == 1: #picking cut cube
                target_pose = torch.tensor([[0.5, 0, -0.5, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device=self.cuda_device)
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    #choose which station to place on the cube
                    self.gripper_inner_state = 2
                    if self.station_state_inner_middle == 0 and (self.station_state_outer_middle == 0 and self.gripper_outer_state == 0):
                        no_load = self.materials.inner_hoop_processing_index < 0 and self.materials.outer_hoop_processing_index < 0
                        if no_load or self.materials.inner_hoop_processing_index >= 0:
                            self.gripper_inner_task = 2 #place on inner station
                            self.materials.inner_cube_processing_index = pick_up_place_cube_index
                        else:
                            self.gripper_inner_task = 3 #place on outer station
                            self.materials.outer_cube_processing_index = pick_up_place_cube_index
                    elif self.station_state_inner_middle == 0:
                        self.gripper_inner_task = 2 #place on inner station
                        self.materials.inner_cube_processing_index = pick_up_place_cube_index
                    else:
                        assert self.station_state_outer_middle == 0
                        self.gripper_inner_task = 3 #place on outer station
                        self.materials.outer_cube_processing_index = pick_up_place_cube_index
                    # if self.station_state_inner_middle == 0 and (self.station_state_outer_middle == 0 and self.gripper_outer_state == 0):
                    #     no_load = self.materials.inner_hoop_processing_index < 0 and self.materials.outer_hoop_processing_index < 0
                    #     if no_load or self.materials.outer_hoop_processing_index >= 0:
                    #         self.gripper_inner_task = 3 #place on outer station
                    #         self.materials.outer_cube_processing_index = pick_up_place_cube_index
                    #     else:
                    #         self.gripper_inner_task = 2 #place on inner station
                    #         self.materials.inner_cube_processing_index = pick_up_place_cube_index
                    # elif self.station_state_outer_middle == 0:
                    #     self.gripper_inner_task = 3 #place on outer station
                    #     self.materials.outer_cube_processing_index = pick_up_place_cube_index
                    # else:
                    #     assert self.station_state_inner_middle == 0
                    #     self.gripper_inner_task = 2 #place on inner station
                    #     self.materials.inner_cube_processing_index = pick_up_place_cube_index
            elif self.gripper_inner_task == 4: #pick_product_from_inner
                target_pose = torch.tensor([[-2.55, -1, -0.8, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device=self.cuda_device)
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_inner_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_inner_task = 6 #place_product from inner
                    self.station_state_inner_middle = -1
            elif self.gripper_inner_task == 5: #pick_product_from_outer
                target_pose = torch.tensor([[-2.55-3.44, -1, -0.8, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device=self.cuda_device)
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_inner_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_inner_task = 7 #place_product from outer
                    self.station_state_outer_middle = -1
        elif self.gripper_inner_state == 2: #gripper is placeing
            placing_material = None
            if self.gripper_inner_task == 2: #place_cut_to_inner_station
                self.materials.cube_states[pick_up_place_cube_index] = 6
                placing_material = self.materials.cube_list[pick_up_place_cube_index]
                target_pose = torch.tensor([[-2.4, 0, -1.25, 0, 0, 0, 0, 0, 0, 0]], device=self.cuda_device)
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    self.gripper_inner_state = 0
                    self.materials.cube_states[pick_up_place_cube_index] = 7
                    self.station_state_inner_middle = 2
                    self.materials.pick_up_place_cube_index = -1
            elif self.gripper_inner_task == 3: #place_cut_to_outer_station
                self.materials.cube_states[pick_up_place_cube_index] = 6
                placing_material = self.materials.cube_list[pick_up_place_cube_index]
                target_pose = torch.tensor([[-5.7, 0, -1.25, 0, 0, 0, 0, 0, 0, 0]], device=self.cuda_device)
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    self.gripper_inner_state = 0
                    self.materials.cube_states[pick_up_place_cube_index] = 8
                    self.station_state_outer_middle = 2
                    self.materials.pick_up_place_cube_index = -1
            elif self.gripper_inner_task == 6: #place_product_from_inner
                self.materials.cube_states[self.materials.inner_cube_processing_index] = 13
                placing_material = self.materials.product_list[self.materials.inner_cube_processing_index]
                orientations = torch.tensor([[ 1, 0,0,0.0]], device=self.cuda_device)
                translate_tensor = torch.tensor([[0.,  0, -2.]], device=self.cuda_device)
                target_pose = torch.tensor([[-9.7, -1, -1, 0, 0, 0, 0, 0, 0, 0]], device=self.cuda_device)
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    "a product is produced and placed on the robot, then do resetting"
                    self.materials.cube_states[self.materials.inner_cube_processing_index] = -2
                    self.gripper_inner_state = 0
                    self.materials.product_states[self.materials.inner_cube_processing_index] = 1 # product is collected
                    collecting_box_idx = self.task_manager.task_in_dic['collect_product']['box_idx'] 
                    assert collecting_box_idx >=0, "box idx error"
                    self.task_manager.boxs.product_idx_list[collecting_box_idx].append(self.materials.inner_cube_processing_index)
                    self.task_manager.boxs.counts[collecting_box_idx] += 1
                    self.materials.inner_cube_processing_index = -1
            elif self.gripper_inner_task == 7: #place_product_from_outer
                self.materials.cube_states[self.materials.outer_cube_processing_index] = 13
                placing_material = self.materials.product_list[self.materials.outer_cube_processing_index]
                orientations = torch.tensor([[1,0,0,0.0]], device=self.cuda_device)
                translate_tensor = torch.tensor([[0, 0., -2.]], device=self.cuda_device)
                target_pose = torch.tensor([[-9.7, -1, -1, 0, 0, 0, 0, 0, 0, 0]], device=self.cuda_device)
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    "a product is produced and placed on the robot, then do resetting"
                    self.materials.cube_states[self.materials.outer_cube_processing_index] = -2
                    self.gripper_inner_state = 0
                    self.materials.product_states[self.materials.outer_cube_processing_index] = 1 # product is collected
                    collecting_box_idx = self.task_manager.task_in_dic['collect_product']['box_idx'] 
                    assert collecting_box_idx >=0, "box idx error"
                    self.task_manager.boxs.product_idx_list[collecting_box_idx].append(self.materials.outer_cube_processing_index)
                    self.task_manager.boxs.counts[collecting_box_idx] += 1
                    self.materials.outer_cube_processing_index = -1

            # ref_pose[0] += torch.tensor([[0,   0,   -0.3]], device=self.cuda_device)
            placing_material.set_world_poses(positions=positions+translate_tensor, orientations=orientations)
            placing_material.set_velocities(torch.zeros((1,6), device=self.cuda_device))
            
            # self.materials.cube_list[pick_up_place_cube_index].set_world_poses(
            #     positions=ref_pose[0]+torch.tensor([[-1,   0,   0]], device=self.cuda_device), 
            #     orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device=self.cuda_device))
        # elif self.gripper_inner_state == 3:
        #     #gripper picked material
        #     a = 1
            # if self.gripper_inner_task == 
        return next_pos_inner
    
    def post_outer_gripper_step(self, next_pos_outer, gripper_pose_outer, gripper_pose_inner):
        '''if any station have weld upper tube request, activate the outer gripper'''
        pick_up_upper_tube_index = self.materials.pick_up_place_upper_tube_index
        outer_initial_pose = torch.zeros(size=(1,10), device=self.cuda_device)
        target_pose = torch.zeros(size=(1,10), device=self.cuda_device)
        if self.gripper_outer_state == 0:
            self.gripper_outer_task = 0
            if self.station_state_inner_middle == 7 and self.check_grippers_in_safe_distance(gripper_pose_inner, gripper_pose_outer): #welded_right
                # check inner gripper reset and wont collide with outer gripper
                self.materials.pick_up_place_upper_tube_index = self.materials.inner_upper_tube_processing_index
                self.gripper_outer_state = 1 #Picking
                self.gripper_outer_task = 1 #pick_upper_tube for inner station
            elif self.station_state_outer_middle == 7 and self.check_grippers_in_safe_distance(gripper_pose_inner, gripper_pose_outer):
                self.materials.pick_up_place_upper_tube_index = self.materials.outer_upper_tube_processing_index
                self.gripper_outer_state = 1 #Picking
                self.gripper_outer_task = 2 #pick_upper_tube for inner station
            next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], outer_initial_pose[0], 'reset')
        elif self.gripper_outer_state == 1:
            #gripper is picking
            if self.gripper_outer_task == 1:
                #picking upper cube for inner station
                target_pose = torch.tensor([[6.45, 0, -1.5, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device=self.cuda_device)
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_outer_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_outer_task = 3
            if self.gripper_outer_task == 2:
                #picking upper cube for outer station
                target_pose = torch.tensor([[6.45, 0, -1.5, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device=self.cuda_device)
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_outer_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_outer_task = 4
        elif self.gripper_outer_state == 2:
            #gripper is placeing
            position, orientation = self.obj_part_7_manipulator.get_world_poses()
            if self.gripper_outer_task == 3:
                #place upper tube to_inner_station
                target_pose = torch.tensor([[8.3, 0, -0.25, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device=self.cuda_device)
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'place')
                if move_done:
                    if self.station_state_inner_middle == 7:
                        self.station_state_inner_middle = 8 #welding_middle

                    elif self.station_state_inner_middle == 9: #welded_middle 
                        #if done reset the outer gripper
                        self.gripper_outer_state = 0
                else:
                    # orientation = torch.tensor([[ 1.0000e+00,  9.0108e-17, -1.9728e-17,  1.0443e-17]], device=self.cuda_device)
                    orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device=self.cuda_device)
                    self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(
                        positions=position+torch.tensor([[0,   0,   -1.7]], device=self.cuda_device), orientations=orientation)
                    self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            elif self.gripper_outer_task == 4:
                #place upper tube to outer station
                target_pose = torch.tensor([[5.2, 0, -0.25, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device=self.cuda_device)
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'place')
                if move_done:
                    if self.station_state_outer_middle == 7:
                        self.station_state_outer_middle = 8 #welding_middle
                    elif self.station_state_outer_middle == 9: #welded_middle 
                        #if done reset the outer gripper
                        self.gripper_outer_state = 0
                else:
                    # orientation = torch.tensor([[ 1.0000e+00,  9.0108e-17, -1.9728e-17,  1.0443e-17]], device=self.cuda_device)
                    orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device=self.cuda_device)
                    self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(
                        positions=position+torch.tensor([[0,   0,   -1.7]], device=self.cuda_device), orientations=orientation)
                    self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))

        return next_pos_outer
    
    def check_grippers_in_safe_distance(self, gripper_inner, gripper_outer):
        
        return (torch.abs(gripper_inner[0][0] + 10.73 - gripper_outer[0][0]) > 4 and self.gripper_inner_state == 0)

    def merge_two_grippers_pose(self, pose, pose_inner, pose_outer):
        pose[0:7:2] = pose_outer[:4]
        pose[1:8:2] = pose_inner[:4]
        pose[8:12] = pose_outer[4:8]
        pose[12:16] = pose_inner[4:8]
        pose[16:18] = pose_outer[8:]
        pose[18:] = pose_inner[8:]
        return pose
    # def get_material_pose_by_ref_pose(self, ref_pose, delta_pos):

    def get_gripper_moving_pose(self, gripper_pose : torch.Tensor, target_pose : torch.Tensor, task):
        #for one env pose generation
        ####debug
        # gripper_pose = torch.zeros((10), device=self.cuda_device)
        # target_pose = torch.tensor([-0.01148, 0, -1.36, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045], device=self.cuda_device)

        #warning, revolution is 0 when doing picking task
        THRESHOLD_A = 0.05
        THRESHOLD_B = 0.04
        threshold = torch.tensor([THRESHOLD_A]*3 + [THRESHOLD_B]*7, device=self.cuda_device)
        dofs = gripper_pose.shape[0]
        next_gripper_pose = torch.zeros(dofs, device=self.cuda_device)
        new_target_pose = torch.zeros(dofs, device=self.cuda_device)
        delta_pose = target_pose - gripper_pose
        move_done = torch.where(torch.abs(delta_pose)<threshold, True, False)
        new_target_pose = torch.zeros(dofs, device=self.cuda_device)
        next_gripper_pose = torch.zeros(dofs, device=self.cuda_device)
        next_delta_pose = torch.zeros(dofs, device=self.cuda_device)
        #todo
        # manipulator_reset_pose = torch.zeros(6, device=self.cuda_device)
        manipulator_reset_pose = torch.tensor([1.8084e-02, -2.9407e-02, -2.6935e-02, -1.6032e-02,  3.3368e-02,  3.2771e-02], device=self.cuda_device)
        delta_m = manipulator_reset_pose - gripper_pose[4:]
        reset_done_m = torch.where(torch.abs(delta_m)<THRESHOLD_B, True, False).all()
        '''todo, manipulator faces reseting and move done problems'''
        reset_done_m = True
        move_done[4:] = True
        '''todo, manipulator faces reseting and move done problems'''
        reset_done_revolution = torch.abs(gripper_pose[3]-0)<THRESHOLD_B
        reset_done_up_down = torch.abs(gripper_pose[2]-0)<THRESHOLD_A

        if task == 'place':
            reset_done_m = True
            reset_done_revolution = True
        if move_done.all():
            next_gripper_pose = target_pose
            return next_gripper_pose, next_delta_pose, True
        elif move_done[:4].all():
            #if in out, left right, up down, revolution done, control manipulator
            new_target_pose = target_pose
            # return self.get_gripper_pose_helper(gripper_pose, target_pose), False
        elif move_done[:3].all():
            if  reset_done_m:
                #move revolution, freeze others
                new_target_pose[:3] = gripper_pose[:3]
                new_target_pose[4:] = manipulator_reset_pose
                new_target_pose[3] = target_pose[3]
            else:
                #freeze [:4], do the manipulator reset
                new_target_pose[:4] = gripper_pose[:4]
                new_target_pose[4:] = manipulator_reset_pose
        elif move_done[:2].all():
            #check manipulator reset done
            if reset_done_m:
                new_target_pose[4:] = manipulator_reset_pose
                #check revolution reset done
                if reset_done_revolution:
                    # do the up down 
                    new_target_pose[:4] = gripper_pose[:4]
                    new_target_pose[2] = target_pose[2] 
                    new_target_pose[3] = 0
                else:
                    #freeze [:3] and reset revolution
                    new_target_pose[:3] = gripper_pose[:3]
                    new_target_pose[3] = 0
            else:
                #freeze [:4], do the manipulator reset
                new_target_pose[:4] = gripper_pose[:4]
                new_target_pose[4:] = manipulator_reset_pose
        else:
            #check manipulator reset done
            if reset_done_m:
                new_target_pose[4:] = manipulator_reset_pose
                #check revolution reset done
                if reset_done_revolution:
                    # new_target_pose[3] = gripper_pose[3]
                    new_target_pose[3] = 0
                    # check the up down 
                    if reset_done_up_down:
                        #do in out and left right
                        # new_target_pose[2] = gripper_pose[2]
                        new_target_pose[2] = 0
                        new_target_pose[:2] = target_pose[:2]
                    else:
                        new_target_pose[:2] = gripper_pose[:2]
                        new_target_pose[2] = 0
                else:
                    #freeze [:3]
                    new_target_pose[:3] = gripper_pose[:3]
                    new_target_pose[3] = 0
            else:
                #freeze [:4], do the manipulator reset
                new_target_pose[:4] = gripper_pose[:4]
                new_target_pose[4:] = manipulator_reset_pose
        next_gripper_pose, delta_pose = self.get_next_pose_helper(gripper_pose, new_target_pose, self.operator_gripper)
        return next_gripper_pose, delta_pose, False

    def get_next_pose_helper(self, pose, target_pose, operator_gripper):
        delta_pose = target_pose - pose
        sign = torch.sign(delta_pose)
        next_pose = sign*operator_gripper + pose
        next_pose_not_reach_target = torch.where((target_pose - next_pose)*delta_pose>0, True, False)
        next_pose = torch.where(next_pose_not_reach_target, next_pose, target_pose)
        delta_pose = next_pose - pose
        return next_pose, delta_pose
    
    def post_weld_station_step(self):
        #inner station step
        weld_station_inner_pose = self.obj_11_station_0.get_joint_positions()
        inner_revolution_target = self.post_weld_station_inner_hoop_step(weld_station_inner_pose[:, 0])
        inner_mid_target_A, inner_mid_target_B = self.post_weld_station_inner_cube_step(weld_station_inner_pose[:, 1], weld_station_inner_pose[:, 2])
        inner_right_target = self.post_weld_station_inner_bending_tube_step(weld_station_inner_pose[:, 3])
        self.post_weld_station_inner_tube_step()

        target_pose = torch.tensor([[inner_revolution_target,  inner_mid_target_A,  inner_mid_target_B, inner_right_target]], device=self.cuda_device)
        next_pose, _ = self.get_next_pose_helper(weld_station_inner_pose, target_pose, self.operator_station)
        self.obj_11_station_0.set_joint_positions(next_pose)
        self.obj_11_station_0.set_joint_velocities(torch.zeros(4, device=self.cuda_device))

        #outer station step
        weld_station_outer_pose = self.obj_11_station_1.get_joint_positions()
        outer_revolution_target = self.post_weld_station_outer_hoop_step(weld_station_outer_pose[:, 0])
        outer_mid_target_A, outer_mid_target_B = self.post_weld_station_outer_cube_step(weld_station_outer_pose[:, 1], weld_station_outer_pose[:, 2])
        outer_right_target = self.post_weld_station_outer_bending_tube_step(weld_station_outer_pose[:, 3])
        self.post_weld_station_outer_tube_step()

        target_pose = torch.tensor([[outer_revolution_target,  outer_mid_target_A,  outer_mid_target_B, outer_right_target]], device=self.cuda_device)
        next_pose, _ = self.get_next_pose_helper(weld_station_outer_pose, target_pose, self.operator_station)
        self.obj_11_station_1.set_joint_positions(next_pose)
        self.obj_11_station_1.set_joint_velocities(torch.zeros(4, device=self.cuda_device))

        return
    
    def post_weld_station_inner_hoop_step(self, dof_inner_revolution):
        THRESHOLD = 0.1
        reset_revolution_pose = 1.5
        inner_revolution_target = 1.5
        inner_hoop_index = self.materials.inner_hoop_processing_index
        # hoop_world_pose_position, hoop_world_pose_orientation = self.materials_hoop_0.get_world_poses()
        if self.station_state_inner_left == 0: #reset_empty
            #station is free now, find existing process group task
            # inner_revolution_target = 1.5
            if inner_hoop_index >= 0:
                assert self.materials.hoop_states[inner_hoop_index] == 3, "error" #in list
                self.station_state_inner_left = 1
                self.materials.hoop_states[inner_hoop_index] = 4
        elif self.station_state_inner_left == 1: #loading
            # inner_revolution_target = 1.5
            # if self.put_hoop_on_weld_station_inner(self.materials.inner_hoop_processing_index):
            #     self.station_state_inner_left = 2
            #     self.materials.hoop_states[self.materials.inner_hoop_processing_index] = 5
            ''
        elif self.station_state_inner_left == 2: #rotating
            if self.station_state_inner_middle in range(-1, 3):
                #the station start to rotating 
                inner_revolution_target = 0.0
                delta_pose = torch.abs(dof_inner_revolution[0] - inner_revolution_target)
                if delta_pose < THRESHOLD:
                    self.station_state_inner_left = 3
        elif self.station_state_inner_left == 3: #waiting
            #waiting for the station middle is prepared well (and the cube is already placed on the middle station)
            if self.welder_inner_task == 1:
                #the welder task is to weld the left part
                self.station_state_inner_left = 4
        elif self.station_state_inner_left == 4:
            "welding the left part"
        elif self.station_state_inner_left == 5:
            "welded the left part"
        elif self.station_state_inner_left == -1:
            # the station is resetting
            delta_pose = torch.abs(dof_inner_revolution[0] - reset_revolution_pose)
            if delta_pose < THRESHOLD:
                self.station_state_inner_left = 0
            # inner_revolution_target = 1.5
        # ref_pose[0] += torch.tensor([[0,   0,   -0.3]], device=self.cuda_device)
        if self.station_state_inner_left in range(2,6) and inner_hoop_index >= 0:
            hoop_world_pose_position, hoop_world_pose_orientation = self.obj_11_station_0_revolution.get_world_poses()
            matrix = Gf.Matrix4d()
            orientation = hoop_world_pose_orientation[0].cpu()
            matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
            translate = Gf.Vec4d(0,0,0.4,0)*matrix
            translate_tensor = torch.tensor(translate[:3], device=self.cuda_device)
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_world_poses(
                positions=hoop_world_pose_position+translate_tensor, orientations=hoop_world_pose_orientation)
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        elif self.station_state_inner_left == 6:
            #set the hoop underground
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device=self.cuda_device))
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            self.materials.inner_hoop_processing_index = -1
            self.station_state_inner_left = -1
        if self.station_state_inner_left in range(3,6):
            inner_revolution_target = 0.0
       
        return inner_revolution_target
            
    def put_hoop_on_weld_station_inner(self, raw_hoop_index) -> bool:
        #todo 
        return True
    
    def post_weld_station_inner_cube_step(self, dof_inner_middle_A, dof_inner_middle_B):
        THRESHOLD = 0.05
        welding_left_pose_A = 0.0 
        welding_left_pose_B = 0.0 
        target_inner_middle_A = 0.0 #default is reseted state
        target_inner_middle_B = 0.0
        cube_index = self.materials.inner_cube_processing_index
        if self.station_state_inner_middle in range(3,8):
            target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
        if self.station_state_inner_middle == 0 and cube_index >= 0:
            if self.materials.cube_states[cube_index] in range(1, 6): #1:"in_list", 2:"conveying", 3:"conveyed", 4:"cutting", 5:"cut_done", 6:"pick_up_place_cut",
                self.station_state_inner_middle = 1
        elif self.station_state_inner_middle == 1:
            "#waiting for the gripper to place the cut cube on station inner middle"
        elif self.station_state_inner_middle == 2: #placed
            #waiting for the station inner left loaded hoop
            if self.station_state_inner_left == 3: #waiting
                self.station_state_inner_middle = 3
        elif self.station_state_inner_middle == 3: #moving left
            #moving left to start welding left part
            # target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
            if torch.abs(dof_inner_middle_A[0] - welding_left_pose_A) <= THRESHOLD:
                self.station_state_inner_middle = 4
                self.welder_inner_task = 1
                # self.station_state_inner_left = 4
        elif self.station_state_inner_middle == 4: #welding left
            #moved left and wating for the welder finished
            # target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
            pass
        elif self.station_state_inner_middle == 5: #welded_left
            #finished welding left and waiting for the starion right is prepared well
            # target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
            if self.station_state_inner_right == 4: #welding_right
                #start welding right
                self.station_state_inner_middle = 6 
                self.welder_inner_task = 2
        elif self.station_state_inner_middle == 6: #welding_right
            #welding right waiting for the welder finish
            pass
        elif self.station_state_inner_middle == 7: #welded_right
            "post_outer_gripper_step to place the upper tube on cube"
            #change the bending tube pose 
        elif self.station_state_inner_middle == 8: #welding_upper
            "welding upper waiting for the welder finish"
            self.welder_inner_task = 3
        elif self.station_state_inner_middle == 9: #welded_upper
            "finished welding and do the materials merge waiting for the inner gripper to pick up the product"
            #set cube to underground 
            assert self.materials.inner_cube_processing_index >= 0
            self.materials.cube_list[self.materials.inner_cube_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device=self.cuda_device))
            self.materials.cube_list[self.materials.inner_cube_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            #set product position
            position, orientation= (torch.tensor([[-22.04908, 3.40965, 1.0]], device=self.cuda_device), torch.tensor([[ 1,0,0,0.0]], device=self.cuda_device))
            self.materials.product_list[self.materials.inner_cube_processing_index].set_world_poses(positions=position, orientations=orientation)
            self.materials.product_list[self.materials.inner_cube_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))

        elif self.station_state_inner_middle == -1: #resetting middle part
            self.materials.inner_bending_tube_processing_index = -1
            self.materials.inner_upper_tube_processing_index = -1
            if torch.abs(dof_inner_middle_A[0] - target_inner_middle_A) <= THRESHOLD:
                self.station_state_inner_middle = 0
        if self.station_state_inner_middle in range(2,9):
            #set cube pose
            ref_position, ref_orientation = self.obj_11_station_0_middle.get_world_poses()
            self.materials.cube_list[cube_index].set_world_poses(positions=ref_position+torch.tensor([[-1.82, -3.03,   0.92]], device=self.cuda_device), orientations=ref_orientation)
            self.materials.cube_list[cube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))

        return target_inner_middle_A, target_inner_middle_B
    
    def post_weld_station_inner_bending_tube_step(self, dof_inner_right):
        THRESHOLD = 0.05
        welding_right_pose = -2.5
        target_inner_right = 0.0 
        loading_index = self.materials.inner_bending_tube_loading_index
        if self.station_state_inner_right == 0: #reset_empty
            if loading_index >= 0:
                assert self.materials.bending_tube_states[loading_index] == 3
                self.station_state_inner_right = 1     
                self.materials.bending_tube_states[loading_index] = 4   
        elif self.station_state_inner_right == 1: #placing
            #place bending tube on the station right 
            # if self.put_bending_tube_on_weld_station_inner(self.materials.inner_bending_tube_processing_index):
            #     self.station_state_inner_right = 2
            #     self.materials.bending_tube_states[self.materials.inner_bending_tube_processing_index] = 5
            ''
        elif self.station_state_inner_right == 2: #placed
            "waiting for the middle part and welding left task finished"
            if self.station_state_inner_middle == 5: #welded left
                self.station_state_inner_right = 3
                self.materials.inner_bending_tube_processing_index = loading_index
                self.materials.inner_bending_tube_loading_index = -1
        elif self.station_state_inner_right == 3: #moving
            #moving
            if torch.abs(dof_inner_right[0] - welding_right_pose) <= THRESHOLD:
                self.station_state_inner_right = 4
                # self.station_state_inner_left = 4
        elif self.station_state_inner_right == 4: #welding_right_step_1
            "wating for the welding right task finished"
        elif self.station_state_inner_right == -1:
            #do the resetting 
            if torch.abs(dof_inner_right[0] - target_inner_right) <= THRESHOLD:
                self.station_state_inner_right = 0
        if self.station_state_inner_right in range(2, 5):
            if self.station_state_inner_right == 2:
                raw_bending_tube_index = self.materials.inner_bending_tube_loading_index
            else:
                raw_bending_tube_index = self.materials.inner_bending_tube_processing_index
            assert raw_bending_tube_index >= 0
            raw_orientation = torch.tensor([[1, 0,  0,  0]], device=self.cuda_device)
            ref_position, _ = self.obj_11_station_0_right.get_world_poses()
            self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(positions=ref_position+torch.tensor([[-2.47, -3.225, 0.54]], device=self.cuda_device), orientations = raw_orientation)
            self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        if self.station_state_inner_right in range(3,5):
            target_inner_right = welding_right_pose
        return target_inner_right
    
    def put_bending_tube_on_weld_station_inner(self, inner_bending_tube_processing_index):
        return True
    
    def post_weld_station_inner_tube_step(self):
        
        raw_cube_index = self.materials.inner_cube_processing_index
        if self.station_state_inner_middle in range(1, 8) and self.materials.inner_upper_tube_processing_index == -1:
            upper_tube_index = self.materials.find_next_raw_upper_tube_index()
            assert upper_tube_index >= 0
            self.materials.upper_tube_states[upper_tube_index] = 1
            self.materials.inner_upper_tube_processing_index = upper_tube_index

    def put_upper_tube_on_station(self, inner_upper_tube_processing_index):
        return True

    def post_welder_step(self):
        self.post_inner_welder_step()
        self.post_outer_welder_step()
        return 

    def post_inner_welder_step(self):
        THRESHOLD = 0.05
        welding_left_pose = -3.5
        welding_middle_pose = -2
        welding_right_pose = 0
        target = 0.0 
        welder_inner_pose = self.obj_11_welding_0.get_joint_positions()
        if self.welder_inner_state == 0: #free_empty
            #waiting for the weld station prepared well 
            if self.welder_inner_task == 1:
                #welding left part task
                self.welder_inner_state = 1
        elif self.welder_inner_state == 1: #moving_left
            #moving to the welding_left_pose
            target = welding_left_pose
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD:
                self.welder_inner_state = 2
        elif self.welder_inner_state == 2: #welding_left
            #start welding left
            target = welding_left_pose
            self.welder_inner_oper_time += 1
            self.materials.cube_states[self.materials.inner_cube_processing_index] = 9
            if self.welder_inner_oper_time > self.welding_once_time + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 3 
                self.station_state_inner_left = 5 #welded
                self.station_state_inner_middle = 5 #welded_left
                self.materials.hoop_states[self.materials.inner_hoop_processing_index] = -2
                # self.station_state_inner_right = 3 #moving right
                # cube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/cube_" + "{}".format(self.materials.inner_cube_processing_index))
                # hoop_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/hoop_" + "{}".format(self.materials.inner_hoop_processing_index))
                # product_prim = self.create_fixed_joint(self.materials.hoop_list[self.materials.inner_hoop_processing_index], 
                #                                        self.materials.cube_list[self.materials.inner_cube_processing_index], 
                #                                        torch.tensor([[-4.9,   -2,   -3.05]], device=self.cuda_device), joint_name='Hoop')
        elif self.welder_inner_state == 3: #welded_left
            target = welding_left_pose
            if self.welder_inner_task == 2:
                self.welder_inner_state = 4
        elif self.welder_inner_state == 4: #moving_right
            #moving to the welding_right_pose
            target = welding_right_pose
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD:
                self.welder_inner_state = 5
        elif self.welder_inner_state == 5: #welding_right
            self.materials.cube_states[self.materials.inner_cube_processing_index] = 10
            target = welding_right_pose
            self.welder_inner_oper_time += 1
            if self.welder_inner_oper_time > self.welding_once_time + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 6 
                # self.station_state_inner_middle = 7 #welded_right
                self.station_state_inner_right = -1 #welded_right
        elif self.welder_inner_state == 6: #rotate_and_welding
            target = welding_right_pose
            self.welder_inner_oper_time += 1
            # ref_position, ref_orientation = self.materials.cube_list[self.materials.inner_cube_processing_index].get_world_poses()
            # raw_bending_tube_index = self.materials.inner_bending_tube_processing_index
            #initial pose bending tube (tensor([[-23.4063,   3.5746,   2.6908]], device=self.cuda_device), tensor([[-0.0154, -0.0033,  0.9999,  0.0044]], device=self.cuda_device))
            # https://www.cnblogs.com/meteoric_cry/p/7987548.html
            # _, prev_orientation =  self.materials.bending_tube_list[raw_bending_tube_index].get_world_poses()
            # initial_orientation = torch.tensor([[-0.0154, -0.0033,  0.9999,  0.0044]], device=self.cuda_device)
            # next_orientation_tensor = self.welder_inner_oper_time*(ref_orientation - initial_orientation)/10 + initial_orientation
            # prev_matrix = Gf.Matrix4d()
            # prev_matrix.SetRotateOnly(Gf.Quatd(float(prev_orientation[0][0]), float(prev_orientation[0][1]), float(prev_orientation[0][2]), float(prev_orientation[0][3])))
            # rot_theta =  9.0*self.welder_inner_oper_time
            # rot_matrix =  Gf.Matrix4d(0.0, 0.0, 1, 0.0,
            #     0,  np.cos(rot_theta), np.sin(rot_theta), 0.0,
            #     0, -np.sin(rot_theta), np.cos(rot_theta), 0.0,
            #     0,0,0, 1.0)
            # next_rot_matrix = prev_matrix*rot_matrix
            # next_orientation_quat = next_rot_matrix.ExtractRotationQuat()
            # real, imaginary = next_orientation_quat.GetReal(), next_orientation_quat.GetImaginary()
            # next_orientation_tensor = torch.tensor([[real, imaginary[0], imaginary[1], imaginary[2]]])
            # self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(
            # positions=ref_position+torch.tensor([[ 1.3123, -1.0179, -2.4780]], device=self.cuda_device), orientations=next_orientation_tensor)
            # self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(torch.tensor([[-23.4193,   4.5691,   1.4]], device=self.cuda_device) ,
            #                                                                           orientations=torch.tensor([[ 0.0051,  0.0026, -0.7029,  0.7113]], device=self.cuda_device))
            # self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            if self.welder_inner_oper_time > 5 + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 7 
                self.station_state_inner_middle = 7 #welded_right
                self.materials.bending_tube_states[self.materials.inner_bending_tube_processing_index] = -2
                # self.station_state_inner_right = -1 #welded_right
                # cube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/cube_" + "{}".format(self.materials.inner_cube_processing_index))
                # bending_tube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/bending_tube_" + "{}".format(self.materials.inner_bending_tube_processing_index))
                # product_prim = self.create_fixed_joint(self.materials.bending_tube_list[self.materials.inner_bending_tube_processing_index], 
                #                         self.materials.cube_list[self.materials.inner_cube_processing_index], 
                #                         torch.tensor([[0,   0,   0]], device=self.cuda_device), joint_name='BendingTube')
        elif self.welder_inner_state == 7: #welded_right
            target= welding_middle_pose
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD and self.welder_inner_task == 3:
                self.welder_inner_state = 8
        elif self.welder_inner_state == 8: #welding_upper
            self.materials.cube_states[self.materials.inner_cube_processing_index] = 11
            pick_up_upper_tube_index = self.materials.inner_upper_tube_processing_index
            assert pick_up_upper_tube_index >= 0
            bending_tube_index = self.materials.inner_bending_tube_processing_index
            assert bending_tube_index >= 0
            target = welding_middle_pose
            self.welder_inner_oper_time += 1
            if self.welder_inner_oper_time > self.welding_once_time + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.materials.cube_states[self.materials.inner_cube_processing_index] = 12
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 9
                self.station_state_inner_middle = 9 #welded_upper
                self.station_state_inner_left = 6
                self.welder_inner_task =0
                #set bending tube to underground
                self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device=self.cuda_device))
                self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
                # self.gripper_inner_task = 4
                # self.gripper_inner_state = 1
                # cube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/cube_" + "{}".format(self.materials.inner_cube_processing_index))
                # upper_tube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/upper_tube_" + "{}".format(self.materials.inner_upper_tube_processing_index))
                # product_prim = self.create_fixed_joint(self.materials.upper_tube_list[self.materials.inner_upper_tube_processing_index], 
                #                         self.materials.cube_list[self.materials.inner_cube_processing_index], 
                #                         torch.tensor([[0,   0,   0]], device=self.cuda_device), joint_name='UpperTube')
                # product_prim = physxUtils.createJoint(self._stage, "Fixed", upper_tube_prim, cube_prim)
                '''set the upper tube under the ground'''
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device=self.cuda_device))
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            else:
                position = torch.tensor([[-21.5430,   3.4130,   1.2]], device=self.cuda_device)
                orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device=self.cuda_device)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position, orientations=orientation)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        elif self.welder_inner_state == 9: #welded_upper
            #do the reset
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD:
                self.welder_inner_state = 0
   
        if self.welder_inner_state in range(6, 9):
            bending_tube_index = self.materials.inner_bending_tube_processing_index
            self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[-23.33143,   3.1,   1.5]], device=self.cuda_device) , orientations=torch.tensor([[ 0.70710678, -0.70710678,  0.        ,  0.        ]], device=self.cuda_device))
            self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            # ref_position, ref_orientation = self.materials.cube_list[self.materials.inner_cube_processing_index].get_world_poses()
            # raw_bending_tube_index = self.materials.inner_bending_tube_processing_index
            # self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(
            #     positions=ref_position+torch.tensor([[0.0,   0,   0.5]], device=self.cuda_device), 
            #     orientations=ref_orientation+torch.tensor([[0.0,   0,   0.0, 0.0]], device=self.cuda_device))
            # self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        # if self.welder_inner_state in range(8, 10):
        # if self.welder_inner_state in range(8, 9):
        #     pick_up_upper_tube_index = self.materials.inner_upper_tube_processing_index
        #     position = torch.tensor([[-21.5430,   3.4130,   1.1422]], device=self.cuda_device)
        #     orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device=self.cuda_device)
        #     self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position, orientations=orientation)
        #     self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        next_pose, _ = self.get_next_pose_helper(welder_inner_pose[0], target, self.operator_welder)
        self.obj_11_welding_0.set_joint_positions(next_pose)
        self.obj_11_welding_0.set_joint_velocities(torch.zeros(1, device=self.cuda_device))

    def get_trans_matrix_from_pose(self, position, orientation):
        matrix = Gf.Matrix4d()
        matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
        return matrix

    def create_fixed_joint(self, from_prim_view: RigidPrim, to_prim_view: RigidPrim, translate, joint_name):
        physxUtils.createJoint
        # from_prim_view.disable_rigid_body_physics()
        # from_prim_view.prims[0].GetAttribute('physics:collisionEnabled').Set(False)
        from_position, from_orientation = from_prim_view.get_world_poses()
        to_position, to_orientation = to_prim_view.get_world_poses()
        # to_position = from_position
        # from_position = to_position
        # from_position += torch.tensor([[5,   0,   0]], device=self.cuda_device)
        to_position += translate
        from_matrix = self.get_trans_matrix_from_pose(from_position[0].cpu(), from_orientation[0].cpu())
        # from_matrix.SetRotateOnly(Gf.Quatd(-0.5, Gf.Vec3d(-0.5, 0.5, 0.5)))
        to_matrix = self.get_trans_matrix_from_pose(to_position[0].cpu(), to_orientation[0].cpu())
        # to_matrix.SetRotateOnly(Gf.Quatd(0.0, Gf.Vec3d(0.0, -0.7071067811865475, 0.7071067811865476)))
        # translate = Gf.Vec3d(0,0,0)*to_matrix.ExtractRotationMatrix()
        # translate = Gf.Vec3d(-5.5, 1, -1)
        # to_matrix.SetTranslateOnly(translate + to_matrix.ExtractTranslation())
        rel_pose = to_matrix * from_matrix.GetInverse()
        rel_pose = rel_pose.RemoveScaleShear()
        # rel_pose.SetRotateOnly(Gf.Quatd(0.0, Gf.Vec3d(0.7071067811865475, 0.0, 0.7071067811865476)))
        # _matrix.ExtractRotationQuat()
        # _matrix = Gf.Matrix4d(0.0, 0.0, 1, 0.0,
        #     0, -1, 0, 0.0,
        #     1, 0, 0, 0.0,
        #     0.8399995477320195, -1.4471829022912717, 2.065864107281353, 1.0)
        pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
        rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())

        from_path = from_prim_view.prim_paths[0]
        to_path = to_prim_view.prim_paths[0]

        joint_path = to_path + '/FixedJoint' + joint_name
        component = UsdPhysics.FixedJoint.Define(self._stage, joint_path)

        # joint_path = to_path + '/PrismaticJoint'
        # component = UsdPhysics.PrismaticJoint.Define(self._stage, joint_path)
        # component.CreateAxisAttr("X")
        # component.CreateLowerLimitAttr(0.0)
        # component.CreateUpperLimitAttr(0.0)
        component.CreateBody0Rel().SetTargets([Sdf.Path(from_path)])
        component.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
        component.CreateLocalPos0Attr().Set(pos1)
        component.CreateLocalRot0Attr().Set(rot1)
        component.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        component.CreateBreakForceAttr().Set(MAX_FLOAT)
        component.CreateBreakTorqueAttr().Set(MAX_FLOAT)

        return self._stage.GetPrimAtPath(joint_path)
    
    def post_weld_station_outer_hoop_step(self, dof_outer_revolution):
        THRESHOLD = 0.1
        reset_revolution_pose = 1.5
        outer_revolution_target = 1.5
        outer_hoop_index = self.materials.outer_hoop_processing_index
        # hoop_world_pose_position, hoop_world_pose_orientation = self.materials_hoop_0.get_world_poses()
        if self.station_state_outer_left == 0: #reset_empty
            #station is free now, find existing process group task
            # outer_revolution_target = 1.5
            if outer_hoop_index >= 0:
                assert self.materials.hoop_states[outer_hoop_index] == 3, "error" #in list
                self.station_state_outer_left = 1
                self.materials.hoop_states[outer_hoop_index] = 4
        elif self.station_state_outer_left == 1: #loading
            # outer_revolution_target = 1.5
            # if self.put_hoop_on_weld_station_outer(self.materials.outer_hoop_processing_index):
            #     self.station_state_outer_left = 2
            #     self.materials.hoop_states[self.materials.outer_hoop_processing_index] = 5
            ''
        elif self.station_state_outer_left == 2: #rotating
            #the station start to rotating 
            if self.station_state_outer_middle in range(-1, 3):
                outer_revolution_target = 0.0
                delta_pose = torch.abs(dof_outer_revolution[0] - outer_revolution_target)
                if delta_pose < THRESHOLD:
                    self.station_state_outer_left = 3
        elif self.station_state_outer_left == 3: #waiting
            #waiting for the station middle is prepared well (and the cube is already placed on the middle station)
            if self.welder_outer_task == 1:
                #the welder task is to weld the left part
                self.station_state_outer_left = 4
        elif self.station_state_outer_left == 4:
            "welding the left part"
        elif self.station_state_outer_left == 5:
            "welded the left part"
        elif self.station_state_outer_left == -1:
            # the station is resetting
            delta_pose = torch.abs(dof_outer_revolution[0] - reset_revolution_pose)
            if delta_pose < THRESHOLD:
                self.station_state_outer_left = 0
            # outer_revolution_target = 1.5
        # ref_pose[0] += torch.tensor([[0,   0,   -0.3]], device=self.cuda_device)
        if self.station_state_outer_left in range(1,6) and outer_hoop_index >= 0:
            hoop_world_pose_position, hoop_world_pose_orientation = self.obj_11_station_1_revolution.get_world_poses()
            matrix = Gf.Matrix4d()
            orientation = hoop_world_pose_orientation[0].cpu()
            matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
            translate = Gf.Vec4d(0,0,0.4,0)*matrix
            translate_tensor = torch.tensor(translate[:3], device=self.cuda_device)
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_world_poses(
                positions=hoop_world_pose_position+translate_tensor, orientations=hoop_world_pose_orientation)
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        elif self.station_state_outer_left == 6:
            #set the hoop underground
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device=self.cuda_device))
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            self.materials.outer_hoop_processing_index = -1
            self.station_state_outer_left = -1
        if self.station_state_outer_left in range(3,6):
            outer_revolution_target = 0.0
       
        return outer_revolution_target
            
    def put_hoop_on_weld_station_outer(self, raw_hoop_index) -> bool:
        #todo 
        return True
    
    def post_weld_station_outer_cube_step(self, dof_outer_middle_A, dof_outer_middle_B):
        THRESHOLD = 0.05
        welding_left_pose_A = 0.0 
        welding_left_pose_B = 0.0 
        target_outer_middle_A = 0.0 #default is reseted state
        target_outer_middle_B = 0.0
        cube_index = self.materials.outer_cube_processing_index
        if self.station_state_outer_middle in range(3,8):
            target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
        if self.station_state_outer_middle == 0 and cube_index >= 0:
            if self.materials.cube_states[cube_index] in range(1, 6): 
                self.station_state_outer_middle = 1
        elif self.station_state_outer_middle == 1:
            "#waiting for the gripper to place the cut cube on station outer middle"
        elif self.station_state_outer_middle == 2: #placed
            #waiting for the station outer left loaded hoop
            if self.station_state_outer_left == 3: #waiting
                self.station_state_outer_middle = 3
        elif self.station_state_outer_middle == 3: #moving left
            #moving left to start welding left part
            # target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
            if torch.abs(dof_outer_middle_A[0] - welding_left_pose_A) <= THRESHOLD:
                self.station_state_outer_middle = 4
                self.welder_outer_task = 1
                # self.station_state_outer_left = 4
        elif self.station_state_outer_middle == 4: #welding left
            #moved left and wating for the welder finished
            # target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
            pass
        elif self.station_state_outer_middle == 5: #welded_left
            #finished welding left and waiting for the starion right is prepared well
            # target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
            if self.station_state_outer_right == 4: #welding_right
                #start welding right
                self.station_state_outer_middle = 6 
                self.welder_outer_task = 2
        elif self.station_state_outer_middle == 6: #welding_right
            #welding right waiting for the welder finish
            pass
        elif self.station_state_outer_middle == 7: #welded_right
            "post_outer_gripper_step to place the upper tube on cube"
            #change the bending tube pose 
        elif self.station_state_outer_middle == 8: #welding_upper
            "welding upper waiting for the welder finish"
            self.welder_outer_task = 3
        elif self.station_state_outer_middle == 9: #welded_upper
            "finished welding and do the materials merge waiting for the outer gripper to pick up the product"
            assert self.materials.outer_cube_processing_index >= 0
            #set cube to underground 
            self.materials.cube_list[self.materials.outer_cube_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device=self.cuda_device))
            self.materials.cube_list[self.materials.outer_cube_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
            #set product position
            position, orientation= (torch.tensor([[-22.04908, 6.69515, 1.]], device=self.cuda_device), torch.tensor([[ 1,0,0,0.0]], device=self.cuda_device))
            self.materials.product_list[self.materials.outer_cube_processing_index].set_world_poses(positions=position, orientations=orientation)
            self.materials.product_list[self.materials.outer_cube_processing_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        elif self.station_state_outer_middle == -1: #resetting middle part
            self.materials.outer_bending_tube_processing_index = -1
            self.materials.outer_upper_tube_processing_index = -1
            if torch.abs(dof_outer_middle_A[0] - target_outer_middle_A) <= THRESHOLD:
                self.station_state_outer_middle = 0
        if self.station_state_outer_middle in range(2,9):
            #set cube pose
            ref_position, ref_orientation = self.obj_11_station_1_middle.get_world_poses()
            self.materials.cube_list[cube_index].set_world_poses(positions=ref_position+torch.tensor([[-1.82, -3.03,   0.92]], device=self.cuda_device), orientations=ref_orientation)
            self.materials.cube_list[cube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))

        return target_outer_middle_A, target_outer_middle_B
    
    def post_weld_station_outer_bending_tube_step(self, dof_outer_right):
        THRESHOLD = 0.05
        welding_right_pose = -2.5
        target_outer_right = 0.0 
        loading_index = self.materials.outer_bending_tube_loading_index
        if self.station_state_outer_right == 0: #reset_empty
            if loading_index >= 0:
                assert self.materials.bending_tube_states[loading_index] == 3
                self.station_state_outer_right = 1     
                self.materials.bending_tube_states[loading_index] = 4   
        elif self.station_state_outer_right == 1: #placing
            #place bending tube on the station right 
            # if self.put_bending_tube_on_weld_station_outer(self.materials.outer_bending_tube_processing_index):
            #     self.station_state_outer_right = 2
            #     self.materials.bending_tube_states[self.materials.outer_bending_tube_processing_index] = 5
            ''
        elif self.station_state_outer_right == 2: #placed
            "waiting for the middle part and welding left task finished"
            if self.station_state_outer_middle == 5: 
                self.station_state_outer_right = 3
                self.materials.outer_bending_tube_processing_index = self.materials.outer_bending_tube_loading_index
                self.materials.outer_bending_tube_loading_index = -1
        elif self.station_state_outer_right == 3: #moving
            #moving
            if torch.abs(dof_outer_right[0] - welding_right_pose) <= THRESHOLD:
                self.station_state_outer_right = 4
                # self.station_state_outer_left = 4
        elif self.station_state_outer_right == 4: #welding_right_step_1
            "wating for the welding right task finished"
        elif self.station_state_outer_right == -1:
            #do the resetting 
            if torch.abs(dof_outer_right[0] - target_outer_right) <= THRESHOLD:
                self.station_state_outer_right = 0
        if self.station_state_outer_right in range(2, 5):
            if self.station_state_outer_right == 2:
                raw_bending_tube_index = self.materials.outer_bending_tube_loading_index
            else:
                raw_bending_tube_index = self.materials.outer_bending_tube_processing_index
            raw_orientation = torch.tensor([[1, 0,  0,  0]], device=self.cuda_device)
            ref_position, _ = self.obj_11_station_1_right.get_world_poses()
            assert raw_bending_tube_index >= 0
            self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(
                positions=ref_position+torch.tensor([[-2.47, -3.225, 0.54]], device=self.cuda_device), orientations = raw_orientation)
            self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        if self.station_state_outer_right in range(3,5):
            target_outer_right = welding_right_pose
        return target_outer_right
    
    def put_bending_tube_on_weld_station_outer(self, outer_bending_tube_processing_index):
        return True
    
    def post_weld_station_outer_tube_step(self):
        
        raw_cube_index = self.materials.outer_cube_processing_index
        if self.station_state_outer_middle in range(1, 8) and self.materials.outer_upper_tube_processing_index == -1:
            upper_tube_index = self.materials.find_next_raw_upper_tube_index()
            assert upper_tube_index >= 0
            self.materials.upper_tube_states[upper_tube_index] = 1
            self.materials.outer_upper_tube_processing_index = upper_tube_index

    def put_upper_tube_on_station(self, outer_upper_tube_processing_index):
        return True

    def post_outer_welder_step(self):
        THRESHOLD = 0.05
        welding_left_pose = -3.1
        welding_middle_pose = -2
        welding_right_pose = 0
        target = 0.0 
        welder_outer_pose = self.obj_11_welding_1.get_joint_positions()
        if self.welder_outer_state == 0: #free_empty
            #waiting for the weld station prepared well 
            if self.welder_outer_task == 1:
                #welding left part task
                self.welder_outer_state = 1
        elif self.welder_outer_state == 1: #moving_left
            #moving to the welding_left_pose
            target = welding_left_pose
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD:
                self.welder_outer_state = 2
        elif self.welder_outer_state == 2: #welding_left
            #start welding left
            self.materials.cube_states[self.materials.outer_cube_processing_index] = 9
            target = welding_left_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > self.welding_once_time + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 3 
                self.station_state_outer_left = 5 #welded
                self.station_state_outer_middle = 5 #welded_left
                self.materials.hoop_states[self.materials.outer_hoop_processing_index] = -2
                # self.station_state_outer_right = 3 #moving right
        elif self.welder_outer_state == 3: #welded_left
            target = welding_left_pose
            if self.welder_outer_task == 2:
                self.welder_outer_state = 4
        elif self.welder_outer_state == 4: #moving_right
            #moving to the welding_right_pose
            target = welding_right_pose
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD:
                self.welder_outer_state = 5
        elif self.welder_outer_state == 5: #welding_right
            self.materials.cube_states[self.materials.outer_cube_processing_index] = 10
            target = welding_right_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > self.welding_once_time + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 6 
                # self.station_state_outer_middle = 7 #welded_right
                self.station_state_outer_right = -1 #welded_right
        elif self.welder_outer_state == 6: #rotate_and_welding
            target = welding_right_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > 5 + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 7 
                self.station_state_outer_middle = 7 #welded_right
                self.materials.bending_tube_states[self.materials.outer_bending_tube_processing_index] = -2
        elif self.welder_outer_state == 7: #welded_right
            target= welding_middle_pose
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD and self.welder_outer_task == 3:
                self.welder_outer_state = 8
        elif self.welder_outer_state == 8: #welding_upper
            self.materials.cube_states[self.materials.outer_cube_processing_index] = 11
            pick_up_upper_tube_index = self.materials.outer_upper_tube_processing_index
            assert pick_up_upper_tube_index >= 0 
            bending_tube_index = self.materials.outer_bending_tube_processing_index
            assert bending_tube_index >= 0
            target = welding_middle_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > self.welding_once_time + self.machine_random_time:
                self.reset_machine_random_time()
                #task finished
                self.materials.cube_states[self.materials.outer_cube_processing_index] = 12
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 9
                self.station_state_outer_middle = 9 #welded_upper
                self.station_state_outer_left = 6
                self.welder_outer_task =0
                #set bending tube to underground
                self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device=self.cuda_device))
                self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
                #set upper tube to under ground
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device=self.cuda_device))
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))  
                # self.gripper_inner_task = 5 
                # self.gripper_inner_state = 1
            else:
                position = torch.tensor([[-21.5430,   6.8,   1.2]], device=self.cuda_device)
                orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device=self.cuda_device)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position, orientations=orientation)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        elif self.welder_outer_state == 9: #welded_upper
            #do the reset
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD:
                self.welder_outer_state = 0
        if self.welder_outer_state in range(6, 9):
            bending_tube_index = self.materials.outer_bending_tube_processing_index
            self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[-23.33143,   6.44407,   1.5]], device=self.cuda_device) ,
                                                                                      orientations=torch.tensor([[ 0.70710678, -0.70710678,  0.        ,  0.        ]], device=self.cuda_device))
            self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device=self.cuda_device))
        next_pose, _ = self.get_next_pose_helper(welder_outer_pose[0], target, self.operator_welder)
        self.obj_11_welding_1.set_joint_positions(next_pose)
        self.obj_11_welding_1.set_joint_velocities(torch.zeros(1, device=self.cuda_device))

    def get_observations(self) -> dict:
        """Compute observations."""
        obs_dict = {}
        ####1.action mask
        # obs_dict['tasks_state'] = self.task_manager.task_in_vector
        obs_dict['action_mask'] = self.task_mask
        ####2.hoop
        obs_dict['state_depot_hoop'] = torch.tensor([self.state_depot_hoop], dtype=torch.int32)
        obs_dict['have_raw_hoops'] = torch.tensor([self.materials.hoop_states.count(0) > 0], dtype=torch.int32)
        ####3.bending_tube
        obs_dict['state_depot_bending_tube'] = torch.tensor([self.state_depot_bending_tube], dtype=torch.int32)
        obs_dict['have_raw_bending_tube'] = torch.tensor([self.materials.bending_tube_states.count(0) > 0], dtype=torch.int32)
        ####4.station state
        obs_dict['station_state_inner_left'] = torch.tensor([self.station_state_inner_left+1], dtype=torch.int32)
        obs_dict['station_state_inner_right'] = torch.tensor([self.station_state_inner_right+1], dtype=torch.int32)
        obs_dict['station_state_outer_left'] = torch.tensor([self.station_state_outer_left+1], dtype=torch.int32)
        obs_dict['station_state_outer_right'] = torch.tensor([self.station_state_outer_right+1], dtype=torch.int32)
        ####5.cutting_machine
        obs_dict['cutting_machine_state'] = torch.tensor([self.cutting_machine_state], dtype=torch.int32)
        ####6.products
        obs_dict['is_full_products'] = torch.tensor([self.task_manager.boxs.is_full_products()], dtype=torch.int32)
        obs_dict['produce_product_req'] = torch.tensor([self.materials.produce_product_req()], dtype=torch.int32)
        ####7.time_step
        # obs_dict['time_step'] = self.episode_length_buf[0].cpu()/(self.dynamic_episode_len - 1)
        obs_dict['time_step'] = torch.tensor([self.episode_length_buf[0].cpu()/self.max_episode_length], dtype=torch.float32)
        obs_dict['max_env_len'] = torch.tensor([self.dynamic_episode_len/self.max_episode_length], dtype=torch.float32)
        # (self.episode_length_buf[0].cpu()/2000).unsqueeze(0)
        
        ####10.progress
        obs_dict['progress'] = torch.tensor([self.materials.progress()], dtype=torch.float32)
        ####11.raw products left, max is 20
        # obs_dict['raw_products'] = torch.tensor([5], dtype=torch.int32)
        obs_dict['raw_products'] = torch.tensor([self.materials.product_states.count(0)], dtype=torch.int32)

        ####8.worker agv box state TODO
        max_num = 3
        agv_mask = torch.zeros([max_num], dtype=bool)
        worker_mask = torch.zeros([max_num], dtype=bool)
        box_mask = torch.zeros([max_num], dtype=bool)
        dim_phy_fatigue_coe = len(self.task_manager.characters.fatigue_list[0].get_phy_fatigue_coe())
        worker_state_len = 5+dim_phy_fatigue_coe
        worker_token_mask = torch.zeros([max_num*worker_state_len], dtype=bool)
        obs_dict['worker_pose'] = torch.zeros([max_num, 1], dtype=torch.int32)
        obs_dict['worker_state'] = torch.zeros([max_num, 1], dtype=torch.int32)
        obs_dict['worker_task'] = torch.zeros([max_num, 1], dtype=torch.int32)
        obs_dict['worker_fatigue_phy'] = torch.zeros([max_num, 1], dtype=torch.float32)
        obs_dict['worker_fatigue_psy'] = torch.zeros([max_num, 1], dtype=torch.float32)
        obs_dict['phy_fatigue_coe'] = torch.zeros([max_num, dim_phy_fatigue_coe], dtype=torch.float32)
        
        for i in range(max_num):
            if i < self.task_manager.characters.acti_num_charc:
                worker_mask[i] = 1
                worker_token_mask[i*worker_state_len:(i+1)*worker_state_len] = 1
                worker_position = self.task_manager.characters.list[i].get_world_poses()
                wp = world_pose_to_navigation_pose(worker_position)
                wp_str = self.task_manager.find_closest_pose(pose_dic=self.task_manager.characters.poses_dic, ego_pose=wp, in_dis=1000.)
                wp_num = self.task_manager.characters.poses_dic2num[wp_str]
                obs_dict['worker_pose'][i] = wp_num
                obs_dict['worker_state'][i] = self.task_manager.characters.states[i]
                obs_dict['worker_task'][i] = self.task_manager.characters.tasks[i]
                obs_dict['worker_fatigue_phy'][i] = self.task_manager.characters.fatigue_list[i].phy_fatigue
                obs_dict['worker_fatigue_psy'][i] = self.task_manager.characters.fatigue_list[i].psy_fatigue
                obs_dict['phy_fatigue_coe'][i] = torch.tensor(self.task_manager.characters.fatigue_list[i].get_phy_fatigue_coe(), dtype=torch.float32)

        max_num = self.cfg.n_max_robot
        obs_dict['agv_pose'] = torch.zeros([max_num, 1], dtype=torch.int32)
        obs_dict['agv_state'] = torch.zeros([max_num, 1], dtype=torch.int32)
        obs_dict['agv_task'] = torch.zeros([max_num, 1], dtype=torch.int32)
        
        for i in range(max_num):
            if i < self.task_manager.agvs.acti_num_agv:
                agv_mask[i] = 1
                position = self.task_manager.agvs.list[i].get_world_poses()
                p = world_pose_to_navigation_pose(position)
                p_str = self.task_manager.find_closest_pose(pose_dic=self.task_manager.agvs.poses_dic, ego_pose=p, in_dis=1000.)
                p_num = self.task_manager.agvs.poses_dic2num[p_str]
                obs_dict['agv_pose'][i] = p_num
                obs_dict['agv_state'][i] = self.task_manager.agvs.states[i]
                obs_dict['agv_task'][i] = self.task_manager.agvs.tasks[i]

        obs_dict['box_pose'] = torch.zeros([max_num, 1], dtype=torch.int32)
        obs_dict['box_state'] = torch.zeros([max_num, 1], dtype=torch.int32)
        obs_dict['box_task'] = torch.zeros([max_num, 1], dtype=torch.int32)
        
        for i in range(max_num):
            if i < self.task_manager.boxs.acti_num_box:
                box_mask[i] = 1
                position = self.task_manager.boxs.list[i].get_world_poses()
                p = world_pose_to_navigation_pose(position)
                p_str = self.task_manager.find_closest_pose(pose_dic=self.task_manager.boxs.poses_dic, ego_pose=p, in_dis=1000.)
                p_num = self.task_manager.boxs.poses_dic2num[p_str]
                obs_dict['box_pose'][i] = p_num
                obs_dict['box_state'][i] = self.task_manager.boxs.states[i]
                obs_dict['box_task'][i] = self.task_manager.boxs.tasks[i]

        other_token_mask = torch.ones([16], dtype=bool)
        obs_dict['worker_mask'] = worker_mask
        obs_dict['token_mask'] = torch.concatenate([other_token_mask, worker_token_mask, agv_mask.repeat(3), box_mask.repeat(3)])
        '''fatigue info'''
        # human coefficient dict including 10 task， for cost function model
        # time step, task step


        return obs_dict
    
