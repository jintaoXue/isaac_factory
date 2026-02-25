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
from .hc_env_base import HcEnvBase
from .hc_map_route import world_pose_to_navigation_pose
from isaacsim.core.prims import RigidPrim
import omni.physx.scripts.utils as physxUtils
from pxr import Gf, Sdf, Usd, UsdPhysics, UsdGeom, PhysxSchema
from omni.usd import get_world_transform_matrix

from ...utils import quaternion  
import numpy as np
import torch.nn.functional as Fun
from .hc_env_cfg import joint_pos_dic_num02_weldingRobot_part02_robot_arm_and_base, MovingPose

MAX_FLOAT = 3.40282347e38
# import numpy as np

class HcEnv(HcEnvBase):
            
    def step(self, action: torch.Tensor | None, action_extra = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""
        # process actions debug TODO
        if self.env_rule_based_exploration:
            action = self.get_rule_based_action() if self.episode_length_buf[0] > 0 else action
        self.task_manager_step(action, action_extra)
        self._pre_physics_step(action)
        ###TODO only support single env training
        # action_mask
        while True:
            self.reset_step()
            self.episode_length_buf[:] += 1
            self.material_step()
            self.num01_rotaryPipeAutomaticWeldingMachine_step()
            self.num02_weldingRobot_step()
            self.num03_rollerbedCNCPipeIntersectionCuttingMachine_step()
            self.num04_laserCuttingMachine_step()
            self.num05_groovingMachineLarge_step()
            self.num06_groovingMachineSmall_step()
            self.num07_highPressureFoamingMachine_step()

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

            if False and(self.task_mask[1:].count_nonzero() == 0 and self.reset_buf[0] == 0):
                # self.get_rule_based_action()
                self.task_manager_step(actions=torch.zeros([1], dtype=torch.int32))
            else:
                # self.calculate_metrics()
                obs = self.get_observations()
                self.task_manager.obs = obs
                # self.get_fatigue_data()
                break

        return obs, self.reward_buf, self.reset_buf, self.extras, action

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        task_id = actions[0] - 1
        task = self.task_manager.task_dic[task_id.item()]
        self.extras['action_info'] = task
        self.caculate_metric_action(actions)
        return actions

    def task_manager_step(self, actions, action_extra=None):
        self.task_manager.step()
        for charac_idx in range(0, self.task_manager.characters.acti_num_charc):
            self.human_step(charac_idx)
        for agv_idx in range(0, self.task_manager.agvs.acti_num_agv):
            self.mobile_robot_step(agv_idx)
        return

    def material_step(self):
 
        return

    def human_step(self, idx):    

        return
    
    def mobile_robot_step(self, idx):

        return
    
    def num01_rotaryPipeAutomaticWeldingMachine_step(self):

        return

    def num02_weldingRobot_step(self):
        articulation_pose_arm_and_base = self.num02_weldingRobot_part02_robot_arm_and_base.get_joint_positions()

        reset2working = True
        target_pose : list[float] = joint_pos_dic_num02_weldingRobot_part02_robot_arm_and_base["working_pose"]
        target_pose = torch.tensor(target_pose, device=self.device).unsqueeze(0)
        if reset2working:
            if self.moving_pose_num02_weldingRobot_part02_robot_arm_and_base is None:
                self.moving_pose_num02_weldingRobot_part02_robot_arm_and_base = MovingPose(
                    start_pose = articulation_pose_arm_and_base,
                    end_pose = target_pose,
                    time=joint_pos_dic_num02_weldingRobot_part02_robot_arm_and_base["moving_pose_time"],
                )
            if not self.moving_pose_num02_weldingRobot_part02_robot_arm_and_base.is_done():
                next_pose = self.moving_pose_num02_weldingRobot_part02_robot_arm_and_base.get_next_pose()
            else:
                self.moving_pose_num02_weldingRobot_part02_robot_arm_and_base = None
                reset2working = False
                next_pose = target_pose

        articulation_pose_mobile_base_for_material = self.num02_weldingRobot_part04_mobile_base_for_material.get_joint_positions()
        articulation_pose_mobile_base_for_material[:,0] = 2

        self.num02_weldingRobot_part02_robot_arm_and_base.set_joint_positions(next_pose)
        self.num02_weldingRobot_part04_mobile_base_for_material.set_joint_positions(articulation_pose_mobile_base_for_material)

        return

    def num03_rollerbedCNCPipeIntersectionCuttingMachine_step(self):

        return

    def num04_laserCuttingMachine_step(self):

        return

    def num05_groovingMachineLarge_step(self):

        return

    def num06_groovingMachineSmall_step(self):

        return

    def num07_highPressureFoamingMachine_step(self):

        return

    def get_observations(self) -> dict:
        """Compute observations."""
        obs_dict = {}


        return obs_dict
    
