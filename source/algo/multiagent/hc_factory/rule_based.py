# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch import nn

from rl_games.common import vecenv
# from ..utils import vecenv
from rl_games.algos_torch import torch_ext
from .agent_A_product_sequencer import ProductSequencingAgent
from .agent_B_process_task_planner import ProcessTaskPlanningAgent
from .agent_C_human_robot_machine_allocator import HumanRobotMachineAllocationAgent


class RuleBasedMultiAgent():
    def __init__(self, base_name, params):
        config = params['config']
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.product_sequencing_agent = ProductSequencingAgent()
        self.process_task_planning_agent = ProcessTaskPlanningAgent()
        self.human_robot_machine_allocation_agent = HumanRobotMachineAllocationAgent()

    def act(self, obs):
        # 'obs' is a list where each element is a dictionary representing the state of a single environment instance.
        # 'actions' is a list where each element is a dictionary representing the action of a single environment instance.
        num_envs = len(obs)
        actions : list[dict] = []
        actions_extra : list[dict] = []
        for env_id in range(num_envs):
            product_sequencing_action = self.product_sequencing_agent.act(obs[env_id])
            process_task_planning_action = self.process_task_planning_agent.act(obs[env_id])
            human_robot_machine_allocation_action = self.human_robot_machine_allocation_agent.act(obs[env_id])
            action = {}
            action_extra = {}
            action['product_sequencing'] = product_sequencing_action
            action['process_task_planning'] = process_task_planning_action
            action['human_robot_machine_allocation'] = human_robot_machine_allocation_action
            actions.append(action)
            actions_extra.append(action_extra)
        return actions, actions_extra

    def train(self):
        # 'obs' is a list where each element is a dictionary representing the state of a single environment instance.
        obs : list[dict] = self.vec_env.reset()
        while True:
            action, action_extra = self.act(obs)
            next_obs = self.vec_env.step(action, action_extra)
            obs = next_obs