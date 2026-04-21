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



class RuleBasedAgent():
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

    
    
    def play_steps(self):
        while True:
            obs : dict = self.vec_env.reset()
            with torch.no_grad():
                next_obs, rewards, dones, infos, action = self.vec_env.step(action)
            return next_obs, rewards, dones, infos, action


    def train(self):
        while True:
            action = None
            action_extra = {}
            self.vec_env.step(action, action_extra)