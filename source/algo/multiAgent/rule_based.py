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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .memory import ReplayMemory, CostfuncMemory
from .model import DQNTrans, SafeDQNTrans
from tqdm import trange
import time
from omegaconf import DictConfig
from ..utils import data
import wandb
import copy


class RuleBasedAgent():
    def __init__(self, base_name, params):
        self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
        self.env_config = params.get('env_config', {})
        self.num_actors = params.get('num_actors', 1)
        self.env_name = params['env_name']
        print("Env name:", self.env_name)
    
    
    def play_steps(self):
        while True:
            obs : dict = self.vec_env.reset()
            with torch.no_grad():
                next_obs, rewards, dones, infos, action = self.vec_env.step(action)
            return next_obs, rewards, dones, infos, action
