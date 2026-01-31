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
from .model import DQNTransNoduel
from tqdm import trange
import time
from omegaconf import DictConfig
from ..utils import data
import wandb
import copy


class DqnAgent():
    def __init__(self, base_name, params):

        self.config : DictConfig = params['config']
        self.base_init()
        config = self.config
        print(config)
        #######pramameters for model update and act
        self.Vmin = config.get('V_min', -10)
        self.Vmax = config.get('V_max', 10)
        self.n = config['multi_step']
        self.gamma = config['gamma']
        self.norm_clip = config.get('norm_clip', 10)

        '''Params for agent training'''
        self.update_frequency = config.get('update_frequency', 400)
        self.update_frequency_sfl = config.get('update_frequency_sfl', 1000)
        # self.update_frequency_sfl = config.get('update_frequency', 100)
        self.evaluate_interval = config.get('evaluate_interval', 400)
        self.target_update = config.get('target_update', int(2e3))
        self.max_steps = config.get("max_steps", int(2.8e6))
        self.max_epochs = config.get("max_epochs", int(1e11))
        self.batch_size = config.get('batch_size', 512)
        self.num_warmup_steps = config.get('num_warmup_steps', int(5e4))
        self.cost_num_warmup_steps = config.get('cost_num_warmup_steps', int(5e3))
        self.use_cost_num_steps = config.get('use_cost_num_steps', int(1.5e5))
        self.use_prediction_net = config.get('use_prediction_net', False)
        #########debug
        # self.update_frequency = config.get('update_frequency', 100)
        # self.update_frequency_sfl = config.get('update_frequency_sfl', 200)
        # self.evaluate_interval = config.get('evaluate_interval', 5)
        # self.num_warmup_steps = config.get('num_warmup_steps', int(300))
        # self.batch_size = 64
        # self.cost_num_warmup_steps = config.get('cost_num_warmup_steps', int(200))
        # self.use_cost_num_steps = config.get('use_cost_num_steps', int(3000))
        '''End of agent training'''

        self.demonstration_steps = config.get('demonstration_steps', int(0))
        self.num_steps_per_epoch = config.get("num_steps_per_epoch", 100)
        # self.horizon_length = config.get("horizon_length", 1000) # temporary, in future we will use other approach
        print(self.batch_size, self.num_actors, self.num_agents)
        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)
        #########buffer
        self.priority_weight_increase = (1 - config['priority_weight']) / (self.max_steps - self.num_warmup_steps)
        self.replay_buffer = ReplayMemory(config, config["replay_buffer_size"])
        # self.costfunc_buffer = CostfuncMemory(config["replay_buffer_size"])
        ####### net
        self.only_train_cost_net = self.config['only_train_cost']

        self.online_net = DQNTransNoduel(config, self.actions_num).to(device=self._device)
        if self._test and not self.env_rule_based_exploration:
            weights = torch.load(self.train_dir + self._load_dir + self._load_name, weights_only=True)
            self.online_net.load_state_dict(weights['net'])
        self.online_net.train()
        # self.target_net = DQN(config, self.actions_num).to(device=self._device)
        self.target_net = DQNTransNoduel(config, self.actions_num).to(device=self._device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False
        #####
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=config['learning_rate'], eps=config['adam_eps'])
        # self.optimiser = optim.Adam(self.online_net.trainable_params_rl, lr=config['learning_rate'], eps=config['adam_eps'])
        # self.cost_optimiser = optim.Adam(self.online_net.trainable_params_sft, lr=config['learning_rate_sft'], eps=config['adam_eps'])
        self.loss_criterion = nn.MSELoss(reduction= 'none')
        self.use_wandb = config.get('wandb_activate', False)
        if self.use_wandb:
            self.init_wandb_logger()
    # def load_networks(self, params):
    #     builder = model_builder.ModelBuilder()
    #     config['network'] = builder.load(params)
    def setdefault(self, dict: dict, key, default):
        if key in dict:
            return
        else:
            dict.__setitem__(key, default)

    def base_init(self):

        self.setdefault(self.config, key='device', default='cuda:0')
        ########for replay buffer args initialize
        self.setdefault(self.config, key='replay_buffer_size', default=int(5e5))
        self.setdefault(self.config, key='history_length', default=1)
        self.setdefault(self.config, key='gamma', default=0.99)
        self.setdefault(self.config, key='multi_step', default=1)
        self.setdefault(self.config, key='priority_exponent', default=0.5)
        self.setdefault(self.config, key='priority_weight', default=0.4)
        ########for neural network args initialize
        self.setdefault(self.config, key='architecture', default='canonical')
        self.setdefault(self.config, key='hidden_size', default=512)
        self.setdefault(self.config, key='noisy_std', default=0.1)
        ######for optimizer initialize
        # self.setdefault(self.config, key='learning_rate', default=0.0000625)
        self.setdefault(self.config, key='learning_rate', default=1e-4)
        self.setdefault(self.config, key='adam_eps', default=1.5e-4)
        config = self.config
        ####TODO
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.action_space = self.env_info['action_space']
        self.actions_num = self.action_space.shape[0]
        self.obs = None
        self._device = config['device']
        ##test
        self._test = config['test']
        self._load_dir = config['load_dir']
        self._load_name = config['load_name']
        self.env_rule_based_exploration = config.get('env_rule_based_exploration', False)
        #temporary for Isaac gym compatibility
        print('Env info:')
        print(self.env_info)

        # self.rewards_shaper = config['reward_shaper']
        
        # self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        # self.is_train = config.get('is_train', True)
        # self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        # self.save_best_after = config.get('save_best_after', 500)
        # self.print_stats = config.get('print_stats', True)
        # self.rnn_states = None
        # self.name = base_name

        # self.save_freq = config.get('save_frequency', 0)

        # self.network = config['network']
        self.num_agents = self.env_info.get('agents', 1)

        self.games_to_track = config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        # self.game_overworks = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        # self.train_avgs = []

        self.eval_progress_avgs = []
        self.eval_env_len_avgs = []
        self.progress_avgs = []
        self.env_len_avgs = []
        for i in range(0, self.config['max_num_worker']):
            list_a = []
            list_b = []
            list_c = []
            list_d = []
            for j in range(0, self.config['max_num_robot']):
                list_a.append(torch_ext.AverageMeter(1, self.games_to_track).to(self._device))
                list_b.append(torch_ext.AverageMeter(1, self.games_to_track).to(self._device))
                list_c.append(torch_ext.AverageMeter(1, self.games_to_track).to(self._device))
                list_d.append(torch_ext.AverageMeter(1, self.games_to_track).to(self._device))
            # self.train_avgs.append(list_a)
            self.eval_progress_avgs.append(list_a)
            self.eval_env_len_avgs.append(list_b)
            self.progress_avgs.append(list_c)
            self.env_len_avgs.append(list_d)
    
        # self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)
        self.step_num = 0
        self.step_num_sfl = 0
        self.epoch_num = 0
        self.episode_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0
        
        self.evaluate_step_num = 0
        self.evaluate_episode_num = 0
        self.evaluate_use_cost_step = -1

        self.train_dir = config.get('train_dir', 'runs')
        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, config['full_experiment_name'])

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        # self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        # os.makedirs(self.summaries_dir, exist_ok=True)

        # self.writer = SummaryWriter('runs/' + config['name'] + time_now)
        print("Run Directory:", self.experiment_dir)

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_wandb_logger(self):

        wandb.define_metric("Train/step")
        wandb.define_metric("Train/buffer_size", step_metric="Train/step")

        wandb.define_metric("SuperviseTrain/step")
        wandb.define_metric("SuperviseTrain/loss", step_metric="SuperviseTrain/step")
        wandb.define_metric("SuperviseTrain/loss_compare", step_metric="SuperviseTrain/step")
        wandb.define_metric("SuperviseTrain/buffer_size", step_metric="SuperviseTrain/step")
        wandb.define_metric("SuperviseTrain/EpOverCost", step_metric="SuperviseTrain/step")


        wandb.define_metric("Train/Mrewards", step_metric="Train/step")
        wandb.define_metric("Train/MLen", step_metric="Train/step")
        wandb.define_metric("Metrics/step_episode", step_metric="Train/step")
        wandb.define_metric("Metrics/EpRet", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpLen", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpEnvLen", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpFilterPredictLoss", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpFilterPredictAccu", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpPredictLossCompare", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpFilterRecoverCoeAccu", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpFilterFatigueCoeAccu", step_metric="Metrics/step_episode")

        wandb.define_metric("Metrics/EpTime", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpProgress", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpRetAction", step_metric="Metrics/step_episode")

        wandb.define_metric("Train/step")
        wandb.define_metric("Train/buffer_size", step_metric="Train/step")
        wandb.define_metric("Train/loss", step_metric="Train/step")
        wandb.define_metric("Train/train_epoch", step_metric="Train/step")
        
        for i in range(0, self.config['max_num_worker']):
            for j in range(0, self.config['max_num_robot']):
                wandb.define_metric(f'Avg_progress/{i+1}_{j+1}', step_metric="Train/step")
                wandb.define_metric(f'Avg_env_len/{i+1}_{j+1}', step_metric="Train/step")
        
        total = sum([param.nelement() for param in self.online_net.parameters()])
        # print("Number of parameters: %.2fM" % (total/1e6))
        param_table = wandb.Table(columns=["online_net_size", "num_warm_up_steps", "cost_num_warmup_steps", "use_cost_num_steps"], 
                                  data=[[total, self.num_warmup_steps, self.cost_num_warmup_steps, self.use_cost_num_steps]])
        wandb.log({"Parameter": param_table})

        #evaluate
        wandb.define_metric("Evaluate/step")
        wandb.define_metric("Evaluate/step_episode", step_metric="Evaluate/step")
        wandb.define_metric("Evaluate/EpRet", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpLen", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpEnvLen", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpTime", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpProgress", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpRetAction", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/Savepth", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpMoveHuman", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpMoveRobot", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpOverCost", step_metric="Evaluate/step_episode")
        #wandb.define_metric("Evaluate/EpPredictLoss", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpPredictLossCompare", step_metric="Evaluate/step_episode")

        wandb.define_metric("Evaluate/EpFilterPredictLoss", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpFilterRecoverCoeAccu", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpFilterFatigueCoeAccu", step_metric="Evaluate/step_episode")
        if self.config['other_filters']:
            wandb.define_metric("Evaluate/EpFilterPredictLoss_kf", step_metric="Evaluate/step_episode")
            wandb.define_metric("Evaluate/EpFilterRecoverCoeAccu_kf", step_metric="Evaluate/step_episode")
            wandb.define_metric("Evaluate/EpFilterFatigueCoeAccu_kf", step_metric="Evaluate/step_episode")
            wandb.define_metric("Evaluate/EpFilterPredictLoss_ekf", step_metric="Evaluate/step_episode")
            wandb.define_metric("Evaluate/EpFilterRecoverCoeAccu_ekf", step_metric="Evaluate/step_episode")
            wandb.define_metric("Evaluate/EpFilterFatigueCoeAccu_ekf", step_metric="Evaluate/step_episode")
        

        for i in range(0, self.config['max_num_worker']):
            for j in range(0, self.config['max_num_robot']):
                wandb.define_metric(f'Eval_avg_progress/{i+1}_{j+1}', step_metric="Evaluate/step")
                wandb.define_metric(f'Eval_avg_env_len/{i+1}_{j+1}', step_metric="Evaluate/step")
        # self.evaluate_table = wandb.Table(columns=["env_length", "action_seq", "progress"])

        #test
        self.test_table = wandb.Table(columns=["worker_initial_pose", "robot_initial_pose", "box_initial_pose", "progress", "env_length", "human_move", "robot_move"])
        self.test_table2 = wandb.Table(columns=["num_worker", "num_robot&box", "max", "min", "mean", "human_mean", "robot_mean"])
        self.test_table3 = wandb.Table(columns=["time_step", "action_list"])
        return
    
    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            if self.use_prediction_net:
                action = self.online_net(data.func(state, 'unsqueeze', 0), self.step_num_sfl>=self.use_cost_num_steps)
                return action.argmax(1).unsqueeze(0)
            else:
                action = self.online_net(data.func(state, 'unsqueeze', 0))
                return action.argmax(1).unsqueeze(0)
            # return (self.online_net(data.func(state, 'unsqueeze', 0)) * self.support).sum(2)

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return self.act_random(state) if np.random.random() < epsilon else self.act(state)
    
    def act_random(self, state):
        action_mask = state['action_mask']
        indexs = action_mask.nonzero()
        index = torch.randint(low=0, high = len(indexs), size = (1,), device=self._device) 
        action = indexs[index]
        return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0), self.step_num_sfl>=self.use_cost_num_steps) * self.support).sum(2).max(1)[0].item()

    def set_train(self):
        self.online_net.train()

    def set_eval(self):
        self.online_net.eval()

    def init_tensors(self):

        batch_size = self.num_agents * self.num_actors
        self.temp_current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.temp_dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_rewards_action = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.current_ep_time = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)        
        
        self.current_overworks = torch.zeros(batch_size, dtype=torch.float32, device=self._device)

        self.evaluate_current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.evaluate_current_rewards_action = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.evaluate_current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.evaluate_current_ep_time = torch.zeros(batch_size, dtype=torch.float32, device=self._device)

        self.count_task_times = torch.zeros([self.config["max_num_worker"], self.config["max_num_robot"]], dtype=torch.float32, device=self._device)
        self.count_task_success = torch.zeros([self.config["max_num_worker"], self.config["max_num_robot"]], dtype=torch.float32, device=self._device)
        self.task_succ_rate = torch.zeros([self.config["max_num_worker"], self.config["max_num_robot"]], dtype=torch.float32, device=self._device)

    @property
    def device(self):
        return self._device

    def get_weights(self):
        print("Loading weights")
        state = {'net':self.online_net.state_dict()}
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.online_net.load_state_dict(weights['actor'])

    def get_full_state_weights(self):
        print("Loading full weights")
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimiser.state_dict()       

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']

        self.optimiser.load_state_dict(weights['optimizer'])
        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("rainbow restore")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1.0 - tau) * target_param.data)

    def update(self, _):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = self.replay_buffer.sample(self.batch_size)
        states = data.stack_from_array(states.squeeze(), device=self._device)
        next_states = data.stack_from_array(next_states.squeeze(), device=self._device)
        # Calculate current state probabilities (online network noise already sampled)
        # log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        # log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)
        q = self.online_net(states)  # probabilities log p(s_t, ·; θonline)
        q_a = q[range(self.batch_size), actions]  # p(s_t, a_t; θonline)
        
        with torch.no_grad():
            # Calculate nth next state probabilities
            self.target_net.reset_noise()  # Sample new target net noise
            qns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            argmax_indices_ns = qns.argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θtarget))]
            qns_a = qns[range(self.batch_size), argmax_indices_ns]  # Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            y = returns + self.gamma*qns_a*nonterminals.squeeze() 
            
        loss = self.loss_criterion(q_a ,y).mean(dim=0)
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()

        self.replay_buffer.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        return loss

    def update_cost_func(self):
        # Sample transitions
        states = self.costfunc_buffer.sample(self.batch_size)
        states = data.stack_from_array(states.squeeze(), device=self._device)
        # fatigue_prediction = self.online_net.cost_forward(states)*states['prediction_mask']
        fatigue_prediction = self.online_net.cost_forward(states)
        # next_fatigue = torch.cat((states['next_phy_fatigue'], states['next_psy_fatigue']), dim=1)
        # loss = self.loss_criterion(fatigue_prediction[torch.arange(self.batch_size), states['action']], next_fatigue).mean()
        delta_fatigue = states['next_phy_fatigue']- states['phy_fatigue']
        loss = self.loss_criterion(fatigue_prediction[:,:,[0]][torch.arange(self.batch_size), states['action']], delta_fatigue).mean()
        self.online_net.zero_grad()
        loss.backward()
        self.cost_optimiser.step()
        # (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        with torch.no_grad():
            # delta_fatigue = torch.cat((states['next_phy_fatigue']- states['phy_fatigue'], states['next_psy_fatigue'] - states['psy_fatigue']), dim=1)
            # delta_fatigue_compare = torch.cat((states['phy_delta_predict'], states['psy_delta_predict']), dim=1)
            delta_fatigue_compare = states['phy_delta_predict']
            loss_compare = self.loss_criterion(delta_fatigue, delta_fatigue_compare).mean()

        return loss, loss_compare

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
            obs = obs.to(self._device)
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self._device)
        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        # if not obs_is_dict or 'obs' not in obs:    
        #     upd_obs = {'obs' : upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions, action_extra = None):
        # actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, actions = self.vec_env.step(actions, action_extra) # (obs_space) -> (n, obs_space)

        return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos, actions


    def env_reset(self, num_worker=None, num_robot=None, evaluate=False):
        with torch.no_grad():
            obs = self.vec_env.reset(num_worker, num_robot, evaluate)

        obs = self.obs_to_tensors(obs)

        return obs

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
    
    def train_epoch(self, num_worker=None, num_robot=None):
        temporary_buffer, reward_extra, repeat_times = self.play_steps(num_worker, num_robot)
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        loss = None
        if self.only_train_cost_net:
            return step_time, None, total_update_time, total_time, loss
        for j in range(repeat_times):
            fatigue_data_list = []
            for i in range(len(temporary_buffer)):
                random_exploration = self.step_num < self.num_warmup_steps
                self.set_train()
                if self.step_num % self.update_frequency == 0:
                    self.reset_noise()
                #debug TODO
                # action = None
                step_start = time.time()
                obs, action, rewards, dones, infos = temporary_buffer[i]
                # if self.reward_clip > 0:
                #     reward = max(min(reward, self.reward_clip), -self.reward_clip)  # Clip rewards
                step_end = time.time()
                #TODO only support num_agents == 1
                assert self.num_agents == 1, ('only support num_agents == 1')
                self.step_num += self.num_actors * 1
                self.current_rewards += rewards+reward_extra
                # print("rewards: {}, reward_extra: {}, current_rewards: {}".format(rewards, reward_extra, self.current_rewards))
                self.current_rewards_action += infos["rew_action"]
                self.current_lengths += 1
                self.current_ep_time += (step_end - step_start)
                total_time += (step_end - step_start)
                step_time += (step_end - step_start)

                all_done_indices = dones.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                self.game_rewards.update(self.current_rewards[done_indices])
                self.game_lengths.update(self.current_lengths[done_indices])

                not_dones = 1.0 - dones.float()
                obs_cpu = {}
                for key, value in obs.items():
                    obs_cpu[key] = value.cpu()
                action_cpu = action.squeeze().cpu()
                rewards_cpu = rewards.squeeze().cpu()
                dones_cpu = dones.squeeze().cpu()
                self.replay_buffer.append(obs_cpu, action_cpu, rewards_cpu+reward_extra, dones_cpu)
                if 'fatigue_data' in infos:
                    fatigue_data = infos['fatigue_data']
                    for _data in fatigue_data:
                        fatigue_data_list.append(_data)
                if dones[0]:
                    self.episode_num += 1

                    if self.use_wandb:
                        wandb.log({
                            "Train/step": self.step_num,
                            'Metrics/step_episode': self.episode_num,
                            'Metrics/EpRet': self.current_rewards,
                            'Metrics/EpLen': self.current_lengths,
                            'Metrics/EpEnvLen': infos['env_length'],
                            "Metrics/EpTime": self.current_ep_time,
                            "Metrics/EpProgress": infos['progress'],
                            "Metrics/EpRetAction": self.current_rewards_action,
                        })
                    if len(fatigue_data_list)>0:
                        EpLossCompare, dict_loss_pf_filter, dict_loss_kf_filter, dict_loss_ekf_filter = self.get_fatigue_related_predtion_loss(fatigue_data_list)
                        if self.use_wandb:
                            wandb.log({
                            "Metrics/EpFilterPredictLoss": dict_loss_pf_filter['EpFilterPredictLoss'],
                            "Metrics/EpFilterRecoverCoeAccu": dict_loss_pf_filter['FilterRecoverCoeAccu'],
                            "Metrics/EpFilterFatigueCoeAccu": dict_loss_pf_filter['FilterFatigueCoeAccu'],
                            # "Evaluate/EpPredictLoss": torch.sqrt(EpLoss).item(),
                            "Metrics/EpPredictLossCompare": EpLossCompare, 
                        })
                    # next_obs = self.env_reset()   
                    # if not random_exploration and self.episode_num % self.evaluate_interval == 0:
                    if self.episode_num % self.evaluate_interval == 0:
                    # if True:
                        #TODO debug
                        # pass
                        success_list= []
                        w, r = 1, 1
                        self.obs = self.env_reset(1,1, evaluate=True)
                        for _i in range(self.config["max_num_worker"]):
                            for _j in range(self.config["max_num_robot"]):
                                r += 1
                                if r>self.config["max_num_robot"]:
                                    r = 1
                                    w += 1
                                if w>self.config["max_num_worker"]:
                                    w, r = None, None
                                success_list.append(self.evaluate_epoch(test=False, reset_n_worker=w, reset_n_robot=r))
                        if np.all(success_list):
                            # checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num) + '_len_' + str(infos['env_length'].item()) + '_rew_' + "{:.2f}".format(self.evaluate_current_rewards.item())
                            checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num)
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            if self.use_wandb:
                                wandb.log({"Evaluate/Savepth": self.episode_num,
                                })
                self.current_rewards = self.current_rewards * not_dones
                self.current_lengths = self.current_lengths * not_dones
                self.current_ep_time = self.current_ep_time * not_dones
                self.current_rewards_action = self.current_rewards_action * not_dones
                # if isinstance(next_obs, dict):    
                #     next_obs_processed = next_obs['obs']

                # rewards = self.rewards_shaper(rewards)
                ####TODO refine replay buffer
                # self.replay_buffer.append(obs, action, torch.unsqueeze(rewards, 1), next_obs_processed, torch.unsqueeze(dones, 1))

                # self.obs = next_obs.copy()
            # obs_copy = {}
            # infos_copy = {}
            # for key, value in obs.items():
            #     obs_copy[key] = value.copy()
            # for key, value in infos.items():
            #     infos_copy[key] = value.copy()
            # action_cpu = action.squeeze().cpu()
            # rewards_cpu = rewards.squeeze().cpu()
            # dones_cpu = dones.squeeze().cpu()
                update_time = 0
                if not random_exploration:
                    self.replay_buffer.priority_weight = min(self.replay_buffer.priority_weight + self.priority_weight_increase, 1)
                    if self.step_num % self.update_frequency == 0:
                        self.set_train()
                        update_time_start = time.time()
                        loss = self.update(self.step_num_sfl > self.use_cost_num_steps)
                        update_time_end = time.time()
                        update_time = update_time_end - update_time_start
                        if self.use_wandb:
                            wandb.log({
                                    'Train/step': self.step_num,
                                    "Train/loss": loss.mean().item(),
                                })
                        time_now = datetime.now().strftime("_%d-%H-%M-%S")   
                        print("RL traning loss:{}".format(loss.mean().item()) + "time_now:{}".format(time_now))

                # Update target network
                if self.step_num % self.target_update == 0:
                    self.update_target_net()

                total_update_time += update_time

            total_time_end = time.time()
            total_time = total_time_end - total_time_start
            play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, loss
    
    def play_steps(self, num_worker=None, num_robot=None):
        temporary_buffer = []
        fatigue_data_list = []
        while True:
            obs : dict = self.obs
            random_exploration = self.step_num < self.num_warmup_steps
            self.set_train()
            action_extra = {}

            if random_exploration:
                if self.step_num < self.demonstration_steps:
                    action = None
                else: 
                    action = self.act_random(obs)
            else:
                with torch.no_grad():
                    action = self.act(obs)
            with torch.no_grad():
                next_obs, rewards, dones, infos, action = self.env_step(action, action_extra)

            if 'fatigue_data' in infos:
                fatigue_data = infos['fatigue_data']
                for _data in fatigue_data:
                    fatigue_data_list.append(_data)

            if self.use_prediction_net:
                if self.step_num_sfl >= self.cost_num_warmup_steps:
                    if self.step_num_sfl % self.update_frequency_sfl == 0:
                        self.set_train()
                        loss, loss_compare = self.update_cost_func()
                        if self.use_wandb:
                            wandb.log({
                                    'SuperviseTrain/step': self.step_num_sfl,
                                    "SuperviseTrain/loss": torch.sqrt(loss).item(),
                                    "SuperviseTrain/loss_compare": torch.sqrt(loss_compare).item(),
                                })
                        time_now = datetime.now().strftime("_%d-%H-%M-%S")   
                        print("supervise traning loss:{}，".format(loss.mean().item()) + " time_now:{}".format(time_now))
                self.step_num_sfl += 1
            #debug
            # if self.costfunc_buffer.total_num() == 3:
            #     batch_data = self.costfunc_buffer.sample(3)
            #     data.stack_from_array(batch_data, device=self._device)
            assert self.num_agents == 1, ('only support num_agents == 1')
            if infos['overwork']:
                self.current_overworks += 1
            self.temp_current_lengths += 1
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            # no_timeouts = self.temp_current_lengths <= self.horizon_length
            # dones = dones * no_timeouts

            # obs_copy = {}
            # infos_copy = {}
            # for key, value in obs.items():
            #     obs_copy[key] = value.copy()
            # for key, value in infos.items():
            #     infos_copy[key] = value.copy()
            # action_cpu = action.squeeze().cpu()
            # rewards_cpu = rewards.squeeze().cpu()
            # dones_cpu = dones.squeeze().cpu()
            ##cost weight
            if not self.config['use_fatigue_mask']:
                cost_value = infos['cost_value']
                rewards -= cost_value
            temporary_buffer.append((copy.deepcopy(obs), copy.deepcopy(action), copy.deepcopy(rewards), copy.deepcopy(dones), copy.deepcopy(infos)))
            done_flag = copy.deepcopy(dones) 
            if done_flag[0]:
                print_info = infos['print_info']
                if len(fatigue_data_list)>0:
                    EpLossCompare, dict_loss_pf_filter, dict_loss_kf_filter, dict_loss_ekf_filter = self.get_fatigue_related_predtion_loss(fatigue_data_list)
                    EpFilterPredictLoss = dict_loss_pf_filter['EpFilterPredictLoss']
                    FilterRecoverCoeLoss = dict_loss_pf_filter['FilterRecoverCoeAccu']
                    FilterFatigueCoeLoss = dict_loss_pf_filter['FilterFatigueCoeAccu']
                    print(print_info + " Comp_loss:{:.3}".format(EpLossCompare) + " Fat_predict_loss:{:.3}".format(EpFilterPredictLoss) + \
                        " Fat_coe_accu:{:.3}".format(FilterFatigueCoeLoss) + " Rec_coe_accu:{:.3}".format(FilterRecoverCoeLoss))
                else: 
                    print(print_info)
                if self.use_wandb:
                    wandb.log({
                            'SuperviseTrain/step': self.step_num_sfl,
                            "SuperviseTrain/EpOverCost": self.current_overworks,
                        })
                    _num_worker, _num_robot = infos['num_worker'], infos['num_robot']
                    if infos['env_length'] < infos['max_env_len']-1 and infos['progress'] == 1:
                        task_success = True
                    else: 
                        task_success = False
                    self.progress_avgs[_num_worker-1][_num_robot-1].update(torch.tensor([task_success], dtype=torch.float32, device=self._device))
                    self.env_len_avgs[_num_worker-1][_num_robot-1].update(torch.tensor([infos['env_length']], dtype=torch.float32, device=self._device))
                    wandb.log({f'Avg_progress/{_num_worker}_{_num_robot}': self.progress_avgs[_num_worker-1][_num_robot-1].get_mean()})
                    wandb.log({f'Avg_env_len/{_num_worker}_{_num_robot}': self.env_len_avgs[_num_worker-1][_num_robot-1].get_mean()})

                next_obs = self.env_reset(num_worker=num_worker, num_robot=num_robot)
            not_dones = 1.0 - done_flag.float()
            self.temp_current_lengths = self.temp_current_lengths * not_dones
            self.current_overworks = self.current_overworks * not_dones
            self.obs = next_obs.copy()
            reward_extra = -0.01
            repeat_times = 1
            if done_flag[0]:
                _,_,_,_,_infos = temporary_buffer[-1]
                goal_finished = _infos['env_length'] < _infos['max_env_len']-1 and _infos['progress'] == 1
                # if self.current_overworks > 0:
                #     reward_extra += -0.03
                if goal_finished:
                    num_worker, num_robot = infos['num_worker'], infos['num_robot']
                    if self.env_len_avgs[num_worker-1][num_robot-1].__len__() > 0:
                        reward_extra += 0.05*(self.env_len_avgs[num_worker-1][num_robot-1].get_mean() - _infos['env_length'])/self.env_len_avgs[num_worker-1][num_robot-1].get_mean()
                        repeat_times = 10
                else:
                    reward_extra += -0.05
                    if len(temporary_buffer) > 100:
                        reward_extra *= 0.2

                # print("reward_extra:{}, env_len:{}".format(reward_extra, _infos['env_length']))
                if not random_exploration or goal_finished:
                    #when doing random exploration, when want find solution for each setting
                    break
        return temporary_buffer, reward_extra, repeat_times
    
    def evaluate_epoch(self, test=False, reset_n_worker=None, reset_n_robot=None):
        total_time_start = time.time()
        total_time = 0
        step_time = 0.0
        action_info_list = []
        fatigue_data_list = []
        task_success = False
        if test:
            time_step_list = []
        while True:
            self.set_eval()
            obs : dict = self.obs
            if self.env_rule_based_exploration:
                action = None
            else:
                with torch.no_grad():
                    action = self.act(obs)
            step_start = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos, action = self.env_step(action)
            # if self.reward_clip > 0:
            #     reward = max(min(reward, self.reward_clip), -self.reward_clip)  # Clip rewards
            step_end = time.time()
            #TODO only support num_agents == 1
            assert self.num_agents == 1, ('only support num_agents == 1')
            self.evaluate_step_num += self.num_actors * 1
            self.evaluate_current_rewards += rewards
            self.evaluate_current_rewards_action += infos["rew_action"]
            action_info_list.append(infos["action_info"])
            if test:
                time_step_list.append(infos["time_step"])
            self.evaluate_current_lengths += 1
            self.evaluate_current_ep_time += (step_end - step_start)
            total_time += (step_end - step_start)
            step_time += (step_end - step_start)
            # no_timeouts = self.evaluate_current_lengths != self.horizon_length
            # dones = dones * no_timeouts
            not_dones = 1.0 - dones.float()
            dones_flag = copy.deepcopy(dones)
            if 'fatigue_data' in infos:
                fatigue_data = infos['fatigue_data']
                for _data in fatigue_data:
                    fatigue_data_list.append(_data)
            if infos['overwork']:
                self.current_overworks += 1
            # use_cost_func = self.step_num_sfl > self.use_cost_num_steps
            # if self.evaluate_use_cost_step < 0 and use_cost_func:
            #     self.evaluate_use_cost_step = self.evaluate_step_num             
            if dones_flag[0]:
                print_info = infos['print_info']          
                if len(fatigue_data_list)>0:
                    EpLossCompare, dict_loss_pf_filter, dict_loss_kf_filter, dict_loss_ekf_filter = self.get_fatigue_related_predtion_loss(fatigue_data_list)
                    EpFilterPredictLoss = dict_loss_pf_filter['EpFilterPredictLoss']
                    FilterRecoverCoeLoss = dict_loss_pf_filter['FilterRecoverCoeAccu']
                    FilterFatigueCoeLoss = dict_loss_pf_filter['FilterFatigueCoeAccu']
                    if self.use_wandb:                    
                        wandb.log({
                            "Evaluate/EpFilterPredictLoss": EpFilterPredictLoss,
                            "Evaluate/EpFilterRecoverCoeAccu": FilterRecoverCoeLoss,
                            "Evaluate/EpFilterFatigueCoeAccu": FilterFatigueCoeLoss,
                            # "Evaluate/EpPredictLoss": torch.sqrt(EpLoss).item(),
                            "Evaluate/EpPredictLossCompare": EpLossCompare, 
                        })
                        if self.config['other_filters']:
                            wandb.log({
                                "Evaluate/EpFilterPredictLoss_kf": dict_loss_kf_filter['EpFilterPredictLoss_kf'],
                                "Evaluate/EpFilterRecoverCoeAccu_kf": dict_loss_kf_filter['FilterRecoverCoeAccu_kf'],
                                "Evaluate/EpFilterFatigueCoeAccu_kf": dict_loss_kf_filter['FilterFatigueCoeAccu_kf'],
                            })
                            wandb.log({
                                "Evaluate/EpFilterPredictLoss_ekf": dict_loss_ekf_filter['EpFilterPredictLoss_ekf'],
                                "Evaluate/EpFilterRecoverCoeAccu_ekf": dict_loss_ekf_filter['FilterRecoverCoeAccu_ekf'],
                                "Evaluate/EpFilterFatigueCoeAccu_ekf": dict_loss_ekf_filter['FilterFatigueCoeAccu_ekf'],
                            })
                    print(print_info + " Comp_loss:{:.3}".format(EpLossCompare) + \
                    " Fat_predict_loss:{:.3}".format(EpFilterPredictLoss) + \
                        " Fat_coe_accu:{:.3}".format(FilterFatigueCoeLoss) + " Rec_coe_accu:{:.3}".format(FilterRecoverCoeLoss))
                else:
                    print(print_info)
                    # print(print_info + " use_cost_func:{},".format(use_cost_func) + " evaluate_use_cost_step:{}".format(self.evaluate_use_cost_step))
                if self.use_wandb:
                    wandb.log({
                        'Evaluate/step': self.evaluate_step_num,
                        'Evaluate/step_episode': self.evaluate_episode_num,
                        'Evaluate/EpRet': self.evaluate_current_rewards,
                        'Evaluate/EpEnvLen': infos['env_length'],
                        'Evaluate/EpLen': self.evaluate_current_lengths,
                        "Evaluate/EpTime": self.evaluate_current_ep_time,
                        "Evaluate/EpProgress": infos['progress'],
                        "Evaluate/EpRetAction": self.evaluate_current_rewards_action,
                        "Evaluate/EpOverCost": self.current_overworks,
                    })      
                    if infos['env_length'] < infos['max_env_len']-1 and infos['progress'] == 1:
                        task_success = True
                    num_worker, num_robot = infos['num_worker'], infos['num_robot']
                    self.eval_progress_avgs[num_worker-1][num_robot-1].update(torch.tensor([task_success], dtype=torch.float32, device=self._device))
                    self.eval_env_len_avgs[num_worker-1][num_robot-1].update(torch.tensor([infos['env_length']], dtype=torch.float32, device=self._device))
                    wandb.log({f'Eval_avg_progress/{num_worker}_{num_robot}': self.eval_progress_avgs[num_worker-1][num_robot-1].get_mean()})
                    wandb.log({f'Eval_avg_env_len/{num_worker}_{num_robot}': self.eval_env_len_avgs[num_worker-1][num_robot-1].get_mean()})
                        # self.evaluate_table.add_data(infos['env_length'], ' '.join(action_info_list), infos['progress'])
                        # wandb.log({"Action": self.evaluate_table}) 
                        # if not test:
                        #     # checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num) + '_len_' + str(infos['env_length'].item()) + '_rew_' + "{:.2f}".format(self.evaluate_current_rewards.item())
                        #     checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num)
                        #     self.save(os.path.join(self.nn_dir, checkpoint_name)) 
                    if test:
                        self.test_table.add_data(infos['worker_initial_pose'] , infos["robot_initial_pose"], infos['box_initial_pose'], infos['progress'], infos['env_length'], infos['human_move'], infos['agv_move'])
                        self.test_table3.add_data(' '.join(time_step_list), ' '.join(action_info_list))
                action_info_list = []
                next_obs = self.env_reset(num_worker=reset_n_worker, num_robot=reset_n_robot, evaluate=True) 
            self.evaluate_current_rewards = self.evaluate_current_rewards * not_dones
            self.evaluate_current_lengths = self.evaluate_current_lengths * not_dones
            self.evaluate_current_ep_time = self.evaluate_current_ep_time * not_dones
            self.evaluate_current_rewards_action = self.evaluate_current_rewards_action * not_dones
            self.current_overworks = self.current_overworks * not_dones
            self.obs = next_obs.copy()
            if dones_flag[0]:
                self.evaluate_episode_num += 1  
                break

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        return task_success
    
    def train(self):
        self.init_tensors()
        total_time = 0
        # rep_count = 0
        self.obs = self.env_reset()
        while True:
            self.epoch_num += 1
            if self._test:     
                if self.config["test_all_settings"]:
                    for w in range(self.config["max_num_worker"]):
                        for r in range(self.config["max_num_robot"]):
                            for i in range(self.config['test_times']):
                                self.evaluate_epoch(test=True)
                            if self.use_wandb:
                                index = w*self.config["max_num_worker"]+r
                                time_span = self.test_table.get_column("env_length")[index*self.config['test_times']: (index+1)*self.config['test_times']]
                                human_move = self.test_table.get_column("human_move")[index*self.config['test_times']: (index+1)*self.config['test_times']]
                                robot_move = self.test_table.get_column("robot_move")[index*self.config['test_times']: (index+1)*self.config['test_times']]
                                self.test_table2.add_data(w+1, r+1, np.max(time_span), np.min(time_span), np.mean(time_span), np.mean(human_move), np.mean(robot_move))
                    if self.use_wandb:
                        wandb.log({"Instances": self.test_table}) 
                        wandb.log({"Instances2": self.test_table2}) 
                        wandb.log({"Actions": self.test_table3}) 
                        wandb.finish()
                else:
                    for i in range(self.config['test_times']):
                        self.evaluate_epoch(test=True)
                break
            else:
                for w in range(self.config["max_num_worker"]):
                    for r in range(self.config["max_num_robot"]):
                        step_time, play_time, update_time, epoch_total_time, loss = self.train_epoch(w+1,r+1)
                        # step_time, play_time, update_time, epoch_total_time, loss = self.train_epoch(3,3)

                        # total_time += epoch_total_time
                        # self.step_num += self.num_steps_per_epoch

                        # fps_step = self.num_steps_per_epoch / step_time
                        # fps_step_inference = self.num_steps_per_epoch / play_time
                        # fps_total = self.num_steps_per_epoch / epoch_total_time

                        if self.use_wandb:
                            wandb.log({
                                    "Train/step": self.step_num,
                                    "Train/train_epoch": self.epoch_num,
                                    'Train/buffer_size': self.replay_buffer.transitions.index,
                                })  
                            if self.game_rewards.current_size > 0:
                                wandb.log({
                                    'Train/Mrewards': self.game_rewards.get_mean(),
                                    'Train/MLen': self.game_lengths.get_mean(),
                                })  
                if self.step_num > self.max_steps:
                    if self.use_wandb:
                        wandb.finish()
                    break

    def get_fatigue_related_predtion_loss(self, fatigue_data_list):
        with torch.no_grad():
            fatigue_datas = data.stack_from_array(fatigue_data_list, device=self._device)
            delta_fatigue = fatigue_datas['next_phy_fatigue'] - fatigue_datas['phy_fatigue']
            EpLossCompare = self.loss_criterion(delta_fatigue, fatigue_datas['phy_delta_predict']).mean()
            EpFilterPredictLoss = self.loss_criterion(delta_fatigue, fatigue_datas['filter_phy_delta_predict']).mean()
            # EpFilterPredictAccu = fatigue_datas['filter_phy_fat_accuracy'].mean()
            FilterRecoverCoeAccu = fatigue_datas['filter_phy_rec_coe_accuracy'].mean()
            FilterFatigueCoeAccu = fatigue_datas['filter_phy_fat_coe_accuracy'].mean()
            if self.config['other_filters']:
                EpFilterPredictLoss_kf = self.loss_criterion(delta_fatigue, fatigue_datas['filter_phy_delta_predict_kf']).mean()
                # EpFilterPredictAccu_kf = fatigue_datas['filter_phy_fat_accuracy_kf'].mean()
                FilterRecoverCoeAccu_kf = fatigue_datas['filter_phy_rec_coe_accuracy_kf'].mean()
                FilterFatigueCoeAccu_kf = fatigue_datas['filter_phy_fat_coe_accuracy_kf'].mean()
                EpFilterPredictLoss_ekf = self.loss_criterion(delta_fatigue, fatigue_datas['filter_phy_delta_predict_ekf']).mean()
                # EpFilterPredictAccu_ekf = fatigue_datas['filter_phy_fat_accuracy_ekf'].mean()
                FilterRecoverCoeAccu_ekf = fatigue_datas['filter_phy_rec_coe_accuracy_ekf'].mean()
                FilterFatigueCoeAccu_ekf = fatigue_datas['filter_phy_fat_coe_accuracy_ekf'].mean() 
            
            else:
                EpFilterPredictLoss_kf = 0
                # EpFilterPredictAccu_kf = 0
                FilterRecoverCoeAccu_kf = 0
                FilterFatigueCoeAccu_kf = 0
                EpFilterPredictLoss_ekf = 0
                # EpFilterPredictAccu_ekf = 0
                FilterRecoverCoeAccu_ekf = 0
                FilterFatigueCoeAccu_ekf = 0
    
        dict_loss_pf_filter = {'EpFilterPredictLoss': EpFilterPredictLoss, 'FilterRecoverCoeAccu': FilterRecoverCoeAccu, 'FilterFatigueCoeAccu': FilterFatigueCoeAccu}
        dict_loss_kf_filter = {'EpFilterPredictLoss_kf': EpFilterPredictLoss_kf, 'FilterRecoverCoeAccu_kf': FilterRecoverCoeAccu_kf, 'FilterFatigueCoeAccu_kf': FilterFatigueCoeAccu_kf}
        dict_loss_ekf_filter = {'EpFilterPredictLoss_ekf': EpFilterPredictLoss_ekf, 'FilterRecoverCoeAccu_ekf': FilterRecoverCoeAccu_ekf, 'FilterFatigueCoeAccu_ekf': FilterFatigueCoeAccu_ekf}
        return EpLossCompare, dict_loss_pf_filter, dict_loss_kf_filter, dict_loss_ekf_filter
