# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch import nn

from rl_games.common import vecenv
from rl_games.algos_torch import torch_ext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from omniisaacgymenvs.algo.rainbow.memory import ReplayMemory
from omniisaacgymenvs.algo.rainbow.model import DQN
from tqdm import trange
import time
from omegaconf import DictConfig
from omniisaacgymenvs.utils.data import data
import wandb
import copy
class RainbowAgent():
    def __init__(self, base_name, params):

        self.config : DictConfig = params['config']
        self.base_init(base_name)
        config = self.config
        print(config)
        #######pramameters for model update and act
        self.atoms = config['atoms']
        self.Vmin = config.get('V_min', -10)
        self.Vmax = config.get('V_max', 10)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device=self._device)  # Support (range) of z
        self.delta_z = (self.Vmax - self.Vmin) / (self.atoms - 1)
        self.n = config['multi_step']
        self.discount = config['discount']
        self.norm_clip = config.get('norm_clip', 10)
        ###########for agent training
        self.update_frequency = config.get('update_frequency', 100)
        self.evaluate_interval = config.get('evaluate_interval', 10)
        self.target_update = config.get('target_update', int(1e2))
        self.max_steps = config.get("max_steps", int(5e9))
        self.max_epochs = config.get("max_epochs", int(1e11))
        self.batch_size = config.get('batch_size', 512)
        self.num_warmup_steps = config.get('num_warmup_steps', int(20e3))
        self.demonstration_steps = config.get('demonstration_steps', int(1))
        self.num_steps_per_epoch = config.get("num_steps_per_epoch", 100)
        self.max_env_steps = config.get("horizon_length", 1000) # temporary, in future we will use other approach
        # self.env_rule_based_exploration = config.get('env_rule_based_exploration', True)
        print(self.batch_size, self.num_actors, self.num_agents)
        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)
        #########buffer
        self.priority_weight_increase = (1 - config['priority_weight']) / (self.max_steps - self.num_warmup_steps)
        self.replay_buffer = ReplayMemory(config, config["replay_buffer_size"])
        ####### net
        self.online_net = DQN(config, self.actions_num).to(device=self._device)
        if self._evaluate:
            weights = torch.load(self.train_dir + self._load_dir + self._load_name, weights_only=True)
            self.online_net.load_state_dict(weights['net'])
        self.online_net.train()
        self.target_net = DQN(config, self.actions_num).to(device=self._device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False
        #####
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=config['learning_rate'], eps=config['adam_eps'])
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

    def base_init(self, base_name):

        self.setdefault(self.config, key='device', default='cuda:0')
        ########for replay buffer args initialize 
        self.setdefault(self.config, key='atoms', default=51) 
        self.setdefault(self.config, key='replay_buffer_size', default=int(1e7))
        self.setdefault(self.config, key='history_length', default=1)
        self.setdefault(self.config, key='discount', default=0.99)
        self.setdefault(self.config, key='multi_step', default=1)
        self.setdefault(self.config, key='priority_exponent', default=0.5)
        self.setdefault(self.config, key='priority_weight', default=0.4)
        ########for neural network args initialize
        self.setdefault(self.config, key='architecture', default='canonical')
        self.setdefault(self.config, key='hidden_size', default=512)
        self.setdefault(self.config, key='noisy_std', default=0.1)
        ######for optimizer initialize
        # self.setdefault(self.config, key='learning_rate', default=0.0000625)
        self.setdefault(self.config, key='learning_rate', default=1e-3)
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
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]
        self.observation_space = self.env_info['observation_space']
        self.obs_shape = self.observation_space.shape
        self.obs = None
        self._device = config['device']
        ##evaluate
        self._evaluate = config['evaluate']
        self._load_dir = config['load_dir']
        self._load_name = config['load_name']
        #temporary for Isaac gym compatibility
        print('Env info:')
        print(self.env_info)

        # self.rewards_shaper = config['reward_shaper']
        
        # self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        # self.is_train = config.get('is_train', True)
        # self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        self.save_best_after = config.get('save_best_after', 500)
        # self.print_stats = config.get('print_stats', True)
        # self.rnn_states = None
        # self.name = base_name

        self.save_freq = config.get('save_frequency', 0)

        # self.network = config['network']
        self.num_agents = self.env_info.get('agents', 1)

        self.games_to_track = config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
    
        # self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)
        self.step_num = 0
        self.epoch_num = 0
        self.episode_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0
        
        self.evaluate_step_num = 0
        self.evaluate_episode_num = 0
        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        # full_experiment_name = config.get('full_experiment_name', None)
        # if full_experiment_name:
        #     print(f'Exact experiment name requested from command line: {full_experiment_name}')
        #     self.experiment_name = full_experiment_name
        # else:
        #     self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        # time_now = datetime.now().strftime("_%d-%H-%M-%S")
        time_now = self.config['time_str']
        self.experiment_name = config['name'] + pbt_str + time_now
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter('runs/' + config['name'] + time_now)
        print("Run Directory:", config['name'] + time_now)

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_wandb_logger(self):
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/buffer_size", step_metric="Train/step")
        wandb.define_metric("Train/loss", step_metric="Train/step")
        wandb.define_metric("Train/train_epoch", step_metric="Train/step")

        wandb.define_metric("Train/Mrewards", step_metric="Train/step")
        wandb.define_metric("Train/MLen", step_metric="Train/step")
        wandb.define_metric("Metrics/step_episode", step_metric="Train/step")
        wandb.define_metric("Metrics/EpRet", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpLen", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpEnvLen", step_metric="Metrics/step_episode")

        wandb.define_metric("Metrics/EpTime", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpProgress", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpRetAction", step_metric="Metrics/step_episode")

 
        total = sum([param.nelement() for param in self.online_net.parameters()])
        # print("Number of parameters: %.2fM" % (total/1e6))
        param_table = wandb.Table(columns=["online_net_size", "num_warm_up_steps"], data=[[total, self.num_warmup_steps]])
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
        self.evaluate_table = wandb.Table(columns=["env_length", "action_seq", "progress"])
        return
    
    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return (self.online_net(data.func(state, 'unsqueeze', 0)) * self.support).sum(2).argmax(1)
            # return (self.online_net(data.func(state, 'unsqueeze', 0)) * self.support).sum(2)

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def set_train(self):
        self.online_net.train()

    def set_eval(self):
        self.online_net.eval()

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_rewards_action = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.current_ep_time = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)        
        
        self.evaluate_current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.evaluate_current_rewards_action = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.evaluate_current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.evaluate_current_ep_time = torch.zeros(batch_size, dtype=torch.float32, device=self._device)

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

    def update(self, step):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = self.replay_buffer.sample(self.batch_size)
        states = data.stack_from_array(states.squeeze(), device=self._device)
        next_states = data.stack_from_array(next_states.squeeze(), device=self._device)
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = torch.zeros(self.batch_size, self.atoms, device=self._device)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()

        self.replay_buffer.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        return loss

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

    def env_step(self, actions):
        # actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, actions = self.vec_env.step(actions) # (obs_space) -> (n, obs_space)

        return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos, actions


    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()

        obs = self.obs_to_tensors(obs)

        return obs

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
    
    def train_epoch(self):
        # random_exploration = self.epoch_num < self.num_warmup_steps
        return self.play_steps()
    
    def play_steps(self, random_exploration = False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        loss = None
        for s in range(int(self.num_steps_per_epoch/self.num_actors)):
            obs : dict = self.obs
            random_exploration = self.step_num < self.num_warmup_steps
            self.set_train()
            if self.step_num % self.update_frequency == 0:
                self.reset_noise()
            if random_exploration:
                if self.step_num < self.demonstration_steps:
                    action = None
                else: 
                    action_mask = obs['action_mask']
                    indexs = action_mask.nonzero()
                    index = torch.randint(low=0, high = len(indexs), size = (1,), device=self._device) 
                    action = indexs[index]
            else:
                with torch.no_grad():
                    action = self.act(obs).unsqueeze(0)
            #debug TODO
            # action = None
            step_start = time.time()

            with torch.no_grad():
                next_obs, rewards, dones, infos, action = self.env_step(action)
            # if self.reward_clip > 0:
            #     reward = max(min(reward, self.reward_clip), -self.reward_clip)  # Clip rewards
            step_end = time.time()
            #TODO only support num_agents == 1
            assert self.num_agents == 1, ('only support num_agents == 1')
            self.step_num += self.num_actors * 1
            self.current_rewards += rewards
            self.current_rewards_action += infos["rew_action"]
            self.current_lengths += 1
            self.current_ep_time += (step_end - step_start)
            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            no_timeouts = self.current_lengths <= self.max_env_steps
            dones = dones * no_timeouts
            not_dones = 1.0 - dones.float()

            obs_cpu = {}
            for key, value in obs.items():
                obs_cpu[key] = value.cpu()
            action_cpu = action.squeeze().cpu()
            rewards_cpu = rewards.squeeze().cpu()
            dones_cpu = dones.squeeze().cpu()
            self.replay_buffer.append(obs_cpu, action_cpu, rewards_cpu, dones_cpu)

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
                next_obs = self.env_reset()   
                if self.episode_num % self.evaluate_interval == 0:
                    self.evaluate_epoch()
            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones
            self.current_ep_time = self.current_ep_time * not_dones
            self.current_rewards_action = self.current_rewards_action * not_dones
            # if isinstance(next_obs, dict):    
            #     next_obs_processed = next_obs['obs']

            # rewards = self.rewards_shaper(rewards)
            ####TODO refine replay buffer
            # self.replay_buffer.append(obs, action, torch.unsqueeze(rewards, 1), next_obs_processed, torch.unsqueeze(dones, 1))

            self.obs = copy.deepcopy(next_obs)

            update_time = 0
            if not random_exploration:
                self.replay_buffer.priority_weight = min(self.replay_buffer.priority_weight + self.priority_weight_increase, 1)
                if self.step_num % self.update_frequency == 0:
                    self.set_train()
                    update_time_start = time.time()
                    loss = self.update(self.epoch_num)
                    update_time_end = time.time()
                    update_time = update_time_end - update_time_start
                    if self.use_wandb:
                        wandb.log({
                                'Train/step': self.step_num,
                                "Train/loss": loss.mean().item(),
                            })
                        time_now = datetime.now().strftime("_%d-%H-%M-%S")   
                    print("time_now:{}".format(time_now) +" traning loss:", loss.mean().item())

            # Update target network
            if self.step_num % self.target_update == 0:
                self.update_target_net()

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, loss
    
    def evaluate_epoch(self):
        total_time_start = time.time()
        total_time = 0
        step_time = 0.0
        action_info_list = []
        while True:
            self.set_eval()
            obs : dict = self.obs
            with torch.no_grad():
                action = self.act(obs).unsqueeze(0)

            step_start = time.time()
            # action = None
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
            self.evaluate_current_lengths += 1
            self.evaluate_current_ep_time += (step_end - step_start)
            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            no_timeouts = self.evaluate_current_lengths != self.max_env_steps
            dones = dones * no_timeouts
            not_dones = 1.0 - dones.float()            
            if dones[0]:
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
                    })   
                    if infos['env_length'] < infos['max_env_len']-1 and infos['progress'] == 1:
                        self.evaluate_table.add_data(infos['env_length'], ' '.join(action_info_list), infos['progress'])
                        wandb.log({"Actions": self.evaluate_table}) 
                        checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num) + '_len_' + str(infos['env_length'].item()) + '_rew_' + "{:.2f}".format(self.evaluate_current_rewards.item())
                        self.save(os.path.join(self.nn_dir, checkpoint_name)) 
                action_info_list = []
                next_obs = self.env_reset() 
            self.evaluate_current_rewards = self.evaluate_current_rewards * not_dones
            self.evaluate_current_lengths = self.evaluate_current_lengths * not_dones
            self.evaluate_current_ep_time = self.evaluate_current_ep_time * not_dones
            self.evaluate_current_rewards_action = self.evaluate_current_rewards_action * not_dones
            self.obs = next_obs.copy()
            if dones[0]:
                self.evaluate_episode_num += 1  
                break

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        return 
    
    def train(self):
        self.init_tensors()
        total_time = 0
        # rep_count = 0
        self.obs = self.env_reset()
        while True:
            self.epoch_num += 1
            if self._evaluate:
                self.evaluate_epoch()
            else:
                step_time, play_time, update_time, epoch_total_time, loss = self.train_epoch()
                total_time += epoch_total_time
                # self.step_num += self.num_steps_per_epoch

                fps_step = self.num_steps_per_epoch / step_time
                fps_step_inference = self.num_steps_per_epoch / play_time
                fps_total = self.num_steps_per_epoch / epoch_total_time

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
                    # if self.step_num >= self.num_warmup_steps and loss is not None:
                    #     assert type(loss.mean().item()) == float, "not approprate loss"
                    #     wandb.log({
                    #             # 'Train/buffer_size': self.replay_buffer.transitions.index,
                    #             "Train/Loss": loss.mean().item(),
                    #             # "Train/Loss": loss.mean().cpu().detach().numpy().item(),
                    #         })                  

                self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.step_num)
                # self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.step_num)
                # self.writer.add_scalar('performance/step_fps', fps_step, self.step_num)
                # self.writer.add_scalar('performance/rl_update_time', update_time, self.step_num)
                # self.writer.add_scalar('performance/step_inference_time', play_time, self.step_num)
                # self.writer.add_scalar('performance/step_time', step_time, self.step_num)

                # if self.epoch_num >= self.num_warmup_steps and loss is not None:
                    # self.writer.add_scalar('losses/loss', torch_ext.mean_list(loss).item(), self.step_num)

                # self.writer.add_scalar('info/epochs', self.epoch_num, self.step_num)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    # mean_lengths = self.game_lengths.get_mean()

                    # self.writer.add_scalar('rewards/step', mean_rewards, self.step_num)
                    # self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    # self.writer.add_scalar('episode_lengths/step', mean_lengths, self.step_num)
                    # self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num) + '_rew_' + str(mean_rewards)

                    should_exit = False

                    # if self.save_freq > 0:
                    #     if self.epoch_num % self.save_freq == 0:
                    #         self.save(os.path.join(self.nn_dir, checkpoint_name))
                                # Save model parameters on current device (don't move model between devices)

                    if mean_rewards > self.last_mean_rewards and self.episode_num >= self.save_best_after:
                        # print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards
                        # self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                            print('Maximum reward achieved. Network won!')
                            # self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                    if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                        if self.game_rewards.current_size == 0:
                            print('WARNING: Max epochs reached before any env terminated at least once')
                            mean_rewards = -np.inf

                        # self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
                        #     + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                        print('MAX EPOCHS NUM!')
                        should_exit = True

                    if self.step_num >= self.max_steps and self.max_steps != -1:
                        if self.game_rewards.current_size == 0:
                            print('WARNING: Max steps reached before any env terminated at least once')
                            mean_rewards = -np.inf

                        # self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_step_' + str(self.step_num) \
                        #     + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                        print('MAX STEPS NUM!')
                        should_exit = True

                    update_time = 0
                    #TODO 
                    should_exit = False
                    if should_exit:
                        return self.last_mean_rewards, self.epoch_num


