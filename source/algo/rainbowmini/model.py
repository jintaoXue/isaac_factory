# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass
@dataclass
class DimState :
    action_mask: int = 10

    # state_depot_hoop: int = 1
    types_state_depot_hoop: int = 3

    # have_raw_hoops: int = 1
    types_have_raw_hoops: int = 2

    # state_depot_bending_tube: int = 1
    types_state_depot_bending_tube: int = 3

    # have_raw_bending_tube: int = 1
    types_have_raw_bending_tube: int = 2

    # station_state_inner_left: int = 1
    types_station_state_inner_left: int = 8

    # station_state_inner_right: int = 1
    types_station_state_inner_right: int = 6

    # station_state_outer_left: int = 1
    types_station_state_outer_left: int = 8

    # station_state_outer_right: int = 1
    types_station_state_outer_right: int = 6

    # cutting_machine_state: int = 1
    types_cutting_machine_state: int = 3

    # is_full_products: int = 1
    types_is_full_products: int = 2

    # produce_product_req: int = 1
    types_produce_product_req: int = 2

    time_step: int = 1
    progress: int = 1
    max_env_len: int = 1
    #raw_product_embd
    types_raw_product_embd: int = 20

    types_worker_state: int = 7
    types_worker_task: int = 11
    types_worker_pose: int = 12
    types_agv_state: int = 4
    types_agv_task: int = 7
    types_agv_pose: int = 10
    types_box_state: int = 3
    types_box_task: int = 4
    
    #(worker+agv+box)*(state+task+pose)*(max_num=3)
    max_num_entity: int = 3
    src_seq_len: int = 16 + 3*3*max_num_entity

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  def __init__(self, config, action_space):
    super(DQN, self).__init__()
    self.action_space = action_space
    self.fe = FeatureExtractorV1(config, DimState())
    self.fm = FeatureMapper(self.fe.dim_feature, 1024)    
    self.fc_h_v = NoisyLinear(self.fm.dim_output, config['hidden_size'], std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(self.fm.dim_output, config['hidden_size'], std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(config['hidden_size'], 1, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(config['hidden_size'], action_space, std_init=config['noisy_std'])
    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
  def forward(self, x, log=False):
    action_mask = x['action_mask']
    x = self.fe(x)
    x = self.fm(x)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # q = torch.nn.functional.normalize(q, dim=1)
    assert log==False, "rainbowmini only support log False"
    # if log:  # Use log softmax for numerical stability
    #   q = F.log_softmax(q, dim=1)  # Log probabilities with action over second dimension
    # else:
    q = torch.clamp(q, min=self.Vmin, max=self.Vmax)
    q = (action_mask-1)*(-self.Vmin) + q*action_mask
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class FeatureExtractorV1(nn.Module):
    def __init__(self, cfg, dimstate : DimState):
        super().__init__()
        # self.model_id = model_id
        # self.method_name = config.method_name
        # self.model_dir = config.model_dir
        # self.model_num = int(config.model_num)
        self.dim_state = dimstate
        self.device = cfg['device']
        self.dtype = torch.float32
        hidden_size = cfg['hidden_size']

        self.action_mask_embedding= nn.Sequential(
            nn.Linear(dimstate.action_mask, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.state_depot_hoop_embedding = nn.Embedding(dimstate.types_state_depot_hoop, hidden_size)
        self.have_raw_hoops_embedding = nn.Embedding(dimstate.types_have_raw_hoops, hidden_size)
        self.state_depot_bending_tube_embedding= nn.Embedding(dimstate.types_state_depot_bending_tube, hidden_size)
        self.have_raw_bending_tube_embedding = nn.Embedding(dimstate.types_have_raw_bending_tube, hidden_size)
        self.station_state_inner_left_embedding = nn.Embedding(dimstate.types_station_state_inner_left, hidden_size)
        self.station_state_inner_right_embedding = nn.Embedding(dimstate.types_station_state_inner_right, hidden_size)
        self.station_state_outer_left_embedding = nn.Embedding(dimstate.types_station_state_outer_left, hidden_size)
        self.station_state_outer_right_embedding = nn.Embedding(dimstate.types_station_state_outer_right, hidden_size)
        self.cutting_machine_state_embedding = nn.Embedding(dimstate.types_cutting_machine_state, hidden_size)
        self.is_full_products_embedding = nn.Embedding(dimstate.types_is_full_products, hidden_size)
        self.produce_product_req_embedding = nn.Embedding(dimstate.types_produce_product_req, hidden_size)
        self.time_step_embedding = nn.Sequential(
            nn.Linear(dimstate.time_step, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.progress_emb = nn.Sequential(
            nn.Linear(dimstate.progress, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.worker_state_embd = nn.Embedding(dimstate.types_worker_state, hidden_size)
        self.worker_task_embd = nn.Embedding(dimstate.types_worker_task, hidden_size)
        self.worker_pose_embd = nn.Embedding(dimstate.types_worker_pose, hidden_size)
        self.agv_state_embd = nn.Embedding(dimstate.types_agv_state, hidden_size)
        self.agv_task_embd = nn.Embedding(dimstate.types_agv_task, hidden_size)
        self.agv_pose_embd = nn.Embedding(dimstate.types_agv_pose, hidden_size)
        self.box_state_embd = nn.Embedding(dimstate.types_box_state, hidden_size)
        self.box_task_embd = nn.Embedding(dimstate.types_box_task, hidden_size)
        self.type_embedding = VectorizedEmbedding(hidden_size)
        
        self.global_head = MultiheadAttentionGlobalHead(hidden_size, nhead=4, dropout=0.1)
        self.dim_feature = hidden_size

    def forward(self, state, **kwargs):
        
        # assert (torch.where(state['state_depot_hoop']>=0, True, False).all() and torch.where(state['state_depot_hoop']<self.dim_state.types_state_depot_hoop, True, False).all()),'0'
        # assert  (torch.where(state['have_raw_hoops']>=0, True, False).all() and torch.where(state['have_raw_hoops']<self.dim_state.types_have_raw_hoops, True, False).all()),'1'
        # assert  (torch.where(state['state_depot_bending_tube']>=0, True, False).all() and torch.where(state['state_depot_bending_tube']<self.dim_state.types_state_depot_bending_tube, True, False).all()),'2'
        # assert  (torch.where(state['have_raw_bending_tube']>=0, True, False).all() and torch.where(state['have_raw_bending_tube']<self.dim_state.types_have_raw_bending_tube, True, False).all()),'3'
        # if not (torch.where(state['station_state_inner_left']>=0, True, False).all() and torch.where(state['station_state_inner_left']<self.dim_state.types_station_state_inner_left, True, False).all()):
        #   print(5)
        # if not (torch.where(state['station_state_inner_right']>=0, True, False).all() and torch.where(state['station_state_inner_right']<self.dim_state.types_station_state_inner_right, True, False).all()):
        #   print(6)
        # if not (torch.where(state['station_state_outer_left']>=0, True, False).all() and torch.where(state['station_state_outer_left']<self.dim_state.types_station_state_outer_left, True, False).all()):
        #   print(7)
        # if not (torch.where(state['station_state_outer_right']>=0, True, False).all() and torch.where(state['station_state_outer_right']<self.dim_state.types_station_state_outer_right, True, False).all()):
        #   print(8)
        # if not (torch.where(state['cutting_machine_state']>=0, True, False).all() and torch.where(state['cutting_machine_state']<self.dim_state.types_cutting_machine_state, True, False).all()):
        #   print(9)
        # if not (torch.where(state['is_full_products']>=0, True, False).all() and torch.where(state['is_full_products']<self.dim_state.types_is_full_products, True, False).all()):
        #   print(10)
        # if not (torch.where(state['produce_product_req']>=0, True, False).all() and torch.where(state['produce_product_req']<self.dim_state.types_produce_product_req, True, False).all()):
        #   print(11)
        # if not (torch.where(state['worker_state_0']>=0, True, False).all() and torch.where(state['worker_state_0']<self.dim_state.types_worker_state, True, False).all()):
        #   print(12)
        # if not (torch.where(state['worker_state_1']>=0, True, False).all() and torch.where(state['worker_state_1']<self.dim_state.types_worker_state, True, False).all()):
        #   print(13)
        # if not (torch.where(state['worker_task_0']>=0, True, False).all() and torch.where(state['worker_task_0']<self.dim_state.types_worker_task, True, False).all()):
        #   print(14)
        # if not (torch.where(state['worker_task_1']>=0, True, False).all() and torch.where(state['worker_task_1']<self.dim_state.types_worker_task, True, False).all()):
        #   print(15)
        # if not (torch.where(state['worker_pose_0']>=0, True, False).all() and torch.where(state['worker_pose_0']<self.dim_state.types_worker_pose, True, False).all()):
        #   print(16)
        # if not (torch.where(state['worker_pose_1']>=0, True, False).all() and torch.where(state['worker_pose_1']<self.dim_state.types_worker_pose, True, False).all()):
        #   print(17)
        # if not (torch.where(state['agv_state_0']>=0, True, False).all() and torch.where(state['agv_state_0']<self.dim_state.types_agv_state, True, False).all()):
        #   print(18)
        # if not (torch.where(state['agv_state_1']>=0, True, False).all() and torch.where(state['agv_state_1']<self.dim_state.types_agv_state, True, False).all()):
        #   print(19)
        # if not (torch.where(state['agv_task_0']>=0, True, False).all() and torch.where(state['agv_task_0']<self.dim_state.types_agv_task, True, False).all()):
        #   print(20)
        # if not (torch.where(state['agv_task_1']>=0, True, False).all() and torch.where(state['agv_task_1']<self.dim_state.types_agv_task, True, False).all()):
        #   print(21)
        # if not (torch.where(state['agv_pose_0']>=0, True, False).all() and torch.where(state['agv_pose_0']<self.dim_state.types_agv_pose, True, False).all()):
        #   print(22)
        # if not (torch.where(state['agv_pose_1']>=0, True, False).all() and torch.where(state['agv_pose_1']<self.dim_state.types_agv_pose, True, False).all()):
        #   print(23)
        # if not (torch.where(state['box_state_0']>=0, True, False).all() and torch.where(state['box_state_0']<self.dim_state.types_box_state, True, False).all()):
        #   print(24)
        # if not (torch.where(state['box_state_1']>=0, True, False).all() and torch.where(state['box_state_1']<self.dim_state.types_box_state, True, False).all()):
        #   print(25)
        # if not (torch.where(state['box_task_0']>=0, True, False).all() and torch.where(state['box_task_0']<self.dim_state.types_box_task, True, False).all()):
        #   print(26)
        # if not (torch.where(state['box_task_1']>=0, True, False).all() and torch.where(state['box_task_1']<self.dim_state.types_box_task, True, False).all()):
        #   print(27)
        # if not (torch.where(state['box_pose_0']>=0, True, False).all() and torch.where(state['box_pose_0']<self.dim_state.types_agv_pose, True, False).all()):
        #   print(28)
        # if not (torch.where(state['box_pose_1']>=0, True, False).all() and torch.where(state['box_pose_1']<self.dim_state.types_agv_pose, True, False).all()):
        #   print(29)

        action_mask_embedding = self.action_mask_embedding(state['action_mask'])
        state_depot_hoop_embedding= self.state_depot_hoop_embedding(state['state_depot_hoop'])
        have_raw_hoops_embedding= self.have_raw_hoops_embedding(state['have_raw_hoops'])
        state_depot_bending_tube_embedding= self.state_depot_bending_tube_embedding(state['state_depot_bending_tube'])
        have_raw_bending_tube_embedding = self.have_raw_bending_tube_embedding(state['have_raw_bending_tube'])
        station_state_inner_left_embedding = self.station_state_inner_left_embedding(state['station_state_inner_left'])
        station_state_inner_right_embedding = self.station_state_inner_right_embedding(state['station_state_inner_right'])
        station_state_outer_left_embedding = self.station_state_outer_left_embedding(state['station_state_outer_left'])
        station_state_outer_right_embedding = self.station_state_outer_right_embedding(state['station_state_outer_right'])
        cutting_machine_state_embedding = self.cutting_machine_state_embedding(state['cutting_machine_state'])
        is_full_products_embedding = self.is_full_products_embedding(state['is_full_products'])
        produce_product_req_embedding = self.produce_product_req_embedding(state['produce_product_req'])
        time_step_embedding = self.time_step_embedding(state['time_step'])
        progress = self.progress_emb(state['progress'])

        worker_state_0 = self.worker_state_embd(state['worker_state_0'])
        worker_state_1 = self.worker_state_embd(state['worker_state_1'])
        worker_task_0 = self.worker_task_embd(state['worker_task_0'])
        worker_task_1 = self.worker_task_embd(state['worker_task_1'])
        worker_pose_0 = self.worker_pose_embd(state['worker_pose_0'])
        worker_pose_1 = self.worker_pose_embd(state['worker_pose_1'])

        agv_state_0 = self.agv_state_embd(state['agv_state_0'])
        agv_state_1 = self.agv_state_embd(state['agv_state_1'])
        agv_task_0 = self.agv_task_embd(state['agv_task_0'])
        agv_task_1 = self.agv_task_embd(state['agv_task_1'])
        agv_pose_0 = self.agv_pose_embd(state['agv_pose_0'])
        agv_pose_1 = self.agv_pose_embd(state['agv_pose_1'])

        box_state_0 = self.box_state_embd(state['box_state_0'])
        box_state_1 = self.box_state_embd(state['box_state_1'])
        box_task_0 = self.box_task_embd(state['box_task_0'])
        box_task_1 = self.box_task_embd(state['box_task_1'])
        box_pose_0 = self.agv_pose_embd(state['box_pose_0'])
        box_pose_1 = self.agv_pose_embd(state['box_pose_1'])

        type_embedding = self.type_embedding(state)

        ###########################################################################################
        ###########################################################################################

        all_embs = torch.cat([action_mask_embedding.unsqueeze(1), state_depot_hoop_embedding, have_raw_hoops_embedding, state_depot_bending_tube_embedding, 
                              have_raw_bending_tube_embedding, station_state_inner_left_embedding, station_state_inner_right_embedding, 
                              station_state_outer_left_embedding, station_state_outer_right_embedding, cutting_machine_state_embedding, 
                              is_full_products_embedding, produce_product_req_embedding, time_step_embedding.unsqueeze(1), progress.unsqueeze(1), 
                              worker_state_0, worker_task_0, worker_pose_0, worker_state_1, worker_task_1, worker_pose_1, 
                              agv_state_0, agv_task_0, agv_pose_0, agv_state_1, agv_task_1, agv_pose_1,
                              box_state_0, box_task_0, box_pose_0, box_state_1, box_task_1, box_pose_1], dim=1)
        type_embedding = self.type_embedding(state)
        outputs, attns = self.global_head(all_embs, type_embedding)
        # self.attention = attns.detach().clone().cpu()
        return outputs



class FeatureExtractorV2(nn.Module):
    def __init__(self, cfg, dimstate : DimState):
        super().__init__()
        # self.model_id = model_id
        # self.method_name = config.method_name
        # self.model_dir = config.model_dir
        # self.model_num = int(config.model_num)
        self.device = cfg['device']
        self.dtype = torch.float32
        hidden_size = cfg['hidden_size']
        self.num_lstm_layers = 1
        self.bi = True
        self.action_mask_embedding= nn.LSTM(dimstate.action_mask, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.state_depot_hoop_embedding = nn.LSTM(dimstate.state_depot_hoop, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.have_raw_hoops_embedding = nn.LSTM(dimstate.have_raw_hoops, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.state_depot_bending_tube_embedding= nn.LSTM(dimstate.state_depot_bending_tube, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.have_raw_bending_tube_embedding = nn.LSTM(dimstate.have_raw_bending_tube, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.station_state_inner_left_embedding = nn.LSTM(dimstate.station_state_inner_left, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.station_state_inner_right_embedding = nn.LSTM(dimstate.station_state_inner_right, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.station_state_outer_left_embedding = nn.LSTM(dimstate.station_state_outer_left, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.station_state_outer_right_embedding = nn.LSTM(dimstate.station_state_outer_right, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.cutting_machine_state_embedding = nn.LSTM(dimstate.cutting_machine_state, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.is_full_products_embedding = nn.LSTM(dimstate.is_full_products, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.produce_product_req_embedding = nn.LSTM(dimstate.produce_product_req, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)
        self.time_step_embedding = nn.LSTM(dimstate.time_step, hidden_size, self.num_lstm_layers, bidirectional=self.bi, batch_first=True)

        self.dim_feature = hidden_size if not self.bi else hidden_size*2
        self.type_embedding = VectorizedEmbedding(self.dim_feature)
        self.global_head = MultiheadAttentionGlobalHead(self.dim_feature, nhead=4, dropout=0.1)

    def forward(self, state, **kwargs):

        action_mask_embedding, (h_n, h_c)  = self.action_mask_embedding(state['action_mask'])
        state_depot_hoop_embedding, _ = self.state_depot_hoop_embedding(state['state_depot_hoop'])
        have_raw_hoops_embedding, _ = self.have_raw_hoops_embedding(state['have_raw_hoops'])
        state_depot_bending_tube_embedding, _ = self.state_depot_bending_tube_embedding(state['state_depot_bending_tube'])
        have_raw_bending_tube_embedding, _ = self.have_raw_bending_tube_embedding(state['have_raw_bending_tube'])
        station_state_inner_left_embedding, _ = self.station_state_inner_left_embedding(state['station_state_inner_left'])
        station_state_inner_right_embedding, _ = self.station_state_inner_right_embedding(state['station_state_inner_right'])
        station_state_outer_left_embedding, _ = self.station_state_outer_left_embedding(state['station_state_outer_left'])
        station_state_outer_right_embedding, _ = self.station_state_outer_right_embedding(state['station_state_outer_right'])
        cutting_machine_state_embedding, _ = self.cutting_machine_state_embedding(state['cutting_machine_state'])
        is_full_products_embedding, _ = self.is_full_products_embedding(state['is_full_products'])
        produce_product_req_embedding, _ = self.produce_product_req_embedding(state['produce_product_req'])
        time_step_embedding, _ = self.time_step_embedding(state['time_step'])
        type_embedding = self.type_embedding(state)

        ###########################################################################################
        ###########################################################################################

        all_embs = torch.cat([action_mask_embedding.unsqueeze(1), state_depot_hoop_embedding.unsqueeze(1), have_raw_hoops_embedding.unsqueeze(1), state_depot_bending_tube_embedding.unsqueeze(1), 
                              have_raw_bending_tube_embedding.unsqueeze(1), station_state_inner_left_embedding.unsqueeze(1), station_state_inner_right_embedding.unsqueeze(1), 
                              station_state_outer_left_embedding.unsqueeze(1), station_state_outer_right_embedding.unsqueeze(1), cutting_machine_state_embedding.unsqueeze(1), 
                              is_full_products_embedding.unsqueeze(1), produce_product_req_embedding.unsqueeze(1), time_step_embedding.unsqueeze(1)], dim=1)
        type_embedding = self.type_embedding(state)
        outputs, attns = self.global_head(all_embs, type_embedding)
        # self.attention = attns.detach().clone().cpu()
        return outputs



class VectorizedEmbedding(nn.Module):
    def __init__(self, dim_embedding: int):
        """A module which associates learnable embeddings to types

        :param dim_embedding: features of the embedding
        :type dim_embedding: int
        """
        super().__init__()
        self.state_types = {
            "action_mask": 0,
            "state_depot_hoop": 1,
            "have_raw_hoops": 2,
            "state_depot_bending_tube": 3,
            'have_raw_bending_tube': 4,
            'station_state_inner_left': 5,
            'station_state_inner_right': 6,
            'station_state_outer_left': 7,
            'station_state_outer_right': 8,
            'cutting_machine_state': 9,
            'is_full_products': 10,
            'produce_product_req': 11,
            'time_step':12,
            'progress':13,
            'worker_state_0':14,
            'worker_task_0':15, 
            'worker_pose_0':16,

            'worker_state_1':17,
            'worker_task_1':18,
            'worker_pose_1':19,

            'agv_state_0':20,
            'agv_task_0':21,  
            'agv_pose_0':22,

            'agv_state_1':23,
            'agv_task_1':24,
            'agv_pose_1':25,

            'box_state_0':26,
            'box_task_0':27,
            'box_pose_0':28,

            'box_state_1':29,
            'box_task_1':30,
            'box_pose_1':31,
        }
        self.dim_embedding = dim_embedding
        self.embedding = nn.Embedding(len(self.state_types), dim_embedding)

        # Torch script did like dicts as Tensor selectors, so we are going more primitive.
        # self.PERCEPTION_LABEL_CAR: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        # self.PERCEPTION_LABEL_PEDESTRIAN: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_PEDESTRIAN"]
        # self.PERCEPTION_LABEL_CYCLIST: int = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CYCLIST"]

    def forward(self, state):
        """
        Model forward: embed the given elements based on their type.

        Assumptions:
        - agent of interest is the first one in the batch
        - other agents follow
        - then we have polylines (lanes)
        """

        with torch.no_grad():
            batch_size = state['action_mask'].shape[0]
            #total_len = 12
            total_len = len(self.state_types)

            indices = torch.full(
                (batch_size, total_len),
                fill_value=-1,
                dtype=torch.long,
                device=state['action_mask'].device,
            )

            indices[:, 0].fill_(self.state_types["action_mask"])
            indices[:, 1].fill_(self.state_types["state_depot_hoop"])
            indices[:, 2].fill_(self.state_types["have_raw_hoops"])
            indices[:, 3].fill_(self.state_types["state_depot_bending_tube"])
            indices[:, 4].fill_(self.state_types["have_raw_bending_tube"])
            indices[:, 5].fill_(self.state_types["station_state_inner_left"])
            indices[:, 6].fill_(self.state_types["station_state_inner_right"])
            indices[:, 7].fill_(self.state_types["station_state_outer_left"])
            indices[:, 8].fill_(self.state_types["station_state_outer_right"])
            indices[:, 9].fill_(self.state_types["cutting_machine_state"])
            indices[:, 10].fill_(self.state_types["is_full_products"])
            indices[:, 11].fill_(self.state_types["produce_product_req"])
            indices[:, 12].fill_(self.state_types["time_step"])
            indices[:, 13].fill_(self.state_types["progress"])
            indices[:, 14].fill_(self.state_types["worker_state_0"])
            indices[:, 15].fill_(self.state_types["worker_task_0"])
            indices[:, 16].fill_(self.state_types["worker_pose_0"])
            indices[:, 17].fill_(self.state_types["worker_state_1"])
            indices[:, 18].fill_(self.state_types["worker_task_1"])
            indices[:, 19].fill_(self.state_types["worker_pose_1"])
            indices[:, 20].fill_(self.state_types["agv_state_0"])
            indices[:, 21].fill_(self.state_types["agv_task_0"])
            indices[:, 22].fill_(self.state_types["agv_pose_0"])
            indices[:, 23].fill_(self.state_types["agv_state_1"])
            indices[:, 24].fill_(self.state_types["agv_task_1"])
            indices[:, 25].fill_(self.state_types["agv_pose_1"])
            indices[:, 26].fill_(self.state_types["box_state_0"])
            indices[:, 27].fill_(self.state_types["box_task_0"])
            indices[:, 28].fill_(self.state_types["box_pose_0"])
            indices[:, 29].fill_(self.state_types["box_state_1"])
            indices[:, 30].fill_(self.state_types["box_task_1"])
            indices[:, 31].fill_(self.state_types["box_pose_1"])

        return self.embedding.forward(indices)
    

class FeatureMapper(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, dim_output),
        )
        self.dim_output = dim_output
    def forward(self, x):
        return self.fm(x)


class MultiheadAttentionGlobalHead(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.encoder = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, inputs: torch.Tensor, type_embedding: torch.Tensor):
        inputs = inputs.transpose(0, 1)
        type_embedding = type_embedding.transpose(0, 1)
        outputs, attns = self.encoder(inputs[[0]], inputs + type_embedding, inputs)
        return outputs.squeeze(0), attns.squeeze(1)
    


class FeatureEmbeddingBlock(nn.Module):
  def __init__(self, hidden_size, dimstate : DimState):
    super().__init__()
    self.dim_state = dimstate
    self.dtype = torch.float32
    self.hidden_size = hidden_size
    self.action_mask_embedding= nn.Sequential(
        nn.Linear(dimstate.action_mask, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.time_step_embedding = nn.Sequential(
        nn.Linear(dimstate.time_step, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.max_env_len_ebd = nn.Sequential(
        nn.Linear(dimstate.max_env_len, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.progress_emb = nn.Sequential(
        nn.Linear(dimstate.progress, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.state_depot_hoop_embedding = nn.Embedding(dimstate.types_state_depot_hoop, hidden_size)
    self.have_raw_hoops_embedding = nn.Embedding(dimstate.types_have_raw_hoops, hidden_size)
    self.state_depot_bending_tube_embedding= nn.Embedding(dimstate.types_state_depot_bending_tube, hidden_size)
    self.have_raw_bending_tube_embedding = nn.Embedding(dimstate.types_have_raw_bending_tube, hidden_size)
    self.station_state_inner_left_embedding = nn.Embedding(dimstate.types_station_state_inner_left, hidden_size)
    self.station_state_inner_right_embedding = nn.Embedding(dimstate.types_station_state_inner_right, hidden_size)
    self.station_state_outer_left_embedding = nn.Embedding(dimstate.types_station_state_outer_left, hidden_size)
    self.station_state_outer_right_embedding = nn.Embedding(dimstate.types_station_state_outer_right, hidden_size)
    self.cutting_machine_state_embedding = nn.Embedding(dimstate.types_cutting_machine_state, hidden_size)
    self.is_full_products_embedding = nn.Embedding(dimstate.types_is_full_products, hidden_size)
    self.produce_product_req_embedding = nn.Embedding(dimstate.types_produce_product_req, hidden_size)
    self.raw_product_embd= nn.Embedding(dimstate.types_raw_product_embd, hidden_size)
    self.worker_state_embd = nn.Embedding(dimstate.types_worker_state, hidden_size)
    self.worker_task_embd = nn.Embedding(dimstate.types_worker_task, hidden_size)
    self.worker_pose_embd = nn.Embedding(dimstate.types_worker_pose, hidden_size)
    self.agv_state_embd = nn.Embedding(dimstate.types_agv_state, hidden_size)
    self.agv_task_embd = nn.Embedding(dimstate.types_agv_task, hidden_size)
    self.agv_pose_embd = nn.Embedding(dimstate.types_agv_pose, hidden_size)
    self.box_state_embd = nn.Embedding(dimstate.types_box_state, hidden_size)
    self.box_task_embd = nn.Embedding(dimstate.types_box_task, hidden_size)

  def forward(self, state, **kwargs):

    action_mask_embedding = self.action_mask_embedding(state['action_mask'])
    time_step_embedding = self.time_step_embedding(state['time_step'])
    progress = self.progress_emb(state['progress'])
    max_env_len_ebd = self.max_env_len_ebd(state['max_env_len'])
    #operation: * math.sqrt(self.hidden_size) according to the paper attention is all you need
    state_depot_hoop_embedding= self.state_depot_hoop_embedding(state['state_depot_hoop'])
    have_raw_hoops_embedding= self.have_raw_hoops_embedding(state['have_raw_hoops'])
    state_depot_bending_tube_embedding= self.state_depot_bending_tube_embedding(state['state_depot_bending_tube'])
    have_raw_bending_tube_embedding = self.have_raw_bending_tube_embedding(state['have_raw_bending_tube'])
    station_state_inner_left_embedding = self.station_state_inner_left_embedding(state['station_state_inner_left'])
    station_state_inner_right_embedding = self.station_state_inner_right_embedding(state['station_state_inner_right'])
    station_state_outer_left_embedding = self.station_state_outer_left_embedding(state['station_state_outer_left'])
    station_state_outer_right_embedding = self.station_state_outer_right_embedding(state['station_state_outer_right'])
    cutting_machine_state_embedding = self.cutting_machine_state_embedding(state['cutting_machine_state'])
    is_full_products_embedding = self.is_full_products_embedding(state['is_full_products'])
    produce_product_req_embedding = self.produce_product_req_embedding(state['produce_product_req'])
    raw_product_embd = self.raw_product_embd(state['raw_products'])
    
    worker_state = self.worker_state_embd(state['worker_state']).squeeze(2) 
    worker_task = self.worker_task_embd(state['worker_task']).squeeze(2) 
    worker_pose = self.worker_pose_embd(state['worker_pose']).squeeze(2) 

    agv_state = self.agv_state_embd(state['agv_state']).squeeze(2) 
    agv_task = self.agv_task_embd(state['agv_task']).squeeze(2) 
    agv_pose = self.agv_pose_embd(state['agv_pose']).squeeze(2) 

    box_state = self.box_state_embd(state['box_state']).squeeze(2) 
    box_task = self.box_task_embd(state['box_task']).squeeze(2) 
    box_pose = self.agv_pose_embd(state['box_pose']).squeeze(2) 


    ###########################################################################################
    ###########################################################################################

    all_embs = torch.cat([action_mask_embedding.unsqueeze(1), state_depot_hoop_embedding, have_raw_hoops_embedding, state_depot_bending_tube_embedding, 
                          have_raw_bending_tube_embedding, station_state_inner_left_embedding, station_state_inner_right_embedding, 
                          station_state_outer_left_embedding, station_state_outer_right_embedding, cutting_machine_state_embedding, 
                          is_full_products_embedding, produce_product_req_embedding, raw_product_embd, max_env_len_ebd.unsqueeze(1), time_step_embedding.unsqueeze(1), progress.unsqueeze(1), 
                          worker_state, worker_task, worker_pose, 
                          agv_state, agv_task, agv_pose, 
                          box_state, box_task, box_pose,], dim=1)
    mask = state['token_mask'].unsqueeze(-1).repeat(1,1,self.hidden_size)
    
    return all_embs*mask*math.sqrt(self.hidden_size)

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.encoder = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, q, k, v, mask):
        outputs, attns = self.encoder(q, k, v, mask)
        # outputs, attns = self.encoder(q.transpose(0,1), k.transpose(0,1), v.transpose(0,1), key_padding_mask = mask)
        return outputs
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        # self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: FeatureEmbeddingBlock, tgt_embed: InputEmbeddings, 
          src_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt).unsqueeze(1)
        # tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(self, state):
        src = self.encode(state, None)
        tgt = self.decode(src, None, state['action_mask'], None)
        return self.projection_layer(tgt)
    

def build_transformer(dim_state: DimState, d_model: int=512, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = FeatureEmbeddingBlock(d_model, dim_state)
    tgt_embed = nn.Sequential(
            nn.Linear(dim_state.action_mask, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, dim_state.src_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(2):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(1):
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    # decoder = None
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = FeatureMapper(d_model, d_model)
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

class DQNTrans(nn.Module):
  def __init__(self, config, action_space):
    super(DQNTrans, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.transformer : Transformer = build_transformer(DimState(), hidden_size) 
    self.fc_h_v = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])
    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
    
  def forward(self, x, log=False):
    action_mask = x['action_mask']
    x = self.transformer(x)
    x = x.squeeze(1) # squeeze the query sequence
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # q = torch.nn.functional.normalize(q, dim=1)
    assert log==False, "rainbowmini only support log False"
    # if log:  # Use log softmax for numerical stability
    #   q = F.log_softmax(q, dim=1)  # Log probabilities with action over second dimension
    # else:
    q = torch.clamp(q, min=self.Vmin, max=self.Vmax)
    q = (action_mask-1)*(-self.Vmin) + q*action_mask
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
