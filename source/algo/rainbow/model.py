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
    state_depot_hoop: int = 1
    have_raw_hoops: int = 1
    state_depot_bending_tube: int = 1
    have_raw_bending_tube: int = 1
    station_state_inner_left: int = 1
    station_state_inner_right: int = 1
    station_state_outer_left: int = 1
    station_state_outer_right: int = 1
    cutting_machine_state: int = 1
    is_full_products: int = 1
    produce_product_req: int = 1
    time_step: int = 1



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
    self.atoms = config['atoms']
    self.action_space = action_space
    self.fe = FeatureExtractor(config, DimState())
    self.fm = FeatureMapper(self.fe.dim_feature, 1024)    
    self.fc_h_v = NoisyLinear(self.fm.dim_output, config['hidden_size'], std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(self.fm.dim_output, config['hidden_size'], std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(config['hidden_size'], self.atoms, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(config['hidden_size'], action_space * self.atoms, std_init=config['noisy_std'])

  def forward(self, x, log=False):
    action_mask = x['action_mask']
    x = self.fe(x)
    x = self.fm(x)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams

    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    action_mask = torch.unsqueeze(action_mask, -1).repeat(1,1,self.atoms)
    prob = ((action_mask[:,:,[0]]-1)*-1)
    q = action_mask*q
    q[:,:,[0]] += prob
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

class FeatureExtractor(nn.Module):
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
        type_embedding = self.type_embedding(state)

        ###########################################################################################
        ###########################################################################################

        all_embs = torch.cat([action_mask_embedding.unsqueeze(1), state_depot_hoop_embedding.unsqueeze(1), have_raw_hoops_embedding.unsqueeze(1), state_depot_bending_tube_embedding.unsqueeze(1), 
                              have_raw_bending_tube_embedding.unsqueeze(1), station_state_inner_left_embedding.unsqueeze(1), station_state_inner_right_embedding.unsqueeze(1), 
                              station_state_outer_left_embedding.unsqueeze(1), station_state_outer_right_embedding.unsqueeze(1), cutting_machine_state_embedding.unsqueeze(1), 
                              is_full_products_embedding.unsqueeze(1), produce_product_req_embedding.unsqueeze(1)], dim=1)
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