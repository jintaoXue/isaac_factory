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
    worker_fatigue_dim: int = 1
    fatigue_coe_dim: int = 1 #the number of coefficients is 10, each coefficient's dim is 1
    types_agv_state: int = 4
    types_agv_task: int = 7
    types_agv_pose: int = 10
    types_box_state: int = 3
    types_box_task: int = 4
    
    #2*3*max_num_entity = (agv+box)*(state+task+pose)*(max_num_entity=3)
    #1*5*max_num_entity = (human)*(state+task+pose+phy_fatigue+psy_fatigue+phy_fatigue_coe)*(max_num_entity=3)
    max_num_entity: int = 3
    src_seq_len: int = 16 + 2*3*max_num_entity + 1*(5+10)*max_num_entity
    #action_embedding 10 + phy 1 + psy 1 + idx 1 + coefficients 10
    cost_seq_len: int = 23
    # device_str = "cuda:0" 

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=1):
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
    
class CostFeatureMapper(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 512), nn.ReLU(),
            nn.Linear(512, dim_output),
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
    # self.phy_recover_coe_embedding= nn.Sequential(
    #     nn.Linear(dimstate.action_mask, hidden_size), nn.ReLU(),
    #     nn.Linear(hidden_size, hidden_size),
    # )
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
    self.worker_phy_fatigue_embd= nn.Sequential(
        nn.Linear(dimstate.worker_fatigue_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.worker_psy_fatigue_embd= nn.Sequential(
        nn.Linear(dimstate.worker_fatigue_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.phy_fatigue_coe_embedding= nn.Sequential(
        nn.Linear(dimstate.fatigue_coe_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
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
    worker_phy_fatigue_embd = self.worker_phy_fatigue_embd(state['worker_fatigue_phy']).squeeze(2)
    worker_psy_fatigue_embd = self.worker_psy_fatigue_embd(state['worker_fatigue_psy']).squeeze(2)
    #we embedding each fatigue fatigue coefficient to hidden_size (batch_size, num_worker, dim) -> (batch_size, num_worker*dim, 1) -> (batch_size, num_worker*dim, hidden_size)
    worker_phy_fatigue_coe = self.phy_fatigue_coe_embedding(state['phy_fatigue_coe'].unsqueeze(-1))

    agv_state = self.agv_state_embd(state['agv_state']).squeeze(2) 
    agv_task = self.agv_task_embd(state['agv_task']).squeeze(2) 
    agv_pose = self.agv_pose_embd(state['agv_pose']).squeeze(2) 

    box_state = self.box_state_embd(state['box_state']).squeeze(2) 
    box_task = self.box_task_embd(state['box_task']).squeeze(2) 
    box_pose = self.agv_pose_embd(state['box_pose']).squeeze(2) 


    ###########################################################################################
    ###########################################################################################
    _shape = worker_phy_fatigue_coe.shape
    all_embs = torch.cat([action_mask_embedding.unsqueeze(1), state_depot_hoop_embedding, have_raw_hoops_embedding, state_depot_bending_tube_embedding, 
                          have_raw_bending_tube_embedding, station_state_inner_left_embedding, station_state_inner_right_embedding, 
                          station_state_outer_left_embedding, station_state_outer_right_embedding, cutting_machine_state_embedding, 
                          is_full_products_embedding, produce_product_req_embedding, raw_product_embd, max_env_len_ebd.unsqueeze(1), time_step_embedding.unsqueeze(1), progress.unsqueeze(1), 
                          worker_state, worker_task, worker_pose, worker_phy_fatigue_embd, worker_psy_fatigue_embd, worker_phy_fatigue_coe.view(_shape[0], _shape[1]*_shape[2], _shape[3]),
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
    
class CostProjectionLayer(nn.Module):

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
    

def build_transformer(dim_state: DimState, d_model: int=512, h: int=8, dropout: float=0.1, d_ff: int=1024) -> Transformer:
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


class DQNTransNoduel(nn.Module):
  def __init__(self, config, action_space):
    super(DQNTransNoduel, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.transformer : Transformer = build_transformer(DimState(), hidden_size) 
    self.fc_h_a = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])
    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
    
  def forward(self, x, log=False):
    action_mask = x['action_mask']
    x = self.transformer(x)
    x = x.squeeze(1) # squeeze the query sequence
    q = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
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



################ with cost function net ############################################################################
################ with cost function net ############################################################################
################ with cost function net ############################################################################
################ with cost function net ############################################################################


class CostFeatureEmbeddingBlock(nn.Module):
  def __init__(self, max_num_worker, hidden_size, dimstate : DimState):
    super().__init__()
    self.dim_state = dimstate
    self.dtype = torch.float32
    self.hidden_size = hidden_size
    self.worker_phy_fatigue_embd= nn.Sequential(
        nn.Linear(dimstate.worker_fatigue_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.worker_psy_fatigue_embd= nn.Sequential(
        nn.Linear(dimstate.worker_fatigue_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.action = torch.arange(dimstate.action_mask, dtype = torch.int32).unsqueeze(0)
    self.action_embedding= nn.Embedding(dimstate.action_mask, hidden_size)
    self.worker_idx_embedding= nn.Embedding(dimstate.max_num_entity, hidden_size)
    self.phy_fatigue_coe_embedding= nn.Sequential(
        nn.Linear(dimstate.fatigue_coe_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    )
    self.max_n_human = max_num_worker

  def forward(self, state, **kwargs):
    
    worker_idx_embedding = self.worker_idx_embedding(state['charac_idx'])
    batch_size = worker_idx_embedding.shape[0]
    action_embedding = self.action_embedding(self.action.repeat([batch_size, 1]))
    worker_phy_fatigue_embd = self.worker_phy_fatigue_embd(state['phy_fatigue'])
    worker_psy_fatigue_embd = self.worker_psy_fatigue_embd(state['psy_fatigue'])
    # torch.index_select(state['phy_fatigue_coe'], 1, state['charac_idx'])
    _shape = state['phy_fatigue_coe'].shape
    fatigue_coe = state['phy_fatigue_coe'].gather(1, state['charac_idx'].unsqueeze(-1).unsqueeze(-1).repeat(1,1,_shape[-1])).squeeze(-2)
    fatigue_coe_embedding = self.phy_fatigue_coe_embedding(fatigue_coe.unsqueeze(-1))
    ###########################################################################################

    # all_embs = torch.cat([worker_phy_fatigue_embd.unsqueeze(1), worker_psy_fatigue_embd.unsqueeze(1), action_embedding.unsqueeze(1), worker_idx_embedding.unsqueeze(1)], dim=1)
    all_embs = torch.cat([action_embedding, worker_phy_fatigue_embd.unsqueeze(1), worker_psy_fatigue_embd.unsqueeze(1), worker_idx_embedding.unsqueeze(1), fatigue_coe_embedding], dim=1)

    return all_embs*math.sqrt(self.hidden_size)
  
  def forward_predict(self, state):
    batch_size = state['worker_fatigue_phy'].shape[0]
    # worker_mask = state['worker_mask'].unsqueeze(-1).repeat(1,1,self.hidden_size)
    action_embedding = self.action_embedding(self.action.repeat([batch_size, 1]))
    predict_list = []
    
    for i in range(self.max_n_human):
        worker_idx_embedding = self.worker_idx_embedding(torch.tensor([i]).repeat(batch_size))
        worker_phy_fatigue_embd = self.worker_phy_fatigue_embd(state['worker_fatigue_phy'][:,i])
        worker_psy_fatigue_embd = self.worker_psy_fatigue_embd(state['worker_fatigue_psy'][:,i])
        fatigue_coe = state['phy_fatigue_coe'][:,i]
        fatigue_coe_embedding = self.phy_fatigue_coe_embedding(fatigue_coe.unsqueeze(-1))
        all_embs = torch.cat([action_embedding, worker_phy_fatigue_embd.unsqueeze(1), worker_psy_fatigue_embd.unsqueeze(1), worker_idx_embedding.unsqueeze(1), fatigue_coe_embedding], dim=1)
        # all_embs = torch.cat([action_embedding, worker_phy_fatigue_embd.unsqueeze(1), worker_psy_fatigue_embd.unsqueeze(1), worker_idx_embedding.unsqueeze(1)], dim=1)
        _predict = all_embs*math.sqrt(self.hidden_size)
        predict_list.append(_predict)
    return predict_list


class CostTransformer(nn.Module):

    def __init__(self, encoder: Encoder, cost_decoder: Decoder, decoder: Decoder, src_embed: FeatureEmbeddingBlock, cost_tgt_embed: CostFeatureEmbeddingBlock, tgt_embed: InputEmbeddings, 
          src_pos: PositionalEncoding, cost_tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer, cost_projection_layer: CostProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.cost_decoder = cost_decoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.cost_tgt_embed = cost_tgt_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.cost_tgt_pos = cost_tgt_pos
        self.projection_layer = projection_layer
        self.cost_projection_layer = cost_projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)self.cost_decoder(cost_tgt, encoder_output, None, None)
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
    
    def decode_cost(self, encoder_output: torch.Tensor, cost_state: torch.Tensor):
        cost_tgt = self.cost_tgt_embed(cost_state)
        cost_tgt = self.cost_tgt_pos(cost_tgt)
        cost_tgt = self.cost_decoder(cost_tgt, encoder_output, None, None)[:,:DimState.action_mask]
        return self.cost_projection_layer(cost_tgt)
    
    def predict_cost(self, encoder_output: torch.Tensor, cost_state: torch.Tensor):
        cost_tgt_list = self.cost_tgt_embed.forward_predict(cost_state)
        _list = []
        for _predict in cost_tgt_list:
           _predict = self.cost_tgt_pos(_predict)
           _predict = self.cost_decoder(_predict, encoder_output, None, None)[:,:DimState.action_mask] 
           _predict = self.cost_projection_layer(_predict).squeeze(-1).unsqueeze(-2)
           _list.append(_predict)
        return torch.cat(_list, dim=-2)
           
        


def build_cost_net(max_num_worker, dim_state: DimState, d_model: int=512, h: int=8, dropout: float=0.1, d_ff: int=1024) -> CostTransformer:
    # Create the embedding layers
    src_embed = FeatureEmbeddingBlock(d_model, dim_state)
    tgt_embed = nn.Sequential(
            nn.Linear(dim_state.action_mask, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    cost_tgt_embed = CostFeatureEmbeddingBlock(max_num_worker, d_model, dim_state)
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, dim_state.src_seq_len, dropout)
    cost_tgt_pos = PositionalEncoding(d_model, dim_state.cost_seq_len, dropout)
    
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
    
    # Create the cost func decoder blocks
    costfunc_decoder_blocks = []
    for _ in range(1):
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_cross_attention_block, feed_forward_block, dropout)
        costfunc_decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    # decoder = None
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    cost_decoder = Decoder(d_model, nn.ModuleList(costfunc_decoder_blocks))
    
    # Create the projection layer
    projection_layer = FeatureMapper(d_model, d_model)
    cost_projection_layer = CostFeatureMapper(d_model, 2)
    # Create the transformer
    transformer = CostTransformer(encoder, cost_decoder, decoder, src_embed, cost_tgt_embed, tgt_embed, src_pos, cost_tgt_pos, projection_layer, cost_projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


class SafeDQNTrans(nn.Module):
  def __init__(self, config, action_space):
    super(SafeDQNTrans, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.transformer : CostTransformer = build_cost_net(config['max_num_worker'], DimState(), hidden_size) 
    self.fc_h_v = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])
    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
    self.ftg_thresh_phy = config.get('ftg_thresh_phy', 0.95)
    self.ftg_thresh_psy = config.get('ftg_thresh_psy', 0.8)

    cost_training_dict = {'transformer.encoder', 'transformer.cost_decoder', 'transformer.cost_tgt_embed', 'transformer.cost_projection_layer'}
    self.trainable_params_sft = []
    self.trainable_params_rl = []
    for name, p in self.named_parameters():
        if any([1 for sub_name in cost_training_dict if sub_name in name]):
          self.trainable_params_sft.append(p)
        else:
          self.trainable_params_rl.append(p)
    
  def forward(self, x, use_cost_function=False, log=False):

    if use_cost_function:
        # print('use_cost function')
        delta_predict = self.predict_cost(x)
        # cost_mask = torch.where(cost_predict[..., 0] < self.ftg_thresh_phy, 1, 0)*torch.where(cost_predict[..., 1] < self.ftg_thresh_psy, 1, 0)
        predict = delta_predict[..., 0] + x['worker_fatigue_phy'][..., 0].unsqueeze(1).repeat(1, self.action_space, 1)
        cost_mask = torch.where(predict < self.ftg_thresh_phy, 1, 0)
        worker_mask = x['worker_mask'].unsqueeze(1).repeat(1, self.action_space, 1)
        cost_mask = cost_mask*worker_mask
        action_mask = x['action_mask']*torch.any(cost_mask, dim=-1)
    else:
        action_mask = x['action_mask']
        cost_mask = None

    x = self.transformer(x) # (batch_size, 1, d_model)
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
    return q, cost_mask
    
  def predict_cost(self, x):
    with torch.no_grad():
        src = self.transformer.encode(x, None)
        return self.transformer.predict_cost(src, x)
  
  def cost_forward(self, x, log=False):
    src = self.transformer.encode(x, None)
    return self.transformer.decode_cost(src, cost_state=x)
  
#   def forward_together(self, x, log=False):
#     src = self.transformer.encode(x, None)
#     tgt = self.transformer.decode(src, None, x['action_mask'], None)
#     x = self.transformer.projection_layer(tgt)
#     cost = self.transformer.decode_cost(src, cost_state=x)
    
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

        



#################333333333333333333333333############################
######################## PPO actor critic########################
######################## PPO actor critic########################
######################## PPO actor critic########################
######################## PPO actor critic########################



class ActorCriticTransformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: FeatureEmbeddingBlock, tgt_embed: InputEmbeddings, 
          src_pos: PositionalEncoding, projection_layer: ProjectionLayer, critic_decoder: Decoder, projection_layer_critic: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.critic_decoder = critic_decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer
        self.projection_layer_critic = projection_layer_critic

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
    
    def decode_critic(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        with torch.no_grad():
            tgt = self.tgt_embed(tgt).unsqueeze(1)
        # tgt = self.tgt_pos(tgt)
        return self.critic_decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(self, state):
        src = self.encode(state, None)
        tgt = self.decode(src, None, state['action_mask'], None)
        return self.projection_layer(tgt)
        
    def forward_critic(self, state):
        with torch.no_grad():
            src = self.encode(state, None)
        critic_tgt = self.decode_critic(src, None, state['action_mask'], None)
        return self.projection_layer_critic(critic_tgt)



def build_actor_critic_transformer(dim_state: DimState, d_model: int=512, h: int=8, dropout: float=0.1, d_ff: int=1024) -> Transformer:
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

        # Create the decoder blocks
    critic_decoder_blocks = []
    for _ in range(1):
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_cross_attention_block, feed_forward_block, dropout)
        critic_decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    # decoder = None
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    # critic_decoder
    critic_decoder = Decoder(d_model, nn.ModuleList(critic_decoder_blocks))
    
    # Create the projection layer
    projection_layer = FeatureMapper(d_model, d_model)
    projection_layer_critic = FeatureMapper(d_model, d_model)
    # Create the transformer
    transformer = ActorCriticTransformer(encoder, decoder, src_embed, tgt_embed, src_pos, projection_layer, critic_decoder, projection_layer_critic)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


class PPOTrans(nn.Module):
  def __init__(self, config, action_space):
    super(PPOTrans, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.transformer : ActorCriticTransformer = build_actor_critic_transformer(DimState(), hidden_size) 
    self.fc_h_v = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])

    self.fc_h_v_critic = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a_critic = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v_critic = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a_critic = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])

    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
    self.ftg_thresh_phy = config.get('ftg_thresh_phy', 0.95)
    self.ftg_thresh_psy = config.get('ftg_thresh_psy', 0.8)

    critic_training_dict = {'transformer.critic_decoder', 'transformer.projection_layer_critic'}
    # critic_training_dict = {'transformer.encoder', 'transformer.critic_decoder', 'transformer.projection_layer_critic'}
    self.trainable_params_sft = []
    self.trainable_params_rl = []
    for name, p in self.named_parameters():
        if any([1 for sub_name in critic_training_dict if sub_name in name]):
          self.trainable_params_sft.append(p)
        else:
          self.trainable_params_rl.append(p)
    
  def forward(self, x, log=False):
    #actor network
    action_mask = x['action_mask']
    x = self.transformer(x)
    x = x.squeeze(1) # squeeze the query sequence
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # q = torch.nn.functional.normalize(q, dim=1)
    assert log==False, "rainbowmini only support log False"
    q = torch.clamp(q, min=self.Vmin, max=self.Vmax)
    q = (action_mask-1)*(-self.Vmin) + q*action_mask
    n = torch.tanh(q)
    action_prob = F.softmax(n, dim=1)
    return action_prob

  
  def forward_critic(self, x, log=False):

    x = self.transformer.forward_critic(x)
    x = x.squeeze(1) # squeeze the query sequence
    v = self.fc_z_v_critic(F.relu(self.fc_h_v_critic(x)))  # Value stream
    a = self.fc_z_a_critic(F.relu(self.fc_h_a_critic(x)))  # Advantage stream
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # q = torch.nn.functional.normalize(q, dim=1)
    assert log==False, "rainbowmini only support log False"
    return q
     
    
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()






##########################44444444444444444444########################
######################## PPO actor critic cost########################
######################## PPO actor critic cost########################
######################## PPO actor critic cost########################
######################## PPO actor critic cost########################


class ActorCriticCostTransformer(ActorCriticTransformer):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: FeatureEmbeddingBlock, tgt_embed: InputEmbeddings, 
          src_pos: PositionalEncoding, projection_layer: ProjectionLayer, critic_decoder: Decoder, projection_layer_critic: ProjectionLayer, cost_decoder: Decoder, projection_layer_cost: ProjectionLayer) -> None:
        super().__init__(encoder, decoder, src_embed, tgt_embed, src_pos, projection_layer, critic_decoder, projection_layer_critic)
        self.cost_decoder = cost_decoder
        self.projection_layer_cost = projection_layer_cost
    

    def decode_cost(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        with torch.no_grad():
            tgt = self.tgt_embed(tgt).unsqueeze(1)
        # tgt = self.tgt_pos(tgt)
        return self.cost_decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward_cost(self, state):
        with torch.no_grad():
            src = self.encode(state, None)
        cost_tgt = self.decode_cost(src, None, state['action_mask'], None)
        return self.projection_layer_cost(cost_tgt)



def build_actor_critic_cost_transformer(dim_state: DimState, d_model: int=512, h: int=8, dropout: float=0.1, d_ff: int=1024) -> ActorCriticCostTransformer:
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

    # Create the critic decoder blocks
    critic_decoder_blocks = []
    for _ in range(1):
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_cross_attention_block, feed_forward_block, dropout)
        critic_decoder_blocks.append(decoder_block)
    
    # Create the cost decoder blocks
    cost_decoder_blocks = []
    for _ in range(1):
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_cross_attention_block, feed_forward_block, dropout)
        cost_decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    # decoder = None
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    # critic_decoder
    critic_decoder = Decoder(d_model, nn.ModuleList(critic_decoder_blocks))
    cost_decoder = Decoder(d_model, nn.ModuleList(cost_decoder_blocks))
    
    # Create the projection layer
    projection_layer = FeatureMapper(d_model, d_model)
    projection_layer_critic = FeatureMapper(d_model, d_model)
    projection_layer_cost = FeatureMapper(d_model, d_model)
    # Create the transformer
    transformer = ActorCriticCostTransformer(encoder, decoder, src_embed, tgt_embed, src_pos, projection_layer, critic_decoder, projection_layer_critic, cost_decoder, projection_layer_cost)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


class SafePPOTrans(nn.Module):
  
  def __init__(self, config, action_space):
    super(SafePPOTrans, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.transformer : ActorCriticCostTransformer = build_actor_critic_cost_transformer(DimState(), hidden_size) 
    self.fc_h_v = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])

    self.fc_h_v_critic = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a_critic = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v_critic = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a_critic = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])

    self.fc_h_v_cost = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a_cost = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v_cost = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a_cost = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])

    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
    self.ftg_thresh_phy = config.get('ftg_thresh_phy', 0.95)
    self.ftg_thresh_psy = config.get('ftg_thresh_psy', 0.8)

    critic_training_dict = {'transformer.critic_decoder', 'transformer.projection_layer_critic', 
      'fc_h_v_critic', 'fc_h_a_critic', 'fc_z_v_critic', 'fc_z_a_critic'}
    cost_training_dict = {'transformer.cost_decoder', 'transformer.projection_layer_cost', 
      'fc_h_v_cost', 'fc_h_a_cost', 'fc_z_v_cost', 'fc_z_a_cost'}
    self.trainable_params_sft = []
    self.trainable_params_rl = []
    self.trainable_params_cost = []
    for name, p in self.named_parameters():
        if any([1 for sub_name in critic_training_dict if sub_name in name]):
          self.trainable_params_sft.append(p)
        elif any([1 for sub_name in cost_training_dict if sub_name in name]):
          self.trainable_params_cost.append(p)
        else:
           self.trainable_params_rl.append(p)

  def forward(self, x, log=False):
    #actor network
    action_mask = x['action_mask']
    x = self.transformer(x)
    x = x.squeeze(1) # squeeze the query sequence
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # q = torch.nn.functional.normalize(q, dim=1)
    assert log==False, "rainbowmini only support log False"
    q = torch.clamp(q, min=self.Vmin, max=self.Vmax)
    q = (action_mask-1)*(-self.Vmin) + q*action_mask
    n = torch.tanh(q)
    action_prob = F.softmax(n, dim=1)
    return action_prob

  
  def forward_critic(self, x, log=False):

    x = self.transformer.forward_critic(x)
    x = x.squeeze(1) # squeeze the query sequence
    v = self.fc_z_v_critic(F.relu(self.fc_h_v_critic(x)))  # Value stream
    a = self.fc_z_a_critic(F.relu(self.fc_h_a_critic(x)))  # Advantage stream
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # q = torch.nn.functional.normalize(q, dim=1)
    assert log==False, "rainbowmini only support log False"
    return q
     
    
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

  def forward_cost(self, x, log=False):

    x = self.transformer.forward_cost(x)
    x = x.squeeze(1) # squeeze the query sequence
    v = self.fc_z_v_cost(F.relu(self.fc_h_v_cost(x)))  # Value stream
    a = self.fc_z_a_cost(F.relu(self.fc_h_a_cost(x)))  # Advantage stream
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # q = torch.nn.functional.normalize(q, dim=1)
    assert log==False, "rainbowmini only support log False"
    q = torch.clamp(q, min=self.Vmin, max=self.Vmax)
    return q
     
    
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()



##### ablation study #####
##### ablation study #####
##### ablation study #####
##### MLP model #####

class MLPBlock(nn.Module):
  def __init__(self, hidden_size: int, seq_len: int = 79):
    """
    Args:
        hidden_size: 隐藏层大小 (512)
        seq_len: 序列长度 (79)
    """
    super(MLPBlock, self).__init__()
    self.feature_embedding_block = FeatureEmbeddingBlock(hidden_size, DimState())
    # 使用线性层压缩序列维度 (79 -> 1)
    self.seq_compress = nn.Linear(seq_len, 1)
    
    self.mlp_proj_layer = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )
    
  def forward(self, x):
    x = self.feature_embedding_block(x)  # (batch, 79, 512)
    
    # 使用线性层压缩: (batch, 79, 512) -> (batch, 512, 79) -> (batch, 512, 1) -> (batch, 1, 512)
    x = x.transpose(1, 2)  # (batch, 512, 79)
    x = self.seq_compress(x)  # (batch, 512, 1)
    x = x.transpose(1, 2)  # (batch, 1, 512)
    
    x = self.mlp_proj_layer(x)  # (batch, 1, 512)
    return x

class SafeDqnMLP(nn.Module):
  def __init__(self, config, action_space):
    super(SafeDqnMLP, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.mlp_block : MLPBlock = MLPBlock(hidden_size)
    for p in self.mlp_block.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)
    self.fc_h_v = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])
    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
    self.ftg_thresh_phy = config.get('ftg_thresh_phy', 0.95)
    self.ftg_thresh_psy = config.get('ftg_thresh_psy', 0.8)

    cost_training_dict = {'transformer.encoder', 'transformer.cost_decoder', 'transformer.cost_tgt_embed', 'transformer.cost_projection_layer'}
    self.trainable_params_sft = []
    self.trainable_params_rl = []
    for name, p in self.named_parameters():
        if any([1 for sub_name in cost_training_dict if sub_name in name]):
          self.trainable_params_sft.append(p)
        else:
          self.trainable_params_rl.append(p)
    
  def forward(self, x, use_cost_function=False, log=False):

    if use_cost_function:
        # print('use_cost function')
        delta_predict = self.predict_cost(x)
        # cost_mask = torch.where(cost_predict[..., 0] < self.ftg_thresh_phy, 1, 0)*torch.where(cost_predict[..., 1] < self.ftg_thresh_psy, 1, 0)
        predict = delta_predict[..., 0] + x['worker_fatigue_phy'][..., 0].unsqueeze(1).repeat(1, self.action_space, 1)
        cost_mask = torch.where(predict < self.ftg_thresh_phy, 1, 0)
        worker_mask = x['worker_mask'].unsqueeze(1).repeat(1, self.action_space, 1)
        cost_mask = cost_mask*worker_mask
        action_mask = x['action_mask']*torch.any(cost_mask, dim=-1)
    else:
        action_mask = x['action_mask']
        cost_mask = None

    x = self.mlp_block(x) # (batch_size, 1, d_model)
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
    return q, cost_mask
    
  def predict_cost(self, x):
    assert False, "predict_cost is not implemented"
    with torch.no_grad():
        src = self.mlp_block.encode(x, None)
        return self.transformer.predict_cost(src, x)
  
  def cost_forward(self, x, log=False):
    assert False, "cost_forward is not implemented"
    src = self.mlp_block.encode(x, None)
    return self.mlp_block.decode_cost(src, cost_state=x)
    
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()




######### only self-attention model #########


class SelfAttentionModel(nn.Module):

    def __init__(self, encoder: Encoder, src_embed: FeatureEmbeddingBlock, 
          src_pos: PositionalEncoding, seq_len: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.seq_compress = nn.Linear(seq_len, 1)

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        x = x.transpose(1, 2)  # (batch, 512, 79)
        x = self.seq_compress(x)  # (batch, 512, 1)
        x = x.transpose(1, 2)  # (batch, 1, 512)
        return x
    
    def forward(self, state):
        src = self.encode(state, None)
        return self.project(src) # (batch, 1, 512)



def build_self_attention_model(dim_state: DimState, d_model: int=512, h: int=8, dropout: float=0.1, d_ff: int=2048) -> SelfAttentionModel:
    # Create the embedding layers
    src_embed = FeatureEmbeddingBlock(d_model, dim_state)
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, dim_state.src_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(2):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    # decoder = None
    
    # Create the self-attention model
    self_attention_model = SelfAttentionModel(encoder, src_embed, src_pos, dim_state.src_seq_len)
    
    # Initialize the parameters
    for p in self_attention_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return self_attention_model


class OnlySelfAttention(nn.Module):
  def __init__(self, config, action_space):
    super(OnlySelfAttention, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.self_attention_model : SelfAttentionModel = build_self_attention_model(DimState(), hidden_size)
    self.fc_h_v = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])
    self.Vmin = config.get('V_min', -20)
    self.Vmax = config.get('V_max', 20)
    self.ftg_thresh_phy = config.get('ftg_thresh_phy', 0.95)
    self.ftg_thresh_psy = config.get('ftg_thresh_psy', 0.8)

    cost_training_dict = {'transformer.encoder', 'transformer.cost_decoder', 'transformer.cost_tgt_embed', 'transformer.cost_projection_layer'}
    self.trainable_params_sft = []
    self.trainable_params_rl = []
    for name, p in self.named_parameters():
        if any([1 for sub_name in cost_training_dict if sub_name in name]):
          self.trainable_params_sft.append(p)
        else:
          self.trainable_params_rl.append(p)
    
  def forward(self, x, use_cost_function=False, log=False):

    if use_cost_function:
        # print('use_cost function')
        delta_predict = self.predict_cost(x)
        # cost_mask = torch.where(cost_predict[..., 0] < self.ftg_thresh_phy, 1, 0)*torch.where(cost_predict[..., 1] < self.ftg_thresh_psy, 1, 0)
        predict = delta_predict[..., 0] + x['worker_fatigue_phy'][..., 0].unsqueeze(1).repeat(1, self.action_space, 1)
        cost_mask = torch.where(predict < self.ftg_thresh_phy, 1, 0)
        worker_mask = x['worker_mask'].unsqueeze(1).repeat(1, self.action_space, 1)
        cost_mask = cost_mask*worker_mask
        action_mask = x['action_mask']*torch.any(cost_mask, dim=-1)
    else:
        action_mask = x['action_mask']
        cost_mask = None

    x = self.self_attention_model(x) # (batch_size, 1, d_model)
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
    return q, cost_mask
    
  def predict_cost(self, x):
    assert False, "predict_cost is not implemented"
    with torch.no_grad():
        src = self.self_attention_model.encode(x, None)
        return self.self_attention_model.predict_cost(src, x)
  
  def cost_forward(self, x, log=False):
    assert False, "cost_forward is not implemented"
    src = self.self_attention_model.encode(x, None)
    return self.self_attention_model.decode_cost(src, cost_state=x)
    
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()