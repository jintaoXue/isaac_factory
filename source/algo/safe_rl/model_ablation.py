# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import FeatureEmbeddingBlock, PositionalEncoding, MultiheadAttentionBlock, FeedForwardBlock, Encoder, EncoderBlock, DimState, NoisyLinear
from .model import CostTransformer, build_cost_net
##### ablation study #####
##### ablation study #####
##### ablation study #####
##### MLP model #####

class MLPBlock(nn.Module):
  def __init__(self, hidden_size: int, seq_len: int = 79, compress_method: str = 'linear'):
    """
    Args:
        hidden_size: 隐藏层大小 (512)
        seq_len: 序列长度 (79)
        compress_method: 压缩方法，可选:
            - 'linear': 线性层压缩 (默认)
            - 'mean': 平均池化
            - 'max': 最大池化
            - 'attention': 注意力池化
            - 'weighted_mean': 加权平均
            - 'first': 取第一个元素
            - 'last': 取最后一个元素
    """
    super(MLPBlock, self).__init__()
    self.feature_embedding_block = FeatureEmbeddingBlock(hidden_size, DimState())
    self.compress_method = compress_method
    self.seq_len = seq_len
    self.hidden_size = hidden_size
    if compress_method == 'linear':
      # 使用线性层压缩序列维度 (79 -> 1)
      self.seq_compress = nn.Sequential(
        nn.Linear(hidden_size, 1),
        nn.Sigmoid(),
      )
      self.seq_compress2 = nn.Sequential(
        nn.Linear(seq_len, hidden_size),
        nn.Sigmoid(),
      )
    elif compress_method == 'attention':
      # 注意力池化: 使用可学习的查询向量
      self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_size))
      self.attention_scale = math.sqrt(hidden_size)
    elif compress_method == 'weighted_mean':
      # 加权平均: 可学习的权重
      self.seq_weights = nn.Parameter(torch.ones(1, seq_len, 1))
    
    self.mlp_proj_layer = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.Sigmoid(),
    )
    
  def forward(self, x):
    x = self.feature_embedding_block(x)  # (batch, 79, 512)
    
    # 根据压缩方法处理序列维度
    if self.compress_method == 'linear':
      # 使用线性层压缩: (batch, 79, 512) -> (batch, 512, 79) -> (batch, 512, 1) -> (batch, 1, 512)
      x = x/self.hidden_size
      x = x.mean(dim=2, keepdim=False)# (batch, seq_len)
      x = self.seq_compress2(x)  # (batch, hidden_size)
    elif self.compress_method == 'mean':
      # 平均池化: (batch, 79, 512) -> (batch, 1, 512)
      x = x.mean(dim=1, keepdim=True)
    elif self.compress_method == 'max':
      # 最大池化: (batch, 79, 512) -> (batch, 1, 512)
      x = x.max(dim=1, keepdim=True)[0]
    elif self.compress_method == 'attention':
      # 注意力池化: (batch, 79, 512) -> (batch, 1, 512)
      query = self.attention_query.expand(x.size(0), -1, -1)  # (batch, 1, 512)
      scores = torch.matmul(query, x.transpose(1, 2)) / self.attention_scale  # (batch, 1, 79)
      attn_weights = F.softmax(scores, dim=-1)  # (batch, 1, 79)
      x = torch.matmul(attn_weights, x)  # (batch, 1, 512)
    elif self.compress_method == 'weighted_mean':
      # 加权平均: (batch, 79, 512) -> (batch, 1, 512)
      weights = F.softmax(self.seq_weights, dim=1)  # (1, 79, 1)
      x = (x * weights).sum(dim=1, keepdim=True)  # (batch, 1, 512)
    elif self.compress_method == 'first':
      # 取第一个元素: (batch, 79, 512) -> (batch, 1, 512)
      x = x[:, 0:1, :]
    elif self.compress_method == 'last':
      # 取最后一个元素: (batch, 79, 512) -> (batch, 1, 512)
      x = x[:, -1:, :]
    else:
      raise ValueError(f"Unknown compress_method: {self.compress_method}")
    
    x = self.mlp_proj_layer(x)  # (batch, 1, 512)
    return x

class SafeDqnMLP(nn.Module):
  def __init__(self, config, action_space):
    super(SafeDqnMLP, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.mlp_block : MLPBlock = MLPBlock(hidden_size)
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

    x = self.mlp_block(x) # (batch_size, hidden_size)
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
######### only self-attention model #########
######### only self-attention model ################################################################


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



## no noisy net ################################################################
## no noisy net ################################################################ 
## no noisy net ################################################################ 


class noNoisyLinear(NoisyLinear):
  def __init__(self, in_features, out_features, std_init=0.1):
    super(noNoisyLinear, self).__init__(in_features, out_features, std_init=std_init)

  def forward(self, input):
    return F.linear(input, self.weight_mu, self.bias_mu)


class noNoisyDQNTrans(nn.Module):
  def __init__(self, config, action_space):
    super(noNoisyDQNTrans, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.transformer : CostTransformer = build_cost_net(config['max_num_worker'], DimState(), hidden_size) 
    self.fc_h_v = noNoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_h_a = noNoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = noNoisyLinear(hidden_size, 1, std_init=config['noisy_std'])
    self.fc_z_a = noNoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])
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



####### no dueling net ################################################################
####### no dueling net ################################################################
####### no dueling net ################################################################





class noDuelingDQNTrans(nn.Module):
  def __init__(self, config, action_space):
    super(noDuelingDQNTrans, self).__init__()
    self.action_space = action_space
    hidden_size = config['hidden_size']
    self.transformer : CostTransformer = build_cost_net(config['max_num_worker'], DimState(), hidden_size) 
    self.fc_h_v = NoisyLinear(hidden_size, hidden_size, std_init=config['noisy_std'])
    self.fc_z_v = NoisyLinear(hidden_size, action_space, std_init=config['noisy_std'])
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
    q = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
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
