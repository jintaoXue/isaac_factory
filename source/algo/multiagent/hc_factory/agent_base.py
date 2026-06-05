from __future__ import annotations
from pydoc import doc
import torch
from abc import ABC, abstractmethod

class AgentBase(ABC):

    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device

    @abstractmethod
    def act(self, env_state_action_dict: dict):
        pass
    
    def keep_first_one(self, t):
        ones_indices = (t == 1).nonzero(as_tuple=True)[0]
        result = torch.zeros_like(t)
        
        if ones_indices.numel() > 0:
            first_one_idx = ones_indices[0]
            result[first_one_idx] = 1
            
        return result
    
    def keep_last_one(self, t):
        ones_indices = (t == 1).nonzero(as_tuple=True)[0]
        result = torch.zeros_like(t)
        
        if ones_indices.numel() > 0:
            last_one_idx = ones_indices[-1]
            result[last_one_idx] = 1
            
        return result
