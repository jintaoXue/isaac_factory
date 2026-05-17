
from ..env_asset_cfg.cfg_storage import CfgStorage
from ..env_asset_cfg.cfg_material_product import CfgRegistrationInfos, CfgProductOrder
import torch
import copy

class TaskManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_storage = CfgStorage

    def reset(self, env_state_action_dict) -> dict:
        #production progress reset
        env_state_action_dict["progress"]["product_order"] = copy.deepcopy(CfgProductOrder)
        env_state_action_dict["progress"]["not_started"] = copy.deepcopy(CfgRegistrationInfos)
        env_state_action_dict["progress"]["next_product"] = None
        env_state_action_dict["progress"]["next_product_index"] = None
        env_state_action_dict["progress"]["producing"] = []
        env_state_action_dict["progress"]["producing_indexs"] = []
        env_state_action_dict["progress"]["finished"] = {}
        env_state_action_dict["progress"]["ongoing_task_records"] = {}
    
    def step(self, env_state_action_dict: dict) -> dict:
        action = env_state_action_dict['action']
        action_product_sequencing = action["product_sequencing"]
           
        env_state_action_dict["progress"]["next_product"] = self.decode_action_product_sequencing(action_product_sequencing)

        return env_state_action_dict
    
    def decode_action_product_sequencing(self, action_product_sequencing : torch.tensor, env_state_action_dict):
        # Return a torch tensor filled with zeros that matches the shape of action_product_sequencing,
        # placed on the correct device.
        if action_product_sequencing none have 1, 
            env_state_action_dict["progress"]["next_product"] = None
            env_state_action_dict["progress"]["next_product_index"] = None
        else:
            find the produtc type and availiable product index in material batch
        return torch.zeros_like(action_product_sequencing, device=self.cuda_device)

