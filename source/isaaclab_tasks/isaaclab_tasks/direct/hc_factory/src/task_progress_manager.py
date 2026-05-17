
from ..env_asset_cfg.cfg_storage import CfgStorage
import torch
import copy

class TaskManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_storage = CfgStorage

    def reset(self, env_state_action_dict: dict) -> dict:

        self.reset_progress_info(env_state_action_dict)
        return env_state_action_dict
    
    def step(self, env_state_action_dict: dict) -> dict:
        self.step_progress_info(env_state_action_dict)

        return env_state_action_dict

    def reset_progress_info(self, env_state_action_dict) -> dict:
        #production progress reset
        env_state_action_dict["progress"]["product_order"] = copy.deepcopy(self.product_material_manager.cfg_product_order)
        env_state_action_dict["progress"]["not_started"] = self.product_material_manager.generate_order_not_started_dict(self.env_state_action_dict)
        env_state_action_dict["progress"]["next_product"] = None
        env_state_action_dict["progress"]["next_product_index"] = None
        env_state_action_dict["progress"]["producing"] = []
        env_state_action_dict["progress"]["producing_indexs"] = []
        env_state_action_dict["progress"]["finished"] = {}

    def step_progress_info(self, env_state_action_dict) -> dict:
        action = env_state_action_dict['action']
        action_product_sequencing = action["product_sequencing"]
        if action_product_sequencing:   
            env_state_action_dict["progress"]["next_product"] = action_product_sequencing