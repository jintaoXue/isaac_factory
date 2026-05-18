
from ..env_asset_cfg.cfg_storage import CfgStorage
from ..env_asset_cfg.cfg_material_product import CfgRegistrationInfos, CfgProductOrder, CfgProductProcess
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll
import torch
import copy

class TaskManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_storage = CfgStorage
        self.decode_action_to_task_records = {
            "product_selection": {"product": "none", "product_index" : "none"},
            "process_task_planning": {"task": "none", "task_index" : 0},
        }
        self.inverse_index_to_task_name = {v: k for k, v in CfgProcessTaskGalleryInAll.items()}

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

        self.decode_action_product_sequencing(env_state_action_dict)
        self.decode_action_to_task_records["product_selection"] = self.decode_action_product_selection(env_state_action_dict)
        self.decode_action_to_task_records["process_task_planning"] = self.decode_action_process_task_planning(env_state_action_dict)
        self.decode_action_to_task_records["human_robot_allocation"] = self.decode_action_human_robot_allocation(env_state_action_dict)

        return env_state_action_dict
    
    def decode_action_product_sequencing(self, env_state_action_dict):
        action_product_sequencing = env_state_action_dict["action"]["product_sequencing"]
        # Return a torch tensor filled with zeros that matches the shape of action_product_sequencing,
        # placed on the correct device.
        if action_product_sequencing.sum() == 0:
            env_state_action_dict["progress"]["next_product"] = None
            env_state_action_dict["progress"]["next_product_index"] = None
        else:
            product_type_index = action_product_sequencing.nonzero()[0][0]
            product_type = CfgProductProcess.keys()[product_type_index]
            material_states_dict = env_state_action_dict["material"]
            for material_name, material_state in material_states_dict.items():
                key_variables = material_state["key_variables"]
                current_task = material_state["current_task"]
                if key_variables["type_name"] == product_type and current_task == "none":
                    env_state_action_dict["progress"]["next_product"] = product_type
                    env_state_action_dict["progress"]["next_product_index"] = key_variables["idx"]
                    break
    
    def decode_action_product_selection(self, env_state_action_dict):
        # action shape is (1 + self.parallel_producing_limit,)
        action_product_selection = env_state_action_dict["action"]["product_selection"]
        decoded_action = {"product": "none", "product_index" : "none"}
        if action_product_selection.sum() == 0:
            return decoded_action
        else:
            _index = action_product_selection.nonzero()[0][0]
            if _index == action_product_selection.shape[0] - 1:
                decoded_action["product"] = env_state_action_dict["progress"]["next_product"]
                decoded_action["product_index"] = env_state_action_dict["progress"]["next_product_index"]
            else:
                decoded_action["product"] = env_state_action_dict["progress"]["producing"][_index]
                decoded_action["product_index"] = env_state_action_dict["progress"]["producing_indexs"][_index]
        return decoded_action

    def decode_action_process_task_planning(self, env_state_action_dict):
        action_process_task_planning = env_state_action_dict["action"]["process_task_planning"]
        decoded_action = {"task": "none", "task_index" : 0}
        if action_process_task_planning.sum() == 0:
            return decoded_action
        else:
            _index = action_process_task_planning.nonzero()[0][0]
            decoded_action["task"] = self.inverse_index_to_task_name[_index]
            decoded_action["task_index"] = _index
        return decoded_action
    
    def decode_action_human_robot_allocation(self, env_state_action_dict):
        action_human_robot_allocation = env_state_action_dict["action"]["human_robot_allocation"]
        #shape is (upper_bound_num_human,)
        action_human = action_human_robot_allocation["human"]
        #shape is (upper_bound_num_robot,)
        action_robot = action_human_robot_allocation["robot"]
        decoded_action = {"human": "none", "human_index": "none", "robot": "none", "robot_index" : "none"}
        if action_human.sum() != 0:
            _index = action_human.nonzero()[0][0]
            decoded_action["human"] = self.inverse_index_to_human_name[_index]
            decoded_action["human_index"] = _index
        if action_robot.sum() != 0:
            _index = action_robot.nonzero()[0][0]
            decoded_action["robot"] = self.inverse_index_to_robot_name[_index]
            decoded_action["robot_index"] = _index
        return decoded_action