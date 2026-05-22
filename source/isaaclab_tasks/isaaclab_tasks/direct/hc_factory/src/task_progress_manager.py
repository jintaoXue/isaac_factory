
from ..env_asset_cfg.cfg_storage import CfgStorage
from ..env_asset_cfg.cfg_material_product import CfgRegistrationInfos, CfgProductOrder, CfgProductProcess
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll, CfgProcessTaskToMachineMapping, CfgSubtaskGallery
import torch
import copy

class TaskManager:
    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device
        self.cfg_storage = CfgStorage
        self.decode_action_to_task_record = {
            "product_selection": {"product": "none", "product_index" : "none", "new_product_selected": False, "storage_name": "none"},
            "process_task_planning": {"task": "none", "task_index" : 0},
            "human_robot_allocation": {"human": "none", "human_index": "none", "robot": "none", "robot_index" : "none"}
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
        self.decode_action_to_task_record["product_selection"] = self.decode_action_product_selection(env_state_action_dict)
        self.decode_action_to_task_record["process_task_planning"] = self.decode_action_process_task_planning(env_state_action_dict)
        self.decode_action_to_task_record["human_robot_allocation"] = self.decode_action_human_robot_allocation(env_state_action_dict)
        self.step_new_generated_task_record(env_state_action_dict)
        self.step_task_records(env_state_action_dict)
        return env_state_action_dict

    def decode_action_product_sequencing(self, env_state_action_dict):
        action_product_sequencing = env_state_action_dict["action"]["product_sequencing"]
        # Return a torch tensor filled with zeros that matches the shape of action_product_sequencing,
        # placed on the correct device.
        if action_product_sequencing.sum() != 0:
            product_type_index = action_product_sequencing.nonzero()[0][0]
            product_type = list(CfgProductProcess.keys())[product_type_index]
            #shape is (num_material_batch, )
            material_states_dict = env_state_action_dict["material"]
            #find the next_product_index in material_batch list that is the right product type
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
        decoded_action = {"product": "none", "product_index" : "none", "new_product_selected": False, "storage_name": "none"}
        if action_product_selection.sum() == 0:
            return decoded_action
        else:
            _index = action_product_selection.nonzero()[0][0]
            _index = _index.item()
            if _index == action_product_selection.shape[0] - 1:
                decoded_action["product"] = env_state_action_dict["progress"]["next_product"]
                decoded_action["product_index"] = env_state_action_dict["progress"]["next_product_index"]
                decoded_action["new_product_selected"] = True
            else:
                decoded_action["product"] = env_state_action_dict["progress"]["producing"][_index]
                decoded_action["product_index"] = env_state_action_dict["progress"]["producing_indexs"][_index]
            material_name = f"num_{decoded_action['product_index']:02d}_{decoded_action['product']}"
            decoded_action["storage_name"] = env_state_action_dict["material"][material_name]["storage_name"]
        return decoded_action

    def decode_action_process_task_planning(self, env_state_action_dict):
        action_process_task_planning = env_state_action_dict["action"]["process_task_planning"]
        decoded_action = {"task": "none", "task_index" : 0} 
        if action_process_task_planning.sum() == 0:
            return decoded_action
        else:
            _index = action_process_task_planning.nonzero()[0][0]
            _index = _index.item()
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
            _index = _index.item()
            key_name = list(env_state_action_dict["human"].keys())[_index]
            decoded_action["human"] = env_state_action_dict["human"][key_name]["key_variables"]["type_name"]
            decoded_action["human_index"] = _index
        if action_robot.sum() != 0:
            _index = action_robot.nonzero()[0][0]
            _index = _index.item()
            key_name = list(env_state_action_dict["robot"].keys())[_index]
            decoded_action["robot"] = env_state_action_dict["robot"][key_name]["key_variables"]["type_name"]
            decoded_action["robot_index"] = _index
        return decoded_action

    def step_new_generated_task_record(self, env_state_action_dict):
        ### add new task record
        new_task_record : dict | None = self.process_decoded_task_record_from_action(env_state_action_dict)
        if new_task_record is not None:
            #TODO: check if the product index is already in the ongoing task records
            # assert new_task_record["product_index"] not in env_state_action_dict["progress"]["ongoing_task_records"], "The product index should not be in the ongoing task records"
            if new_task_record["product_index"] not in env_state_action_dict["progress"]["ongoing_task_records"]:
                env_state_action_dict["progress"]["ongoing_task_records"][new_task_record["product_index"]] = new_task_record
                self.apply_new_task_record_to_human_robot_machine(env_state_action_dict, new_task_record)
    
    def process_decoded_task_record_from_action(self, env_state_action_dict):
        task_record = {}
        if self.decode_action_to_task_record["product_selection"]["product"] == "none" or \
            self.decode_action_to_task_record["process_task_planning"]["task"] == "none":
            # means no valuable record, including task, human, or robot, is set up
            return None
        assert self.decode_action_to_task_record["human_robot_allocation"]["human"] != "none", "Human availablity should be check by mask before the task can be selected"
        #product
        task_record["product"] = self.decode_action_to_task_record["product_selection"]["product"]
        task_record["product_index"] = self.decode_action_to_task_record["product_selection"]["product_index"]
        task_record["new_product_selected"] = self.decode_action_to_task_record["product_selection"]["new_product_selected"]
        
        #human and robot
        task_record["human"] = self.decode_action_to_task_record["human_robot_allocation"]["human"]
        task_record["human_index"] = self.decode_action_to_task_record["human_robot_allocation"]["human_index"]
        task_record["robot"] = self.decode_action_to_task_record["human_robot_allocation"]["robot"]
        task_record["robot_index"] = self.decode_action_to_task_record["human_robot_allocation"]["robot_index"]
        #task
        task_record["task_done"] = False #default is False, will be set to True when the task is done
        task_record["task"] = self.decode_action_to_task_record["process_task_planning"]["task"]
        task_record["task_index"] = self.decode_action_to_task_record["process_task_planning"]["task_index"]

        if task_record["logistic_machine"] != "none":
            task_record["for_logistic"] = True
        else:
            task_record["for_logistic"] = False
        ##storage
        product_name = f"num_{task_record['product_index']:02d}_{task_record['product']}"
        storage_name = env_state_action_dict["material"][product_name]["storage_name"]
        task_record["storage_name"] = storage_name
        ##machine
        task_record["target_machine"] = CfgProcessTaskToMachineMapping[task_record["task"]]["target_machine"]
        states = env_state_action_dict["machine"][task_record["target_machine"]]["state"]
        first_free_workstation_index = states.index('free')
        task_record["target_machine_workstation_key"] = \
            list(env_state_action_dict["machine"][task_record["target_machine"]]["key_variables"]["working_area_ids"].keys())[first_free_workstation_index]
        task_record["logistic_machine"] = CfgProcessTaskToMachineMapping[task_record["task"]]["logistic_machine"]
        
        task_record["subtasks"] = self.initialze_subtasks(env_state_action_dict, task_record)
        return task_record

    def initialze_subtasks(self, env_state_action_dict, task_record):
        if task_record["for_logistic"]:
            if task_record["robot"] != "none":
                subtasks = CfgSubtaskGallery["logistic"]["have_AGV"]
            else:
                subtasks = CfgSubtaskGallery["logistic"]["only_have_gantry"]
        else:
            subtasks = CfgSubtaskGallery["processing"]
        return subtasks

    def apply_new_task_record_to_human_robot_machine(self, env_state_action_dict, task_record):
        #apply the new task record to the human, robot, and machine
        #human
        human_type = task_record["human"]
        human_idx = task_record["human_index"]
        human_key = f"num_{human_idx:02d}_{human_type}"
        assert env_state_action_dict["human"][human_key]["ongoing_task_record"] == {}, "The ongoing task record should be empty"
        env_state_action_dict["human"][human_key]["ongoing_task_record"] = task_record
        #robot
        robot_type = task_record["robot"]
        robot_idx = task_record["robot_index"]
        robot_key = f"num_{robot_idx:02d}_{robot_type}"
        assert env_state_action_dict["robot"][robot_key]["ongoing_task_record"] == {}, "The ongoing task record should be empty"
        env_state_action_dict["robot"][robot_key]["ongoing_task_record"] = task_record
        #machine
        machine_type = task_record["target_machine"]
        assert env_state_action_dict["machine"][machine_type]["ongoing_task_record"] == {}, "The ongoing task record should be empty"
        env_state_action_dict["machine"][machine_type]["ongoing_task_record"] = task_record

    def step_task_records(self, env_state_action_dict):

        ongoing_task_records : dict = env_state_action_dict["progress"]["ongoing_task_records"]
        for product_index, task_record in ongoing_task_records.items():
            product_type = task_record["product"]
            if task_record["task_done"]:
                #delete the task_record from ongoing_task_records
                del ongoing_task_records[product_index]
                # Remove only one occurrence of product_type from the producing list, even if there are duplicates
                producing_list = env_state_action_dict["progress"]["producing"]
                producing_indexs = env_state_action_dict["progress"]["producing_indexs"]
                assert len(producing_list) == len(producing_indexs), "The length of producing list and producing indexs should be the same"
                for i, prod in enumerate(producing_list):
                    if prod == product_type:
                        del producing_list[i]
                        del producing_indexs[i]
                        if product_type not in env_state_action_dict["progress"]["finished"]:
                            env_state_action_dict["progress"]["finished"][product_type] = 0
                        else:
                            env_state_action_dict["progress"]["finished"][product_type] += 1
                        break
            elif task_record["new_product_selected"] == True:
                task_record['new_product_selected'] = False
                env_state_action_dict["progress"]["producing"].append(product_type)
                env_state_action_dict["progress"]["producing_indexs"].append(product_index)
                env_state_action_dict["progress"]["next_product"] = None
                env_state_action_dict["progress"]["next_product_index"] = None
                env_state_action_dict["progress"]["not_started"][product_type] -= 1
        return env_state_action_dict