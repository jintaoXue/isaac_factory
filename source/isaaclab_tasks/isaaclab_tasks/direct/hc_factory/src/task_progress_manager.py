
from ..env_asset_cfg.cfg_storage import CfgStorage
from ..env_asset_cfg.cfg_material_product import CfgRegistrationInfos, CfgProductOrder, CfgProductProcess
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll, CfgProcessTaskGalleryDetailedClassified, CfgSubtaskGallery, TaskRecordTemplate
import torch
import copy

class TaskManager:
    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device
        self.cfg_storage = CfgStorage
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
        new_task_record = copy.deepcopy(TaskRecordTemplate)
        self.decode_action_product_sequencing(env_state_action_dict)
        self.decode_action_product_selection(env_state_action_dict, new_task_record)
        self.decode_action_process_task_planning(env_state_action_dict, new_task_record)
        self.decode_action_human_robot_allocation(env_state_action_dict, new_task_record)
        self.update_new_task_record(env_state_action_dict, new_task_record)
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
                finished_task = material_state["finished_task"]
                if key_variables["type_name"] == product_type and finished_task == "none":
                    env_state_action_dict["progress"]["next_product"] = product_type
                    env_state_action_dict["progress"]["next_product_index"] = key_variables["idx"]
                    break

    def decode_action_product_selection(self, env_state_action_dict, new_task_record):
        # action shape is (1 + self.parallel_producing_limit,)
        action_product_selection = env_state_action_dict["action"]["product_selection"]
        
        if action_product_selection.sum() == 0:
            return
        else:
            _index = action_product_selection.nonzero()[0][0]
            _index = _index.item()
            if _index == action_product_selection.shape[0] - 1:
                new_task_record["product"] = env_state_action_dict["progress"]["next_product"]
                new_task_record["product_index"] = env_state_action_dict["progress"]["next_product_index"]
                new_task_record["new_product_selected"] = True
            else:
                new_task_record["product"] = env_state_action_dict["progress"]["producing"][_index]
                new_task_record["product_index"] = env_state_action_dict["progress"]["producing_indexs"][_index]
            material_name = f"num_{new_task_record['product_index']:02d}_{new_task_record['product']}"
            new_task_record["storage_name"] = env_state_action_dict["material"][material_name]["storage_name"]
        return new_task_record

    def decode_action_process_task_planning(self, env_state_action_dict, new_task_record):
        action_process_task_planning = env_state_action_dict["action"]["process_task_planning"]
        if action_process_task_planning.sum() == 0:
            action_process_task_planning[0] = 1
        else:
            _index = action_process_task_planning.nonzero()[0][0]
            _index = _index.item()
            new_task_record["task"] = self.inverse_index_to_task_name[_index]
            new_task_record["task_index"] = _index
            new_task_record["task_type"] = CfgProcessTaskGalleryDetailedClassified[new_task_record["product"]][new_task_record["task"]]["task_type"]
        return new_task_record
    
    def decode_action_human_robot_allocation(self, env_state_action_dict, new_task_record):
        action_human_robot_allocation = env_state_action_dict["action"]["human_robot_allocation"]
        #shape is (upper_bound_num_human,)
        action_human = action_human_robot_allocation["human"]
        #shape is (upper_bound_num_robot,)
        action_robot = action_human_robot_allocation["robot"]   
        if action_human.sum() != 0:
            _index = action_human.nonzero()[0][0]
            _index = _index.item()
            key_name = list(env_state_action_dict["human"].keys())[_index]
            new_task_record["human"] = env_state_action_dict["human"][key_name]["key_variables"]["type_name"]
            new_task_record["human_index"] = _index
        if action_robot.sum() != 0 and new_task_record["task_type"] == "logistic":
            _index = action_robot.nonzero()[0][0]
            _index = _index.item()
            key_name = list(env_state_action_dict["robot"].keys())[_index]
            new_task_record["robot"] = env_state_action_dict["robot"][key_name]["key_variables"]["type_name"]
            new_task_record["robot_index"] = _index
        return new_task_record

    def update_new_task_record(self, env_state_action_dict, new_task_record):
        if new_task_record["product"] is None or new_task_record["task"] == "none":
            return
            # means no valuable record, including task, human, or robot, is set up
        assert new_task_record["human"] != None, "Human availablity should be check by mask before the task can be selected"
        assert new_task_record["product_index"] not in env_state_action_dict["progress"]["ongoing_task_records"], "The product index should not be in the ongoing task records"
        product_type = new_task_record["product"]
        ##machine information
        new_task_record["target_machine"] = CfgProcessTaskGalleryDetailedClassified[product_type][new_task_record["task"]]["target_machine"]
        # assert new_task_record["target_machine"] != None, "Target machine should be set"
        states = env_state_action_dict["machine"][new_task_record["target_machine"]]["state"]
        workstation_index = None
        if new_task_record["task_type"] == "logistic":
            ## find the free workstation index for logistic task
            workstation_index = states.index('free')
        elif new_task_record["task_type"] == "processing":
            ## find the ready workstation index for logistic task
            for i, state in enumerate(states):
                pre_name = state.split('_')[0]
                task_name = state.split('_')[1]
                if pre_name == "materialReadyFor" and task_name == new_task_record["task"]:
                    workstation_index = i
                    break
        else:
            raise ValueError(f"Invalid task type: {new_task_record['task_type']}")
        new_task_record["chosen_machine_workstation"] = \
            list(env_state_action_dict["machine"][new_task_record["target_machine"]]["key_variables"]["working_area_ids"].keys())[workstation_index]
        new_task_record["chosen_workstation_index"] = workstation_index
        new_task_record["logistic_machine"] = CfgProcessTaskGalleryDetailedClassified[product_type]["logistic_machine"]
        if new_task_record["task_type"] == "logistic":
            chosen_free_gantry_index = self._find_free_gantry(env_state_action_dict, new_task_record)
            assert chosen_free_gantry_index is not None, "Free gantry index should be found"
            new_task_record["chosen_gantry_index"] = chosen_free_gantry_index
        #subtasks information
        new_task_record["subtasks_dict"] : dict = copy.deepcopy(self.initialze_subtasks(env_state_action_dict, new_task_record))
        
        env_state_action_dict["progress"]["ongoing_task_records"][new_task_record["product_index"]] = new_task_record
        self.apply_new_task_record_to_human_robot_machine_material(env_state_action_dict, new_task_record)

    def _find_free_gantry(self, env_state_action_dict, task_record):
        assert task_record["logistic_machine"] == "num07_gantry_group", "now only num07_gantry_group is valid logistic machine"
        gantry_states : list[str] = env_state_action_dict["machine"][task_record["logistic_machine"]]["state"]

        def index_of(value, in_list):
            return next((idx for idx, item in enumerate(in_list) if item == value), None)

        chosen_free_gantry_index = index_of("free", gantry_states)
        return chosen_free_gantry_index

    def initialze_subtasks(self, env_state_action_dict, task_record):
        if task_record["task_type"] == "logistic":
            if task_record["robot"] != None:
                subtasks = copy.deepcopy(CfgSubtaskGallery[task_record["product"]][task_record["task"]]["have_AGV"])
            else:
                subtasks = copy.deepcopy(CfgSubtaskGallery[task_record["product"]][task_record["task"]]["only_have_gantry"])
            #1. set the material start area for subtasks dict
            product_name = f"num_{task_record['product_index']:02d}_{task_record['product']}"
            required_logistic_material = subtasks["logistic_submaterial"]
            state_material = env_state_action_dict["material"][product_name]["submaterials"][required_logistic_material]
            assert state_material["storage_name"] is not None, "The storage name should be initialized in material.py"
            assert state_material["storage_name"] != "disappear", "The material still not appeared"
            assert subtasks["material_start_area"] != subtasks["material_goal_area"], "The material start area and goal area should be different, \
                otherwise dont need to logistic this material"
            subtasks["material_start_area"] = state_material["storage_name"]
            #2. update the working area ids by specifying the machine type and workstation key
            # (1) update the start area ids
            if subtasks["material_start_area"] in env_state_action_dict["machine"]:
                subtasks["start_area_ids"] = env_state_action_dict["machine"][subtasks["material_start_area"]]["key_variables"]["working_area_ids"]
            elif subtasks["material_start_area"] in env_state_action_dict["storage"]:
                subtasks["start_area_ids"] = env_state_action_dict["storage"][subtasks["material_start_area"]]["key_variables"]["working_area_ids"]
            else:
                raise ValueError(f"Invalid material start area: {subtasks['material_start_area']}")
            # (2) update the goal area ids
            machine_workstation_key = task_record["chosen_machine_workstation"]
            subtasks["goal_area_ids"] = subtasks["goal_area_ids"][machine_workstation_key]

        elif task_record["task_type"] == "processing":
            #1. set the goal area for subtasks dict
            subtasks = copy.deepcopy(CfgSubtaskGallery[task_record["product"]][task_record["task"]])
            # only update the start area ids, the goal area ids is updated during the processing task
            assert subtasks["material_start_area"] in env_state_action_dict["machine"]
            machine_workstation_key = task_record["chosen_machine_workstation"]
            subtasks["start_area_ids"] = subtasks["start_area_ids"][machine_workstation_key]
        else:
            raise ValueError(f"Invalid task type: {task_record['task_type']}")
        return subtasks

    def apply_new_task_record_to_human_robot_machine_material(self, env_state_action_dict, task_record):
        #apply the new task record to the human, robot, and machine

        #machine
        machine_type = task_record["target_machine"]
        if machine_type != None:
            assert env_state_action_dict["machine"][machine_type]["ongoing_task_record_index"] == None, "The ongoing task record should be empty"
            chosen_workstation_index = task_record["chosen_workstation_index"]
            env_state_action_dict["machine"][machine_type]["ongoing_task_record_index"][chosen_workstation_index] = task_record["product_index"]
        #human
        human_type = task_record["human"]
        if human_type != None:
            human_idx = task_record["human_index"]
            human_key = f"num_{human_idx:02d}_{human_type}"
            assert env_state_action_dict["human"][human_key]["ongoing_task_record_index"] == None, "The ongoing task record should be empty"
            env_state_action_dict["human"][human_key]["ongoing_task_record_index"] = task_record["product_index"]
        #robot
        robot_type = task_record["robot"]
        if robot_type != None:
            robot_idx = task_record["robot_index"]
            robot_key = f"num_{robot_idx:02d}_{robot_type}"
            assert env_state_action_dict["robot"][robot_key]["ongoing_task_record_index"] == None, "The ongoing task record should be empty"
            env_state_action_dict["robot"][robot_key]["ongoing_task_record_index"] = task_record["product_index"]

        logistic_machine_type = task_record["logistic_machine"]
        if logistic_machine_type != None:
            assert logistic_machine_type == "num07_gantry_group", "now only num07_gantry_group is valid logistic machine"
            env_state_action_dict["machine"][logistic_machine_type]["ongoing_task_record_index"] = task_record["product_index"]
        #material
        material_type = task_record["product"]
        assert material_type != None, "The material type should not be none"
        material_name = f"num_{task_record['product_index']:02d}_{material_type}"
        assert env_state_action_dict["material"][material_name]["ongoing_task_record_index"] == None, "The ongoing task record should be empty"
        env_state_action_dict["material"][material_name]["ongoing_task_record_index"] = task_record["product_index"]        

    def step_task_records(self, env_state_action_dict):

        ongoing_task_records : dict = env_state_action_dict["progress"]["ongoing_task_records"]
        for product_index, task_record in ongoing_task_records.items():
            product_type = task_record["product"]
            if self._check_subtask_and_task_done(env_state_action_dict, task_record):
                #delete the task_record from ongoing_task_records
                if self._is_the_last_one_task_done(task_record):
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
                del ongoing_task_records[product_index]
            elif task_record["new_product_selected"] == True:
                task_record['new_product_selected'] = False
                env_state_action_dict["progress"]["producing"].append(product_type)
                env_state_action_dict["progress"]["producing_indexs"].append(product_index)
                env_state_action_dict["progress"]["next_product"] = None
                env_state_action_dict["progress"]["next_product_index"] = None
                env_state_action_dict["progress"]["not_started"][product_type] -= 1

        return env_state_action_dict

    def _update_task_record_when_doing_subtask(self, env_state_action_dict, task_record):
        if not task_record["for_logistic"]:
            ## processing task, 1 is gantry, if gantry is none, means gantry is not needed
            if task_record["subtasks_dict"]["ongoing"][1] == "none":
                return
            elif task_record["subtasks_dict"]["ongoing"][1] == "finding_free_gantry" and task_record["subtasks_dict"]["finished"][1] == False:
                if task_record["chosen_free_gantry_index"] is None:
                    task_record = self._find_free_gantry(env_state_action_dict, task_record)
                else:
                    #finded the free gantry
                    task_record["subtasks_dict"]["finished"][1] = True
        return task_record

    def _check_target_area_update(self, env_state_action_dict, task_record):
        if task_record["subtasks_dict"]["index_to_decide_target_area_type"] is None:
            return task_record
        if task_record["subtasks_dict"]["target_area_type"] is not None:
            return task_record
        assert task_record["task_type"] == "processing", "now only processing task can decide the target area"
        #### decide the target area that the processed material will be put on
        ### 1. check next subtask target
        task = task_record["task"]
        product_type = task_record["product"]
        if CfgProcessTaskGalleryDetailedClassified[product_type][task]["is_final_task"] == True:
            task_record["subtasks_dict"]["target_area_type"] = "storage"
        else:
            next_subtask_target = task_record["subtasks_dict"]["subtasks"][task_record["subtasks_dict"]["ongoing_index"]][task_record["subtasks_dict"]["index_to_decide_target_area_type"]]
            task_record["subtasks_dict"]["target_area_type"] = next_subtask_target
        return task_record

    def _check_subtask_and_task_done(self, env_state_action_dict, task_record):
        task_record = self._update_task_record_when_doing_subtask(env_state_action_dict, task_record)
        task_record = self._check_target_area_update(env_state_action_dict, task_record)
        finished : list[bool] =  task_record["subtasks_dict"]["finished"]
        if all(finished) == True:
            if task_record["subtasks_dict"]["ongoing_index"] == task_record["subtasks_dict"]["num_subtasks"]:
                ## all subtasks are done
                task_record["task_done"] = True
                return True
            else:
                task_record["subtasks_dict"]["ongoing_index"] += 1
                task_record["subtasks_dict"]["ongoing"] = task_record["subtasks_dict"]["subtasks"][task_record["subtasks_dict"]["ongoing_index"]]
                for bool_value in finished:
                    bool_value = False
                return False

    
    def _is_the_last_one_task_done(self, task_record: dict) -> bool:
        product_type = task_record["product"]
        task = task_record["task"]
        return bool(CfgProcessTaskGalleryDetailedClassified[product_type][task]["is_final_task"])

    def _find_free_storage(self, env_state_action_dict, storage_class: str):
        storages = env_state_action_dict["storage"]
        for storage_name, value in storages.items():
            if value["class_name"] == storage_class:
                # If this material isn't supported, or storage is already full, skip
                if material_type not in supporting_materials or value["state"] == "full":
                    continue
                # If storage is partially filled with a different material type, skip
                if value["state"] == "partial" and material_type != value["material_type"]:
                    continue
                if storage is free or not full and have the same material type:
                    return storage_name
                else:
                    continue
        error_message = f"No free storage found for {storage_class}"
        raise ValueError(error_message)