from isaacsim.core.prims import RigidPrim
import omni.usd
from pxr import Usd, UsdSkel, Gf, Sdf
from ..env_asset_cfg.cfg_human import CfgHuman, CfgHumanRegistrationInfos
from ..env_asset_cfg.cfg_route.cfg_route import RouteOptionalInitPointsInMap, OptionalInitPointIds
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll, CfgSubtaskPredefinedTimeGallery, CfgProcessTaskToMachineMapping
import torch
import copy
import random

class HumanManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_human = CfgHuman
        self.cfg_registration_infos = CfgHumanRegistrationInfos
        self.optional_init_points_in_map = RouteOptionalInitPointsInMap["human_xyz"]
        self.optional_init_points_ids = OptionalInitPointIds["human"]
        self.human_list: list[Human] = []
        self._register_human_list()
        self.upper_bound_num_human = self.cfg_human["NumUpperBound"]

    def reset(self, env_state_action_dict: dict) -> dict:
        num_humans = len(self.human_list)
        num_points = int(self.optional_init_points_in_map.shape[0])
        if num_points < num_humans:
            raise ValueError(
                f"Not enough init points for humans: points={num_points}, humans={num_humans}."
            )
        perm = torch.randperm(num_points, device=self.optional_init_points_in_map.device)
        shuffled_init_points_in_map = self.optional_init_points_in_map[perm]
        shuffled_init_points_ids = self.optional_init_points_ids[perm]
        for human, i in zip(self.human_list, range(num_humans)):
            human.reset(env_state_action_dict, shuffled_init_points_in_map[i].unsqueeze(0), shuffled_init_points_ids[i])
        
        self.update_task_availability_mask(env_state_action_dict)
        self.update_self_availability_mask(env_state_action_dict)

        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for human in self.human_list:
            human.step(env_state_action_dict)
        self.update_task_availability_mask(env_state_action_dict)
        return env_state_action_dict
    
    def update_task_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for human availability for selection by human-robot machine allocator agent
        mask = torch.zeros(len(CfgProcessTaskGalleryInAll), dtype=torch.int32, device=self.cuda_device)
        mask[0] = 1 # "none" task is always available
        for human, i in zip(self.human_list, range(len(self.human_list))):
            if human.state['state'] == "free":
                mask[:] = 1
                break
        env_state_action_dict["human"]["task_availability_mask"] = mask
        return env_state_action_dict

    def update_self_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for human availability
        # shape (upper_bound_num_human,) 
        mask = torch.zeros(self.upper_bound_num_human, dtype=torch.int32, device=self.cuda_device)
        for human, i in zip(self.human_list, range(len(self.human_list))):
            if human.state['state'] == "free":
                mask[i] = 1
        env_state_action_dict["human"]["self_availability_mask"] = mask
        return env_state_action_dict

    def _register_human_list(self):
        for type_name, n in self.cfg_registration_infos.items():
            cls = globals()[type_name]
            for idx in range(n):
                self.human_list.append(cls(idx, self.cfg_human[type_name], self.env_id, self.cuda_device))

class Human:
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        # static variables
        self.idx = idx
        self.cfg = copy.deepcopy(cfg)
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id
        self.reset_state : str = copy.deepcopy(cfg["reset_state"])
        self.reset_state["key_variables"] = self.iter_key_variables()
        self.skeleton: UsdSkel.Skeleton | None = None
        self.prim: RigidPrim | None = None
        self.cuda_device = cuda_device
        self._register_skeleton()
        self._register_rigid_prim()
        ### dynmaic variables
        self.state : str = None

    def _register_skeleton(self):
        meta = self.meta_registeration_info
        stage = omni.usd.get_context().get_stage()
        prim_path = meta["skeleton_prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}")
        skeleton_prim = stage.GetPrimAtPath(prim_path)
        self.skeleton = UsdSkel.Skeleton(skeleton_prim)
        # Example of reading joint translations
        joints = self.skeleton.GetJointsAttr().Get()
        return
    
    def _register_rigid_prim(self):
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["rigid_prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}"),
            name=f"env_{self.env_id}_{meta['name'].format(idx=f'{self.idx:02d}')}",
            reset_xform_properties=False,
        )
        return
    
    def iter_key_variables(self):
        return {
            "type_name": self.type_name,
            "idx": self.idx,
        }

    def reset(self, env_state_action_dict: dict, init_point_in_map: torch.tensor, init_point_id: int) -> dict:
        self.state : str = copy.deepcopy(self.reset_state)
        self.state["current_area_id"] = init_point_id
        env_state_action_dict["human"][f"num_{self.idx:02d}_{self.type_name}"] = self.state
        self.reset_to_random_map_point(env_state_action_dict, init_point_in_map)
        return env_state_action_dict

    def reset_to_random_map_point(self, env_state_action_dict: dict, init_point_in_map: torch.tensor) -> dict:
        name = f"num_{self.idx:02d}_{self.type_name}"
        env_state_action_dict["rigid_prims"][name] = {
            "object": self.prim,
            "position": init_point_in_map,
            "orientation": torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.cuda_device).unsqueeze(0),
        }
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        task_record_index : int = env_state_action_dict["human"][f"num_{self.idx:02d}_{self.type_name}"]["ongoing_task_record_index"]
        if task_record_index is None:
            return
        task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
        assert task_record["human_index"] == self.idx, "The human index should be the same as the human index in the task record"

        if self.state["state"] == "free":
            #human is chosen to work on the task
            self.state["state"] = 'working_' + task_record["task"]
        
        if task_record["for_logistic"]:
            self.step_logistic(env_state_action_dict, task_record)
        else:
            self.step_processing(env_state_action_dict, task_record)
        return env_state_action_dict
    
    def step_logistic(self, env_state_action_dict: dict, task_record: dict) -> dict:
        subtasks = task_record["subtasks"]
        subtask = subtasks["ongoing"]
        human_subtask = subtask[0]
        #type 1, only have human and gantry + Type
        #TODO: check change subtasks value can change task records value
        if task_record["robot"] == "none":
            if human_subtask == "go_to_storage":
                self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_type="storage")
            elif human_subtask == "material_on_gantry":
                self._time_counting_subtask(subtasks, human_subtask)
            elif human_subtask == "control_gantry_while_going_to_target_machine":
                self._subtask_control_gantry_while_going_to_target_machine(
                    env_state_action_dict, task_record, subtasks, human_subtask
                )
            elif human_subtask == "material_on_target_area":
                self._time_counting_subtask(subtasks, human_subtask)

            #type 2, extra subtasks have human, AGV, and gantry
            elif human_subtask == "control_gantry":
                self._time_counting_subtask(subtasks, human_subtask)
            elif human_subtask == "material_on_robot":
                self._time_counting_subtask(subtasks, human_subtask)
            elif human_subtask == "go_to_target_machine":
                self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_type="machine")
            else:
                raise ValueError(f"Invalid human subtask for logistic with robot: {human_subtask}")

    def step_processing(self, env_state_action_dict: dict, task_record: dict) -> dict:
        subtasks = task_record["subtasks"]
        subtask = subtasks["ongoing"]
        human_subtask = subtask[0]
        
        if human_subtask == "go_to_target_machine":
            self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_type="machine")
        elif human_subtask == "control_machine":
            self._time_counting_subtask(env_state_action_dict, task_record, subtasks, human_subtask)
        else:
            raise ValueError(f"Invalid human subtask for processing: {human_subtask}")

    def _subtask_go_to_target(self, env_state_action_dict: dict, task_record: dict, subtasks: dict, target_type: str) -> None:
        if self.state["target_area_id"] is None:
            if target_type == "machine":
                workstation_areas = env_state_action_dict["machine"][task_record["target_machine"]]["key_variables"]["working_area_ids"][task_record["target_machine_workstation_key"]]
                self.state["target_area_id"] = random.choice(workstation_areas["human_working_areas_ids"])
            elif target_type == "storage":
                storage_key_variables = env_state_action_dict["storage"][task_record["storage_name"]]["key_variables"]
                self.state["target_area_id"] = random.choice(storage_key_variables["human_working_areas_ids"])
            else:
                raise ValueError(f"Invalid target type: {target_type}")
        if self.state["target_area_id"] == self.state["current_area_id"]:
            subtasks["finished"][1] = True

    def _subtask_control_gantry_while_going_to_target_machine(
        self, env_state_action_dict: dict, task_record: dict, subtasks: dict, human_subtask: str
    ) -> None:
        if self.state["target_area_id"] is None:
            workstation_areas = env_state_action_dict["machine"][task_record["target_machine"]]["key_variables"]["working_area_ids"][task_record["target_machine_workstation_key"]]
            self.state["target_area_id"] = random.choice(workstation_areas["human_working_areas_ids"])
        if self.state["subtask_time_counter"] < CfgSubtaskPredefinedTimeGallery[human_subtask] and not subtasks["finished"][1]:
            self.state["subtask_time_counter"] += 1
        elif self.state["target_area_id"] == self.state["current_area_id"]:
            subtasks["finished"][1] = True
            self.state["subtask_time_counter"] = 0
            
    def _time_counting_subtask(self, subtasks: dict, human_subtask: str, finished_index: int = 0) -> None:
        if self.state["subtask_time_counter"] < CfgSubtaskPredefinedTimeGallery[human_subtask] and not subtasks["finished"][finished_index]:
            self.state["subtask_time_counter"] += 1
        else:
            subtasks["finished"][finished_index] = True
            self.state["subtask_time_counter"] = 0


class NormalHuman(Human):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)