from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_robot import CfgRobot, CfgRobotRegistrationInfos
from ..env_asset_cfg.route.cfg_route import RouteOptionalInitPointsInMap, OptionalInitPointIds
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll

import copy
import torch
import random

class RobotManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_robot = CfgRobot
        self.cfg_registration_infos = CfgRobotRegistrationInfos
        self.optional_init_points_in_map = RouteOptionalInitPointsInMap["robot_xyz"]
        self.optional_init_points_ids = OptionalInitPointIds["robot"]
        self.robot_list: list[Robot] = []
        self._register_robot_list()
        self.upper_bound_num_robot = self.cfg_robot["NumUpperBound"]

    def reset(self, env_state_action_dict: dict) -> dict:
        num_robots = len(self.robot_list)
        num_points = int(self.optional_init_points_in_map.shape[0])
        if num_points < num_robots:
            raise ValueError(
                f"Not enough init points for robots: points={num_points}, robots={num_robots}."
            )
        perm = torch.randperm(num_points, device=self.optional_init_points_in_map.device)
        shuffled_init_points_in_map = self.optional_init_points_in_map[perm]
        # self.optional_init_points_ids is a list (Python, not a torch tensor) so use list comprehension, not tensor indexing
        shuffled_init_points_ids: list[int] = [self.optional_init_points_ids[i] for i in perm.tolist()]
        for robot, i in zip(self.robot_list, range(num_robots)):
            robot.reset(env_state_action_dict, shuffled_init_points_in_map[i].unsqueeze(0), shuffled_init_points_ids[i])
        
        # self.update_task_availability_mask(env_state_action_dict)
        # self.update_self_availability_mask(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for robot in self.robot_list:
            robot.step(env_state_action_dict)
        # self.update_task_availability_mask(env_state_action_dict)
        # self.update_self_availability_mask(env_state_action_dict)
        return env_state_action_dict

    def _register_robot_list(self):
        for type_name, n in self.cfg_registration_infos.items():
            cls = globals()[type_name]
            for idx in range(n):
                self.robot_list.append(cls(idx, self.cfg_robot[type_name], self.env_id, self.cuda_device))

    def update_task_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for robot availability for selection by human-robot machine allocator agent
        mask = torch.zeros(len(CfgProcessTaskGalleryInAll), dtype=torch.int32, device=self.cuda_device)
        mask[0] = 1 # "none" task is always available
        for robot, i in zip(self.robot_list, range(len(self.robot_list))):
            if robot.state['state'] == "free":
                mask[:] = 1
                break
        env_state_action_dict["agent_action_mask"]["robot"]["task_availability_mask"] = mask
        return env_state_action_dict

    def update_self_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for robot availability
        # shape (upper_bound_num_robot,) 
        mask = torch.zeros(self.upper_bound_num_robot, dtype=torch.int32, device=self.cuda_device)
        for robot, i in zip(self.robot_list, range(len(self.robot_list))):
            if robot.state['state'] == "free":
                mask[i] = 1
        env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"] = mask
        return env_state_action_dict


class Robot:
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
        self.prim: RigidPrim | None = None
        self.cuda_device = cuda_device
        self._register_rigid_prim()
        ### dynmaic variables
        self.state : str = None

    def _register_rigid_prim(self):
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}"),
            name=f"env_{self.env_id}_{meta['name'].format(idx=f'{self.idx:02d}')}",
            reset_xform_properties=False,
        ) 

    def iter_key_variables(self):
        return {
            "type_name": self.type_name,
            "idx": self.idx,
        }

    def reset(self, env_state_action_dict: dict, init_point_in_map: torch.tensor, init_point_id: int) -> dict:
        self.state : str = copy.deepcopy(self.reset_state)
        self.state["current_area_id"] = init_point_id
        env_state_action_dict["robot"][f"num_{self.idx:02d}_{self.type_name}"] = self.state
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
        task_record_index : int = env_state_action_dict["robot"][f"num_{self.idx:02d}_{self.type_name}"]["ongoing_task_record_index"]
        if task_record_index is None:
            return
        task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
        assert task_record["robot_index"] == self.idx, "The robot index should be the same as the robot index in the task record"
    
        assert self.state["state"] != "free", "The robot should be working on the task, and be defined in task_progress_manager.py"
        
        #now robot is only for logistic
        self.step_logistic(env_state_action_dict, task_record)
        return env_state_action_dict

    def step_logistic(self, env_state_action_dict: dict, task_record: dict) -> dict:
        subtasks = task_record["subtasks_dict"]
        subtask = subtasks["ongoing"]
        robot_subtask = subtask[3]
        if robot_subtask == "go_to_material":
            self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_area_type = "start")
        elif robot_subtask == "wait":
            subtasks["finished"][3] = True
        elif robot_subtask == "carry_to_goal_area":
            self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_area_type="goal")
        elif robot_subtask == "done":
            self._task_done(env_state_action_dict, task_record, subtasks)
        else:
            raise ValueError(f"Invalid robot subtask for logistic: {robot_subtask}")
        return env_state_action_dict
            
    def _subtask_go_to_target(self, env_state_action_dict: dict, task_record: dict, subtasks: dict, target_area_type: str) -> None:
        if subtasks["finished"][3] == True:
            return
        
        if self.state["target_area_id"] is None:
            if target_area_type == "start":
                assert task_record["subtasks_dict"]["start_area_ids"] is not None, "The start area ids should be initialized in task_progress_manager.py"
                target_area_id = task_record["subtasks_dict"]["start_area_ids"]["robot_parking_areas_ids"][0]
            elif target_area_type == "goal":
                assert task_record["subtasks_dict"]["goal_area_ids"] is not None, "The goal area ids should be initialized in task_progress_manager.py"
                target_area_id = task_record["subtasks_dict"]["goal_area_ids"]["robot_parking_areas_ids"][0]
            else:
                raise ValueError(f"Invalid target area type: {target_area_type}")
            self.state["target_area_id"] = target_area_id
        elif self.state["target_area_id"] == self.state["current_area_id"]:
            subtasks["finished"][3] = True
            self.state["target_area_id"] = None
            self.state["route_index"] = 0
            self.state["route_length"] = 0
            self.state["generated_route"] = []
        else:
            # the human is going to the target area by route planner in route.py
            pass
        return env_state_action_dict

    def _task_done(self, env_state_action_dict: dict, task_record: dict, subtasks: dict) -> None:
        subtasks["finished"][3] = True
        ### reset the robot in advance
        current_area_id = self.state["current_area_id"]
        self.state : str = copy.deepcopy(self.reset_state)
        self.state["current_area_id"] = current_area_id
        env_state_action_dict["robot"][f"num_{self.idx:02d}_{self.type_name}"] = self.state
        return env_state_action_dict

class AGV(Robot):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)