from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_robot import CfgRobot, CfgRobotRegistrationInfos
from ..env_asset_cfg.cfg_route.cfg_route import RouteOptionalInitPointsInMap
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll
import copy
import torch


class RobotManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_robot = CfgRobot
        self.cfg_registration_infos = CfgRobotRegistrationInfos
        self.optional_init_points_in_map = RouteOptionalInitPointsInMap["robot_xyz"]
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
        for robot, i in zip(self.robot_list, range(num_robots)):
            robot.reset(env_state_action_dict, shuffled_init_points_in_map[i].unsqueeze(0))
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for robot in self.robot_list:
            robot.step(env_state_action_dict)
        return env_state_action_dict

    def _register_robot_list(self):
        for type_name, n in self.cfg_registration_infos.items():
            cls = globals()[type_name]
            for idx in range(n):
                self.robot_list.append(cls(idx, self.cfg_robot[type_name], self.env_id, self.cuda_device))

    def update_robot_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for robot availability for selection by human-robot machine allocator agent
        mask = torch.ones(len(CfgProcessTaskGalleryInAll), dtype=torch.int32, device=self.cuda_device)
        env_state_action_dict["robot"]["task_availability_mask"] = mask
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
        self.state_gallery = cfg["state_gallery"]
        self.reset_state : str = copy.deepcopy(cfg["reset_state"])
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

    def reset(self, env_state_action_dict: dict, init_point_in_map: torch.tensor) -> dict:
        self.state : str = copy.deepcopy(self.reset_state)
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
        return env_state_action_dict


class AGV(Robot):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)