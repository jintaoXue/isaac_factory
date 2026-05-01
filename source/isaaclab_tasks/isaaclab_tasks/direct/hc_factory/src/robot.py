from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_robot import CfgRobot, CfgRobotRegistrationInfos
from ..env_asset_cfg.cfg_route.cfg_route import RouteOptionalInitPointsInMap
import copy
import torch


class RobotManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_robot = CfgRobot
        self.cfg_registration_infos = CfgRobotRegistrationInfos
        self.robot_list: list[Robot] = []
        self._register_robot_list()

    def reset(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["state_robot"] = self
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["state_robot"] = self
        return env_state_action_dict

    def _register_robot_list(self):
        for type_name, n in self.cfg_registration_infos.items():
            cls = globals()[type_name]
            for idx in range(n):
                self.robot_list.append(cls(idx, self.cfg_robot[type_name], self.env_id, self.cuda_device))


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
        self.reset_state = copy.deepcopy(cfg["reset_state"])
        self.optional_init_points_in_map = RouteOptionalInitPointsInMap["robot_xyz"]
        self.prim: RigidPrim | None = None
        self.cuda_device = cuda_device
        self._register_rigid_prim()
        ### dynmaic variables
        self.state : dict = None

    def _register_rigid_prim(self):
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}"),
            name=f"env_{self.env_id}_{meta['name'].format(idx=f'{self.idx:02d}')}",
            reset_xform_properties=False,
        ) 

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state : dict = copy.deepcopy(self.reset_state)
        env_state_action_dict["robot"][f"num_{self.idx:02d}_{self.type_name}"] = self.state
        self.reset_to_random_map_point(env_state_action_dict)
        return env_state_action_dict
    
    def reset_to_random_map_point(self, env_state_action_dict: dict) -> dict:
        random_point_idx = torch.randint(0, self.optional_init_points_in_map.shape[0], (1,))
        name = f"num_{self.idx:02d}_{self.type_name}"
        env_state_action_dict["rigid_prims"][name] = {
            "object": self.prim,
            "position": self.optional_init_points_in_map[random_point_idx],
            "orientation": torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.cuda_device),
        }
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict


class AGV(Robot):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)