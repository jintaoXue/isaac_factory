from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_robot import CfgRobot, CfgRobotRegistrationInfos
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
        self.idx = idx
        self.cfg = cfg
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id
        self.state_gallery = cfg["state_gallery"]
        self.reset_state = cfg["reset_state"]
        self.state : dict = None
        self.prim: RigidPrim | None = None
        self.cuda_device = cuda_device
        self._register_rigid_prim()

    def _register_rigid_prim(self):
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}"),
            name=f"env_{self.env_id}_{meta['name'].format(idx=f'{self.idx:02d}')}",
            reset_xform_properties=False,
        ) 

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state : dict = self.reset_state.copy()
        env_state_action_dict["state_robot"][f"{self.type_name}_{self.idx:02d}"] = self.state
        return env_state_action_dict
    
    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict


class AGV(Robot):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)