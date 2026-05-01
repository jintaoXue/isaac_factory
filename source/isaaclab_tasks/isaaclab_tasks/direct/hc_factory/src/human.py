from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_human import CfgHuman, CfgHumanRegistrationInfos
import torch
import copy

class HumanManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_human = CfgHuman
        self.cfg_registration_infos = CfgHumanRegistrationInfos
        self.human_list: list[Human] = []
        self._register_human_list()

    def reset(self, env_state_action_dict: dict) -> dict:
        for human in self.human_list:
            human.reset(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for human in self.human_list:
            human.step(env_state_action_dict)
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
        self.state_gallery = cfg["state_gallery"]
        self.reset_state = copy.deepcopy(cfg["reset_state"])
        self.optional_init_point_ids_in_map_points_list = cfg["optional_init_point_ids_in_map_points_list"]
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
        env_state_action_dict["human"][f"num_{self.idx:02d}_{self.type_name}"] = self.state

        return env_state_action_dict

    def reset_to_random_point(self, env_state_action_dict: dict) -> dict:
        random_point_id = random.choice(self.optional_init_point_ids_in_map_points_list)
        self.prim.set_local_poses(translations=random_point_id, orientations=None)
        return env_state_action_dict
        
    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict

class NormalHuman(Human):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)