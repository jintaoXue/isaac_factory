from isaacsim.core.prims import RigidPrim
import omni.usd
from pxr import Usd, UsdSkel, Gf, Sdf
from ..env_asset_cfg.cfg_human import CfgHuman, CfgHumanRegistrationInfos
from ..env_asset_cfg.cfg_route.cfg_route import RouteOptionalInitPointsInMap
import torch
import copy

class HumanManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_human = CfgHuman
        self.cfg_registration_infos = CfgHumanRegistrationInfos
        self.optional_init_points_in_map = RouteOptionalInitPointsInMap["human_xyz"]
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
        for human, i in zip(self.human_list, range(num_humans)):
            human.reset(env_state_action_dict, shuffled_init_points_in_map[i].unsqueeze(0))
        
        self.update_human_availability_mask(env_state_action_dict)
        
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for human in self.human_list:
            human.step(env_state_action_dict)
        return env_state_action_dict
    
    def update_human_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for human availability for selection by human-robot machine allocator agent
        mask = torch.zeros(self.upper_bound_num_human, dtype=torch.int32, device=self.cuda_device)
        for human, i in zip(self.human_list, range(len(self.human_list))):
            if human.state['state'] == "free":
                mask[i] = 1
        env_state_action_dict["human"]["availability_mask"] = mask
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
        self.skeleton: UsdSkel.Skeleton | None = None
        self.prim: RigidPrim | None = None
        self.cuda_device = cuda_device
        self._register_skeleton()
        self._register_rigid_prim()
        ### dynmaic variables
        self.state : dict = None

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
        )
        return
    
    def reset(self, env_state_action_dict: dict, init_point_in_map: torch.tensor) -> dict:
        self.state : dict = copy.deepcopy(self.reset_state)
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
        return env_state_action_dict

class NormalHuman(Human):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)