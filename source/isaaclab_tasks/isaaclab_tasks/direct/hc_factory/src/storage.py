from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_storage import CfgStorage, CfgResetStateTemplate, CfgStateGallery,_quat_multiply, _quat_conjugate
import json
import torch
from abc import abstractmethod

class StorageManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_storage = CfgStorage
        self.storage_list: list[Storage] = []
        self._register_storage_list()

    def reset(self, env_state_action_dict: dict) -> dict:
        for storage in self.storage_list:
            storage.reset(env_state_action_dict)
        return env_state_action_dict

    def _register_storage_list(self):
        for class_name, cfg in self.cfg_storage.items():
            num_storage = cfg["num_storage"]
            cls = globals()[class_name]
            for idx in range(num_storage):
                storage_cfg = cfg["storage_cfg_dict"][f"{class_name}_{idx:02d}"]
                self.storage_list.append(cls(idx, storage_cfg, self.env_id, self.cuda_device))

class Storage:
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        # static variables
        self.env_id = env_id
        self.idx = idx
        self.cuda_device = cuda_device
        self.cfg = cfg.copy()
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.class_name = cfg["class_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.capacity = cfg["capacity"]
        self.supporting_materials = cfg["supporting_materials"]
        self.human_working_areas_ids = cfg["human_working_areas_ids"]
        self.robot_parking_areas_ids = cfg["robot_parking_areas_ids"]
        self.gantry_parking_areas_ids = cfg["gantry_parking_areas_ids"]
        self.state_gallery = CfgStateGallery
        self.reset_state = CfgResetStateTemplate
        self.placement_type = cfg["placement_type"]

        self.prim: RigidPrim = None
        self._register_rigid_prim()

        ### dynmaic variables
        self.state : dict = None        
        #We need to copy the original data to avoid modifying the original data
        self.placement_cfg = self.cfg["placement_cfg"].copy()
        self.initialize_placement_cfg()

    def _register_rigid_prim(self):
        if self.class_name == "GroundStorage":
            return
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["prim_paths_expr"].format(i=self.env_id),
            name=f"env_{self.env_id}_{meta['name']}",
            reset_xform_properties=False,
        )

    def iter_key_variables(self):
        return {
            "type_name": self.type_name, 
            "class_name": self.class_name,
            "capacity": self.capacity,
            "supporting_materials": self.supporting_materials,
            "human_working_areas_ids": self.human_working_areas_ids,
            "robot_parking_areas_ids": self.robot_parking_areas_ids,
            "gantry_parking_areas_ids": self.gantry_parking_areas_ids,
            "placement_type": self.placement_type,
        }

    def initialize_placement_cfg(self):
        #the placement cfg is the relative poses of the storage slots to the storage base
        # trans the relative poses to the absolute poses using the storage base pose
        if self.placement_cfg["data_type"] == "relative":
            storage_base_pose = self.prim.get_local_poses()
            storage_base_position = storage_base_pose[0].squeeze(0)
            storage_base_orientation = storage_base_pose[1].squeeze(0)
            for pose in self.placement_cfg["pose_list"]:
                pose["position"] = [
                    pose["position"][0] + storage_base_position[0],
                    pose["position"][1] + storage_base_position[1],
                    pose["position"][2] + storage_base_position[2],
                ]
                pose["orientation"] = [
                    *_quat_multiply(_quat_conjugate(storage_base_orientation), pose["orientation"])
                ]
        elif self.placement_cfg["data_type"] == "absolute":
            pass
        return

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state : dict = self.reset_state.copy()
        self.state["key_variables"] = self.iter_key_variables()
        env_state_action_dict["storage"][f"{self.class_name}_{self.idx:02d}"] = self.state
        return env_state_action_dict
    
    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict

class BlackStorage(Storage):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)

class YellowStorage(Storage):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)


class GroundStorage(Storage):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)
