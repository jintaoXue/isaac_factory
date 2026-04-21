from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_storage import CfgStorage, CfgCommonState
import json


class StorageManager:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.cfg_storage = CfgStorage
        self.storage_list = []
        self._set_up_storage_list()

    def reset(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["storage_state"] = self
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["storage_state"] = self
        return env_state_action_dict

    def _set_up_storage_list(self):
        for class_name, cfg in self.cfg_storage.items():
            num_storage = cfg["num_storage"]
            cls = globals()[class_name]
            for idx in range(num_storage):
                storage_cfg = cfg["storage_cfg_dict"][f"{class_name}_{idx:02d}"]
                self.storage_list.append(cls(idx, storage_cfg, self.env_id))

class Storage:
    def __init__(self, idx: int, cfg: dict, env_id: int):
        self.idx = idx
        self.cfg = cfg
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.class_name = cfg["class_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.capacity = cfg["capacity"]
        self.supporting_materials = cfg["supporting_materials"]
        self.human_working_areas_ids = cfg["human_working_areas_ids"]
        self.robot_parking_areas_ids = cfg["robot_parking_areas_ids"]
        self.gantry_parking_areas_ids = cfg["gantry_parking_areas_ids"]
        self.env_id = env_id
        self.state_gallery = CfgCommonState["state_gallery"]
        self.reset_state = CfgCommonState["reset_state"]
        self.state : dict = None
        self.prim: RigidPrim = None
        self._set_up_rigid_prim()

    def _set_up_rigid_prim(self):
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["prim_paths_expr"].format(i=self.env_id),
            name=f"env_{self.env_id}_{meta['name']}",
            reset_xform_properties=False,
        )
    def reset(self, env_state_action_dict: dict) -> dict:
        self.state : dict = self.reset_state.copy()
        env_state_action_dict["state_storage"][f"{self.class_name}_{self.idx:02d}"] = self.state
        return env_state_action_dict
    
    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict

class BlackStorage(Storage):
    def __init__(self, idx: int, cfg: dict, env_id: int):
        super().__init__(idx, cfg, env_id)

class YellowStorage(Storage):
    def __init__(self, idx: int, cfg: dict, env_id: int):
        super().__init__(idx, cfg, env_id)

class GroundStorage(Storage):
    def __init__(self, idx: int, cfg: dict, env_id: int):
        super().__init__(idx, cfg, env_id)