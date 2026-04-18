from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_storage import CfgStorage
import json


class StorageManager:
    def __init__(self, env_id: int):
        self.cfg_storage = CfgStorage
        self.storage_list = []
        self._set_up_storage_list()

    def _set_up_storage_list(self):
        for type_name, n in self.cfg_storage.items():
            class_name = type_name.split("_")[0]
            cls = globals()[class_name]
            for idx in range(n):
                self.storage_list.append(cls(idx, self.cfg_storage[class_name], self.env_id))
            

class Storage:
    def __init__(self, idx: int, cfg: dict, env_id: int):
        self.idx = idx
        self.cfg = cfg
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.class_name = cfg["type_name"].split("_")[0]
        # self.meta_registeration_info = cfg["meta_registeration_info"]
        self.capacity = cfg["capacity"]
        self.supporting_materials = cfg["supporting_materials"]
        self.human_working_areas_ids = cfg["human_working_areas_ids"]
        self.robot_parking_areas_ids = cfg["robot_parking_areas_ids"]
        self.gantry_parking_areas_ids = cfg["gantry_parking_areas_ids"]
        self.env_id = env_id

class BlackStorage:
    def __init__(self, idx: int, cfg: dict, env_id: int):
        super().__init__(idx, cfg, env_id)

class YellowStorage:
    def __init__(self, idx: int, cfg: dict, env_id: int):
        super().__init__(idx, cfg, env_id)

class GroundStorage:
    def __init__(self, idx: int, cfg: dict, env_id: int):
        super().__init__(idx, cfg, env_id)