from isaacsim.core.prims import RigidPrim
from abc import abstractmethod
from ..env_asset_cfg.cfg_material_product import CfgProductProcess, CfgRegistrationInfos, CfgResetStateTemplate
import torch


class ProductMaterialManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_product_process = CfgProductProcess
        self.cfg_registration_infos = CfgRegistrationInfos
        self.material_batch_list: list[MaterialBatch] = []
        self._register_material_batch_list()

    def _register_material_batch_list(self):
        for material_batch_type_name, num_material_batch in self.cfg_registration_infos.items():
            for idx in range(num_material_batch):
                material_batch_class = globals()[material_batch_type_name]
                material_batch = material_batch_class(idx, self.cfg_product_process[material_batch_type_name], self.env_id, self.cuda_device)
                self.material_batch_list.append(material_batch)
            
    def reset(self, env_state_action_dict: dict) -> dict:
        for material_batch in self.material_batch_list:
            material_batch.reset(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for material_batch in self.material_batch_list:
            material_batch.step(env_state_action_dict)
        return env_state_action_dict

class MaterialBatch:
    def __init__(self, idx_in_material_batch_list: int, cfg: dict, env_id: int, cuda_device: torch.device):
        # static variables
        self.idx = idx_in_material_batch_list
        self.cuda_device = cuda_device
        self.cfg = cfg.copy()
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id
        self.reset_state = cfg["reset_state_template"]
        self.reset_state["key_variables"] = self.iter_key_variables()
        self._register_rigid_prim()
        ### dynmaic variables
        self.state : dict = None

    def _register_rigid_prim(self):
        for obj_name, info in self.meta_registeration_info.items():
            rigid_prim = RigidPrim(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx_in_material_batch_list:02d}"),
                name=f"env_{self.env_id}_{info['name'].format(idx=f'{self.idx_in_material_batch_list:02d}')}",
                reset_xform_properties=False,
            )
            setattr(self, obj_name, rigid_prim)

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state : dict = self.reset_state.copy()
        env_state_action_dict["material"][f"{self.type_name}_{self.idx_in_material_batch_list:02d}"] = self.state
        self.reset_material_to_storage(env_state_action_dict)
        return env_state_action_dict

    def reset_material_to_storage(self, env_state_action_dict: dict) -> dict:
        material_prims : dict = self.iter_raw_materials_prim()
        storages : dict = env_state_action_dict["storage"]
        for material_type, material_prim in material_prims.items():
            for storage_name, storage_state in storages.items():
                supporting_materials = storage_state["key_variables"]["supporting_materials"]
                if storage_state["state"] == "empty" and material_type in supporting_materials:
                    storage_state["state"] = "partial"
                    storage_state["material_type"] = material_type
                    storage_state["material_idx"].append(self.idx)
                    storage_state["num_material"] = 1
                elif storage_state["state"] == "partial" and material_type == storage_state["material_type"]:
                    storage_state["num_material"] += 1
                    storage_state["material_idx"].append(self.idx)
                    if storage_state["num_material"] == storage_state["key_variables"]["capacity"]:
                        storage_state["state"] = "full"
                elif storage_state["state"] == "full":
                    continue
        rigid_prims_values: dict = {}
        for obj_name in self.meta_registeration_info.keys():
            rigid_prim = getattr(self, obj_name, None)
            if rigid_prim is None:
                continue
            rigid_prims_values[obj_name] = {
                "rigid_prim": rigid_prim,
                "positions": rigid_prim.get_world_poses()[0],
                "orientations": rigid_prim.get_world_poses()[1],
            }
        return rigid_prims_values

    def iter_key_variables(self):
        return {
            "type_name": self.type_name,
            "type_id": self.type_id,
        }

    @abstractmethod
    def iter_raw_materials_prim(self): 
        pass

    @abstractmethod
    def iter_integrated_material_prims(self):
        pass
    
    @abstractmethod
    def step(self, env_state_action_dict: dict) -> dict:
        pass


class ProductWaterPipe(MaterialBatch):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        self.product_00_pipe : RigidPrim = None
        self.product_00_flange : RigidPrim = None
        self.product_00_elbow : RigidPrim = None
        self.product_00_semi : RigidPrim = None
        self.product_00_maded : RigidPrim = None
        super().__init__(idx, cfg, env_id, cuda_device)

    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict

    def iter_raw_materials_prim(self):
        return {
            "product_00_pipe": self.product_00_pipe,
            "product_00_flange": self.product_00_flange,
            "product_00_elbow": self.product_00_elbow,
        }

    def iter_integrated_material_prims(self):
        return {
            "product_00_semi": self.product_00_semi,
            "product_00_maded": self.product_00_maded,
        }