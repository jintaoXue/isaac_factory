from isaacsim.core.prims import RigidPrim
from abc import abstractmethod
from ..env_asset_cfg.cfg_material_product import CfgProductProcess, CfgRegistrationInfos
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
        for material_batch in self.iter_material_batches():
            material_batch.step(env_state_action_dict)
        return env_state_action_dict

class MaterialBatch:
    def __init__(self, idx_in_material_batch_list: int, cfg: dict, env_id: int, cuda_device: torch.device):
        self.idx_in_material_batch_list = idx_in_material_batch_list
        self.cfg = cfg
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id
        self.reset_state = cfg["reset_state"]
        self.state : dict = None
        self.cuda_device = cuda_device
        self._register_rigid_prim()

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
        env_state_action_dict["state_material"][f"{self.type_name}_{self.idx_in_material_batch_list:02d}"] = self.state
        return env_state_action_dict

    def reset_rigid_prims(self) -> dict:
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
