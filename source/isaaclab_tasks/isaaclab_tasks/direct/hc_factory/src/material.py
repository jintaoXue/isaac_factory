from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_material_product import CfgProductProcess, CfgRegistrationInfos


class ProductMaterialManager:
    def __init__(self, cfg_product_process: CfgProductProcess, cfg_registration_infos: CfgRegistrationInfos, env_id: int):
        self.env_id = env_id
        self.cfg_product_process = cfg_product_process
        self.cfg_registration_infos = cfg_registration_infos
        self.product_list = []
        self._set_up_product_list()

    def _set_up_product_list(self):
        for product_type_name, num_product in self.cfg_registration_infos.items():
            for idx in range(num_product):
                product_class = globals()[product_type_name]
                product = product_class(idx, self.cfg_product_process[product_type_name], self.env_id)
                self.product_list.append(product)
                
class Material:
    def __init__(self, idx_in_product_list: int, cfg: dict, env_id: int):
        self.idx_in_product_list = idx_in_product_list
        self.cfg = cfg
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id
        
        self.product_00_pipe : RigidPrim = None
        self.product_00_flange : RigidPrim = None
        self.product_00_elbow : RigidPrim = None
        self.product_00_semi : RigidPrim = None
        self.product_00_maded : RigidPrim = None
        self._set_up_rigid_prim()

    def _set_up_rigid_prim(self):
        for obj_name, info in self.meta_registeration_info.items():
            rigid_prim = RigidPrim(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id, idx=self.idx_in_product_list),
                name=info["name"].format(idx=self.idx_in_product_list),
                reset_xform_properties=False,
            )
            setattr(self, obj_name, rigid_prim)
