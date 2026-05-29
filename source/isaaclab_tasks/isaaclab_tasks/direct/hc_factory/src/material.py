from isaacsim.core.prims import RigidPrim
from abc import abstractmethod
from ..env_asset_cfg.cfg_material_product import CfgProductProcess, CfgProductOrder, CfgRegistrationInfos
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll, CfgProcessTaskGalleryDetailedClassified
import copy
import torch


class ProductMaterialManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_product_process = CfgProductProcess
        self.cfg_product_order = CfgProductOrder
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
        self.update_task_availability_mask(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for material_batch in self.material_batch_list:
            material_batch.step(env_state_action_dict)
        self.update_task_availability_mask(env_state_action_dict)
        return env_state_action_dict
    
    def update_task_availability_mask(self, env_state_action_dict: dict) -> dict:
        # CfgProcessTaskGalleryInAll. All product process tasks share a common encoded index space defined by CfgProcessTaskGallery.
        # CfgProcessTaskGalleryDetailedClassified. This contains the task gallery for all product types; each product type has its own process gallery. 
        mask = torch.zeros(len(self.material_batch_list), len(CfgProcessTaskGalleryInAll), dtype=torch.int32, device=self.cuda_device)
        mask[:, 0] = 1 # "none" task is always available
        ongoing_task_records : dict = env_state_action_dict["progress"]["ongoing_task_records"]
        for material_batch, material_batch_index in zip(self.material_batch_list, range(len(self.material_batch_list))):
            if material_batch_index in ongoing_task_records:
                continue
            product_type = material_batch.type_name
            finished_task = material_batch.state["finished_task"]
            one_ProcessTaskGallery = CfgProcessTaskGalleryDetailedClassified[product_type]
            next_allowing_task_index = self.find_product_next_allowing_task_index(finished_task, one_ProcessTaskGallery)
            mask[material_batch_index][next_allowing_task_index] = 1
        env_state_action_dict["material"]["task_availability_mask"] = mask
        return env_state_action_dict
    
    def find_product_next_allowing_task_index(self, finished_task, one_ProcessTaskGallery):
        # Use the ordered task list for the product to determine the next allowable task.
        # If finished_task is the last task in this product's gallery, return "none".
        keys = list(one_ProcessTaskGallery.keys())
        assert finished_task in keys, f"Current task {finished_task} not found in the product's process task gallery."
        current_index = keys.index(finished_task)
        next_index = current_index + 1
        if next_index >= len(keys):
            return CfgProcessTaskGalleryInAll["none"]
        next_task = keys[next_index]
        return CfgProcessTaskGalleryInAll[next_task]

class MaterialBatch:
    def __init__(self, idx_in_material_batch_list: int, cfg: dict, env_id: int, cuda_device: torch.device):
        # static variables
        self.idx = idx_in_material_batch_list
        self.cuda_device = cuda_device
        self.cfg = copy.deepcopy(cfg)
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id
        self.reset_state = copy.deepcopy(cfg["reset_state_template"])
        self.reset_state["key_variables"] = self.iter_key_variables()
        self._register_rigid_prim()
        ### dynmaic variables
        self.state : dict = None

    def _register_rigid_prim(self):
        for obj_name, info in self.meta_registeration_info.items():
            rigid_prim = RigidPrim(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}"),
                name=f"env_{self.env_id}_{info['name'].format(idx=f'{self.idx:02d}')}",
                reset_xform_properties=False,
            )
            setattr(self, obj_name, rigid_prim)

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state : dict = copy.deepcopy(self.reset_state)
        env_state_action_dict["material"][f"num_{self.idx:02d}_{self.type_name}"] = self.state
        self.reset_raw_materials_to_storage(env_state_action_dict)
        self.reset_integrated_materials_to_unappeared(env_state_action_dict)
        return env_state_action_dict

    def reset_integrated_materials_to_unappeared(self, env_state_action_dict: dict) -> dict:
        material_prims : dict = self.iter_integrated_material_prims()
        for material_type, material_prim in material_prims.items():
            material_name = f"num_{self.idx:02d}_{material_type}"
            position = material_prim.get_local_poses()[0]
            ### to set the material to underground
            position[0][2] = -100
            orientation = material_prim.get_local_poses()[1]
            env_state_action_dict["rigid_prims"][material_name] = {
                "object": material_prim,
                "position": position,
                "orientation": orientation,
            }
        return env_state_action_dict

    def reset_raw_materials_to_storage(self, env_state_action_dict: dict) -> dict:
        # The following code resets the placement of raw materials into appropriate storage slots,
        # based on storage capacity, type support, and current fill state. For each raw material type,
        # we try to place it into a storage that can accept it and is not already full. If the storage
        # is partially filled with another material type, we skip it to only mix the same types.
        # We then update the storage's meta state and also register the material's position and orientation.
        material_prims : dict = self.iter_raw_material_prims()
        storages : dict = env_state_action_dict["storage"]
        for material_type, material_prim in material_prims.items():
            material_name = f"num_{self.idx:02d}_{material_type}"
            for storage_name, value in storages.items():
                supporting_materials = value["key_variables"]["supporting_materials"]
                # If this material isn't supported, or storage is already full, skip
                if material_type not in supporting_materials or value["state"] == "full":
                    continue
                # If storage is partially filled with a different material type, skip
                if value["state"] == "partial" and material_type != value["material_type"]:
                    continue
                # Otherwise, place material in this storage
                self.state["storage_name"] = storage_name
                value["material_type"] = material_type
                value["num_material"] += 1
                value["material_idx_list"].append(self.idx)
                capacity = value["key_variables"]["capacity"]
                value["state"] = "full" if value["num_material"] == capacity else "partial"
                # Retrieve the pose (position & orientation) for this material in storage
                pose_list = value["key_variables"]["placement_cfg"]["pose_list"]
                position = pose_list[value["num_material"] - 1]["position"]
                orientation = pose_list[value["num_material"] - 1]["orientation"]
                # Register material's prim, position, and orientation in env_state_action_dict
                env_state_action_dict["rigid_prims"][material_name] = {
                    "object": material_prim,
                    "position": position,
                    "orientation": orientation,
                }
                break
                #usage:
                #material_prim.set_local_poses(translations=position.unsqueeze(0), orientations=orientation.unsqueeze(0))
        return

    def iter_key_variables(self):
        return {
            "type_name": self.type_name,
            "type_id": self.type_id,
            "idx": self.idx,
        }

    @abstractmethod
    def iter_raw_material_prims(self): 
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
        # trace the material batch's state, and update the position
        return env_state_action_dict

    def iter_raw_material_prims(self):
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