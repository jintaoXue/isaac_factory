from isaacsim.core.prims import RigidPrim
from abc import abstractmethod
from ..env_asset_cfg.cfg_material_product import CfgProductProcess, CfgProductOrder, CfgRegistrationInfos
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll, CfgProcessTaskGalleryDetailedClassified
from ..env_asset_cfg.cfg_machine import CfgMachine
from ..env_asset_cfg.cfg_robot import CfgRobot
from .disturbance import should_skip_material_placement
import copy
import torch

import omni.usd

# (subfolder, name_prefix) under product_water_pipe_group/
_PRODUCT_PRIM_SPECS: tuple[tuple[str, str], ...] = (
    ("cubes", "cube"),
    ("hoops", "hoop"),
    ("bending_tubes", "bending_tube"),
    ("products", "semi_product"),
    ("products", "product"),
)


def ensure_product_water_pipe_prims(env_id: int) -> None:
    """Clone missing product material prims when order > USD-authored instances.

    HC_import.usd ships with batch indices 00-04 only. Higher ``CfgRegistrationInfos``
    values are satisfied by duplicating the 00 template at runtime.
    """
    required = int(CfgRegistrationInfos.get("ProductWaterPipe", 0) or 0)
    if required <= 0:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available for product prim provisioning")

    group_base = f"/World/envs/env_{env_id}/obj/HC_factory/product_water_pipe_group"
    template_idx = "00"

    for idx in range(required):
        idx_str = f"{idx:02d}"
        for subfolder, prefix in _PRODUCT_PRIM_SPECS:
            dst = f"{group_base}/{subfolder}/{prefix}_{idx_str}"
            if stage.GetPrimAtPath(dst).IsValid():
                continue
            src = f"{group_base}/{subfolder}/{prefix}_{template_idx}"
            if not stage.GetPrimAtPath(src).IsValid():
                raise RuntimeError(
                    f"Cannot provision {dst}: template prim missing at {src}. "
                    "Extend HC_import.usd or lower CfgRegistrationInfos."
                )
            ok = omni.usd.duplicate_prim(stage, src, dst, duplicate_layers=True)
            if not ok and not stage.GetPrimAtPath(dst).IsValid():
                raise RuntimeError(f"Failed to duplicate product prim {src} -> {dst}")


def _is_storage_location(storage_name: str | None) -> bool:
    return bool(storage_name and "Storage_" in storage_name)


def _effective_storage_capacity(storage: dict) -> int:
    """Logical capacity may exceed authored placement poses; clamp to pose count."""
    cap = int(storage["key_variables"]["capacity"])
    pose_list = storage["key_variables"]["placement_cfg"]["pose_list"]
    return min(cap, len(pose_list))


def _sync_storage_fill_state(storage: dict) -> None:
    capacity = _effective_storage_capacity(storage)
    n = int(storage.get("num_material", 0))
    if n <= 0:
        storage["num_material"] = 0
        storage["state"] = "empty"
        storage["material_type"] = None
    elif n >= capacity:
        storage["num_material"] = capacity
        storage["state"] = "full"
    else:
        storage["state"] = "partial"


def _release_storage_slot(env_state_action_dict: dict, storage_name: str | None, batch_idx: int) -> None:
    if not _is_storage_location(storage_name):
        return
    storage = env_state_action_dict["storage"].get(storage_name)
    if storage is None:
        return
    idx_list = storage.get("material_idx_list", [])
    if batch_idx in idx_list:
        idx_list.remove(batch_idx)
        storage["num_material"] = len(idx_list)
    elif storage.get("num_material", 0) > 0:
        storage["num_material"] -= 1
    _sync_storage_fill_state(storage)


def _occupy_storage_slot(
    env_state_action_dict: dict,
    storage_name: str,
    batch_idx: int,
    material_type: str,
) -> int:
    """Reserve one slot; return 0-based pose index."""
    storage = env_state_action_dict["storage"][storage_name]
    idx_list = storage.setdefault("material_idx_list", [])
    if batch_idx in idx_list:
        return idx_list.index(batch_idx)

    eff_cap = _effective_storage_capacity(storage)
    if len(idx_list) >= eff_cap:
        raise ValueError(
            f"Storage {storage_name} is full ({len(idx_list)}/{eff_cap} slots)"
        )

    idx_list.append(batch_idx)
    storage["num_material"] = len(idx_list)
    storage["material_type"] = material_type
    _sync_storage_fill_state(storage)
    return len(idx_list) - 1


def pick_free_storage(env_state_action_dict: dict, processed_material: str) -> str:
    """Return the first storage with a free slot for ``processed_material``."""
    for storage_name, value in env_state_action_dict["storage"].items():
        supporting_materials = value["key_variables"]["supporting_materials"]
        if processed_material not in supporting_materials:
            continue
        if (
            value.get("state") == "partial"
            and value.get("material_type") not in (None, processed_material)
        ):
            continue
        idx_list = value.get("material_idx_list", [])
        if len(idx_list) >= _effective_storage_capacity(value):
            continue
        return storage_name
    raise ValueError(f"No free storage found for {processed_material}")


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
        # self.update_task_availability_mask(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for material_batch in self.material_batch_list:
            material_batch.step(env_state_action_dict)
        # self.update_task_availability_mask(env_state_action_dict)
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
        env_state_action_dict["agent_action_mask"]["material"]["task_availability_mask"] = mask
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
        self.offset_for_AGV_placement = CfgRobot["AGV"]["offset_for_material_placement"]["position"].to(self.cuda_device)

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
            self.state["submaterials"][material_type]["storage_name"] = "disappear"
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
        for material_type, material_prim in material_prims.items():
            material_name = f"num_{self.idx:02d}_{material_type}"
            if should_skip_material_placement(self.idx, material_type):
                # Shortage: park underground so kitting / downstream starve.
                self.state["submaterials"][material_type]["storage_name"] = "disappear"
                position = material_prim.get_local_poses()[0]
                position[0][2] = -100
                orientation = material_prim.get_local_poses()[1]
                env_state_action_dict["rigid_prims"][material_name] = {
                    "object": material_prim,
                    "position": position,
                    "orientation": orientation,
                }
                continue
            try:
                storage_name = pick_free_storage(env_state_action_dict, material_type)
            except ValueError:
                continue
            self.state["submaterials"][material_type]["storage_name"] = storage_name
            slot_idx = _occupy_storage_slot(
                env_state_action_dict, storage_name, self.idx, material_type
            )
            pose_list = env_state_action_dict["storage"][storage_name]["key_variables"]["placement_cfg"]["pose_list"]
            position = pose_list[slot_idx]["position"]
            orientation = pose_list[slot_idx]["orientation"]
            env_state_action_dict["rigid_prims"][material_name] = {
                "object": material_prim,
                "position": position,
                "orientation": orientation,
            }
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
    
    
    def step(self, env_state_action_dict: dict) -> dict:
        task_record_index : int = env_state_action_dict["material"][f"num_{self.idx:02d}_{self.type_name}"]["ongoing_task_record_index"]
        if task_record_index is None:
            return
        task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
        assert task_record["product_index"] == self.idx, "The product index should be the same as the product index in the task record"
        subtasks = task_record["subtasks_dict"]
        material_subtask = subtasks["material_states_in_subtasks"]
        ongoing_index = subtasks["ongoing_index"]
        if ongoing_index == subtasks["num_subtasks"] - 1:
            ## task done, set the finished task and ongoing task record index to none
            if task_record["task_type"] == "processing":
                if task_record["already_done_next_logistic_task"] == True:
                    self.state["finished_task"] = task_record["next_logistic_task"]
                else:
                    self.state["finished_task"] = task_record["task"]
            elif task_record["task_type"] == "logistic":
                self.state["finished_task"] = task_record["task"]
            else:
                raise ValueError(f"Invalid task type: {task_record['task_type']}")
            self.state["ongoing_task_record_index"] = None
        all_materials = {**self.iter_raw_material_prims(), **self.iter_integrated_material_prims()}
        for material_type, material_prim in all_materials.items():
            material_name = f"num_{self.idx:02d}_{material_type}"
            material_state = material_subtask[material_type][ongoing_index]

            position = env_state_action_dict["rigid_prims"][material_name]["position"]
            orientation = env_state_action_dict["rigid_prims"][material_name]["orientation"]
            old_storage_name = self.state["submaterials"][material_type]["storage_name"]
            storage_name = old_storage_name

            if material_state == "on_start_area":
                pass
            elif material_state == "disappear":
                position[0][2] = -100
                storage_name = "disappear"
            elif material_state == "on_gantry":
                gantry_index = task_record["chosen_gantry_index"]
                gantry_indexs = CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]["gantry_indexs"]
                joint_position = env_state_action_dict["articulations"]["num07_gantry_group"]["joint_position"].clone()[gantry_indexs == gantry_index]
                xy_position_reset = CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]["xy_position_reset"].to(self.cuda_device)[gantry_indexs == gantry_index]
                joint_positions_reset = CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]["joint_positions_reset"].to(self.cuda_device)[gantry_indexs == gantry_index]
                fixed_hook_height : int = CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]["fixed_hook_height"]
                xy_target = joint_position - joint_positions_reset + xy_position_reset
                position = torch.tensor([xy_target[0], xy_target[1], fixed_hook_height], device=self.cuda_device).unsqueeze(0)
                storage_name = "num07_gantry_group"
            elif material_state == "on_robot":
                robot_name = task_record["robot"]
                position = env_state_action_dict["rigid_prims"][robot_name]["position"].clone()
                position[0][2] = position[0][2] + 0.1
                position += self.offset_for_AGV_placement
                orientation = env_state_action_dict["rigid_prims"][robot_name]["orientation"].clone()
                storage_name = robot_name
            elif (material_state == "on_goal_area" and (subtasks["material_goal_area"] in CfgMachine)) or \
                (material_state == "on_machine"):
                if material_state == "on_goal_area":
                    machine_name = subtasks["material_goal_area"]
                    workstation_key = subtasks["goal_area_workstation_key"]
                else:
                    workstation_key = task_record["chosen_machine_workstation"]
                    machine_name = task_record["target_machine"]
                position = CfgMachine[machine_name]["material_placement_cfg"][workstation_key]["position"]
                position = position.to(self.cuda_device).unsqueeze(0)
                orientation = CfgMachine[machine_name]["material_placement_cfg"][workstation_key]["orientation"]
                orientation = orientation.to(self.cuda_device).unsqueeze(0)
                storage_name = workstation_key
            elif material_state == "on_goal_area" and "Storage" in subtasks["material_goal_area"]:
                storage_name = subtasks["material_goal_area"]
                if old_storage_name != storage_name:
                    try:
                        slot_idx = _occupy_storage_slot(
                            env_state_action_dict, storage_name, self.idx, material_type
                        )
                    except ValueError:
                        storage_name = pick_free_storage(env_state_action_dict, material_type)
                        subtasks["material_goal_area"] = storage_name
                        subtasks["goal_area_ids"] = env_state_action_dict["storage"][storage_name][
                            "key_variables"
                        ]["working_area_ids"]
                        slot_idx = _occupy_storage_slot(
                            env_state_action_dict, storage_name, self.idx, material_type
                        )
                    pose_list = env_state_action_dict["storage"][storage_name]["key_variables"]["placement_cfg"]["pose_list"]
                    slot_pose = pose_list[slot_idx]
                    position = slot_pose["position"]
                    orientation = slot_pose["orientation"]
            else:
                raise ValueError(f"Invalid material state: {material_state}")

            if old_storage_name != storage_name and _is_storage_location(old_storage_name):
                _release_storage_slot(env_state_action_dict, old_storage_name, self.idx)

            env_state_action_dict["rigid_prims"][material_name]["position"] = position
            env_state_action_dict["rigid_prims"][material_name]["orientation"] = orientation
            self.state["submaterials"][material_type]["storage_name"] = storage_name


class ProductWaterPipe(MaterialBatch):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        self.product_00_pipe : RigidPrim = None
        self.product_00_flange : RigidPrim = None
        self.product_00_elbow : RigidPrim = None
        self.product_00_semi : RigidPrim = None
        self.product_00_maded : RigidPrim = None
        super().__init__(idx, cfg, env_id, cuda_device)

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