from isaacsim.core.prims import RigidPrim
import omni.usd
from pxr import Usd, UsdSkel, Gf, Sdf
from ..env_asset_cfg.cfg_human import CfgHuman, CfgHumanRegistrationInfos
from ..env_asset_cfg.cfg_machine import CfgMachine
from ..env_asset_cfg.route.cfg_route import RouteOptionalInitPointsInMap, OptionalInitPointIds
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll
from ..env_asset_cfg.cfg_process_subtask_gallery import CfgSubtaskPredefinedTimeGallery, SubtaskTimeNoiseStdSteps
from .utils import HumanAnimation, quat_multiply_wxyz, sample_noisy_steps, yaw_to_quaternion_wxyz
import torch
import copy
import random

class HumanManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_human = CfgHuman
        self.cfg_registration_infos = CfgHumanRegistrationInfos
        self.optional_init_points_in_map = RouteOptionalInitPointsInMap["human_xyz"]
        self.optional_init_points_ids = OptionalInitPointIds["human"]
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
        # self.optional_init_points_ids is a list (Python, not a torch tensor) so use list comprehension, not tensor indexing
        shuffled_init_points_ids: list[int] = [self.optional_init_points_ids[i] for i in perm.tolist()]
 
        for human, i in zip(self.human_list, range(num_humans)):
            human.reset(env_state_action_dict, shuffled_init_points_in_map[i].unsqueeze(0), shuffled_init_points_ids[i])
        
        # self.update_task_availability_mask(env_state_action_dict)
        # self.update_self_availability_mask(env_state_action_dict)

        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for human in self.human_list:
            human.step(env_state_action_dict)
        # self.update_task_availability_mask(env_state_action_dict)
        # self.update_self_availability_mask(env_state_action_dict)

        return env_state_action_dict
    
    def update_task_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for human availability for selection by human-robot machine allocator agent
        mask = torch.zeros(len(CfgProcessTaskGalleryInAll), dtype=torch.int32, device=self.cuda_device)
        mask[0] = 1 # "none" task is always available
        for human, i in zip(self.human_list, range(len(self.human_list))):
            if human.state['state'] == "free":
                mask[:] = 1
                break
        env_state_action_dict["agent_action_mask"]["human"]["task_availability_mask"] = mask
        return env_state_action_dict

    def update_self_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for human availability
        # shape (upper_bound_num_human,) 
        mask = torch.zeros(self.upper_bound_num_human, dtype=torch.int32, device=self.cuda_device)
        for human, i in zip(self.human_list, range(len(self.human_list))):
            if human.state['state'] == "free":
                mask[i] = 1
        env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"] = mask
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
        self.reset_state : str = copy.deepcopy(cfg["reset_state"])
        self.reset_state["key_variables"] = self.iter_key_variables()
        self.skeleton: UsdSkel.Skeleton | None = None
        self.prim: RigidPrim | None = None
        self._joint_names: list[str] = []
        self._joint_name_to_index: dict[str, int] = {}
        self._rest_transforms_base: list[Gf.Matrix4d] = []
        self._current_pose_name: str = "default"
        self._animation = HumanAnimation(cfg.get("animation_cfg", {}))
        self._joint_map: dict = cfg.get("animation_cfg", {}).get("joints", {})
        self.cuda_device = cuda_device
        self._register_skeleton()
        self._register_rigid_prim()
        ### dynmaic variables
        self.state : dict = {}

    def _register_skeleton(self):
        meta = self.meta_registeration_info
        stage = omni.usd.get_context().get_stage()
        prim_path = meta["skeleton_prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}")
        skeleton_prim = stage.GetPrimAtPath(prim_path)
        self.skeleton = UsdSkel.Skeleton(skeleton_prim)
        joints = self.skeleton.GetJointsAttr().Get() or []
        self._joint_names = [str(j) for j in joints]
        self._joint_name_to_index = {name: i for i, name in enumerate(self._joint_names)}
        rest = self.skeleton.GetRestTransformsAttr().Get()
        if rest is None or len(rest) != len(self._joint_names):
            bind = self.skeleton.GetBindTransformsAttr().Get() or []
            if len(bind) == len(self._joint_names):
                rest = bind
            else:
                rest = [Gf.Matrix4d(1.0) for _ in self._joint_names]
        self._rest_transforms_base = [Gf.Matrix4d(m) for m in rest]
        return
    
    def _register_rigid_prim(self):
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["rigid_prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}"),
            name=f"env_{self.env_id}_{meta['name'].format(idx=f'{self.idx:02d}')}",
            reset_xform_properties=False,
        )
        return
    
    def iter_key_variables(self):
        return {
            "type_name": self.type_name,
            "idx": self.idx,
        }

    def reset(self, env_state_action_dict: dict, init_point_in_map: torch.tensor, init_point_id: int) -> dict:
        self.state : dict = copy.deepcopy(self.reset_state)
        self.state["current_area_id"] = init_point_id
        env_state_action_dict["human"][f"num_{self.idx:02d}_{self.type_name}"] = self.state
        self.reset_to_random_map_point(env_state_action_dict, init_point_in_map)
        self._animation.reset()
        self.set_pose("idle")
        self._advance_animation()
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
        task_record_index : int = env_state_action_dict["human"][f"num_{self.idx:02d}_{self.type_name}"]["ongoing_task_record_index"]
        if task_record_index is None:
            self.set_pose("idle")
            self._advance_animation()
            return env_state_action_dict
        task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
        assert task_record["human_index"] == self.idx, "The human index should be the same as the human index in the task record"

        assert self.state["state"] != "free", "The human should be working on the task, and be defined in task_progress_manager.py"
        
        subtasks = task_record["subtasks_dict"]
        subtask = subtasks["ongoing"]
        human_subtask = subtask[0]
        #TODO: check change subtasks value can change task records value
        if human_subtask in ("go_to_material", "go_to_goal_area", "go_to_processing_machine"):
            self.set_pose("walk")
        elif human_subtask in (
            "material_on_gantry",
            "control_gantry",
            "material_on_robot",
            "material_on_goal_area",
            "control_machine",
        ):
            self.set_pose("operate")
        elif human_subtask in ("wait", "done"):
            self.set_pose("idle")
        self._advance_animation()
        if human_subtask == "go_to_material":
            self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_area_type = "start")
        elif human_subtask == "material_on_gantry":
            self._time_counting_subtask(subtasks, human_subtask)
        elif human_subtask == "control_gantry":
            self._time_counting_subtask(subtasks, human_subtask)
        elif human_subtask == "material_on_robot":
            self._time_counting_subtask(subtasks, human_subtask)
        elif human_subtask == "go_to_goal_area":
            self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_area_type = "goal")
        elif human_subtask == "material_on_goal_area":
            self._time_counting_subtask(subtasks, human_subtask)    
        elif human_subtask == "go_to_processing_machine":
            self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, target_area_type = "start")
        elif human_subtask == "control_machine":
            self._let_human_face_to_machine(env_state_action_dict, task_record)
            self._time_counting_subtask(subtasks, human_subtask)
        elif human_subtask == "wait":
            subtasks["finished"][0] = True
        elif human_subtask == "done":
            self._task_done(env_state_action_dict, subtasks)
        else:
            raise ValueError(f"Invalid human subtask for processing: {human_subtask}")
        return env_state_action_dict

    def _subtask_go_to_target(self, env_state_action_dict: dict, task_record: dict, subtasks: dict, target_area_type: str) -> None:
        if subtasks["finished"][0] == True:
            return
        
        if self.state["target_area_id"] is None:
            if target_area_type == "start":
                assert task_record["subtasks_dict"]["start_area_ids"] is not None, "The start area ids should be initialized in task_progress_manager.py"
                target_area_id = task_record["subtasks_dict"]["start_area_ids"]["human_working_areas_ids"][0]
            elif target_area_type == "goal":
                assert task_record["subtasks_dict"]["goal_area_ids"] is not None, "The goal area ids should be initialized in task_progress_manager.py"
                target_area_id = task_record["subtasks_dict"]["goal_area_ids"]["human_working_areas_ids"][0]
            else:
                raise ValueError(f"Invalid target area type: {target_area_type}")
            self.state["target_area_id"] = target_area_id
        elif self.state["target_area_id"] == self.state["current_area_id"]:
            subtasks["finished"][0] = True
            self.state["target_area_id"] = None
            self.state["route_index"] = 0
            self.state["route_length"] = 0
            self.state["generated_route"] = []
        else:
            # the human is going to the target area by route planner in route.py
            pass
        return env_state_action_dict

    def _let_human_face_to_machine(self, env_state_action_dict: dict, task_record: dict) -> None:
        human_area_id = self.state["current_area_id"]
        machine_name = task_record["target_machine"]
        yaw = CfgMachine[machine_name]["human_working_areas_ids_orientation"][human_area_id]
        orientation = yaw_to_quaternion_wxyz(yaw, self.cuda_device)
        human_name = f"num_{self.idx:02d}_{self.type_name}"
        env_state_action_dict["rigid_prims"][human_name]["orientation"] = orientation.unsqueeze(0)
        return

    def _time_counting_subtask(self, subtasks: dict, human_subtask: str) -> None:
        if subtasks["finished"][0] == True:
            return
        if self.state.get("_counting_subtask") != human_subtask:
            self.state["_counting_subtask"] = human_subtask
            self.state["subtask_time_counter"] = 0
            base = CfgSubtaskPredefinedTimeGallery[human_subtask]
            self.state["subtask_time_target"] = sample_noisy_steps(base, SubtaskTimeNoiseStdSteps)
        target = self.state["subtask_time_target"]
        if self.state["subtask_time_counter"] < target:
            self.state["subtask_time_counter"] += 1
        else:
            subtasks["finished"][0] = True
            self.state["subtask_time_counter"] = 0

    def _task_done(self, env_state_action_dict: dict, subtasks: dict) -> None:
        subtasks["finished"][0] = True
        #TODO, check reset successfully
        current_area_id = self.state["current_area_id"]
        self.state : str = copy.deepcopy(self.reset_state)
        self.state["current_area_id"] = current_area_id
        env_state_action_dict["human"][f"num_{self.idx:02d}_{self.type_name}"] = self.state
        return env_state_action_dict

    def _resolve_joint_index(self, joint_name: str) -> int | None:
        if joint_name in self._joint_name_to_index:
            return self._joint_name_to_index[joint_name]
        joint_name_lower = joint_name.lower()
        joint_suffix = joint_name_lower.rsplit("/", 1)[-1]
        for name, index in self._joint_name_to_index.items():
            name_lower = name.lower()
            if name_lower.endswith(joint_name_lower) or joint_name_lower in name_lower:
                return index
            name_suffix = name_lower.rsplit("/", 1)[-1]
            if joint_suffix == name_suffix:
                return index
        return None

    def _rotation_matrix_from_euler_xyz_deg(self, xyz_deg: tuple[float, float, float]) -> Gf.Matrix4d:
        rx = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), float(xyz_deg[0]))
        ry = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), float(xyz_deg[1]))
        rz = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(xyz_deg[2]))
        rot = rz * ry * rx
        mat = Gf.Matrix4d(1.0)
        mat.SetRotate(rot)
        return mat

    def _apply_joint_pose(self, joint_pose: dict[str, tuple[float, float, float]]) -> int:
        if self.skeleton is None or not self._rest_transforms_base:
            return 0
        rest_transforms = [Gf.Matrix4d(m) for m in self._rest_transforms_base]
        hit = 0
        for joint_key, rot_xyz in joint_pose.items():
            joint_name = self._joint_map.get(joint_key, joint_key)
            joint_index = self._resolve_joint_index(joint_name)
            if joint_index is None:
                continue
            delta = self._rotation_matrix_from_euler_xyz_deg(rot_xyz)
            rest_transforms[joint_index] = rest_transforms[joint_index] * delta
            hit += 1
        self.skeleton.GetRestTransformsAttr().Set(rest_transforms)
        return hit

    def _advance_animation(self) -> int:
        joint_pose = self._animation.step()
        if not joint_pose:
            return 0
        return self._apply_joint_pose(joint_pose)

    def set_pose(self, pose_name: str) -> bool:
        pose_name = pose_name.lower()
        if pose_name == "default":
            if self.skeleton is None:
                return False
            self.skeleton.GetRestTransformsAttr().Set(list(self._rest_transforms_base))
            self._current_pose_name = pose_name
            self._animation.reset()
            return True
        if pose_name != self._current_pose_name:
            self._animation.set_animation(pose_name)
            self._current_pose_name = pose_name
        return True

class NormalHuman(Human):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)