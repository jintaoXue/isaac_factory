from isaacsim.core.prims import Articulation
from abc import abstractmethod
from ..env_asset_cfg.cfg_machine import CfgMachine
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll
from .utils import GantryGroupAnimation, PoseAnimation
import copy
import torch

class MachineManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_machine = CfgMachine
        self.num_machine = len(self.cfg_machine)
        self.num00_rotaryPipeAutomaticWeldingMachine = num00_rotaryPipeAutomaticWeldingMachine(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num01_weldingRobot = num01_weldingRobot(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine = num02_rollerbedCNCPipeIntersectionCuttingMachine(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num03_laserCuttingMachine = num03_laserCuttingMachine(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num04_groovingMachineLarge = num04_groovingMachineLarge(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num05_groovingMachineSmall = num05_groovingMachineSmall(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num06_highPressureFoamingMachine = num06_highPressureFoamingMachine(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num07_gantry_group = num07_gantry_group(env_id=self.env_id, cuda_device=self.cuda_device)
        self.num08_workbench = num08_workbench(env_id=self.env_id, cuda_device=self.cuda_device)

    def reset(self, env_state_action_dict: dict) -> dict:
        for machine in self.iter_machines() + self.iter_logistic_machines():
            machine.reset(env_state_action_dict)
        self.update_task_availability_mask(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for machine in self.iter_machines() + self.iter_logistic_machines():
            machine.step(env_state_action_dict)
        self.update_task_availability_mask(env_state_action_dict)
        return env_state_action_dict
    
    def iter_machines(self):
        return (
            self.num00_rotaryPipeAutomaticWeldingMachine,
            self.num01_weldingRobot,
            self.num02_rollerbedCNCPipeIntersectionCuttingMachine,
            self.num03_laserCuttingMachine,
            self.num04_groovingMachineLarge,
            self.num05_groovingMachineSmall,
            self.num06_highPressureFoamingMachine,
            self.num08_workbench,
        )
    def iter_logistic_machines(self):
        return (
            self.num07_gantry_group,
        )
    def update_task_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for machine availability for selection by human-robot machine allocator agent
        # output shape (len(CfgProcessTaskGalleryInAll))
        mask = torch.zeros(len(CfgProcessTaskGalleryInAll), dtype=torch.int32, device=self.cuda_device)
        mask[0] = 1 # "none" task is always available
        have_free_gantry = False
        active_gantry_indices = CfgMachine["num07_gantry_group"]["active_gantry_indices"]
        for machine in self.iter_logistic_machines():
            state: list = machine.state["state"]
            for gantry_index in active_gantry_indices:
                if state[gantry_index] == "free":
                    have_free_gantry = True
                    break
            if have_free_gantry:
                break
        for machine in self.iter_machines():
            assert machine.type_name != "num07_gantry_group", "num07_gantry_group is logistic machine"
            state : list = machine.state['state']
            can_do_logistic_task_names : list = machine.corresponding_logistic_task
            for state_name in state:
                if state_name == "free":
                    if have_free_gantry:
                        for task_name in can_do_logistic_task_names:
                            task_index = CfgProcessTaskGalleryInAll[task_name]
                            mask[task_index] = 1
                elif state_name == "invalid":
                    pass
                else:
                    # Though processing task contains logistic subtasks, 
                    # but will be defined in task_progress_manager.py, so here dont need to consider it
                    pre_name = state_name.split("_")[0]
                    task_name = state_name.split("_", 1)[1]
                    if pre_name == "materialReadyFor":                        
                        task_index = CfgProcessTaskGalleryInAll[task_name]
                        mask[task_index] = 1
                    elif pre_name == "working" or pre_name == "waiting":
                        pass
                    else:
                        raise ValueError(f"Invalid machine state: {state_name}")
        env_state_action_dict["agent_action_mask"]["machine"]["task_availability_mask"] = mask


class Machine:
    def __init__(self, cfg: dict, env_id: int, cuda_device: torch.device):
        # static variables
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.registration_type = cfg["registration_type"]
        self.num_workstations = cfg["num_workstations"]
        self.num_registration_parts = cfg["num_registration_parts"]
        self.registration_infos = cfg["registration_infos"]
        # self.corresponding_process_task = cfg["corresponding_process_task"]
        self.corresponding_logistic_task = cfg["corresponding_logistic_task"]
        self.reset_state = copy.deepcopy(cfg["reset_state"])
        self.working_area_ids = cfg["working_area_ids"]
        self.material_placement_cfg = cfg["material_placement_cfg"]
        self.reset_state["key_variables"] = self.iter_key_variables()
        ### dynmaic variables
        self.state : dict = {}
        self._register_articulation_animation()

    def _register_articulation_animation(self):
        for obj_name, info in self.registration_infos.items():
            articulation = Articulation(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id),
                name=f"env_{self.env_id}_{obj_name}",
                reset_xform_properties=False,
            )
            setattr(self, obj_name, articulation)
            setattr(self, f"animation_{obj_name}", PoseAnimation(
                start_pose=info["joint_positions_reset"],
                end_pose=info["joint_positions_reset"],
                animation_time=info["animation_time"],
                device=self.cuda_device,
            ))

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state : dict = copy.deepcopy(self.reset_state)
        env_state_action_dict["machine"][self.type_name] = self.state
        articulations_values = self.reset_articulations()
        env_state_action_dict["articulations"].update(articulations_values)
        return env_state_action_dict

    def reset_articulations(self) -> dict:
        articulations_values: dict = {}
        for obj_name in self.registration_infos.keys():
            obj = getattr(self, obj_name, None)
            joint_positions_reset : torch.Tensor = self.registration_infos[obj_name]["joint_positions_reset"].to(self.cuda_device)
            articulations_values[obj_name] = {
                "object": obj,
                "joint_position": joint_positions_reset,
            }
        return articulations_values

    def iter_key_variables(self):
        return {
            "type_name": self.type_name, 
            "working_area_ids": self.working_area_ids,
            "num_workstations": self.num_workstations,
            "material_placement_cfg": self.material_placement_cfg,
        }

    def step(self, env_state_action_dict: dict) -> dict:
        assert self.type_name != "num07_gantry_group", "num07_gantry_group have its own step function"
        task_record_index_list : list[int] = env_state_action_dict["machine"][self.type_name]["ongoing_task_record_index"]
        for task_record_index, workstation_index in zip(task_record_index_list, range(self.num_workstations)):
            if task_record_index is None:
                continue
            self._step_one_workstation(env_state_action_dict, task_record_index, workstation_index)
    
    def _step_one_workstation(self, env_state_action_dict: dict, task_record_index: int, workstation_index: int) -> None:
        task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
        chosen_workstation_index = task_record["chosen_workstation_index"]
        assert chosen_workstation_index == workstation_index, "The chosen workstation index should be the same as the workstation index in the task record"
        workstation_state = self.state["state"][chosen_workstation_index]
        task_type = task_record["task_type"]
        subtasks = task_record["subtasks_dict"]
        machine_subtask = subtasks["ongoing"][2]
        
        pre_name = workstation_state.split("_")[0]
        # free, materialReadyFor_task_name, working_task_name, waiting_processing_task, invalid
        assert pre_name == "working", "The machine should be working on the task, and be defined in task_progress_manager.py"

        if machine_subtask == "done":
            self._task_done(env_state_action_dict, task_record)
        elif machine_subtask == "process":
            self._subtask_process(env_state_action_dict, task_record, subtasks)
        elif machine_subtask == "wait":
            subtasks["finished"][2] = True
        else:
            raise ValueError(f"Invalid machine subtask for processing: {machine_subtask}")
        ## animation
        if self.type_name != "num08_workbench":
            chosen_machine_workstation = task_record["chosen_machine_workstation"]
            animation_obj : PoseAnimation = getattr(self, f"animation_{chosen_machine_workstation}", None)
            env_state_action_dict["articulations"][chosen_machine_workstation]["joint_position"] = animation_obj.step_next_pose()


    def _subtask_process(self, env_state_action_dict: dict, task_record: dict, subtasks: dict) -> None:
        if subtasks["finished"][2] == True:
            return
        ### animation
        chosen_workstation_index = task_record["chosen_workstation_index"]
        chosen_machine_workstation = task_record["chosen_machine_workstation"]
        workstation_state = self.state["state"][chosen_workstation_index]
        task_name = workstation_state.split("_", 1)[1]
        assert task_name == task_record["task"], "The workstation ready for task should be the same as the task in the task record"
        if self.type_name != "num08_workbench":
            obj_animation : PoseAnimation = getattr(self, f"animation_{chosen_machine_workstation}", None)
            if self.state["target_joints_position"][chosen_workstation_index] is None:
                self.state["target_joints_position"][chosen_workstation_index] = self.registration_infos[chosen_machine_workstation]["joint_positions_working"].to(self.cuda_device)
                obj_animation.set_target_pose(self.state["target_joints_position"][chosen_workstation_index])
        ## processing the material on the machine
        self.state["processing_time_step"][chosen_workstation_index] += 1
        animation_time = None
        if self.type_name != "num08_workbench":
            animation_time = self.registration_infos[chosen_machine_workstation]["animation_time"]
        elif self.type_name == "num08_workbench":
            animation_time = self.registration_infos["num08_workbench"]["animation_time"]
        if self.state["processing_time_step"][chosen_workstation_index] >= animation_time:
            subtasks["finished"][2] = True
            self.state["processing_time_step"][chosen_workstation_index] = 0
    
    def _task_done(self, env_state_action_dict: dict, task_record: dict) -> None:
        task_record["subtasks_dict"]["finished"][2] = True
        task_type = task_record["task_type"]
        chosen_workstation_index = task_record["chosen_workstation_index"]
        self.state["ongoing_task_record_index"][chosen_workstation_index] = None
        if task_type == "logistic":
            processing_task_name = task_record["task"].removeprefix("logistic_for_")
            self.state["state"][chosen_workstation_index] = "materialReadyFor_" + processing_task_name
            return
        elif task_type == "processing":
            self.state["state"][chosen_workstation_index] = "free"
            # the next processing task is ready for the next workstation
            if task_record["next_chosen_workstation_index"] is not None:
                #the material is already on the machine, so no need to do logistic task for next processing task
                next_target_machine = task_record["next_target_machine"]
                next_chosen_workstation_index = task_record["next_chosen_workstation_index"]
                machine_state = env_state_action_dict["machine"][next_target_machine]["state"]
                machine_state[next_chosen_workstation_index] = "materialReadyFor_" + task_record["next_processing_task"]
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        
        chosen_machine_workstation = task_record["chosen_machine_workstation"]
        # self.state["processing_time_step"][chosen_workstation_index] = 0
        if self.type_name != "num07_gantry_group" and self.type_name != "num08_workbench":
            self.state["target_joints_position"][chosen_workstation_index] = None
            animation_obj : PoseAnimation = getattr(self, f"animation_{chosen_machine_workstation}", None)
            animation_obj.set_target_pose(self.registration_infos[chosen_machine_workstation]["joint_positions_reset"])

        return env_state_action_dict

class num00_rotaryPipeAutomaticWeldingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num00_rotaryPipeAutomaticWeldingMachine_part_01_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_01_station: PoseAnimation = None
        self.num00_rotaryPipeAutomaticWeldingMachine_part_02_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_02_station: PoseAnimation = None        
        super().__init__(cfg=CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"], env_id=env_id, cuda_device=cuda_device)


class num01_weldingRobot(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num01_weldingRobot_part02_robot_arm_and_base = None
        self.animation_num01_weldingRobot_part02_robot_arm_and_base: PoseAnimation = None
        self.num01_weldingRobot_part04_mobile_base_for_material = None
        self.animation_num01_weldingRobot_part04_mobile_base_for_material: PoseAnimation = None
        super().__init__(cfg=CfgMachine["num01_weldingRobot"], env_id=env_id, cuda_device=cuda_device)


class num02_rollerbedCNCPipeIntersectionCuttingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station = None
        self.animation_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station: PoseAnimation = None
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine = None
        self.animation_num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine: PoseAnimation = None
        super().__init__(
            cfg=CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"],
            env_id=env_id,
            cuda_device=cuda_device,
        )


class num03_laserCuttingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num03_laserCuttingMachine = None
        self.animation_num03_laserCuttingMachine: PoseAnimation = None
        super().__init__(cfg=CfgMachine["num03_laserCuttingMachine"], env_id=env_id, cuda_device=cuda_device)


class num04_groovingMachineLarge(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num04_groovingMachineLarge_part01_large_fixed_base = None
        self.animation_num04_groovingMachineLarge_part01_large_fixed_base: PoseAnimation = None
        self.num04_groovingMachineLarge_part02_large_mobile_base = None
        self.animation_num04_groovingMachineLarge_part02_large_mobile_base: PoseAnimation = None
        super().__init__(cfg=CfgMachine["num04_groovingMachineLarge"], env_id=env_id, cuda_device=cuda_device)


class num05_groovingMachineSmall(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num05_groovingMachineSmall_part01_small_fixed_base = None
        self.animation_num05_groovingMachineSmall_part01_small_fixed_base: PoseAnimation = None
        self.num05_groovingMachineSmall_part02_small_mobile_handle = None
        self.animation_num05_groovingMachineSmall_part02_small_mobile_handle: PoseAnimation = None
        super().__init__(cfg=CfgMachine["num05_groovingMachineSmall"], env_id=env_id, cuda_device=cuda_device)


class num06_highPressureFoamingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num06_highPressureFoamingMachine = None
        self.animation_num06_highPressureFoamingMachine: PoseAnimation = None
        super().__init__(
            cfg=CfgMachine["num06_highPressureFoamingMachine"],
            env_id=env_id,
            cuda_device=cuda_device,
        )

class num08_workbench(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num08_workbench = None
        self.animation_num08_workbench: PoseAnimation = None
        super().__init__(cfg=CfgMachine["num08_workbench"], env_id=env_id, cuda_device=cuda_device)
    
    def reset_articulations(self) -> dict:
        # The num08_workbench has 2 workstations, but they share the same articulation
        articulations_values: dict = {}
        obj_name = "num08_workbench"
        obj = getattr(self, obj_name, None)
        workstation_names = ["num08_workbench_station_00", "num08_workbench_station_01"]
        joint_positions_reset : torch.Tensor = self.registration_infos[obj_name]["joint_positions_reset"].to(self.cuda_device)
        # The num08_workbench is actually a manual workbench, so the joint positions are not important.
        # We simply set them to the reset positions to maintain a consistent format.
        articulations_values[workstation_names[0]] = {
            "object": obj,
            "joint_position": joint_positions_reset,
        }
        articulations_values[workstation_names[1]] = {
            "object": obj,
            "joint_position": joint_positions_reset,
        }
        
        return articulations_values


######### logistic machines #########



class num07_gantry_group(Machine):

    ACTIVE_GANTRY_INDICES = CfgMachine["num07_gantry_group"]["active_gantry_indices"]
    INVALID_GANTRY_PARKING_XY = {
        2: torch.tensor([-46.0, 10.18675]),
        3: torch.tensor([-42.0, 10.18675]),
    }

    def __init__(self, env_id: int, cuda_device: torch.device):
        self.num07_gantry_group = None
        self.animation_num07_gantry_group: GantryGroupAnimation = None
        super().__init__(cfg=CfgMachine["num07_gantry_group"], env_id=env_id, cuda_device=cuda_device)
        gantry_info = self.registration_infos["num07_gantry_group"]
        self.joint_position_reset = gantry_info["joint_positions_reset"].to(self.cuda_device)
        self.xy_position_reset = gantry_info["xy_position_reset"].to(self.cuda_device)
        self.gantry_indexs = gantry_info["gantry_indexs"].to(self.cuda_device)
        self.fixed_hook_height: float = gantry_info["fixed_hook_height"]
        self.safe_x_gap: float = gantry_info["safe_x_gap"]
        self._yielding = [False] * self.num_workstations
        self._yield_target_x: list[float | None] = [None] * self.num_workstations

    def _register_articulation_animation(self):
        for obj_name, info in self.registration_infos.items():
            articulation = Articulation(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id),
                name=f"env_{self.env_id}_{obj_name}",
                reset_xform_properties=False,
            )
            setattr(self, obj_name, articulation)
            setattr(
                self,
                f"animation_{obj_name}",
                GantryGroupAnimation(
                    start_pose=info["joint_positions_reset"],
                    end_pose=info["joint_positions_reset"],
                    animation_time=info["animation_time"],
                    device=self.cuda_device,
                    num_gantrys=self.num_workstations,
                ),
            )

    def _joint_to_world_x(self, joint_position: torch.Tensor, gantry_index: int) -> float:
        joint_x = float(joint_position[gantry_index].item())
        reset_joint_x = float(self.joint_position_reset[gantry_index].item())
        reset_world_x = float(self.xy_position_reset[gantry_index].item())
        return joint_x - reset_joint_x + reset_world_x

    def _joint_to_world_y(self, joint_position: torch.Tensor, gantry_index: int) -> float:
        y_joint_index = self.num_workstations + gantry_index
        joint_y = float(joint_position[y_joint_index].item())
        reset_joint_y = float(self.joint_position_reset[y_joint_index].item())
        reset_world_y = float(self.xy_position_reset[y_joint_index].item())
        return joint_y - reset_joint_y + reset_world_y

    def _forbidden_corridor(self, joint_position: torch.Tensor, gantry_index: int) -> tuple[float, float] | None:
        """X interval the gantry occupies or will traverse; None if effectively stationary."""
        animation = self.animation_num07_gantry_group
        current_x = self._joint_to_world_x(joint_position, gantry_index)
        if animation.is_done(gantry_index=gantry_index):
            return current_x, current_x
        target_x = self._joint_to_world_x(animation.end_pose, gantry_index)
        if abs(current_x - target_x) < 0.05:
            return current_x, current_x
        return min(current_x, target_x), max(current_x, target_x)

    def _corridor_clear_low_x(self, corridor: tuple[float, float]) -> float:
        lo, _ = corridor
        return lo - 2.0 * self.safe_x_gap

    def _corridor_clear_high_x(self, corridor: tuple[float, float]) -> float:
        _, hi = corridor
        return hi + 2.0 * self.safe_x_gap

    def _x_blocks_corridor(self, world_x: float, corridor: tuple[float, float]) -> bool:
        lo, hi = corridor
        return (lo - self.safe_x_gap) <= world_x <= (hi + self.safe_x_gap)

    def _x_clears_corridor(self, world_x: float, corridor: tuple[float, float]) -> bool:
        return not self._x_blocks_corridor(world_x, corridor)

    def _yield_moves_away(self, low_x: float, yield_x: float, high_x: float) -> bool:
        """True when yield_x is on the side of low_x away from the higher-priority gantry."""
        if low_x <= high_x:
            return yield_x <= low_x
        return yield_x >= low_x

    def _compute_yield_world_x(
        self, current_x: float, corridor: tuple[float, float], high_x: float
    ) -> float:
        """Yield outside the forbidden corridor, on the side away from the higher-priority gantry."""
        low_side = self._corridor_clear_low_x(corridor)
        high_side = self._corridor_clear_high_x(corridor)
        preferred = low_side if current_x <= high_x else high_side
        alternate = high_side if current_x <= high_x else low_side
        if self._x_clears_corridor(preferred, corridor):
            return preferred
        return alternate

    def _clear_yield(self, gantry_index: int, joint_position: torch.Tensor) -> None:
        if not self._yielding[gantry_index]:
            return
        self._yielding[gantry_index] = False
        self._yield_target_x[gantry_index] = None
        animation = self.animation_num07_gantry_group
        animation.is_yield_move[gantry_index] = False
        task_target = self.state["target_joints_position"][gantry_index]
        if task_target is not None:
            animation.set_target_pose(task_target, gantry_index=gantry_index, current_joint_position=joint_position)

    def _update_yield(self, joint_position: torch.Tensor, env_state_action_dict: dict) -> None:
        animation = self.animation_num07_gantry_group
        priorities = {
            gantry_index: self._gantry_priority(gantry_index, env_state_action_dict)
            for gantry_index in self.ACTIVE_GANTRY_INDICES
        }
        ordered = sorted(self.ACTIVE_GANTRY_INDICES, key=lambda idx: priorities[idx])

        for high_idx in ordered:
            corridor = self._forbidden_corridor(joint_position, high_idx)
            if corridor is None:
                continue
            for low_idx in ordered:
                if priorities[low_idx] <= priorities[high_idx]:
                    continue
                low_x = self._joint_to_world_x(joint_position, low_idx)
                high_x = self._joint_to_world_x(joint_position, high_idx)
                too_close = abs(low_x - high_x) < self.safe_x_gap
                in_corridor = self._x_blocks_corridor(low_x, corridor)
                if not too_close and not in_corridor:
                    continue
                if too_close and not in_corridor:
                    corridor = (high_x, high_x)

                if self._x_clears_corridor(low_x, corridor):
                    if self._yielding[low_idx]:
                        self._clear_yield(low_idx, joint_position)
                    continue

                locked_x = self._yield_target_x[low_idx]
                if (
                    self._yielding[low_idx]
                    and locked_x is not None
                    and not self._x_blocks_corridor(locked_x, corridor)
                    and self._yield_moves_away(low_x, locked_x, high_x)
                ):
                    return

                yield_x = self._compute_yield_world_x(low_x, corridor, high_x)
                if (
                    self._yielding[low_idx]
                    and locked_x is not None
                    and abs(locked_x - yield_x) < 0.05
                ):
                    return

                current_y = self._joint_to_world_y(joint_position, low_idx)
                yield_xy = torch.tensor([yield_x, current_y], dtype=torch.float32, device=self.cuda_device)
                yield_pose = self._get_joint_pose_from_xy_target(
                    joint_position.clone(), yield_xy, gantry_index=low_idx
                )
                animation.set_yield_target_pose(
                    yield_pose, gantry_index=low_idx, current_joint_position=joint_position
                )
                self._yielding[low_idx] = True
                self._yield_target_x[low_idx] = yield_x
                return

        for gantry_index in self.ACTIVE_GANTRY_INDICES:
            if self._yielding[gantry_index]:
                self._clear_yield(gantry_index, joint_position)

    def _gantry_priority(self, gantry_index: int, env_state_action_dict: dict) -> tuple[int, int]:
        task_record_index = self.state["ongoing_task_record_index"][gantry_index]
        if task_record_index is None:
            return (1, gantry_index)
        task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
        start_step = task_record.get("task_start_time_step")
        if start_step is None:
            start_step = 1_000_000_000
        return (0, int(start_step))

    def _sync_invalid_gantries(self, joint_position: torch.Tensor) -> None:
        animation = self.animation_num07_gantry_group
        for gantry_index, parking_xy in self.INVALID_GANTRY_PARKING_XY.items():
            self.state["state"][gantry_index] = "invalid"
            self.state["ongoing_task_record_index"][gantry_index] = None
            parking_xy = parking_xy.to(self.cuda_device)
            current_x = self._joint_to_world_x(joint_position, gantry_index)
            if abs(current_x - float(parking_xy[0].item())) < 0.05 and animation.is_done(gantry_index=gantry_index):
                continue
            parked_pose = self._get_joint_pose_from_xy_target(
                joint_position.clone(), parking_xy, gantry_index=gantry_index
            )
            animation.sync_gantry_pose(parked_pose, gantry_index=gantry_index)
            gantry_mask = self.gantry_indexs == gantry_index
            joint_position[gantry_mask] = parked_pose[gantry_mask]

    def step(self, env_state_action_dict):
        joint_position = env_state_action_dict["articulations"]["num07_gantry_group"]["joint_position"]
        self._sync_invalid_gantries(joint_position)

        for gantry_index in self.ACTIVE_GANTRY_INDICES:
            task_record_index = self.state["ongoing_task_record_index"][gantry_index]
            if task_record_index is None:
                continue

            task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
            assert task_record["chosen_gantry_index"] == gantry_index
            if self.state["state"][gantry_index] == "free":
                self.state["state"][gantry_index] = "working_" + task_record["task"]

            subtasks = task_record["subtasks_dict"]
            gantry_subtask = subtasks["ongoing"][1]
            if gantry_subtask == "go_to_material":
                self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, gantry_index, "start")
            elif gantry_subtask == "go_to_processing_machine":
                self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, gantry_index, "start")
            elif gantry_subtask == "wait":
                subtasks["finished"][1] = True
            elif gantry_subtask == "carry_to_robot":
                self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, gantry_index, "robot_start")
            elif gantry_subtask == "carry_to_goal_area":
                self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, gantry_index, "goal")
            elif gantry_subtask == "move_to_goal_area":
                self._subtask_go_to_target(env_state_action_dict, task_record, subtasks, gantry_index, "goal")
            elif gantry_subtask == "done":
                self._task_done(env_state_action_dict, task_record, gantry_index, joint_position)
            else:
                raise ValueError(f"Invalid gantry subtask for logistic: {gantry_subtask}")

        self._update_yield(joint_position, env_state_action_dict)
        priority_fn = lambda gantry_index: self._gantry_priority(gantry_index, env_state_action_dict)
        world_x_fn = lambda gantry_index, pose: self._joint_to_world_x(pose, gantry_index)
        env_state_action_dict["articulations"]["num07_gantry_group"]["joint_position"] = (
            self.animation_num07_gantry_group.step_next_pose(
                joint_position,
                self.ACTIVE_GANTRY_INDICES,
                self.safe_x_gap,
                world_x_fn,
                priority_fn,
            )
        )
        return env_state_action_dict

    def _subtask_go_to_target(
        self,
        env_state_action_dict: dict,
        task_record: dict,
        subtasks: dict,
        gantry_index: int,
        target_area_type: str,
    ) -> None:
        if subtasks["finished"][1]:
            return

        if self.state["target_area_id"][gantry_index] is None:
            if target_area_type == "start":
                self.state["target_area_id"][gantry_index] = task_record["subtasks_dict"]["start_area_ids"][
                    "gantry_parking_areas_ids"
                ][0]
            elif target_area_type == "robot_start":
                self.state["target_area_id"][gantry_index] = task_record["subtasks_dict"]["start_area_ids"][
                    "robot_parking_areas_ids"
                ][0]
            elif target_area_type == "goal":
                self.state["target_area_id"][gantry_index] = task_record["subtasks_dict"]["goal_area_ids"][
                    "gantry_parking_areas_ids"
                ][0]
            else:
                raise ValueError(f"Invalid target area type: {target_area_type}")

        if self.state["target_joints_position"][gantry_index] is None:
            if self.state["target_area_xy"][gantry_index] is None:
                return
            if self._yielding[gantry_index]:
                return
            joint_position = env_state_action_dict["articulations"]["num07_gantry_group"]["joint_position"]
            self.state["target_joints_position"][gantry_index] = self._get_joint_pose_from_xy_target(
                joint_position.clone(),
                self.state["target_area_xy"][gantry_index],
                gantry_index=gantry_index,
            )
            self.animation_num07_gantry_group.set_target_pose(
                self.state["target_joints_position"][gantry_index],
                gantry_index=gantry_index,
                current_joint_position=joint_position,
            )
            return

        if self.animation_num07_gantry_group.is_done(gantry_index=gantry_index):
            subtasks["finished"][1] = True
            self.state["target_area_id"][gantry_index] = None
            self.state["target_area_xy"][gantry_index] = None
            self.state["target_joints_position"][gantry_index] = None

    def _get_joint_pose_from_xy_target(
        self, joint_position: torch.Tensor, xy_target: torch.Tensor, gantry_index: int
    ) -> torch.Tensor:
        reset = self.joint_position_reset[self.gantry_indexs == gantry_index]
        xy_reset = self.xy_position_reset[self.gantry_indexs == gantry_index]
        target = reset + (xy_target - xy_reset)
        joint_position[self.gantry_indexs == gantry_index] = target
        return joint_position

    def _task_done(
        self,
        env_state_action_dict: dict,
        task_record: dict,
        chosen_gantry_index: int,
        joint_position: torch.Tensor,
    ) -> None:
        task_record["subtasks_dict"]["finished"][1] = True
        self.state["state"][chosen_gantry_index] = "free"
        self.state["ongoing_task_record_index"][chosen_gantry_index] = None
        self.state["target_area_id"][chosen_gantry_index] = None
        self.state["target_area_xy"][chosen_gantry_index] = None
        self.state["target_joints_position"][chosen_gantry_index] = None
        self._yielding[chosen_gantry_index] = False
        self._yield_target_x[chosen_gantry_index] = None
        home_xy = torch.tensor(
            [
                float(self.xy_position_reset[chosen_gantry_index].item()),
                float(self.xy_position_reset[self.num_workstations + chosen_gantry_index].item()),
            ],
            dtype=torch.float32,
            device=self.cuda_device,
        )
        home_pose = self._get_joint_pose_from_xy_target(
            joint_position.clone(), home_xy, gantry_index=chosen_gantry_index
        )
        self.animation_num07_gantry_group.set_target_pose(
            home_pose, gantry_index=chosen_gantry_index, current_joint_position=joint_position
        )
