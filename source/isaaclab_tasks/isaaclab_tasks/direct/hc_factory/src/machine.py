from isaacsim.core.prims import Articulation
from abc import abstractmethod
from ..env_asset_cfg.cfg_machine import CfgMachine
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll
from .utils import GantryGroupAnimation, PoseAnimation
from .disturbance import machine_process_succeeded, sample_machine_process_time, qc_hold_extra_steps
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
        # self.update_task_availability_mask(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for machine in self.iter_machines() + self.iter_logistic_machines():
            machine.step(env_state_action_dict)
        # self.update_task_availability_mask(env_state_action_dict)
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
            rail_busy = any(state[i] not in ("free", "invalid") for i in active_gantry_indices)
            if rail_busy:
                break
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
        ## processing the material on the machine (target sampled once per process with optional noise)
        targets = self.state.setdefault(
            "processing_time_target", [None] * self.num_workstations
        )
        if targets[chosen_workstation_index] is None:
            if self.type_name != "num08_workbench":
                info = self.registration_infos[chosen_machine_workstation]
            else:
                info = self.registration_infos["num08_workbench"]
            base = float(info["animation_time"])
            per_std = float(info.get("animation_time_noise_std", 0.0) or 0.0)
            t = int(env_state_action_dict.get("time_step", 0) or 0)
            targets[chosen_workstation_index] = sample_machine_process_time(
                base, per_std, time_step=t
            )
        self.state["processing_time_step"][chosen_workstation_index] += 1
        animation_time = targets[chosen_workstation_index]
        if self.state["processing_time_step"][chosen_workstation_index] >= animation_time:
            if not task_record.get("_qc_hold_applied"):
                extra = qc_hold_extra_steps(
                    self.type_name,
                    task_record.get("task") or "",
                    int(env_state_action_dict.get("time_step", 0) or 0),
                )
                if extra > 0:
                    task_record["_qc_hold_applied"] = True
                    targets[chosen_workstation_index] = int(animation_time) + extra
                    return
            if machine_process_succeeded():
                subtasks["finished"][2] = True
                self.state["processing_time_step"][chosen_workstation_index] = 0
                targets[chosen_workstation_index] = None
            else:
                # Rework: re-sample a longer remaining process window.
                self.state["processing_time_step"][chosen_workstation_index] = 0
                if self.type_name != "num08_workbench":
                    info = self.registration_infos[chosen_machine_workstation]
                else:
                    info = self.registration_infos["num08_workbench"]
                base = float(info["animation_time"])
                per_std = float(info.get("animation_time_noise_std", 0.0) or 0.0)
                t = int(env_state_action_dict.get("time_step", 0) or 0)
                targets[chosen_workstation_index] = sample_machine_process_time(
                    base, per_std, time_step=t
                )
    
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
    TRAVEL_SUBTASKS = frozenset(
        {
            "go_to_material",
            "go_to_processing_machine",
            "carry_to_robot",
            "carry_to_goal_area",
            "move_to_goal_area",
        }
    )
    INVALID_GANTRY_PARKING_XY = {
        2: torch.tensor([-46.0, 10.18675]),
        3: torch.tensor([-42.0, 10.18675]),
    }
    # Logic steps a lower-priority gantry may stay in yield before forced unlock.
    YIELD_TIMEOUT_STEPS = 400
    # Passed to GantryGroupAnimation.step_next_pose for mutual safe_x_gap deadlocks.
    BLOCK_TIMEOUT_STEPS = 400

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
        self._yield_steps = [0] * self.num_workstations
        self._locked_siding_x: list[float | None] = [None] * self.num_workstations

    def reset(self, env_state_action_dict: dict) -> dict:
        self._yielding = [False] * self.num_workstations
        self._yield_target_x = [None] * self.num_workstations
        self._yield_steps = [0] * self.num_workstations
        self._locked_siding_x = [None] * self.num_workstations
        if self.animation_num07_gantry_group is not None:
            self.animation_num07_gantry_group.blocked_steps = [0] * self.num_workstations
        return super().reset(env_state_action_dict)

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
                    device=self.cuda_device,
                    num_gantrys=self.num_workstations,
                    move_speed=info["move_speed"],
                    move_dt=info.get("move_dt", 1.0),
                    loaded_speed_scale=info.get("loaded_speed_scale", 0.5),
                    move_speed_noise_std=info.get("move_speed_noise_std", 0.0),
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

    def _home_world_xy(self, gantry_index: int) -> tuple[float, float]:
        return (
            float(self.xy_position_reset[gantry_index].item()),
            float(self.xy_position_reset[self.num_workstations + gantry_index].item()),
        )

    def _forbidden_corridor(self, joint_position: torch.Tensor, gantry_index: int) -> tuple[float, float] | None:
        """X interval the gantry occupies or will traverse, including the task target.

        Must include ``target_area_xy`` / ``target_joints_position`` so an idle
        gantry can clear ``carry_to_robot`` before the working animation starts.
        """
        current_x = self._joint_to_world_x(joint_position, gantry_index)
        xs = [current_x]
        animation = self.animation_num07_gantry_group
        if animation is not None and not animation.is_done(gantry_index=gantry_index):
            xs.append(self._joint_to_world_x(animation.end_pose, gantry_index))
        target_joints = self.state["target_joints_position"][gantry_index]
        if target_joints is not None:
            xs.append(self._joint_to_world_x(target_joints, gantry_index))
        target_xy = self.state["target_area_xy"][gantry_index]
        if target_xy is not None:
            xs.append(float(target_xy[0].item()))
        lo, hi = min(xs), max(xs)
        if abs(hi - lo) < 0.05:
            return lo, lo
        return lo, hi

    def _corridor_clear_low_x(self, corridor: tuple[float, float]) -> float:
        lo, _ = corridor
        return lo - 2.0 * self.safe_x_gap

    def _corridor_clear_high_x(self, corridor: tuple[float, float]) -> float:
        _, hi = corridor
        return hi + 2.0 * self.safe_x_gap

    def _x_blocks_corridor(
        self, world_x: float, corridor: tuple[float, float], margin: float | None = None
    ) -> bool:
        lo, hi = corridor
        gap = self.safe_x_gap if margin is None else margin
        return (lo - gap) <= world_x <= (hi + gap)

    def _x_clears_corridor(
        self, world_x: float, corridor: tuple[float, float], margin: float | None = None
    ) -> bool:
        return not self._x_blocks_corridor(world_x, corridor, margin=margin)

    def _siding_margin(self) -> float:
        """Park/yield stay-put margin. 1×gap treats gantry home (36.7) as clear of AGV drop (32.4)."""
        return 2.0 * self.safe_x_gap

    def _task_world_x(self, gantry_index: int) -> float | None:
        target_xy = self.state["target_area_xy"][gantry_index]
        if target_xy is not None:
            return float(target_xy[0].item())
        target_joints = self.state["target_joints_position"][gantry_index]
        if target_joints is not None:
            return self._joint_to_world_x(target_joints, gantry_index)
        return None

    def _segment_hits_corridor(self, x0: float, x1: float, corridor: tuple[float, float]) -> bool:
        lo, hi = min(x0, x1), max(x0, x1)
        c_lo, c_hi = corridor
        return lo <= (c_hi + self.safe_x_gap) and hi >= (c_lo - self.safe_x_gap)

    def _resume_conflicts_with_peers(self, joint_position: torch.Tensor, gantry_index: int) -> bool:
        """True if leaving yield would drive this gantry back into a tasked peer's corridor."""
        task_x = self._task_world_x(gantry_index)
        current_x = self._joint_to_world_x(joint_position, gantry_index)
        for other_index in self.ACTIVE_GANTRY_INDICES:
            if other_index == gantry_index:
                continue
            if self.state["ongoing_task_record_index"][other_index] is None:
                continue
            corridor = self._forbidden_corridor(joint_position, other_index)
            if corridor is None:
                continue
            if self._x_blocks_corridor(current_x, corridor, margin=self._siding_margin()):
                return True
            if task_x is not None and self._segment_hits_corridor(current_x, task_x, corridor):
                return True
        return False

    def _hold_yield_at(
        self,
        gantry_index: int,
        yield_x: float,
        joint_position: torch.Tensor,
    ) -> None:
        animation = self.animation_num07_gantry_group
        locked_x = self._yield_target_x[gantry_index]
        if (
            self._yielding[gantry_index]
            and locked_x is not None
            and abs(locked_x - yield_x) < 0.05
        ):
            return
        current_y = self._joint_to_world_y(joint_position, gantry_index)
        yield_xy = torch.tensor([yield_x, current_y], dtype=torch.float32, device=self.cuda_device)
        yield_pose = self._get_joint_pose_from_xy_target(
            joint_position.clone(), yield_xy, gantry_index=gantry_index
        )
        animation.set_yield_target_pose(
            yield_pose, gantry_index=gantry_index, current_joint_position=joint_position
        )
        self._yielding[gantry_index] = True
        self._yield_target_x[gantry_index] = yield_x

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
        self._yield_steps[gantry_index] = 0
        animation = self.animation_num07_gantry_group
        animation.is_yield_move[gantry_index] = False
        task_target = self.state["target_joints_position"][gantry_index]
        if task_target is not None:
            animation.set_target_pose(
                task_target,
                gantry_index=gantry_index,
                current_joint_position=joint_position,
                loaded=animation.move_loaded[gantry_index],
            )

    def _force_clear_yield(self, gantry_index: int, joint_position: torch.Tensor, reason: str) -> None:
        """Timeout unlock: stop endless yield so the gantry can resume its task target."""
        if not self._yielding[gantry_index]:
            self._yield_steps[gantry_index] = 0
            return
        print(
            f"[GantryUnlock] clear yield gantry_{gantry_index} after "
            f"{self._yield_steps[gantry_index]} steps ({reason})"
        )
        self._clear_yield(gantry_index, joint_position)

    def _resolve_yield_timeouts(self, joint_position: torch.Tensor) -> None:
        for gantry_index in self.ACTIVE_GANTRY_INDICES:
            if not self._yielding[gantry_index]:
                self._yield_steps[gantry_index] = 0
                continue
            # Idle gantries are parked, not yielded; never timeout them back
            # into a working gantry's corridor.
            if self.state["ongoing_task_record_index"][gantry_index] is None:
                self._yield_steps[gantry_index] = 0
                continue
            # Working gantries must also stay aside while resume would re-enter
            # a peer's carry_to_robot / go_to_* corridor. Timeout-into-corridor
            # is the 2-crane deadlock (both tasked, one yields, 400 steps later
            # drives back in, neither completes, watchdog fires).
            if self._resume_conflicts_with_peers(joint_position, gantry_index):
                self._yield_steps[gantry_index] = 0
                continue
            self._yield_steps[gantry_index] += 1
            if self._yield_steps[gantry_index] >= self.YIELD_TIMEOUT_STEPS:
                self._force_clear_yield(
                    gantry_index, joint_position, reason="yield_timeout"
                )

    def _update_yield(self, joint_position: torch.Tensor, env_state_action_dict: dict) -> None:
        """Shared rail uses one-worker mutex; do not sidestep/timeout into each other."""
        del joint_position, env_state_action_dict
        return

    def _gantry_priority(self, gantry_index: int, env_state_action_dict: dict) -> tuple[int, int]:
        task_record_index = self.state["ongoing_task_record_index"][gantry_index]
        if task_record_index is None:
            return (1, gantry_index)
        task_record = env_state_action_dict["progress"]["ongoing_task_records"][task_record_index]
        start_step = task_record.get("task_start_time_step")
        if start_step is None:
            start_step = 1_000_000_000
        return (0, int(start_step))

    def _current_gantry_subtask(self, env_state_action_dict: dict, gantry_index: int) -> str | None:
        rid = self.state["ongoing_task_record_index"][gantry_index]
        if rid is None:
            return None
        rec = (env_state_action_dict.get("progress") or {}).get("ongoing_task_records") or {}
        task_record = rec.get(rid)
        if task_record is None:
            return None
        ongoing = (task_record.get("subtasks_dict") or {}).get("ongoing") or []
        return ongoing[1] if len(ongoing) > 1 else None

    def _rail_owner_index(self, env_state_action_dict: dict) -> int | None:
        """Highest-priority gantry currently in a travel subtask; others must wait."""
        traveling: list[tuple[tuple[int, int], int]] = []
        for gantry_index in self.ACTIVE_GANTRY_INDICES:
            if self._current_gantry_subtask(env_state_action_dict, gantry_index) not in self.TRAVEL_SUBTASKS:
                continue
            traveling.append((self._gantry_priority(gantry_index, env_state_action_dict), gantry_index))
        if not traveling:
            return None
        traveling.sort()
        return traveling[0][1]

    def _pause_non_owners(self, env_state_action_dict: dict, joint_position: torch.Tensor) -> None:
        """Only the rail owner may keep a travel target; others drop it and yield."""
        owner = self._rail_owner_index(env_state_action_dict)
        if owner is not None and self._yielding[owner]:
            self._clear_yield(owner, joint_position)
        for gantry_index in self.ACTIVE_GANTRY_INDICES:
            if owner is None or gantry_index == owner:
                continue
            if self._current_gantry_subtask(env_state_action_dict, gantry_index) not in self.TRAVEL_SUBTASKS:
                continue
            self.state["target_area_id"][gantry_index] = None
            self.state["target_area_xy"][gantry_index] = None
            self.state["target_joints_position"][gantry_index] = None

    def _idle_blocked_at(
        self,
        joint_position: torch.Tensor,
        gantry_index: int,
        world_x: float,
        margin: float,
    ) -> bool:
        for other_index in self.ACTIVE_GANTRY_INDICES:
            if other_index == gantry_index:
                continue
            if self.state["ongoing_task_record_index"][other_index] is None:
                continue
            corridor = self._forbidden_corridor(joint_position, other_index)
            if corridor is None:
                continue
            if self._x_blocks_corridor(world_x, corridor, margin=margin):
                return True
        return False

    def _gantry_rail_side(self, gantry_index: int) -> str:
        homes = [self._home_world_xy(i)[0] for i in self.ACTIVE_GANTRY_INDICES]
        home_x, _ = self._home_world_xy(gantry_index)
        mid = sum(homes) / max(len(homes), 1)
        return "east" if home_x >= mid else "west"

    def _worker_target_xs(self) -> list[float]:
        """Task destinations only. Do not track current_x — that made idle siding chase the worker."""
        xs: list[float] = []
        for other_index in self.ACTIVE_GANTRY_INDICES:
            if self.state["ongoing_task_record_index"][other_index] is None:
                continue
            task_x = self._task_world_x(other_index)
            if task_x is not None:
                xs.append(task_x)
        return xs

    def _dedicated_siding_x(
        self, gantry_index: int, joint_position: torch.Tensor | None = None
    ) -> float:
        """Park outside worker *destinations*, then only push further out.

        Cutting drop x≈13.1; grooving drop x≈-4.8. Following current_x made the
        west siding crawl east during carry_to_robot and reprint every step.
        """
        del joint_position
        homes = [self._home_world_xy(i)[0] for i in self.ACTIVE_GANTRY_INDICES]
        home_x, _ = self._home_world_xy(gantry_index)
        if len(self.ACTIVE_GANTRY_INDICES) <= 1:
            return home_x
        margin = 2.5 * self.safe_x_gap
        side = self._gantry_rail_side(gantry_index)
        park_x = max(homes) + self._siding_margin() if side == "east" else min(homes) - margin
        targets = self._worker_target_xs()
        if targets:
            if side == "east":
                park_x = max(park_x, max(targets) + margin)
            else:
                park_x = min(park_x, min(targets) - margin)
        locked = self._locked_siding_x[gantry_index]
        if locked is not None:
            park_x = max(park_x, locked) if side == "east" else min(park_x, locked)
        self._locked_siding_x[gantry_index] = park_x
        return park_x

    def _idle_should_move_to_siding(
        self, joint_position: torch.Tensor, gantry_index: int, park_x: float
    ) -> bool:
        """Park unless that path would drive into a worker the idle crane is currently clear of.

        If the idle crane already sits in a worker corridor (home 16.7 vs drop 13.1),
        it must leave — that is the 2-crane freeze after carry_to_robot.
        """
        current_x = self._joint_to_world_x(joint_position, gantry_index)
        blocking_worker = False
        would_enter_worker = False
        for other_index in self.ACTIVE_GANTRY_INDICES:
            if other_index == gantry_index:
                continue
            if self.state["ongoing_task_record_index"][other_index] is None:
                continue
            corridor = self._forbidden_corridor(joint_position, other_index)
            if corridor is None:
                continue
            if self._x_blocks_corridor(current_x, corridor, margin=self.safe_x_gap):
                blocking_worker = True
            elif self._segment_hits_corridor(current_x, park_x, corridor):
                would_enter_worker = True
        if blocking_worker:
            return True
        if would_enter_worker:
            return False
        return True

    def _idle_park_x(self, joint_position: torch.Tensor, gantry_index: int) -> float:
        return self._dedicated_siding_x(gantry_index, joint_position)

    def _clear_idle_yield_flags(self, gantry_index: int) -> None:
        self._yielding[gantry_index] = False
        self._yield_target_x[gantry_index] = None
        self._yield_steps[gantry_index] = 0
        animation = self.animation_num07_gantry_group
        if animation is not None:
            animation.is_yield_move[gantry_index] = False

    def _park_idle_gantries(self, joint_position: torch.Tensor) -> None:
        """Park untasked cranes at dedicated rail ends. Stay invalid if L2 froze them.

        Free idle cranes must also go to their siding: leaving gantry_1 at home
        x≈16.7 blocks gantry_0 from the cutting-machine drop at x≈13.1.
        Do not drive a clear idle crane *into* a worker corridor.
        """
        animation = self.animation_num07_gantry_group
        if animation is None:
            return
        for gantry_index in self.ACTIVE_GANTRY_INDICES:
            if gantry_index in self.INVALID_GANTRY_PARKING_XY:
                continue
            if self.state["state"][gantry_index] == "invalid":
                continue
            if self.state["ongoing_task_record_index"][gantry_index] is not None:
                continue
            state = self.state["state"][gantry_index]
            if state not in ("waiting_park", "free"):
                continue
            peer_working = any(
                other != gantry_index
                and self.state["ongoing_task_record_index"][other] is not None
                for other in self.ACTIVE_GANTRY_INDICES
            )
            # Stay at home until a peer is actually working. Parking both cranes
            # at reset sent gantry_0 east, then back west, for no clearance gain.
            if state == "free" and not peer_working:
                continue
            home_x, home_y = self._home_world_xy(gantry_index)
            current_x = self._joint_to_world_x(joint_position, gantry_index)
            park_x = self._dedicated_siding_x(gantry_index, joint_position)
            if state == "free" and not self._idle_should_move_to_siding(
                joint_position, gantry_index, park_x
            ):
                continue
            if abs(current_x - park_x) < 0.05 and animation.is_done(gantry_index=gantry_index):
                self.state["state"][gantry_index] = "free"
                if self._yielding[gantry_index]:
                    self._clear_idle_yield_flags(gantry_index)
                continue
            if not animation.is_done(gantry_index=gantry_index):
                dest_x = self._joint_to_world_x(animation.end_pose, gantry_index)
                if abs(dest_x - park_x) < 0.5:
                    continue
            if self._yielding[gantry_index]:
                self._clear_idle_yield_flags(gantry_index)
            park_xy = torch.tensor(
                [park_x, home_y], dtype=torch.float32, device=self.cuda_device
            )
            park_pose = self._get_joint_pose_from_xy_target(
                joint_position.clone(), park_xy, gantry_index=gantry_index
            )
            animation.set_target_pose(
                park_pose,
                gantry_index=gantry_index,
                current_joint_position=joint_position,
                loaded=False,
            )
            if abs(park_x - home_x) >= 0.05:
                print(
                    f"[GantryPark] gantry_{gantry_index} home={home_x:.2f} -> "
                    f"{park_x:.2f} (rail siding)"
                )

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
        # Unlock stale yields before assigning new task targets this step.
        self._resolve_yield_timeouts(joint_position)
        self._pause_non_owners(env_state_action_dict, joint_position)

        for gantry_index in self.ACTIVE_GANTRY_INDICES:
            task_record_index = self.state["ongoing_task_record_index"][gantry_index]
            if task_record_index is None:
                continue
            self._locked_siding_x[gantry_index] = None

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

        self._park_idle_gantries(joint_position)
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
                block_timeout=self.BLOCK_TIMEOUT_STEPS,
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

        owner = self._rail_owner_index(env_state_action_dict)
        if owner is not None and owner != gantry_index:
            # Shared rail: only one travel at a time. Stay parked until we own it.
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
            # Only pause target acquisition while actively yielding; after yield timeout
            # _resolve_yield_timeouts clears the flag so we can proceed.
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
                loaded=self._gantry_subtask_loaded(subtasks["ongoing"][1]),
            )
            return

        if self.animation_num07_gantry_group.is_done(gantry_index=gantry_index):
            # Ignore completion of a pure yield sidestep; wait for real task animation.
            if self._yielding[gantry_index] or self.animation_num07_gantry_group.is_yield_move[gantry_index]:
                return
            subtasks["finished"][1] = True
            self.state["target_area_id"][gantry_index] = None
            self.state["target_area_xy"][gantry_index] = None
            self.state["target_joints_position"][gantry_index] = None

    @staticmethod
    def _gantry_subtask_loaded(gantry_subtask: str) -> bool:
        return gantry_subtask in ("carry_to_robot", "carry_to_goal_area")

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
        self.state["state"][chosen_gantry_index] = "waiting_park"
        self.state["ongoing_task_record_index"][chosen_gantry_index] = None
        self.state["target_area_id"][chosen_gantry_index] = None
        self.state["target_area_xy"][chosen_gantry_index] = None
        self.state["target_joints_position"][chosen_gantry_index] = None
        self._yielding[chosen_gantry_index] = False
        self._yield_target_x[chosen_gantry_index] = None
        self._yield_steps[chosen_gantry_index] = 0
        siding_x = self._dedicated_siding_x(chosen_gantry_index, joint_position)
        _, home_y = self._home_world_xy(chosen_gantry_index)
        siding_xy = torch.tensor(
            [siding_x, home_y],
            dtype=torch.float32,
            device=self.cuda_device,
        )
        siding_pose = self._get_joint_pose_from_xy_target(
            joint_position.clone(), siding_xy, gantry_index=chosen_gantry_index
        )
        self.animation_num07_gantry_group.set_target_pose(
            siding_pose,
            gantry_index=chosen_gantry_index,
            current_joint_position=joint_position,
            loaded=False,
        )
