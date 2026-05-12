from isaacsim.core.prims import Articulation
from abc import abstractmethod
from ..env_asset_cfg.cfg_machine import CfgMachine
from .utils import PoseAnimation
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
        for machine in self.iter_machines():
            machine.reset(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for machine in self.iter_machines():
            machine.step(env_state_action_dict)
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
            self.num07_gantry_group,
            self.num08_workbench,
        )

    def update_machine_availability_mask(self, env_state_action_dict: dict) -> dict:
        # mask for machine availability for selection by human-robot machine allocator agent
        mask = torch.zeros(self.num_machine, dtype=torch.int32, device=self.cuda_device)
        for machine, i in zip(self.iter_machines(), range(self.num_machine)):
            state : list = machine.state
            if "free" in state:
                mask[i] = 1
        env_state_action_dict["machine"]["availability_mask"] = mask
        return env_state_action_dict


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
        self.state_gallery = cfg["state_gallery"]
        self.reset_state = copy.deepcopy(cfg["reset_state"])
        ### dynmaic variables
        self.state : list = None
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

    @abstractmethod
    def step(self, env_state_action_dict: dict) -> dict:
        pass


class num00_rotaryPipeAutomaticWeldingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num00_rotaryPipeAutomaticWeldingMachine_part_01_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_01_station: PoseAnimation = None
        self.num00_rotaryPipeAutomaticWeldingMachine_part_02_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_02_station: PoseAnimation = None        
        super().__init__(cfg=CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"], env_id=env_id, cuda_device=cuda_device)

    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict


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


class num07_gantry_group(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num07_gantry_group = None
        self.animation_num07_gantry_group: PoseAnimation = None
        super().__init__(cfg=CfgMachine["num07_gantry_group"], env_id=env_id, cuda_device=cuda_device)
        # shape 8x1 tensor and joint_position has 8 elements: [x0, x1, x2, x3, y0, y1, y2, y3] for 4 subgantrys
        self.joint_position_reset = self.registration_infos["num07_gantry_group"]["joint_positions_reset"].to(self.cuda_device)
        self.xy_position_reset = self.registration_infos["num07_gantry_group"]["xy_position_reset"].to(self.cuda_device)
        #"gantry_indexs": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]), number 4 subgantrys in total, each subgantry has a gantry and a hook, the gantry moves in xy plane and the hook moves in z axis, the subgantry indexs indicate which subgantry each joint belongs to
        self.gantry_indexs = self.registration_infos["num07_gantry_group"]["gantry_indexs"]
        self.fixed_hook_height = self.registration_infos["num07_gantry_group"]["fixed_hook_height"]

    def step(self, env_state_action_dict):
        joint_position = env_state_action_dict["articulations"]["num07_gantry_group"]["joint_position"]
        xyz_1 : list = [-50.45092570343084, 10.18675, 0.5]
        xyz_2 : list = [-46, 10.18675, 0.5]
        xyz_3 : list = [-42, 10.18675, 0.5]

        target_joint_position_1 = self.get_joint_pose_from_xy_target(joint_position, xyz_1[:2], gantry_index = 1)
        target_joint_position_2 = self.get_joint_pose_from_xy_target(joint_position, xyz_2[:2], gantry_index = 2)
        target_joint_position_3 = self.get_joint_pose_from_xy_target(joint_position, xyz_3[:2], gantry_index = 3)

        final_target_joint_position = joint_position.clone()
        final_target_joint_position[self.gantry_indexs == 1] = target_joint_position_1[self.gantry_indexs == 1]
        final_target_joint_position[self.gantry_indexs == 2] = target_joint_position_2[self.gantry_indexs == 2]
        final_target_joint_position[self.gantry_indexs == 3] = target_joint_position_3[self.gantry_indexs == 3]
        env_state_action_dict["articulations"]["num07_gantry_group"]["joint_position"] = final_target_joint_position
        return
    
    def get_joint_pose_from_xy_target(self, joint_position: torch.Tensor, position_xy: list, gantry_index: int) -> torch.Tensor:
        # compute the target joint position for the gantry, with specified target xy position and subgantry index
        # the subgantry index indicates which subgantry the target position belongs to, and the gantry will move in xy plane to reach the target position, while the hook will keep a fixed height
        target_joint_position = joint_position.clone()
        # Assume joint_position has 8 elements: [x0, x1, x2, x3, y0, y1, y2, y3] for 4 subgantrys
        # where x0 is x for subgantry 0, y0 is y for subgantry 0, etc.
        # For the selected subgantry, update x and y joints to reach position_xy
        x_index = gantry_index
        y_index = gantry_index + 4
        # Calculate offset from reset position
        reset_x = self.xy_position_reset[x_index]
        reset_y = self.xy_position_reset[y_index]
        target_x = position_xy[0]
        target_y = position_xy[1]
        # Update joints (assuming linear mapping, adjust as needed for actual kinematics)
        target_joint_position[x_index] = self.joint_position_reset[x_index] + (target_x - reset_x)
        target_joint_position[y_index] = self.joint_position_reset[y_index] + (target_y - reset_y)
        # Hook height remains fixed, so no change to z joints if any
        return target_joint_position
    
    def gantrys_collision_check(self, joint_position: torch.Tensor) -> bool:
        # Implement collision check logic for the gantry group based on joint positions
        # x0 > x1 > x2 > x3 to avoid collision between gantrys, adjust the threshold as needed
        return False

class num08_workbench(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num08_workbench = None
        self.animation_num08_workbench: PoseAnimation = None
        super().__init__(cfg=CfgMachine["num08_workbench"], env_id=env_id, cuda_device=cuda_device)
