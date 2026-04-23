from isaacsim.core.prims import Articulation
from abc import abstractmethod
from ..env_asset_cfg.cfg_machine import CfgMachine
from .utils import PoseAnimation
import torch

class MachineManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_machine = CfgMachine
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


class Machine:
    def __init__(self, cfg: dict, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.registration_type = cfg["registration_type"]
        self.num_workstations = cfg["num_workstations"]
        self.num_registration_parts = cfg["num_registration_parts"]
        self.registration_infos = cfg["registration_infos"]
        self.state_gallery = cfg["state_gallery"]
        self.reset_state = cfg["reset_state"]
        self.state : dict = None

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
        env_state_action_dict["state_machine"][self.type_name] = self.reset_state.copy()
        articulations_values = self.reset_articulations()
        env_state_action_dict["articulations"].update(articulations_values)
        return env_state_action_dict

    def reset_articulations(self) -> dict:
        articulations_values: dict = {}
        for obj_name in self.registration_infos.keys():
            articulation = getattr(self, obj_name, None)
            if articulation is None:
                continue
            joint_positions_reset : torch.Tensor = self.registration_infos[obj_name]["joint_positions_reset"].to(self.cuda_device)
            articulations_values[obj_name] = {
                "articulation": articulation,
                "joint_positions": joint_positions_reset,
            }
        return articulations_values

    @abstractmethod
    def step(self, env_state_action_dict: dict) -> dict:
        pass


class num00_rotaryPipeAutomaticWeldingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):        
        super().__init__(cfg=CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"], env_id=env_id, cuda_device=cuda_device)
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num00_rotaryPipeAutomaticWeldingMachine_part_01_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_01_station: PoseAnimation = None
        self.num00_rotaryPipeAutomaticWeldingMachine_part_02_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_02_station: PoseAnimation = None
        self._register_articulation_animation()

    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict


class num01_weldingRobot(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(cfg=CfgMachine["num01_weldingRobot"], env_id=env_id, cuda_device=cuda_device)
        self.num01_weldingRobot_part02_robot_arm_and_base = None
        self.animation_num01_weldingRobot_part02_robot_arm_and_base: PoseAnimation = None
        self.num01_weldingRobot_part04_mobile_base_for_material = None
        self.animation_num01_weldingRobot_part04_mobile_base_for_material: PoseAnimation = None
        self._register_articulation_animation()


class num02_rollerbedCNCPipeIntersectionCuttingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(
            cfg=CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"],
            env_id=env_id,
            cuda_device=cuda_device,
        )
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station = None
        self.animation_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station: PoseAnimation = None
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine = None
        self.animation_num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine: PoseAnimation = None
        self._register_articulation_animation()


class num03_laserCuttingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(cfg=CfgMachine["num03_laserCuttingMachine"], env_id=env_id, cuda_device=cuda_device)
        self.num03_laserCuttingMachine = None
        self.animation_num03_laserCuttingMachine: PoseAnimation = None
        self._register_articulation_animation()


class num04_groovingMachineLarge(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(cfg=CfgMachine["num04_groovingMachineLarge"], env_id=env_id, cuda_device=cuda_device)
        self.num04_groovingMachineLarge_part01_large_fixed_base = None
        self.animation_num04_groovingMachineLarge_part01_large_fixed_base: PoseAnimation = None
        self.num04_groovingMachineLarge_part02_large_mobile_base = None
        self.animation_num04_groovingMachineLarge_part02_large_mobile_base: PoseAnimation = None
        self._register_articulation_animation()


class num05_groovingMachineSmall(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(cfg=CfgMachine["num05_groovingMachineSmall"], env_id=env_id, cuda_device=cuda_device)
        self.num05_groovingMachineSmall_part01_small_fixed_base = None
        self.animation_num05_groovingMachineSmall_part01_small_fixed_base: PoseAnimation = None
        self.num05_groovingMachineSmall_part02_small_mobile_handle = None
        self.animation_num05_groovingMachineSmall_part02_small_mobile_handle: PoseAnimation = None
        self._register_articulation_animation()


class num06_highPressureFoamingMachine(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(
            cfg=CfgMachine["num06_highPressureFoamingMachine"],
            env_id=env_id,
            cuda_device=cuda_device,
        )
        self.num06_highPressureFoamingMachine = None
        self.animation_num06_highPressureFoamingMachine: PoseAnimation = None
        self._register_articulation_animation()


class num07_gantry_group(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(cfg=CfgMachine["num07_gantry_group"], env_id=env_id, cuda_device=cuda_device)
        self.num07_gantry_group = None
        self.animation_num07_gantry_group: PoseAnimation = None
        self._register_articulation_animation()


class num08_workbench(Machine):

    def __init__(self, env_id: int, cuda_device: torch.device):
        super().__init__(cfg=CfgMachine["num08_workbench"], env_id=env_id, cuda_device=cuda_device)
        self.num08_workbench = None
        self.animation_num08_workbench: PoseAnimation = None
        self._register_articulation_animation()
