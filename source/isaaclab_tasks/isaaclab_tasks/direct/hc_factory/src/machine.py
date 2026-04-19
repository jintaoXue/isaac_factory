from isaacsim.core.prims import Articulation
from abc import abstractmethod
from ..env_asset_cfg.cfg_machine import CfgMachine
from .utils import PoseAnimation



class MachineManager:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.cfg_machine = CfgMachine
        self.num00_rotaryPipeAutomaticWeldingMachine = num00_rotaryPipeAutomaticWeldingMachine(env_id=self.env_id)
        self.num01_weldingRobot = num01_weldingRobot(env_id=self.env_id)
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine = num02_rollerbedCNCPipeIntersectionCuttingMachine(env_id=self.env_id)
        self.num03_laserCuttingMachine = num03_laserCuttingMachine(env_id=self.env_id)
        self.num04_groovingMachineLarge = num04_groovingMachineLarge(env_id=self.env_id)
        self.num05_groovingMachineSmall = num05_groovingMachineSmall(env_id=self.env_id)
        self.num06_highPressureFoamingMachine = num06_highPressureFoamingMachine(env_id=self.env_id)
        self.num07_gantry_group = num07_gantry_group(env_id=self.env_id)
        self.num08_workbench = num08_workbench(env_id=self.env_id)

    def reset(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["machine_state"] = self
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["machine_state"] = self
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
    def __init__(self, cfg: dict, env_id: int):
        self.env_id = env_id
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.registration_type = cfg["registration_type"]
        self.num_workstations = cfg["num_workstations"]
        self.num_registration_parts = cfg["num_registration_parts"]
        self.registration_infos = cfg["registration_infos"]
        self.human_working_areas_ids = cfg["human_working_areas_ids"]
        self.robot_parking_areas_ids = cfg["robot_parking_areas_ids"]
        self.gantry_parking_areas_ids = cfg["gantry_parking_areas_ids"]
        self.state_set = cfg["state_set"]
    def _set_up_articulation(self):
        for obj_name, info in self.registration_infos.items():
            articulation = Articulation(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id),
                name=obj_name,
                reset_xform_properties=False,
            )
            setattr(self, obj_name, articulation)
            setattr(self, f"animation_{obj_name}", PoseAnimation(
                start_pose=info["joint_positions_reset"],
                end_pose=info["joint_positions_reset"],
                time=info["animation_time"],
            ))
    @abstractmethod
    def reset(self, env_state_action_dict: dict) -> dict:
        pass
    @abstractmethod
    def step(self, env_state_action_dict: dict) -> dict:
        pass

class num00_rotaryPipeAutomaticWeldingMachine(Machine):

    def __init__(self, env_id: int):        
        super().__init__(name="num00_rotaryPipeAutomaticWeldingMachine", cfg=CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"], env_id=env_id)
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg.py 的 registeration_infos_combined keys
        self.num00_rotaryPipeAutomaticWeldingMachine_part_01_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_01_station: PoseAnimation = None
        self.num00_rotaryPipeAutomaticWeldingMachine_part_02_station = None
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_02_station: PoseAnimation = None
        self._set_up_articulation()

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state_dict = {
            01_station: "free",
            02_station: "free",
        }
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_01_station
        self.animation_num00_rotaryPipeAutomaticWeldingMachine_part_02_station
        return env_state_action_dict


    def step(self, env_state_action_dict: dict) -> dict:
        return env_state_action_dict


class num01_weldingRobot(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num01_weldingRobot", cfg=CfgMachine["num01_weldingRobot"], env_id=env_id)
        self.num01_weldingRobot_part02_robot_arm_and_base = None
        self.animation_num01_weldingRobot_part02_robot_arm_and_base: PoseAnimation = None
        self.num01_weldingRobot_part04_mobile_base_for_material = None
        self.animation_num01_weldingRobot_part04_mobile_base_for_material: PoseAnimation = None
        self._set_up_articulation()


class num02_rollerbedCNCPipeIntersectionCuttingMachine(Machine):

    def __init__(self, env_id: int):
        super().__init__(
            name="num02_rollerbedCNCPipeIntersectionCuttingMachine",
            cfg=CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"],
            env_id=env_id,
        )
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station = None
        self.animation_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station: PoseAnimation = None
        self.num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine = None
        self.animation_num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine: PoseAnimation = None
        self._set_up_articulation()


class num03_laserCuttingMachine(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num03_laserCuttingMachine", cfg=CfgMachine["num03_laserCuttingMachine"], env_id=env_id)
        self.num03_laserCuttingMachine = None
        self.animation_num03_laserCuttingMachine: PoseAnimation = None
        self._set_up_articulation()


class num04_groovingMachineLarge(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num04_groovingMachineLarge", cfg=CfgMachine["num04_groovingMachineLarge"], env_id=env_id)
        self.num04_groovingMachineLarge_part01_large_fixed_base = None
        self.animation_num04_groovingMachineLarge_part01_large_fixed_base: PoseAnimation = None
        self.num04_groovingMachineLarge_part02_large_mobile_base = None
        self.animation_num04_groovingMachineLarge_part02_large_mobile_base: PoseAnimation = None
        self._set_up_articulation()


class num05_groovingMachineSmall(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num05_groovingMachineSmall", cfg=CfgMachine["num05_groovingMachineSmall"], env_id=env_id)
        self.num05_groovingMachineSmall_part01_small_fixed_base = None
        self.animation_num05_groovingMachineSmall_part01_small_fixed_base: PoseAnimation = None
        self.num05_groovingMachineSmall_part02_small_mobile_handle = None
        self.animation_num05_groovingMachineSmall_part02_small_mobile_handle: PoseAnimation = None
        self._set_up_articulation()


class num06_highPressureFoamingMachine(Machine):

    def __init__(self, env_id: int):
        super().__init__(
            name="num06_highPressureFoamingMachine",
            cfg=CfgMachine["num06_highPressureFoamingMachine"],
            env_id=env_id,
        )
        self.num06_highPressureFoamingMachine = None
        self.animation_num06_highPressureFoamingMachine: PoseAnimation = None
        self._set_up_articulation()


class num07_gantry_group(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num07_gantry_group", cfg=CfgMachine["num07_gantry_group"], env_id=env_id)
        self.num07_gantry_group = None
        self.animation_num07_gantry_group: PoseAnimation = None
        self._set_up_articulation()


class num08_workbench(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num08_workbench", cfg=CfgMachine["num08_workbench"], env_id=env_id)
        self.num08_workbench = None
        self.animation_num08_workbench: PoseAnimation = None
        self._set_up_articulation()
