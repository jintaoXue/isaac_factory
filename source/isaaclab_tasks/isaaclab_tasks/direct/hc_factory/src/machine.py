from isaacsim.core.prims import Articulation
from ..env_asset_cfg.cfg_machine import CfgMachine
from .utils import PoseAnimation


class Machine:
    def __init__(self, name: str, cfg_machine: CfgMachine, env_id: int):
        self.env_id = env_id
        self.name = name
        self.registration_type = cfg_machine["registration_type"]
        self.num_workstations = cfg_machine["num_workstations"]
        self.num_registration_parts = cfg_machine["num_registration_parts"]
        self.registration_infos = cfg_machine["registration_infos"]
        
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


class num01_rotaryPipeAutomaticWeldingMachine(Machine):

    def __init__(self, env_id: int):        
        super().__init__(name="num01_rotaryPipeAutomaticWeldingMachine", cfg_machine=CfgMachine["num01_rotaryPipeAutomaticWeldingMachine"], env_id=env_id)
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg_machine.py 的 registeration_infos_combined keys
        self.num01_rotaryPipeAutomaticWeldingMachine_part_01_station = None
        self.animation_num01_rotaryPipeAutomaticWeldingMachine_part_01_station: PoseAnimation = None
        self.num01_rotaryPipeAutomaticWeldingMachine_part_02_station = None
        self.animation_num01_rotaryPipeAutomaticWeldingMachine_part_02_station: PoseAnimation = None
        self._set_up_articulation()


class num02_weldingRobot(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num02_weldingRobot", cfg_machine=CfgMachine["num02_weldingRobot"], env_id=env_id)
        self.num02_weldingRobot_part02_robot_arm_and_base = None
        self.animation_num02_weldingRobot_part02_robot_arm_and_base: PoseAnimation = None
        self.num02_weldingRobot_part04_mobile_base_for_material = None
        self.animation_num02_weldingRobot_part04_mobile_base_for_material: PoseAnimation = None
        self._set_up_articulation()


class num03_rollerbedCNCPipeIntersectionCuttingMachine(Machine):

    def __init__(self, env_id: int):
        super().__init__(
            name="num03_rollerbedCNCPipeIntersectionCuttingMachine",
            cfg_machine=CfgMachine["num03_rollerbedCNCPipeIntersectionCuttingMachine"],
            env_id=env_id,
        )
        self.num03_rollerbedCNCPipeIntersectionCuttingMachine_part01_station = None
        self.animation_num03_rollerbedCNCPipeIntersectionCuttingMachine_part01_station: PoseAnimation = None
        self.num03_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine = None
        self.animation_num03_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine: PoseAnimation = None
        self._set_up_articulation()


class num04_laserCuttingMachine(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num04_laserCuttingMachine", cfg_machine=CfgMachine["num04_laserCuttingMachine"], env_id=env_id)
        self.num04_laserCuttingMachine = None
        self.animation_num04_laserCuttingMachine: PoseAnimation = None
        self._set_up_articulation()


class num05_groovingMachineLarge(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num05_groovingMachineLarge", cfg_machine=CfgMachine["num05_groovingMachineLarge"], env_id=env_id)
        self.num05_groovingMachineLarge_part01_large_fixed_base = None
        self.animation_num05_groovingMachineLarge_part01_large_fixed_base: PoseAnimation = None
        self.num05_groovingMachineLarge_part02_large_mobile_base = None
        self.animation_num05_groovingMachineLarge_part02_large_mobile_base: PoseAnimation = None
        self._set_up_articulation()


class num06_groovingMachineSmall(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num06_groovingMachineSmall", cfg_machine=CfgMachine["num06_groovingMachineSmall"], env_id=env_id)
        self.num06_groovingMachineSmall_part01_small_fixed_base = None
        self.animation_num06_groovingMachineSmall_part01_small_fixed_base: PoseAnimation = None
        self.num06_groovingMachineSmall_part02_small_mobile_handle = None
        self.animation_num06_groovingMachineSmall_part02_small_mobile_handle: PoseAnimation = None
        self._set_up_articulation()


class num07_highPressureFoamingMachine(Machine):

    def __init__(self, env_id: int):
        super().__init__(
            name="num07_highPressureFoamingMachine",
            cfg_machine=CfgMachine["num07_highPressureFoamingMachine"],
            env_id=env_id,
        )
        self.num07_highPressureFoamingMachine = None
        self.animation_num07_highPressureFoamingMachine: PoseAnimation = None
        self._set_up_articulation()


class num08_gantry_group(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num08_gantry_group", cfg_machine=CfgMachine["num08_gantry_group"], env_id=env_id)
        self.num08_gantry_group = None
        self.animation_num08_gantry_group: PoseAnimation = None
        self._set_up_articulation()


class num09_workbench(Machine):

    def __init__(self, env_id: int):
        super().__init__(name="num09_workbench", cfg_machine=CfgMachine["num09_workbench"], env_id=env_id)
        self.num09_workbench = None
        self.animation_num09_workbench: PoseAnimation = None
        self._set_up_articulation()
