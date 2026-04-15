from isaacsim.core.prims import Articulation
from ..cfgs.cfg_machine import CfgMachines
from .utils import PoseAnimation
class Machine:
    def __init__(self, name: str, cfg_machine: dict):
        self.name = name
        self.registration_type = cfg_machine["registration_type"]
        self.num_workstations = cfg_machine["num_workstations"]
        self.num_registration_parts = cfg_machine["num_registration_parts"]
        self.registration_infos = cfg_machine["registration_infos"]
        

class num01_rotaryPipeAutomaticWeldingMachine(Machine):
    def __init__(self):        
        super().__init__(name="num01_rotaryPipeAutomaticWeldingMachine", cfg_machine=CfgMachines["num01_rotaryPipeAutomaticWeldingMachine"])
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg_machine.py 的 registeration_infos_combined keys
        self.num01_rotaryPipeAutomaticWeldingMachine_part_01_station = None
        self.animation_num01_rotaryPipeAutomaticWeldingMachine_part_01_station: PoseAnimation = None
        self.num01_rotaryPipeAutomaticWeldingMachine_part_02_station = None
        self.animation_num01_rotaryPipeAutomaticWeldingMachine_part_02_station: PoseAnimation = None
        # 再根据配置创建 Articulation
        for obj_name, info in combined.items():
            articulation = Articulation(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id),
                name=obj_name,
                reset_xform_properties=bool(info.get("reset_xform_properties", False)),
            )
            setattr(self, obj_name, articulation)

            setattr(self, f"animation_{obj_name}", PoseAnimation(
                start_pose=info["joint_positions_reset"],
                end_pose=info["joint_positions_reset"],
                time=info["animation_time"],
            ))