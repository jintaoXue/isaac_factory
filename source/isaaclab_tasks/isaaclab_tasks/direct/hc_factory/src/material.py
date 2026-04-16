from isaacsim.core.prims import RigidPrim
from ..cfgs.cfg_material_product import CfgProductProcess, CfgProductionOrder


class MaterialManager:
    def __init__(self, name: str, cfg_material: CfgProductProcess, env_id: int):
        self.env_id = env_id
        self.name = name
        self.registration_type = cfg_material["registration_type"]
        self.num_workstations = cfg_material["num_workstations"]
        self.num_registration_parts = cfg_material["num_registration_parts"]
        self.registration_infos = cfg_material["registration_infos"]
        
    def _set_up_articulation(self):
        for obj_name, info in self.registration_infos.items():
            rigid_prim = RigidPrim(
                prim_paths_expr=info["prim_paths_expr"].format(i=self.env_id),
                name=obj_name,
                reset_xform_properties=False,
            )
            setattr(self, obj_name, rigid_prim)


class num01_rotaryPipeAutomaticWeldingMachine(Material):

    def __init__(self, env_id: int):        
        super().__init__(name="num01_rotaryPipeAutomaticWeldingMachine", cfg_machine=CfgMachines["num01_rotaryPipeAutomaticWeldingMachine"], env_id=env_id)
        # ===== 显式声明（更直观：一眼能看到有哪些对象会挂到 self 上）=====
        # 这些名称来自 cfg_machine.py 的 registeration_infos_combined keys
        self.num01_rotaryPipeAutomaticWeldingMachine_part_01_station = None
        self.animation_num01_rotaryPipeAutomaticWeldingMachine_part_01_station: PoseAnimation = None
        self.num01_rotaryPipeAutomaticWeldingMachine_part_02_station = None
        self.animation_num01_rotaryPipeAutomaticWeldingMachine_part_02_station: PoseAnimation = None
        self._set_up_articulation()

