from isaacsim.core.prims import Articulation

class Machine:
    def __init__(self, name: str, num_workstations: int, num_registration_parts: int):
        self.name = name
        self.registration_type = "Articulation"
        self.num_workstations = num_workstations
        self.num_registration_parts = num_registration_parts

class num01_rotaryPipeAutomaticWeldingMachine(Machine):
    def __init__(self):
        super().__init__(name="num01_rotaryPipeAutomaticWeldingMachine", 
        num_workstations=2, num_registration_parts=2)
        self.num01_rotaryPipeAutomaticWeldingMachine_part_01_station = Articulation(cfg=ArticulationCfg(
            prim_paths_expr="/World/envs/.*/obj/HC_factory/num01_rotaryPipeAutomaticWeldingMachine/part_01_station/track_for_mobile_base",
            name="num01_rotaryPipeAutomaticWeldingMachine_part_01_station",
            reset_xform_properties=False,
        )) 