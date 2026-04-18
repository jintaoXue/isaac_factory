from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_human import CfgHuman, CfgHumanRegistrationInfos


class HumanManager:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.cfg_human = CfgHuman
        self.cfg_registration_infos = CfgHumanRegistrationInfos
        self.human_list: list[Human] = []
        self._set_up_human_list()

    def reset(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["human_state"] = self
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["human_state"] = self
        return env_state_action_dict

    def _set_up_human_list(self):
        for type_name, n in self.cfg_registration_infos.items():
            cls = globals()[type_name]
            for idx in range(n):
                self.human_list.append(cls(idx, self.cfg_human[type_name], self.env_id))


class Human:
    def __init__(self, idx: int, cfg: dict, env_id: int):
        self.idx = idx
        self.cfg = cfg
        self.type_id = cfg["type_id"]
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id

        self.prim: RigidPrim | None = None
        self._set_up_rigid_prim()

    def _set_up_rigid_prim(self):
        meta = self.meta_registeration_info
        self.prim = RigidPrim(
            prim_paths_expr=meta["prim_paths_expr"].format(i=self.env_id, idx=self.idx),
            name=meta["name"].format(idx=self.idx),
            reset_xform_properties=False,
        )

