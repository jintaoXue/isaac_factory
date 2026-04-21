from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_route.cfg_route import CfgRoute
import json
from pathlib import Path


class RouteManagerVectorEnv:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.cfg_route = CfgRoute
        self.human_path = Path(self.cfg_route["routes_path_human"]).expanduser()
        self.robot_path = Path(self.cfg_route["routes_path_robot"]).expanduser()
        self.routes_human = json.load(self.human_path.open("r", encoding="utf-8"))
        self.routes_robot = json.load(self.robot_path.open("r", encoding="utf-8"))

    def reset(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["state_route"] = self
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["state_route"] = self
        return env_state_action_dict


