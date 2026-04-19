from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_route.cfg_route import CfgRoute
import json


class RouteManagerVectorEnv:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.cfg_route = CfgRoute
        self.routes_path_human = self.cfg_route["routes_path_human"]
        self.routes_path_robot = self.cfg_route["routes_path_robot"]
        self.routes_human = json.load(open(self.routes_path_human))
        self.routes_robot = json.load(open(self.routes_path_robot))

    def reset(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["route_state"] = self
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict["route_state"] = self
        return env_state_action_dict


