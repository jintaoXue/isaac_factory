from isaacsim.core.prims import RigidPrim
from ..env_asset_cfg.cfg_route.cfg_route import CfgRoute
import json


class RouteManagerVectorEnv:
    def __init__(self):
        self.cfg_route = CfgRoute
        self.routes_path_human = self.cfg_route["routes_path_human"]
        self.routes_path_robot = self.cfg_route["routes_path_robot"]


