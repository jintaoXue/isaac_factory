from __future__ import annotations
from pydoc import doc

from ..env_asset_cfg.cfg_material_product import CfgProductProcess


CfgProductSequencerAgent = {
    "num_product_types": len(CfgProductProcess),
    "product_types": {**{k: i for i, k in enumerate(CfgProductProcess.keys())}},
}

CfgProductSelectorAgent = {

}

CfgProcessTaskPlannerAgent = {
    
}
       
CfgHumanRobotMachineAllocatorAgent = {
}