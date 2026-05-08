from __future__ import annotations
from pydoc import doc

from ..env_asset_cfg.cfg_material_product import CfgProductProcess


cfgAlgoMultiAgentMasker = {
    #action space: the number of products that can be produced in parallel
    "parallel_producing_limit": 5,
}

CfgProductSequencerAgent = {
    "product_types": list(CfgProductProcess.keys()),
    "num_product_types": len(CfgProductProcess),
    "action_space": {"None": 0, **{k: i+1 for i, k in enumerate(CfgProductProcess.keys())}},
}

CfgProductSelectorAgent = {

}

CfgProcessTaskPlannerAgent = {
    
}
       
CfgHumanRobotMachineAllocatorAgent = {
}