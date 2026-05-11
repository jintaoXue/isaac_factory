from __future__ import annotations
from pydoc import doc

from ..env_asset_cfg.cfg_material_product import CfgProductProcess
from ..env_asset_cfg.cfg_hc_env import HcVectorEnvCfg

cfgAlgoMultiAgentMasker = {
    #action space: the number of products that can be produced in parallel
    "parallel_producing_limit": HcVectorEnvCfg().parallel_producing_limit,
}

CfgProductSequencerAgent = {
    "num_product_types": len(CfgProductProcess),
    "product_types": {**{k: i+1 for i, k in enumerate(CfgProductProcess.keys())}},
}

CfgProductSelectorAgent = {

}

CfgProcessTaskPlannerAgent = {
    
}
       
CfgHumanRobotMachineAllocatorAgent = {
}