from ..algo_cfg.cfg_multiagent import cfgAlgoMultiAgentMasker, CfgProductSequencerAgent
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGallery
import torch

class AlgoMultiAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device
        self.parallel_producing_limit = cfgAlgoMultiAgentMasker["parallel_producing_limit"]
        self.agent_A_Product_sequencer = ProductSequencerAgentMasker(self.cuda_device)
        self.agent_B_Product_selection = ProductSelectorAgentMasker(self.cuda_device)
        self.agent_C_Process_task_planning = ProcessTaskPlannerAgentMasker(self.cuda_device)
        self.agent_D_Human_robot_machine_allocation = HumanRobotMachineAllocatorAgentMasker(self.cuda_device)

    def generate_agents_mask(self, env_state_action_dict) -> dict:
        for agent in self.iter_agents():
            agent.generate_mask(env_state_action_dict)

    def iter_agents(self):
        return (
            self.agent_A_Product_sequencer,
            self.agent_B_Product_selection,
            self.agent_C_Process_task_planning,
            self.agent_D_Human_robot_machine_allocation,
        )

class ProductSequencerAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device
        self.product_types = CfgProductSequencerAgent["product_types"]
        self.num_product_types = CfgProductSequencerAgent["num_product_types"]

    def generate_mask(self, env_state_action_dict) -> None:
        #mask for product sequencing agent
        not_started : dict = env_state_action_dict["progress"]["not_started"]
        next_product : str = env_state_action_dict["progress"]["next_product"]
        producing : list[str] = env_state_action_dict["progress"]["producing"]
        finished : list[str] = env_state_action_dict["progress"]["finished"]
        
        mask = torch.zeros(self.num_product_types, dtype=torch.int32, device=self.cuda_device)
        if next_product is None and len(not_started.keys()) > 0:
            #can select the product in not_started
            for product in not_started.keys():
                mask[self.product_types[product]] = 1
        env_state_action_dict["agent_action_mask"]["agent_A_product_sequencer"] = mask

class ProductSelectorAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device
        self.parallel_producing_limit = cfgAlgoMultiAgentMasker["parallel_producing_limit"]

    def generate_mask(self, env_state_action_dict) -> None:
        #mask for product selection agent
        # can only select the product in producing or the next product to be produced
        producing : list[str] = env_state_action_dict["progress"]["producing"]
        next_product : str = env_state_action_dict["progress"]["next_product"]
        mask = torch.zeros(self.parallel_producing_limit + 1, dtype=torch.int32, device=self.cuda_device)
        for i in range(len(producing)):
            mask[i] = 1
        # The last position in the mask is for selecting the next product to be produced, 
        # which can only be selected when there are available slots for producing 
        # and there is a next product to be produced.
        if next_product is not None and len(producing) < self.parallel_producing_limit:
            mask[self.parallel_producing_limit] = 1 # can select the next product to be produced
        env_state_action_dict["agent_action_mask"]["agent_B_product_selector"] = mask

class ProcessTaskPlannerAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device
        self.action_space = CfgProcessTaskGallery

    def generate_mask(self, env_state_action_dict) -> None:
        #mask for process task planning agent
        mask = torch.zeros(len(self.action_space), dtype=torch.int32, device=self.cuda_device)

        pass

class HumanRobotMachineAllocatorAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device

    def generate_mask(self, env_state_action_dict) -> None:
        #mask for human-robot-machine allocation agent
        pass

