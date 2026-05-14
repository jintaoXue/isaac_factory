from ..algo_cfg.cfg_multiagent import CfgProductSequencerAgent
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll
from ..env_asset_cfg.cfg_hc_env import HcVectorEnvCfg
import torch

class AlgoMultiAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device
        self.parallel_producing_limit = HcVectorEnvCfg().single_env_parallel_producing_limit
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
        # output shape (self.num_product_types,)
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
        self.parallel_producing_limit = HcVectorEnvCfg().single_env_parallel_producing_limit

    def generate_mask(self, env_state_action_dict) -> None:
        #mask for product selection agent
        # can only select the product in producing or the next product to be produced
        # output shape (1 + self.parallel_producing_limit,)
        producing : list[str] = env_state_action_dict["progress"]["producing"]
        next_product : str = env_state_action_dict["progress"]["next_product"]
        mask = torch.zeros(self.parallel_producing_limit + 1, dtype=torch.int32, device=self.cuda_device)
        products_to_check = ["None"] * (self.parallel_producing_limit + 1)
        for i in range(len(producing)):
            mask[i] = 1
            products_to_check[i] = producing[i]
        # The last position in the mask is for selecting the next product to be produced, 
        # which can only be selected when there are available slots for producing 
        # and there is a next product to be produced.
        if next_product is not None and len(producing) < self.parallel_producing_limit:
            mask[self.parallel_producing_limit] = 1 # can select the next product to be produced
            products_to_check[self.parallel_producing_limit] = next_product
        
        env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]["mask"] = mask
        env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]["products_to_check"] = products_to_check
            
class ProcessTaskPlannerAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device
        self.task_gallery = CfgProcessTaskGalleryInAll
        self.parallel_producing_limit = HcVectorEnvCfg().single_env_parallel_producing_limit

    def generate_mask(self, env_state_action_dict) -> None:
        # Output shape (self.parallel_producing_limit + 1, len(self.task_gallery)) mask for process task planning agent.

        product_selector_mask: torch.Tensor = env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]["mask"]
        # the same length as product_selector_mask, including the currently producing products and the next product to be produced
        products_to_check : list[str] = env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]["products_to_check"] 
        mask = torch.zeros((self.parallel_producing_limit + 1, len(self.task_gallery)), dtype=torch.int32, device=self.cuda_device)

        for i in range(len(product_selector_mask)):
            if product_selector_mask[i] == 0:
                continue
            assert products_to_check[i] != "None", "The product to check for process task planning should not be None when the corresponding product selection mask is 1"
            mask[i] = self._task_mask_for_one_product(products_to_check[i], env_state_action_dict)

        env_state_action_dict["agent_action_mask"]["agent_C_process_task_planner"] = mask

    def _task_mask_for_one_product(self, product: str, env_state_action_dict: dict) -> torch.Tensor:
        task_mask = torch.zeros(len(self.task_gallery), dtype=torch.int32, device=self.cuda_device)
        for i, task in enumerate(self.task_gallery):
            if self._check_task_ready(env_state_action_dict, product, task):
                task_mask[i] = 1
        return task_mask

    def _check_task_ready(self, env_state_action_dict: dict, product: str, task : str) -> bool:
        # TODO: generate task mask rules based on product state, required materials, and process order.
        assert product == "ProductWaterPipe", "Currently only ProductWaterPipe is supported in ProcessTaskPlannerAgentMasker"
        material_task_availability_mask = env_state_action_dict["material"]["task_availability_mask"]
    
    def _check_material_ready(self,)

        return False





class HumanRobotMachineAllocatorAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device

    def generate_mask(self, env_state_action_dict) -> None:
        #mask for human-robot-machine allocation agent
        pass

