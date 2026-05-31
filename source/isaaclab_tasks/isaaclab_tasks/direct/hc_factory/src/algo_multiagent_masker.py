from ..algo_cfg.cfg_multiagent import CfgProductSequencerAgent
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll
from ..env_asset_cfg.cfg_hc_env import HcVectorEnvCfg
import torch
import copy

class AlgoMultiAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device
        self.parallel_producing_limit = HcVectorEnvCfg().single_env_parallel_producing_limit
        self.agent_A_Product_sequencer = ProductSequencerAgentMasker(self.cuda_device)
        self.agent_B_Product_selection = ProductSelectorAgentMasker(self.cuda_device)
        self.agent_C_Process_task_planning = ProcessTaskPlannerAgentMasker(self.cuda_device)
        self.agent_D_Human_robot_machine_allocation = HumanRobotMachineAllocatorAgentMasker(self.cuda_device)

    def reset(self, env_state_action_dict):
        return self.generate_agents_mask(env_state_action_dict)

    def step(self, env_state_action_dict):
        return self.generate_agents_mask(env_state_action_dict)

    def generate_agents_mask(self, env_state_action_dict):
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
        
        mask = torch.zeros(self.num_product_types, dtype=torch.int32, device=self.cuda_device)
        if next_product is None and len(not_started.keys()) > 0:
            #can select the product in not_started
            for product_type in not_started.keys():
                mask[self.product_types[product_type]] = 1
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
        producing_indexs : list[int] = env_state_action_dict["progress"]["producing_indexs"]
        next_product : str = env_state_action_dict["progress"]["next_product"]
        mask = torch.zeros(self.parallel_producing_limit + 1, dtype=torch.int32, device=self.cuda_device)
        keys_of_ongoing_task_records : list[int] = list(env_state_action_dict["progress"]["ongoing_task_records"].keys())
        for i in range(len(producing_indexs)):
            product_index = producing_indexs[i]
            if product_index not in keys_of_ongoing_task_records:
                #means no ongoing task for processing this product, can select this product
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
        self.task_gallery = CfgProcessTaskGalleryInAll
        self.parallel_producing_limit = HcVectorEnvCfg().single_env_parallel_producing_limit

    def generate_mask(self, env_state_action_dict) -> None:
        # Output shape (self.parallel_producing_limit + 1, len(self.task_gallery)) mask for process task planning agent.
        # product_selector_mask shape (1 + self.parallel_producing_limit,)
        product_selector_mask: torch.Tensor = env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]
        producing_indexs : list = env_state_action_dict["progress"]["producing_indexs"]
        next_product_index : int = env_state_action_dict["progress"]["next_product_index"]
        #shape is (material_batch_upper_bound, len(CfgProcessTaskGalleryInAll))
        task_mask_for_all_products = self.get_task_mask_for_all_products(env_state_action_dict)
        # the same length as product_selector_mask, including the currently producing products and the next product to be produced
        mask = torch.zeros((self.parallel_producing_limit + 1, len(self.task_gallery)), dtype=torch.int32, device=self.cuda_device)
        
        #expand dim and repeat to match the shape of task_mask_for_all_products
        product_selector_mask_expanded = product_selector_mask.unsqueeze(1).expand(-1, len(self.task_gallery))
        # first check producing products
        for i in range(len(producing_indexs)): # including the next product to be produced
            index_in_material_batch = producing_indexs[i]
            mask[i] = task_mask_for_all_products[index_in_material_batch] & product_selector_mask_expanded[i]
        if next_product_index is not None:
            mask[self.parallel_producing_limit] = task_mask_for_all_products[next_product_index] & product_selector_mask_expanded[self.parallel_producing_limit]

        env_state_action_dict["agent_action_mask"]["agent_C_process_task_planner"] = mask

    def get_task_mask_for_all_products(self, env_state_action_dict: dict) -> torch.Tensor:
        #shape of human task_availability_mask (len(CfgProcessTaskGalleryInAll),)
        human_task_availability_mask = env_state_action_dict["agent_action_mask"]["human"]["task_availability_mask"]
        #shape of machine task_availability_mask (len(CfgProcessTaskGalleryInAll),)
        machine_task_availability_mask = env_state_action_dict["agent_action_mask"]["machine"]["task_availability_mask"]
        #shape of robot task_availability_mask (len(CfgProcessTaskGalleryInAll),)
        robot_task_availability_mask = env_state_action_dict["agent_action_mask"]["robot"]["task_availability_mask"]
        #shape of material_task_availability_mask (upper_bound_num_material_batch, len(CfgProcessTaskGalleryInAll))
        material_task_availability_mask = env_state_action_dict["agent_action_mask"]["material"]["task_availability_mask"]

        # First, perform bitwise AND on human, robot, machine masks
        combined_mask = human_task_availability_mask & robot_task_availability_mask & machine_task_availability_mask
        # material_task_availability_mask is 2D, expand combined_mask to match dimensions
        num_batches = material_task_availability_mask.shape[0]
        expanded_combined = combined_mask.unsqueeze(0).expand(num_batches, -1)
        # Perform bitwise AND with material mask
        task_mask_for_all_product = expanded_combined & material_task_availability_mask
    
        return task_mask_for_all_product

class HumanRobotMachineAllocatorAgentMasker:
    def __init__(self, cuda_device: torch.device) -> None:
        self.cuda_device = cuda_device

    def generate_mask(self, env_state_action_dict) -> None:
        #mask for human-robot-machine allocation agent
        #output shape (len(CfgProcessTaskGalleryInAll),)
        human_mask = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"]
        robot_mask = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"]

        env_state_action_dict["agent_action_mask"]["agent_D_human_robot_allocator"] = {
            "human_mask": human_mask,
            "robot_mask": robot_mask,
        }

