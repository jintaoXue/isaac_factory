from __future__ import annotations
import torch
from .agent_base import AgentBase

class ProcessTaskPlanningAgent(AgentBase):
    """Process's Task Planning Agent.

    Input:
    - Production schedule (a list of products to be produced in the future) or the next product to be produced
    - Product process with corresponding task dependency graph (a more detailed description of the product process, 
        including the machine, human, and material required for each process step (A task node in the task dependency graph), 
        and subtasks operations sequence logic for each task)
    - env_state_action_dict (dict): Current state and actions of all relevant assets in the environment
          (e.g., humans, robots, machines, storage systems).
          
    Output:
    - The next task to be executed, including the predefined sequence of subtasks to be executed.
    """
    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device
        
    def act(self, env_state_action_dict: dict, product_selection_action: torch.Tensor | None) -> dict:
        ## env start, generate first process task planning
        ##shape (self.parallel_producing_limit + 1, len(CfgProcessTaskGalleryInAll)) mask for process task planning agent.
        task_mask_for_products = env_state_action_dict["agent_action_mask"]["agent_C_process_task_planner"]
        
        count = (product_selection_action == 1).sum().item()
        if count == 0:
            action = torch.zeros((task_mask_for_products.shape[1]), dtype=torch.int32, device=self.cuda_device)
            action[0] = 1 # only "none" task is available when there is no product selected for process task planning
        else:
            assert count == 1, "There should be only one product selected for process task planning, but got multiple."
            mask = task_mask_for_products[product_selection_action.nonzero()[0][0]] # get the task mask for the selected product
            if mask.sum() == 1:
                #only "none" task is available for the selected product
                action = torch.zeros((task_mask_for_products.shape[1]), dtype=torch.int32, device=self.cuda_device)
                action[0] = 1
            else:
                #select the second available task for the selected product according to the task mask
                mask[0] = 0 #set "none" task is not available
                action = self.keep_last_one(mask)
        return action

