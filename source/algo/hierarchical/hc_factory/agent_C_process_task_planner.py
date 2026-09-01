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
        task_mask_for_products = env_state_action_dict["agent_action_mask"]["agent_C_process_task_planner"]
        count = (product_selection_action == 1).sum().item()
        if count == 0:
            return self._none_task_action(task_mask_for_products.shape[1])
        mask = task_mask_for_products[product_selection_action.nonzero()[0][0]]
        return self._select_from_mask(mask, task_mask_for_products.shape[1])

    def act_for_slot(self, cuda_device: torch.device, task_mask: torch.Tensor) -> torch.Tensor:
        return self._select_from_mask(task_mask, task_mask.shape[0])

    def _none_task_action(self, c_dim: int) -> torch.Tensor:
        action = torch.zeros(c_dim, dtype=torch.int32, device=self.cuda_device)
        action[0] = 1
        return action

    def _select_from_mask(self, mask: torch.Tensor, c_dim: int) -> torch.Tensor:
        if mask.sum() == 1:
            return self._none_task_action(c_dim)
        mask = mask.clone()
        mask[0] = 0
        return self.keep_last_one(mask)

