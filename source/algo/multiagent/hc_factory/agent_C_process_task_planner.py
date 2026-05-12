from __future__ import annotations
import torch

class ProcessTaskPlanningAgent:
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

    def act(self, env_state_action_dict: dict, product_selection_action: torch.Tensor | None) -> dict:
        ## env start, generate first process task planning
        # check available human and robot resource
        return None
