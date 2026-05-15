from __future__ import annotations
import torch
from .agent_base import AgentBase

class HumanRobotMachineAllocationAgent(AgentBase):
    """Human–Robot–Machine Allocation Agent.

    Input:
    - The next task to be executed, including the predefined sequence of subtasks to be executed.
    - The current state and actions of all relevant assets in the environment
      (e.g., humans, robots, machines, storage systems).

    Output:
    - Task allocation result: The human, robot, and machine are in charge of executing the task.
    """
    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device

    def act(self, env_state_action_dict: dict, product_selection_action: torch.Tensor | None, process_task_planning_action: torch.Tensor | None) -> dict:
        
        human_availability_mask = env_state_action_dict["human"]["self_availability_mask"]
        robot_availability_mask = env_state_action_dict["robot"]["self_availability_mask"]
        return None

