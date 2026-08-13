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
        human_availability_mask = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"]
        robot_availability_mask = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"]
        return self._allocate(process_task_planning_action, human_availability_mask, robot_availability_mask)

    def act_with_masks(
        self,
        process_task_planning_action: torch.Tensor | None,
        d_masks: dict,
    ) -> dict:
        return self._allocate(
            process_task_planning_action,
            d_masks["human"],
            d_masks["robot"],
        )

    def _allocate(
        self,
        process_task_planning_action: torch.Tensor | None,
        human_availability_mask: torch.Tensor,
        robot_availability_mask: torch.Tensor,
    ) -> dict:
        count = (process_task_planning_action == 1).sum().item()
        assert count == 1, "There should be only one task selected for human-robot-machine allocation, but got multiple."
        if process_task_planning_action[0] == 1:
            return {
                "human": torch.zeros(human_availability_mask.shape[0], dtype=torch.int32, device=self.cuda_device),
                "robot": torch.zeros(robot_availability_mask.shape[0], dtype=torch.int32, device=self.cuda_device),
            }
        return {
            "human": self.keep_first_one(human_availability_mask),
            "robot": self.keep_first_one(robot_availability_mask),
        }

