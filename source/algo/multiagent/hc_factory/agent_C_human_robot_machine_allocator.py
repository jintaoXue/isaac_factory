from __future__ import annotations


class HumanRobotMachineAllocationAgent:
    """Human–Robot–Machine Allocation Agent.

    Input:
    - The next task to be executed, including the predefined sequence of subtasks to be executed.
    - The current state and actions of all relevant assets in the environment
      (e.g., humans, robots, machines, storage systems).

    Output:
    - Task allocation result: The human, robot, and machine are in charge of executing the task.
    """

    def reset(self, task: dict, env_state_action_dict: dict) -> dict:
        return env_state_action_dict

    def step(self, task: dict, env_state_action_dict: dict) -> dict:
        return env_state_action_dict

