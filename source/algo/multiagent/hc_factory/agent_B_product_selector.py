from __future__ import annotations
from pydoc import doc


class ProductSelectionAgent:
    """
    Product Selection Agent

    At each time step, this agent selects the focal product from the current producing list.
    The selected product will serve as the target for process task planning in the subsequent decision stage (agent C task planner).

    Inputs:
        - env_state_action_dict (dict): Current state and actions of all relevant assets in the environment
          (e.g., humans, robots, machines, storage systems).

    Outputs:
        - The focusing product in producing list.
    """

    def act(self, env_state_action_dict: dict) -> str | None:
        #example: env_state_action_dict["progress"]    "progress": {
        not_started : dict = env_state_action_dict["progress"]["not_started"]
        next_product : str = env_state_action_dict["progress"]["next_product"]
        producing : list[str] = env_state_action_dict["progress"]["producing"]
        finished : list[str] = env_state_action_dict["progress"]["finished"]
        
        if next_product is None and len(not_started.keys()) > 0:
            #select the first product in not_started
            action = list(not_started.keys())[0]
        else:
            action = None
        return action