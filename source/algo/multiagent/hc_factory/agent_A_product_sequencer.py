from __future__ import annotations
from pydoc import doc


class ProductSequencingAgent:
    """
    Product Sequencing Agent

    This agent is responsible for determining the optimal order in which products should be manufactured,
    based on the current production order and the state of the environment/assets.

    Inputs:
        - production_order (example: {'ProductWaterPipe': 5}): Dictionary containing product information and requirements to be produced.
        - env_state_action_dict (dict): Current state and actions of all relevant assets in the environment
          (e.g., humans, robots, machines, storage systems).

    Outputs:
        - The next product (or list of products) to be prioritized for production. 
        - Or the schedule of products to be produced in the future.
    """

    def act(self, env_state_action_dict: dict) -> dict:
        #example: env_state_action_dict["progress"]    "progress": {
        not_started : dict = env_state_action_dict["progress"]["not_started"]
        next_product : str = env_state_action_dict["progress"]["next_product"]
        producing : list[str] = env_state_action_dict["progress"]["producing"]
        finished : list[str] = env_state_action_dict["progress"]["finished"]
        
        if next_product and len(not_started.keys()) > 0:
            #select the first product in not_started
            action = not_started.keys()[0]
        else:
            action = None
        return action