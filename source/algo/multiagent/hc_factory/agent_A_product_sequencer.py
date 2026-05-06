from __future__ import annotations


class ProductSequencingAgent:
    """
    Product Sequencing Agent

    This agent is responsible for determining the optimal order in which products should be manufactured,
    based on the current production order and the state of the environment/assets.

    Inputs:
        - production_order (list[dict]): List containing product information and requirements to be produced.
        - env_state_action_dict (dict): Current state and actions of all relevant assets in the environment
          (e.g., humans, robots, machines, storage systems).

    Outputs:
        - The next product (or list of products) to be prioritized for production. 
        - Or the schedule of products to be produced in the future.
    """

    def reset(self, production_order: list[dict], env_state_action_dict: dict) -> list[dict]:

        # For now, return the production order unchanged.
        return production_order

    def act(self, production_order: list[dict], env_state_action_dict: dict) -> list[dict]:

        # Placeholder: Returns production order unchanged.
        return production_order