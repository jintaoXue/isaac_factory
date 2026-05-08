from __future__ import annotations
from pydoc import doc
import torch

class ProductSequencingAgent:
    """
    Product Sequence Planner Agent

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

    def act(self, env_state_action_dict: dict) -> str | None:
        # mask is a tensor shaped (1 + num_product_types,)
        # Each entry is binary: 1 means the corresponding product type may be selected,
        # 0 means it is not eligible right now.
        # The extra first entry mask[0] is reserved for not selecting any product for production in current step.
        mask: torch.Tensor = env_state_action_dict["agent_action_mask"]["agent_A_product_sequencer"]
        assert mask is not None, "Agent action mask for agent_A_product_sequencer is None"
        nonzero = mask.nonzero(as_tuple=True)[0]
        if nonzero.numel() == 0:
            return mask
        # The current strategy is simply select the first product type in the action space 
        # that is eligible according to the mask
        action = mask.clone().detach()
        action[nonzero[0]+1:] = 0
        return action