from __future__ import annotations
from pydoc import doc
import torch

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

    def act(self, env_state_action_dict: dict, product_sequencing_action: torch.Tensor) -> torch.Tensor | None:
        # mask is a tensor shaped (1 + num_product_types,)
        # Each entry is binary: 1 means the corresponding product type may be selected,
        # 0 means it is not eligible right now.
        # The extra first entry mask[0] is reserved for not selecting any product for production in current step.
        producing_products : list[str] = env_state_action_dict["progress"]["producing"]
        next_product : str = env_state_action_dict["progress"]["next_product"]
        mask: torch.Tensor = env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]
        assert mask is not None, "Agent action mask for agent_B_product_selector is None"
        nonzero = mask.nonzero(as_tuple=True)[0]
        if nonzero.numel() == 0:
            return mask
        # The cuurent strategy is simply select the first product in the action space 
        # that is eligible according to the mask
        action = mask.clone().detach()
        action[nonzero[0]+1:] = 0
        return action
