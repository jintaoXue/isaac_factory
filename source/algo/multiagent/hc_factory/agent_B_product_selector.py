from __future__ import annotations
from pydoc import doc
import torch
from .agent_base import AgentBase

class ProductSelectionAgent(AgentBase):
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
    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device

    def act(self, env_state_action_dict: dict, product_sequencing_action: torch.Tensor) -> torch.Tensor | None:
        # mask is a tensor shaped (1 + self.parallel_producing_limit,)
        # Each entry is binary: 1 means the corresponding product type may be selected,
        # 0 means it is not eligible right now.
        mask: torch.Tensor = env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]
        assert mask is not None, "Agent action mask for agent_B_product_selector is None"
        action = self.keep_first_one(mask)
        return action
