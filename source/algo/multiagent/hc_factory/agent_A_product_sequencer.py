from __future__ import annotations
from pydoc import doc
import torch
from .agent_base import AgentBase

class ProductSequencingAgent(AgentBase):
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
    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device

    def act(self, env_state_action_dict: dict) -> torch.Tensor | None:
        # mask is a tensor shaped (1 + num_product_types,)
        # Each entry is binary: 1 means the corresponding product type may be selected,
        # 0 means it is not eligible right now.
        # The extra first entry mask[0] is reserved for not selecting any product for production in current step.
        mask: torch.Tensor = env_state_action_dict["agent_action_mask"]["agent_A_product_sequencer"]
        assert mask is not None, "Agent action mask for agent_A_product_sequencer is None"
        action = self.keep_first_one(mask)
        return action