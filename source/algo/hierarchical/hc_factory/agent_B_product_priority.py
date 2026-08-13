# -*- coding: utf-8 -*-
"""Agent B: priority ranking over eligible WIP / staging slots (FIFO rule baseline)."""
from __future__ import annotations

import torch

from .agent_base import AgentBase


class ProductPriorityAgent(AgentBase):
    """Rank eligible product slots; lower slot index first, staging slot last."""

    def rank_slots(self, eligible_mask: torch.Tensor) -> list[int]:
        indices = (eligible_mask == 1).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return []
        staging = int(eligible_mask.shape[0] - 1)
        producing_slots = [int(i.item()) for i in indices if int(i.item()) != staging]
        producing_slots.sort()
        ordered = producing_slots
        if eligible_mask[staging].item() == 1:
            ordered.append(staging)
        return ordered

    def scores_from_order(
        self,
        eligible_mask: torch.Tensor,
        slot_order: list[int],
        dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        scores = torch.zeros(dim, dtype=torch.float32, device=device)
        n = max(len(slot_order), 1)
        for rank, slot in enumerate(slot_order):
            if 0 <= slot < dim:
                scores[slot] = float(n - rank)
        return scores

    def act(self, env_state_action_dict: dict) -> torch.Tensor:
        """Legacy one-hot: first slot in FIFO order."""
        mask = env_state_action_dict["agent_action_mask"]["agent_B_product_selector"]
        order = self.rank_slots(mask)
        out = torch.zeros_like(mask)
        if order:
            out[order[0]] = 1
        return out
