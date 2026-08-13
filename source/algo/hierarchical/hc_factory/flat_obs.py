"""Flat TPA observation: shared state encoder + all action masks."""

from __future__ import annotations

import torch

from .hier_obs import HierObsEncoder


class FlatObsEncoder(HierObsEncoder):
    """StateEncoder output concatenated with flattened hierarchical masks."""

    def encode_flat(self, env_state_action_dict: dict) -> torch.Tensor:
        z = self.encode_state(env_state_action_dict)
        pre = self.preprocess(env_state_action_dict)
        aam = pre["agent_action_mask"]
        return torch.cat(
            [
                z,
                aam["agent_A_product_sequencer"].float().flatten(),
                aam["agent_B_product_selector"].float().flatten(),
                aam["agent_C_process_task_planner"].float().flatten(),
                aam["human"]["self_availability_mask"].float().flatten(),
                aam["robot"]["self_availability_mask"].float().flatten(),
            ]
        )

    def get_obs_dim_flat(self, env_state_action_dict: dict) -> int:
        with torch.no_grad():
            return int(self.encode_flat(env_state_action_dict).shape[0])
