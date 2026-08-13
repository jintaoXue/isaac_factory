"""Hierarchical observation encoder: preprocess → StateEncoder → agent-specific obs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from .hier_networks import StateEncoder


def _load_preprocess():
    try:
        from isaaclab_tasks.direct.hc_factory.src.data_preprocess_for_buffer import (  # type: ignore
            CFG as PRE_CFG,
            preprocess_for_buffer,
        )

        return PRE_CFG, preprocess_for_buffer
    except Exception:
        path = (
            Path(__file__).resolve().parents[3]
            / "isaaclab_tasks"
            / "isaaclab_tasks"
            / "direct"
            / "hc_factory"
            / "src"
            / "data_preprocess_for_buffer.py"
        )
        spec = importlib.util.spec_from_file_location("data_preprocess_for_buffer", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # dataclass needs module in sys.modules
        import sys

        sys.modules["data_preprocess_for_buffer"] = mod
        spec.loader.exec_module(mod)
        return mod.CFG, mod.preprocess_for_buffer


PRE_CFG, preprocess_for_buffer = _load_preprocess()


class HierObsEncoder(nn.Module):
    """Full preprocess features → shared ``StateEncoder`` → per-agent flat obs."""

    def __init__(
        self,
        cuda_device: torch.device,
        parallel_producing_limit: int = 5,
        state_dim: int = 256,
    ):
        super().__init__()
        self.cuda_device = cuda_device
        self.parallel_producing_limit = parallel_producing_limit
        self.state_dim = state_dim
        self.state_encoder = StateEncoder(
            out_dim=state_dim,
            max_ongoing=PRE_CFG.max_ongoing,
            max_subtasks=PRE_CFG.max_subtasks,
            max_material=PRE_CFG.max_material,
            num_agents=PRE_CFG.num_agents,
            transformer_layers=2,
            transformer_heads=4,
        ).to(cuda_device)

    def preprocess(self, env_state_action_dict: dict) -> dict:
        """Raw env dict → fixed-shape nested tensors on ``cuda_device``."""
        prog = env_state_action_dict.get("progress")
        if isinstance(prog, dict) and "ongoing_tasks" in prog:
            return self._to_device(env_state_action_dict)
        return preprocess_for_buffer(env_state_action_dict, device=self.cuda_device)

    def _to_device(self, pre: dict) -> dict:
        out = {}
        for k, v in pre.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.cuda_device)
            elif isinstance(v, dict):
                out[k] = self._to_device(v)
            else:
                out[k] = v
        return out

    def encode_state(self, env_or_pre: dict) -> torch.Tensor:
        pre = self.preprocess(env_or_pre)
        return self.state_encoder(pre)

    def encode_A(self, env_state_action_dict: dict) -> torch.Tensor:
        z = self.encode_state(env_state_action_dict)
        pre = self.preprocess(env_state_action_dict)
        mask = pre["agent_action_mask"]["agent_A_product_sequencer"].float().flatten()
        return torch.cat([z, mask]).detach()

    def encode_B(self, env_state_action_dict: dict, product_sequencing_action: torch.Tensor | None) -> torch.Tensor:
        z = self.encode_state(env_state_action_dict)
        pre = self.preprocess(env_state_action_dict)
        a_dim = int(pre["agent_action_mask"]["agent_A_product_sequencer"].numel())
        if product_sequencing_action is not None:
            a_action = product_sequencing_action.float().flatten().to(self.cuda_device)
        else:
            a_action = torch.zeros(a_dim, dtype=torch.float32, device=self.cuda_device)
        mask = pre["agent_action_mask"]["agent_B_product_selector"].float().flatten()
        return torch.cat([z, a_action, mask]).detach()

    def encode_C(
        self,
        env_state_action_dict: dict,
        product_selection_action: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode_state(env_state_action_dict)
        pre = self.preprocess(env_state_action_dict)
        aam = pre["agent_action_mask"]
        b_action = product_selection_action.float().flatten().to(self.cuda_device)
        return torch.cat(
            [
                z,
                b_action,
                aam["agent_C_process_task_planner"].float().flatten(),
                aam["human"]["task_availability_mask"].float().flatten(),
                aam["machine"]["task_availability_mask"].float().flatten(),
            ]
        ).detach()

    def encode_D(
        self,
        env_state_action_dict: dict,
        process_task_planning_action: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode_state(env_state_action_dict)
        pre = self.preprocess(env_state_action_dict)
        aam = pre["agent_action_mask"]
        c_action = process_task_planning_action.float().flatten().to(self.cuda_device)
        return torch.cat(
            [
                z,
                c_action,
                aam["human"]["self_availability_mask"].float().flatten(),
                aam["robot"]["self_availability_mask"].float().flatten(),
            ]
        ).detach()

    def get_obs_dim_A(self, env_state_action_dict: dict) -> int:
        return int(self.encode_A(env_state_action_dict).shape[0])

    def get_obs_dim_B(self, env_state_action_dict: dict) -> int:
        return int(self.encode_B(env_state_action_dict, None).shape[0])

    def get_obs_dim_C(self, env_state_action_dict: dict, product_selection_action: torch.Tensor) -> int:
        return int(self.encode_C(env_state_action_dict, product_selection_action).shape[0])

    def get_obs_dim_D(self, env_state_action_dict: dict, process_task_planning_action: torch.Tensor) -> int:
        return int(self.encode_D(env_state_action_dict, process_task_planning_action).shape[0])
