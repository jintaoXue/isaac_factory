"""Single masked DQN agent over flat joint action space."""

from __future__ import annotations

import torch

from .flat_action_space import FlatJointActionSpace
from .hier_rl_agents import MaskedDQNAgent


class RLFlatAgent:
    """One-stage TPA: joint index selection + decode to env action dict."""

    def __init__(
        self,
        obs_encoder,
        action_space: FlatJointActionSpace,
        device: torch.device,
        **dqn_kwargs,
    ) -> None:
        self.obs_encoder = obs_encoder
        self.action_space = action_space
        self.device = device
        self.dqn_kwargs = dqn_kwargs
        self.dqn: MaskedDQNAgent | None = None

    def _ensure_dqn(self, env_state_action_dict: dict) -> None:
        if self.dqn is not None:
            return
        obs_dim = self.obs_encoder.get_obs_dim_flat(env_state_action_dict)
        self.action_space._ensure_dims(env_state_action_dict)
        action_dim = self.action_space.joint_dim
        self.dqn = MaskedDQNAgent("flat_joint", obs_dim, action_dim, self.device, **self.dqn_kwargs)

    def act(
        self,
        env_state_action_dict: dict,
        epsilon: float,
        *,
        pre: dict | None = None,
    ) -> tuple[dict, int | None]:
        self._ensure_dqn(env_state_action_dict)
        obs = self.obs_encoder.encode_flat(env_state_action_dict, pre=pre)
        joint_mask = self.action_space.build_joint_mask(env_state_action_dict)
        joint_idx = self.dqn.select_action(obs, joint_mask, epsilon)
        if joint_idx is None:
            valid = (joint_mask > 0).nonzero(as_tuple=True)[0]
            joint_idx = int(valid[0].item()) if valid.numel() > 0 else 0
        action = self.action_space.decode(joint_idx)
        return action, joint_idx

    def observe_step(
        self,
        env_state_action_dict: dict,
        joint_idx: int | None,
        reward: float,
        next_env_state_action_dict: dict,
        done: bool,
        *,
        learn=True,
    ) -> float | None:
        if joint_idx is None:
            return None
        self._ensure_dqn(env_state_action_dict)
        obs = self.obs_encoder.encode_flat(env_state_action_dict)
        next_obs = self.obs_encoder.encode_flat(next_env_state_action_dict)
        mask = self.action_space.build_joint_mask(env_state_action_dict)
        next_mask = self.action_space.build_joint_mask(next_env_state_action_dict)
        self.dqn.store(obs, int(joint_idx), reward, next_obs, mask, next_mask, done)
        if not learn:
            return None
        return self.dqn.learn()
