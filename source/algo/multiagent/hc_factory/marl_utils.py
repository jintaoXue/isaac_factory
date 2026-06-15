from __future__ import annotations

import random

import torch


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> int:
    """Return the index of the maximum Q-value among valid (mask==1) actions."""
    q_masked = q_values.clone()
    q_masked[mask == 0] = -float("inf")
    return int(q_masked.argmax().item())


def masked_random_action(mask: torch.Tensor) -> int:
    """Uniformly sample one valid action index from the mask."""
    valid = (mask == 1).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return 0
    return int(valid[random.randint(0, valid.numel() - 1)].item())


def index_to_one_hot(index: int, action_dim: int, device: torch.device) -> torch.Tensor:
    """Convert a discrete action index to a one-hot int32 tensor."""
    action = torch.zeros(action_dim, dtype=torch.int32, device=device)
    action[index] = 1
    return action


def one_hot_to_index(action: torch.Tensor) -> int:
    """Convert a one-hot action tensor to its active index."""
    return int(action.nonzero(as_tuple=True)[0][0].item())


def masked_select_action(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> int:
    """Epsilon-greedy action selection with action masking."""
    if random.random() < epsilon:
        return masked_random_action(mask)
    return masked_argmax(q_values, mask)


def compute_team_reward(prev_obs: dict, next_obs: dict) -> float:
    """Progress-based team reward shared by all four agents."""
    prev_finished = sum(len(v) for v in prev_obs["progress"]["finished"].values())
    next_finished = sum(len(v) for v in next_obs["progress"]["finished"].values())
    reward = float(next_finished - prev_finished) * 10.0
    reward -= 0.01
    return reward
