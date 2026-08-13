from __future__ import annotations

import random

import torch


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> int | None:
    """Return the index of the maximum Q-value among valid (mask==1) actions.

    Returns ``None`` if no action is valid.
    """
    if mask.sum() == 0:
        return None
    q_masked = q_values.clone()
    q_masked[mask == 0] = -float("inf")
    return int(q_masked.argmax().item())


def masked_random_action(mask: torch.Tensor) -> int | None:
    """Uniformly sample one valid action index from the mask.

    Returns ``None`` if no action is valid (caller should emit an all-zero action).
    """
    valid = (mask == 1).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return None
    return int(valid[random.randint(0, valid.numel() - 1)].item())


def index_to_one_hot(index: int, action_dim: int, device: torch.device) -> torch.Tensor:
    """Convert a discrete action index to a one-hot int32 tensor."""
    action = torch.zeros(action_dim, dtype=torch.int32, device=device)
    action[index] = 1
    return action


def one_hot_to_index(action: torch.Tensor) -> int:
    """Convert a one-hot action tensor to its active index."""
    nz = action.nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        return 0
    return int(nz[0].item())


def masked_select_action(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> int | None:
    """Epsilon-greedy action selection with action masking. ``None`` = no-op."""
    if mask.sum() == 0:
        return None
    if random.random() < epsilon:
        return masked_random_action(mask)
    return masked_argmax(q_values, mask)


def compute_team_reward(prev_obs: dict, next_obs: dict) -> float:
    """Prefer env-written ``rl.reward``; fallback to finished-count delta."""
    rl = next_obs.get("rl") if isinstance(next_obs, dict) else None
    if isinstance(rl, dict) and "reward" in rl:
        return float(rl["reward"])
    prev_finished = sum(len(v) for v in prev_obs["progress"]["finished"].values())
    next_finished = sum(len(v) for v in next_obs["progress"]["finished"].values())
    reward = float(next_finished - prev_finished) * 10.0
    reward -= 0.01
    return reward


def read_rl_done(env_dict: dict) -> tuple[bool, bool, bool]:
    """Return ``(done, truncated, success)`` from ``env_dict['rl']``."""
    rl = env_dict.get("rl") or {}
    return bool(rl.get("done")), bool(rl.get("truncated")), bool(rl.get("success"))


def detach_pre_to_cpu(pre: dict) -> dict:
    """Deep-copy preprocessed dict tensors to CPU without grad."""
    out: dict = {}
    for key, value in pre.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu()
        elif isinstance(value, dict):
            out[key] = detach_pre_to_cpu(value)
        else:
            out[key] = value
    return out


def pre_to_device(pre: dict, device: torch.device) -> dict:
    """Move nested tensor fields in a preprocessed dict to ``device``."""
    out: dict = {}
    for key, value in pre.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = pre_to_device(value, device)
        else:
            out[key] = value
    return out
