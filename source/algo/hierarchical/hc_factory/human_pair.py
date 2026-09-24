"""Optional D-human pair residual. The original QNetwork stays unchanged."""
from __future__ import annotations

import torch
from torch import nn
from .hier_networks import QNetwork

# Version 1: fatigue, eta, current-task skill, eta*skill, work/recovery rates,
# availability, existence. Do not use stale last-subtask skill_effective.
PAIR_FEATURE_DIM = 8


def human_task_pair_features(pre: dict, task_action: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Replay and online use the same preprocessed observation and C context.

    cfg_human skill_task columns follow CfgProcessTaskGalleryInAll IDs 1..12;
    task ID 0 is none. Contract checked against source tables in CPU tests.
    """
    human = pre["human"]
    task = task_action.float().flatten()
    skills = human["skill_task"].to(task.device).float()[:action_dim]
    if skills.shape != (action_dim, 12) or task.numel() != 13:
        raise ValueError("Pair v1 requires 12 skill columns and 13 task actions (including none)")
    matched = skills @ task[1:]
    def field(name):
        value = human[name].to(task.device).float().flatten()[:action_dim]
        if value.numel() != action_dim:
            raise ValueError(f"Pair observation missing candidate slots: {name}")
        return value
    fatigue = field("fatigue").clamp(0, 1)
    eta = field("efficiency").clamp(0, 1)
    exists = field("mask")
    available = pre["agent_action_mask"]["human"]["self_availability_mask"].to(task.device).float().flatten()
    features = torch.stack((fatigue, eta, matched, eta * matched,
                            field("fatigue_work_rate"), field("fatigue_recover_rate"),
                            available, exists), dim=-1)
    return features * exists.unsqueeze(-1)


class HumanPairQNetwork(QNetwork):
    """Q_old(global, task) + shared MLP(global, task, candidate features)."""
    def __init__(self, obs_dim, action_dim, hidden_dim=128, *, pair_feature_dim=PAIR_FEATURE_DIM, **kwargs):
        self.base_obs_dim = int(obs_dim) - int(action_dim) * int(pair_feature_dim)
        if self.base_obs_dim <= 0 or pair_feature_dim != PAIR_FEATURE_DIM:
            raise ValueError("Invalid pair observation schema")
        super().__init__(self.base_obs_dim, action_dim, hidden_dim, **kwargs)
        self.pair_feature_dim = pair_feature_dim
        self.pair_residual = nn.Sequential(
            nn.Linear(self.base_obs_dim + pair_feature_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.zero_residual()

    def zero_residual(self):
        nn.init.zeros_(self.pair_residual[-1].weight)
        nn.init.zeros_(self.pair_residual[-1].bias)

    def forward(self, obs):
        base = obs[..., :self.base_obs_dim]
        pairs = obs[..., self.base_obs_dim:].reshape(*obs.shape[:-1], self.action_dim, self.pair_feature_dim)
        context = base.unsqueeze(-2).expand(*base.shape[:-1], self.action_dim, self.base_obs_dim)
        residual = self.pair_residual(torch.cat((context, pairs), dim=-1)).squeeze(-1)
        # Last pair feature is candidate existence; masked padded slots get no residual.
        return super().forward(base) + residual * pairs[..., -1]

    def load_compatible_state_dict(self, state):
        if any(k.startswith("pair_residual.") for k in state):
            return self.load_state_dict(state, strict=True)
        # Validate every original key/shape; only the new residual may be absent.
        expected = {k: v for k, v in self.state_dict().items() if not k.startswith("pair_residual.")}
        if set(state) != set(expected) or any(state[k].shape != v.shape for k, v in expected.items()):
            raise RuntimeError("Legacy D-human checkpoint does not match the original base network")
        for layer in self.pair_residual:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.zero_residual()
        return self.load_state_dict({**self.state_dict(), **state}, strict=True)
