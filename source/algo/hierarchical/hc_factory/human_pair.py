"""Optional D-human pair residual. The original QNetwork stays unchanged."""
from __future__ import annotations

import torch
from torch import nn
from .hier_networks import QNetwork

# Version 1: fatigue, eta, current-task skill, eta*skill, work/recovery rates,
# availability, existence. Do not use stale last-subtask skill_effective.
PAIR_FEATURE_DIM = 8
TASK_PAIR_FEATURE_DIM = 5


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
    feature_width = PAIR_FEATURE_DIM

    def __init__(self, obs_dim, action_dim, hidden_dim=128, *, pair_feature_dim=PAIR_FEATURE_DIM, duration_aux=False, **kwargs):
        self.base_obs_dim = int(obs_dim) - int(action_dim) * int(pair_feature_dim)
        if self.base_obs_dim <= 0 or pair_feature_dim != self.feature_width:
            raise ValueError("Invalid pair observation schema")
        super().__init__(self.base_obs_dim, action_dim, hidden_dim, **kwargs)
        self.pair_feature_dim = pair_feature_dim
        self.pair_residual = nn.Sequential(
            nn.Linear(self.base_obs_dim + pair_feature_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.zero_residual()
        if duration_aux:
            self.duration_head = nn.Linear(64, 1)

    def predict_duration(self, obs):
        base = obs[..., :self.base_obs_dim]
        pairs = obs[..., self.base_obs_dim:].reshape(*obs.shape[:-1], self.action_dim, self.pair_feature_dim)
        context = base.unsqueeze(-2).expand(*base.shape[:-1], self.action_dim, self.base_obs_dim)
        shared = self.pair_residual[:2](torch.cat((context, pairs), dim=-1))
        return nn.functional.softplus(self.duration_head(shared).squeeze(-1))

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
        state = dict(state)
        if hasattr(self, "duration_head") and not any(k.startswith("duration_head.") for k in state):
            self.duration_head.reset_parameters()
            state.update({k: v for k, v in self.state_dict().items() if k.startswith("duration_head.")})
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


def task_human_summary(pre: dict) -> torch.Tensor:
    """Per-task max/mean eta*skill, free fraction, best-worker fatigue, valid flag.

    Current simulator permits every free human for every task; task feasibility
    remains enforced by the existing C mask. Padded workers never participate.
    """
    h = pre["human"]
    skills = h["skill_task"].float()
    if skills.ndim != 2 or skills.shape[1] != 12:
        raise ValueError("C pair requires the 12-task skill schema")
    available = pre.get("_task_pair_available", pre["agent_action_mask"]["human"]["self_availability_mask"]).to(skills.device).bool()
    exists = h["mask"].flatten().to(skills.device).bool()
    valid = available.flatten() & exists
    result = skills.new_zeros((13, TASK_PAIR_FEATURE_DIM))
    if valid.any():
        values = h["efficiency"].flatten().to(skills.device)[valid, None] * skills[valid]
        best, indices = values.max(dim=0)
        fatigue = h["fatigue"].flatten().to(skills.device)[valid][indices]
        fraction = valid.sum().float() / exists.sum().clamp_min(1)
        result[1:] = torch.stack((best, values.mean(dim=0), fraction.expand(12), fatigue, torch.ones_like(best)), -1)
    return result


class TaskHumanPairQNetwork(HumanPairQNetwork):
    """Same compatible residual mechanism, with task-side summary features."""
    feature_width = TASK_PAIR_FEATURE_DIM

    def __init__(self, obs_dim, action_dim, hidden_dim=128, **kwargs):
        if action_dim != 13:
            raise ValueError("C pair requires 13 task actions")
        super().__init__(obs_dim, action_dim, hidden_dim,
                         pair_feature_dim=TASK_PAIR_FEATURE_DIM, **kwargs)
