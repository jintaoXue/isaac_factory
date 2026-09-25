"""D/C human–task match scoring (v1): dual-tower + duration-aligned features.

Unlike the shallow pair residual, match scoring is the primary human-assignment
inductive bias: Q = Q_context(base) + score(context, human_feats).

Features use η × skill_task × skill_sub(control_machine), matching timed process
duration stretch (see cfg_human / human._time_counting_subtask).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .hier_networks import QNetwork

# fatigue, η, skill_task, skill_sub_cm, speed, log_speed, work/recover, available, exists
MATCH_FEATURE_DIM = 10
TASK_MATCH_FEATURE_DIM = 5
_LOG_EPS = 1e-3
_SKILL_SUB_PROCESS = "control_machine"


def _skill_sub_column(pre: dict, action_dim: int) -> torch.Tensor:
    """Per-human skill_sub for the dominant timed process subtask."""
    human = pre["human"]
    device = human["skill_task"].device
    if "skill_subtask" in human:
        subs = human["skill_subtask"].to(device).float()
        # Layout follows cfg_human._HUMAN_SUBTASK_NAMES; control_machine is last.
        if subs.ndim == 2 and subs.shape[0] >= action_dim and subs.shape[1] >= 1:
            return subs[:action_dim, -1].clamp(0.35, 1.80)
    return torch.ones(action_dim, device=device, dtype=torch.float32)


def human_task_match_features(pre: dict, task_action: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Duration-aligned (task, human) features for D-human match head."""
    human = pre["human"]
    task = task_action.float().flatten()
    skills = human["skill_task"].to(task.device).float()[:action_dim]
    if skills.shape != (action_dim, 12) or task.numel() != 13:
        raise ValueError("Match v1 requires 12 skill columns and 13 task actions (including none)")

    def field(name: str) -> torch.Tensor:
        value = human[name].to(task.device).float().flatten()[:action_dim]
        if value.numel() != action_dim:
            raise ValueError(f"Match observation missing candidate slots: {name}")
        return value

    exists = field("mask")
    available = pre["agent_action_mask"]["human"]["self_availability_mask"].to(task.device).float().flatten()[:action_dim]
    fatigue = field("fatigue").clamp(0, 1)
    eta = field("efficiency").clamp(0.25, 1)
    skill_t = skills @ task[1:]
    skill_s = _skill_sub_column(pre, action_dim)
    # none-task → zero skill channels (no assignment signal)
    if float(task[0].item()) > 0.5:
        skill_t = torch.zeros_like(skill_t)
        skill_s = torch.zeros_like(skill_s)
    speed = (eta * skill_t * skill_s).clamp(0.05, 1.80)
    log_speed = torch.log(speed + _LOG_EPS)
    features = torch.stack(
        (
            fatigue,
            eta,
            skill_t,
            skill_s,
            speed,
            log_speed,
            field("fatigue_work_rate"),
            field("fatigue_recover_rate"),
            available,
            exists,
        ),
        dim=-1,
    )
    return features * exists.unsqueeze(-1)


def task_human_match_summary(pre: dict) -> torch.Tensor:
    """Per-task free-pool summary using duration-aligned speed."""
    h = pre["human"]
    skills = h["skill_task"].float()
    if skills.ndim != 2 or skills.shape[1] != 12:
        raise ValueError("C match requires the 12-task skill schema")
    available = pre.get(
        "_task_pair_available", pre["agent_action_mask"]["human"]["self_availability_mask"]
    ).to(skills.device).bool()
    exists = h["mask"].flatten().to(skills.device).bool()
    valid = available.flatten() & exists
    result = skills.new_zeros((13, TASK_MATCH_FEATURE_DIM))
    if not valid.any():
        return result
    eta = h["efficiency"].flatten().to(skills.device)[valid]
    fatigue = h["fatigue"].flatten().to(skills.device)[valid]
    skill_s = _skill_sub_column(pre, int(h["mask"].numel()))[valid]
    # speed[t] = η * skill_task[:,t] * skill_sub_cm
    values = eta[:, None] * skills[valid] * skill_s[:, None]
    best, indices = values.max(dim=0)
    best_fatigue = fatigue[indices]
    fraction = valid.sum().float() / exists.sum().clamp_min(1)
    result[1:] = torch.stack(
        (best, values.mean(dim=0), fraction.expand(12), best_fatigue, torch.ones_like(best)),
        dim=-1,
    )
    return result


class HumanMatchQNetwork(QNetwork):
    """Q_context(base) + dual-tower score(context, human) + prior·log_speed."""

    feature_width = MATCH_FEATURE_DIM
    _SPEED_IDX = 4
    _LOG_SPEED_IDX = 5
    _EXISTS_IDX = 9

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=128,
        *,
        pair_feature_dim=MATCH_FEATURE_DIM,
        match_hidden=64,
        match_dim=32,
        prior_init: float = 1.0,
        **kwargs,
    ):
        self.base_obs_dim = int(obs_dim) - int(action_dim) * int(pair_feature_dim)
        if self.base_obs_dim <= 0 or pair_feature_dim != self.feature_width:
            raise ValueError("Invalid match observation schema")
        # Drop pair-only kwargs if any leak through.
        kwargs.pop("duration_aux", None)
        super().__init__(self.base_obs_dim, action_dim, hidden_dim, **kwargs)
        self.pair_feature_dim = int(pair_feature_dim)
        self.match_dim = int(match_dim)
        self.human_tower = nn.Sequential(
            nn.Linear(self.pair_feature_dim, match_hidden),
            nn.ReLU(),
            nn.Linear(match_hidden, match_dim),
        )
        self.ctx_tower = nn.Sequential(
            nn.Linear(self.base_obs_dim, match_hidden),
            nn.ReLU(),
            nn.Linear(match_hidden, match_dim),
        )
        # Direct readout on features (learns to trust speed); last layer zero → safe hotstart.
        self.feat_score = nn.Sequential(
            nn.Linear(self.pair_feature_dim, match_hidden),
            nn.ReLU(),
            nn.Linear(match_hidden, 1),
        )
        nn.init.zeros_(self.feat_score[-1].weight)
        nn.init.zeros_(self.feat_score[-1].bias)
        # Positive prior: prefer higher log_speed immediately after T0 load.
        self.prior_weight = nn.Parameter(torch.tensor(float(prior_init)))

    def forward(self, obs):
        base = obs[..., : self.base_obs_dim]
        pairs = obs[..., self.base_obs_dim :].reshape(
            *obs.shape[:-1], self.action_dim, self.pair_feature_dim
        )
        exists = pairs[..., self._EXISTS_IDX]
        human_h = self.human_tower(pairs)
        ctx_h = self.ctx_tower(base).unsqueeze(-2).expand_as(human_h)
        bilinear = (human_h * ctx_h).sum(dim=-1) / math.sqrt(self.match_dim)
        feat = self.feat_score(pairs).squeeze(-1)
        prior = self.prior_weight * pairs[..., self._LOG_SPEED_IDX]
        score = (bilinear + feat + prior) * exists
        return super().forward(base) + score

    def load_compatible_state_dict(self, state):
        state = dict(state)
        if any(k.startswith(("human_tower.", "ctx_tower.", "feat_score.", "prior_weight")) for k in state):
            return self.load_state_dict(state, strict=True)
        # Strip pair residual keys if someone points a pair ckpt here.
        state = {k: v for k, v in state.items() if not k.startswith("pair_residual.")}
        expected = {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith(("human_tower.", "ctx_tower.", "feat_score.", "prior_weight"))
        }
        if set(state) != set(expected) or any(state[k].shape != v.shape for k, v in expected.items()):
            raise RuntimeError("Legacy D-human checkpoint does not match match-head base network")
        merged = {**self.state_dict(), **state}
        return self.load_state_dict(merged, strict=True)


class TaskMatchQNetwork(HumanMatchQNetwork):
    """C-head match scoring over per-task free-pool summaries."""

    feature_width = TASK_MATCH_FEATURE_DIM
    _EXISTS_IDX = 4
    _LOG_SPEED_IDX = 0  # best speed already in col0; use as prior channel
    _SPEED_IDX = 0

    def __init__(self, obs_dim, action_dim, hidden_dim=128, **kwargs):
        if action_dim != 13:
            raise ValueError("C match requires 13 task actions")
        kwargs.pop("duration_aux", None)
        kwargs.pop("pair_feature_dim", None)
        # Rebuild with task feature width (parent checks feature_width).
        pair_feature_dim = TASK_MATCH_FEATURE_DIM
        self.base_obs_dim = int(obs_dim) - int(action_dim) * int(pair_feature_dim)
        if self.base_obs_dim <= 0:
            raise ValueError("Invalid task-match observation schema")
        QNetwork.__init__(self, self.base_obs_dim, action_dim, hidden_dim, **kwargs)
        self.pair_feature_dim = pair_feature_dim
        match_hidden = 64
        match_dim = 32
        self.match_dim = match_dim
        self.human_tower = nn.Sequential(
            nn.Linear(self.pair_feature_dim, match_hidden),
            nn.ReLU(),
            nn.Linear(match_hidden, match_dim),
        )
        self.ctx_tower = nn.Sequential(
            nn.Linear(self.base_obs_dim, match_hidden),
            nn.ReLU(),
            nn.Linear(match_hidden, match_dim),
        )
        self.feat_score = nn.Sequential(
            nn.Linear(self.pair_feature_dim, match_hidden),
            nn.ReLU(),
            nn.Linear(match_hidden, 1),
        )
        nn.init.zeros_(self.feat_score[-1].weight)
        nn.init.zeros_(self.feat_score[-1].bias)
        self.prior_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, obs):
        base = obs[..., : self.base_obs_dim]
        pairs = obs[..., self.base_obs_dim :].reshape(
            *obs.shape[:-1], self.action_dim, self.pair_feature_dim
        )
        exists = pairs[..., self._EXISTS_IDX]
        # col0 is max speed; log for prior. none-row stays zero.
        log_speed = torch.log(pairs[..., 0].clamp_min(0.0) + _LOG_EPS)
        human_h = self.human_tower(pairs)
        ctx_h = self.ctx_tower(base).unsqueeze(-2).expand_as(human_h)
        bilinear = (human_h * ctx_h).sum(dim=-1) / math.sqrt(self.match_dim)
        feat = self.feat_score(pairs).squeeze(-1)
        prior = self.prior_weight * log_speed
        score = (bilinear + feat + prior) * exists
        return QNetwork.forward(self, base) + score
