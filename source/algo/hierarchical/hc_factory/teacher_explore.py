# -*- coding: utf-8 -*-
"""Frozen teacher policy for E3 guided online exploration (+E).

On the ε-explore branch, mix greedy actions from a frozen T0 copy vs uniform
random (mask-valid). Exploit branch always uses the live student at ε=0.
"""
from __future__ import annotations

import copy
import random
from types import SimpleNamespace
from typing import Any

import torch

from .hier_obs import HierObsEncoder
from .hier_rl_agents import (
    MaskedDQNAgent,
    RLHumanRobotAllocatorAgent,
    RLProcessTaskPlanningAgent,
    RLProductSelectionAgent,
    RLProductSequencingAgent,
)
from .offline_replay import _clear_buffer


def _freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _frozen_dqn_copy(src: MaskedDQNAgent | None) -> MaskedDQNAgent | None:
    if src is None:
        return None
    dst = copy.deepcopy(src)
    _clear_buffer(dst.buffer)
    _freeze_module(dst.q_net)
    _freeze_module(dst.target_net)
    return dst


def build_frozen_teacher(student: Any) -> SimpleNamespace:
    """Clone encoder + Q heads from ``student`` after warmstart load; never train."""
    parallel_limit = int(getattr(student.obs_encoder, "parallel_producing_limit", 10))
    state_dim = int(getattr(student.obs_encoder, "state_dim", 256))
    encoder = HierObsEncoder(
        student.cuda_device,
        parallel_producing_limit=parallel_limit,
        state_dim=state_dim,
    )
    encoder.load_state_dict(student.obs_encoder.state_dict())
    _freeze_module(encoder)

    # Tiny buffers: teacher never stores / learns.
    tiny = {"buffer_capacity": 1, "batch_size": 1}
    kw_a = dict(getattr(student.agent_A, "dqn_kwargs", {}) or {})
    kw_a.update(tiny)
    kw = dict(getattr(student.agent_B, "dqn_kwargs", {}) or {})
    kw.update(tiny)

    teacher = SimpleNamespace()
    teacher.obs_encoder = encoder
    teacher.c_forbid_none_mode = getattr(student, "c_forbid_none_mode", "always")
    teacher.agent_A = RLProductSequencingAgent(encoder, student.cuda_device, **kw_a)
    teacher.agent_B = RLProductSelectionAgent(encoder, student.cuda_device, **kw)
    teacher.agent_B.b_score_rl = bool(getattr(student.agent_B, "b_score_rl", False))
    teacher.agent_C = RLProcessTaskPlanningAgent(encoder, student.cuda_device, **kw)
    teacher.agent_D = RLHumanRobotAllocatorAgent(encoder, student.cuda_device, **kw)

    teacher.agent_A.dqn = _frozen_dqn_copy(student.agent_A.dqn)
    teacher.agent_B.dqn = _frozen_dqn_copy(student.agent_B.dqn)
    teacher.agent_C.dqn = _frozen_dqn_copy(student.agent_C.dqn)
    teacher.agent_D.human_dqn = _frozen_dqn_copy(student.agent_D.human_dqn)
    teacher.agent_D.robot_dqn = _frozen_dqn_copy(student.agent_D.robot_dqn)
    return teacher


def teacher_explore_ratio(
    *,
    env_step: int,
    ratio_start: float,
    ratio_end: float,
    decay_env_steps: int,
) -> float:
    """Linear decay of teacher share within the ε-explore branch."""
    start = float(ratio_start)
    end = float(ratio_end)
    horizon = max(1, int(decay_env_steps))
    t = min(1.0, float(max(0, int(env_step))) / float(horizon))
    return start + (end - start) * t


def choose_explore_policy(
    *,
    epsilon: float,
    teacher_ratio: float,
) -> str:
    """Return ``exploit`` | ``teacher`` | ``random`` for one decision interval."""
    eps = float(epsilon)
    if eps <= 0.0 or random.random() >= eps:
        return "exploit"
    if random.random() < float(teacher_ratio):
        return "teacher"
    return "random"
