"""Checkpoint discovery and loading for hier / flat TPA agents."""

from __future__ import annotations

import glob
import os
import re

import torch


def _latest_step_in_dir(nn_dir: str) -> int | None:
    steps: list[int] = []
    for path in glob.glob(os.path.join(nn_dir, "*_step_*.pth")):
        match = re.search(r"_step_(\d+)\.pth$", os.path.basename(path))
        if match:
            steps.append(int(match.group(1)))
    return max(steps) if steps else None


def resolve_nn_dir(config: dict, checkpoint_path: str | None = None) -> str | None:
    if checkpoint_path:
        path = os.path.abspath(checkpoint_path)
        if os.path.isdir(path):
            if os.path.basename(path) == "nn":
                return path
            nested = os.path.join(path, "nn")
            if os.path.isdir(nested):
                return nested
            return path
        return os.path.dirname(path)

    load_dir = str(config.get("load_dir") or "").strip()
    load_name = str(config.get("load_name") or "").strip()
    if load_dir and load_name:
        return os.path.join(load_dir, load_name, "nn")
    if load_dir:
        nested = os.path.join(load_dir, "nn")
        return nested if os.path.isdir(nested) else load_dir
    train_dir = config.get("train_dir")
    exp_name = config.get("full_experiment_name")
    if train_dir and exp_name:
        return os.path.join(train_dir, exp_name, "nn")
    return None


def resolve_step(config: dict, nn_dir: str | None, checkpoint_path: str | None = None) -> int | None:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        match = re.search(r"_step_(\d+)\.pth$", os.path.basename(checkpoint_path))
        if match:
            return int(match.group(1))
    load_step = config.get("load_step")
    if load_step is not None:
        return int(load_step)
    if nn_dir and os.path.isdir(nn_dir):
        return _latest_step_in_dir(nn_dir)
    return None


def load_hier_checkpoint(hier_agent, config: dict, checkpoint_path: str | None = None) -> str | None:
    nn_dir = resolve_nn_dir(config, checkpoint_path)
    if not nn_dir or not os.path.isdir(nn_dir):
        print(f"[Hier] checkpoint nn dir not found: {nn_dir}")
        return None
    step = resolve_step(config, nn_dir, checkpoint_path)
    if step is None:
        print(f"[Hier] no checkpoint step found under {nn_dir}")
        return None

    enc_path = os.path.join(nn_dir, f"state_encoder_step_{step}.pth")
    if os.path.isfile(enc_path):
        state = torch.load(enc_path, map_location=hier_agent.cuda_device, weights_only=True)
        hier_agent.obs_encoder.load_state_dict(state)
        print(f"[Hier] loaded encoder: {enc_path}")

    for agent, name in [
        (hier_agent.agent_A, "agent_A"),
        (hier_agent.agent_B, "agent_B"),
        (hier_agent.agent_C, "agent_C"),
    ]:
        path = os.path.join(nn_dir, f"{name}_step_{step}.pth")
        if os.path.isfile(path) and agent.dqn is not None:
            agent.dqn.load(path)
            print(f"[Hier] loaded {name}: {path}")

    for attr, fname in [("human_dqn", "agent_D_human"), ("robot_dqn", "agent_D_robot")]:
        path = os.path.join(nn_dir, f"{fname}_step_{step}.pth")
        dqn = getattr(hier_agent.agent_D, attr)
        if os.path.isfile(path) and dqn is not None:
            dqn.load(path)
            print(f"[Hier] loaded {fname}: {path}")

    return enc_path if os.path.isfile(enc_path) else nn_dir


def load_flat_checkpoint(flat_agent, config: dict, checkpoint_path: str | None = None) -> str | None:
    nn_dir = resolve_nn_dir(config, checkpoint_path)
    if not nn_dir or not os.path.isdir(nn_dir):
        print(f"[Flat] checkpoint nn dir not found: {nn_dir}")
        return None
    step = resolve_step(config, nn_dir, checkpoint_path)
    if step is None:
        print(f"[Flat] no checkpoint step found under {nn_dir}")
        return None

    enc_path = os.path.join(nn_dir, f"state_encoder_step_{step}.pth")
    if os.path.isfile(enc_path):
        state = torch.load(enc_path, map_location=flat_agent.cuda_device, weights_only=True)
        flat_agent.obs_encoder.load_state_dict(state)
        print(f"[Flat] loaded encoder: {enc_path}")

    q_path = os.path.join(nn_dir, f"flat_joint_step_{step}.pth")
    if os.path.isfile(q_path) and flat_agent.agent.dqn is not None:
        flat_agent.agent.dqn.load(q_path)
        print(f"[Flat] loaded flat Q-net: {q_path}")
    return q_path if os.path.isfile(q_path) else enc_path
