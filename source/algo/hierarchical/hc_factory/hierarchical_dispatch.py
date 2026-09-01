# -*- coding: utf-8 -*-
"""Build env action via info-pool + A→B→C→D loop (supports parallel CD dispatch)."""
from __future__ import annotations

import torch

from .hc_factory_imports import CfgProcessTaskGalleryInAll

from .agent_A_product_sequencer import ProductSequencingAgent
from .agent_B_product_priority import ProductPriorityAgent
from .agent_C_process_task_planner import ProcessTaskPlanningAgent
from .agent_D_human_robot_allocator import HumanRobotMachineAllocationAgent
from .tpa_info_pool import TpaInfoPool


def _slot_to_one_hot(slot_index: int, dim: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(dim, dtype=torch.int32, device=device)
    if 0 <= slot_index < dim:
        out[slot_index] = 1
    return out


def _empty_allocation(human_dim: int, robot_dim: int, device: torch.device) -> dict:
    return {
        "human": torch.zeros(human_dim, dtype=torch.int32, device=device),
        "robot": torch.zeros(robot_dim, dtype=torch.int32, device=device),
    }


def _c_mask_for_rl_act(
    c_mask: torch.Tensor,
    slot_index: int,
    pool: TpaInfoPool,
    *,
    forbid_none_mode: str = "always",
) -> torch.Tensor:
    """Adjust C action mask before RL act.

    ``always`` (default, Rule-aligned): when real tasks exist, forbid none on any slot.
    Needed for greedy eval with checkpoints that over-prefer none on WIP slots.

    ``staging_only``: forbid none only on staging while ``next_product`` blocks A.
    WIP may still defer; safe only if the policy rarely greedy-selects none on WIP.
    """
    c_mask_for_act = c_mask.clone()
    if int(c_mask_for_act.sum().item()) <= 1:
        return c_mask_for_act
    if forbid_none_mode == "always":
        c_mask_for_act[0] = 0
        return c_mask_for_act
    if forbid_none_mode == "staging_only":
        is_staging = slot_index == pool.staging_slot
        blocks_sequencing = pool.progress.get("next_product") is not None
        if is_staging and blocks_sequencing:
            c_mask_for_act[0] = 0
        return c_mask_for_act
    raise ValueError(f"unknown c_forbid_none_mode={forbid_none_mode!r}")


def build_rule_based_action(
    env_state_action_dict: dict,
    cuda_device: torch.device,
    max_parallel_cd_dispatch: int = 1,
    agent_a: ProductSequencingAgent | None = None,
    agent_b: ProductPriorityAgent | None = None,
    agent_c: ProcessTaskPlanningAgent | None = None,
    agent_d: HumanRobotMachineAllocationAgent | None = None,
) -> dict:
    agent_a = agent_a or ProductSequencingAgent(cuda_device)
    agent_b = agent_b or ProductPriorityAgent(cuda_device)
    agent_c = agent_c or ProcessTaskPlanningAgent(cuda_device)
    agent_d = agent_d or HumanRobotMachineAllocationAgent(cuda_device)

    pool = TpaInfoPool(env_state_action_dict, cuda_device)
    product_sequencing = agent_a.act(env_state_action_dict)
    pool.apply_product_sequencing(product_sequencing)

    eligible = pool.compute_b_eligible_mask()
    slot_order = agent_b.rank_slots(eligible)
    b_dim = eligible.shape[0]
    human_dim = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].shape[0]
    robot_dim = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].shape[0]

    dispatch_list: list[dict] = []
    inverse_task = {v: k for k, v in CfgProcessTaskGalleryInAll.items()}
    k_limit = max(1, int(max_parallel_cd_dispatch))

    for slot_index in slot_order:
        if len(dispatch_list) >= k_limit:
            break
        if slot_index in pool.served_slots:
            continue

        c_mask = pool.get_c_mask_for_slot(slot_index)
        process_task_planning = agent_c.act_for_slot(cuda_device, c_mask)
        if process_task_planning[0] == 1 and c_mask.sum() <= 1:
            continue

        d_masks = pool.get_d_masks()
        human_robot_allocation = agent_d.act_with_masks(process_task_planning, d_masks)
        if process_task_planning[0] != 1:
            if human_robot_allocation["human"].sum() == 0:
                continue

        task_name = TpaInfoPool.task_name_from_planning(process_task_planning, inverse_task)
        if task_name == "none":
            continue

        h_idx, r_idx = TpaInfoPool.allocation_indices(human_robot_allocation)
        pool.consume_dispatch(slot_index, task_name, h_idx, r_idx)

        dispatch_list.append(
            {
                "slot_index": int(slot_index),
                "process_task_planning": process_task_planning,
                "human_robot_allocation": human_robot_allocation,
            }
        )

    product_priority = agent_b.scores_from_order(eligible, slot_order, b_dim, cuda_device)

    if dispatch_list:
        first = dispatch_list[0]
        first_slot = int(first["slot_index"])
        product_selection = _slot_to_one_hot(first_slot, b_dim, cuda_device)
        process_task_planning = first["process_task_planning"]
        human_robot_allocation = first["human_robot_allocation"]
    else:
        product_selection = torch.zeros(b_dim, dtype=torch.int32, device=cuda_device)
        c_dim = env_state_action_dict["agent_action_mask"]["agent_C_process_task_planner"].shape[1]
        process_task_planning = torch.zeros(c_dim, dtype=torch.int32, device=cuda_device)
        process_task_planning[0] = 1
        human_robot_allocation = _empty_allocation(human_dim, robot_dim, cuda_device)

    return {
        "product_sequencing": product_sequencing,
        "product_priority": product_priority,
        "dispatch_list": dispatch_list,
        "product_selection": product_selection,
        "process_task_planning": process_task_planning,
        "human_robot_allocation": human_robot_allocation,
    }


def build_hier_rl_action(
    env_state_action_dict: dict,
    cuda_device: torch.device,
    agents,
    epsilon: float,
    max_parallel_cd_dispatch: int = 1,
    pre: dict | None = None,
) -> dict:
    """RL variant: A/B/C/D agents with same info-pool CD loop (Phase 1: K=1 typical)."""
    if pre is None:
        pre = agents.obs_encoder.preprocess(env_state_action_dict)
    pool = TpaInfoPool(env_state_action_dict, cuda_device)
    product_sequencing = agents.agent_A.act(env_state_action_dict, epsilon, pre=pre)
    pool.apply_product_sequencing(product_sequencing)

    eligible = pool.compute_b_eligible_mask()
    slot_order = agents.agent_B.rank_slots(
        env_state_action_dict, eligible, epsilon, product_sequencing, pre=pre
    )
    b_dim = eligible.shape[0]
    human_dim = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].shape[0]
    robot_dim = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].shape[0]

    dispatch_list: list[dict] = []
    inverse_task = {v: k for k, v in CfgProcessTaskGalleryInAll.items()}
    k_limit = max(1, int(max_parallel_cd_dispatch))

    for slot_index in slot_order:
        if len(dispatch_list) >= k_limit:
            break
        if slot_index in pool.served_slots:
            continue

        c_mask = pool.get_c_mask_for_slot(slot_index)
        forbid_none_mode = getattr(agents, "c_forbid_none_mode", "always")
        c_mask_for_act = _c_mask_for_rl_act(
            c_mask, slot_index, pool, forbid_none_mode=forbid_none_mode
        )
        slot_one_hot = _slot_to_one_hot(slot_index, b_dim, cuda_device)
        process_task_planning = agents.agent_C.act_with_mask(
            env_state_action_dict, slot_one_hot, c_mask_for_act, epsilon, pre=pre
        )
        if process_task_planning[0] == 1 and c_mask.sum() <= 1:
            continue

        d_masks = pool.get_d_masks()
        human_robot_allocation = agents.agent_D.act_with_masks(
            env_state_action_dict, process_task_planning, d_masks, epsilon, pre=pre
        )
        if process_task_planning[0] != 1 and human_robot_allocation["human"].sum() == 0:
            continue

        task_name = TpaInfoPool.task_name_from_planning(process_task_planning, inverse_task)
        if task_name == "none":
            continue

        h_idx, r_idx = TpaInfoPool.allocation_indices(human_robot_allocation)
        pool.consume_dispatch(slot_index, task_name, h_idx, r_idx)
        dispatch_list.append(
            {
                "slot_index": int(slot_index),
                "product_selection": slot_one_hot,
                "process_task_planning": process_task_planning,
                "human_robot_allocation": human_robot_allocation,
            }
        )

    product_priority = agents.agent_B.scores_from_order(eligible, slot_order, b_dim, cuda_device)

    if dispatch_list:
        first = dispatch_list[0]
        first_slot = int(first["slot_index"])
        product_selection = _slot_to_one_hot(first_slot, b_dim, cuda_device)
        process_task_planning = first["process_task_planning"]
        human_robot_allocation = first["human_robot_allocation"]
    else:
        product_selection = torch.zeros(b_dim, dtype=torch.int32, device=cuda_device)
        c_dim = env_state_action_dict["agent_action_mask"]["agent_C_process_task_planner"].shape[1]
        process_task_planning = torch.zeros(c_dim, dtype=torch.int32, device=cuda_device)
        process_task_planning[0] = 1
        human_robot_allocation = _empty_allocation(human_dim, robot_dim, cuda_device)

    return {
        "product_sequencing": product_sequencing,
        "product_priority": product_priority,
        "dispatch_list": dispatch_list,
        "product_selection": product_selection,
        "process_task_planning": process_task_planning,
        "human_robot_allocation": human_robot_allocation,
    }
