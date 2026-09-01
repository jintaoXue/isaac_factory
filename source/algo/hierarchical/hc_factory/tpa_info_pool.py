# -*- coding: utf-8 -*-
"""Algorithm-side info pool for within-step parallel CD dispatch.

Maintains shadow resource state (human/robot/gantry/machine workstations) so
C/D masks stay consistent across multiple dispatches in one env step.
"""
from __future__ import annotations

import copy

import torch

from .hc_factory_imports import (
    AlgoHierarchicalMasker,
    CfgProcessTaskGalleryDetailedClassified,
    CfgProductProcess,
    HcVectorEnvCfg,
    find_free_gantry_index,
    find_workstation_index_for_task,
    staging_slot_index,
    maybe_release_unready_next_product,
    next_gallery_task_name,
    task_required_materials_ready,
    wip_cap,
)


class TpaInfoPool:
    """Per-step working copy of env state for hierarchical A→B→C→D loops."""

    def __init__(self, env_state_action_dict: dict, cuda_device: torch.device):
        self.device = cuda_device
        self.parallel_producing_limit = int(HcVectorEnvCfg().single_env_parallel_producing_limit)
        self.staging_slot = staging_slot_index(self.parallel_producing_limit)

        self.progress = copy.deepcopy(env_state_action_dict["progress"])
        self._env_ongoing_keys = set(env_state_action_dict["progress"]["ongoing_task_records"].keys())

        self.material = copy.deepcopy(env_state_action_dict["material"])
        self.human = copy.deepcopy(env_state_action_dict["human"])
        self.robot = copy.deepcopy(env_state_action_dict["robot"])
        self.machine = copy.deepcopy(env_state_action_dict["machine"])

        self.dispatched_product_indices: set[int] = set()
        self.served_slots: set[int] = set()
        self.used_human_indices: set[int] = set()
        self.used_robot_indices: set[int] = set()

        self._masker = AlgoHierarchicalMasker(cuda_device)
        self._working = self._build_working_dict(env_state_action_dict)
        self._masker.generate_agents_mask(self._working)

    def _build_working_dict(self, env_state_action_dict: dict) -> dict:
        return {
            "progress": self.progress,
            "human": self.human,
            "robot": self.robot,
            "machine": self.machine,
            "material": self.material,
            "agent_action_mask": copy.deepcopy(env_state_action_dict["agent_action_mask"]),
        }

    def apply_product_sequencing(self, product_sequencing: torch.Tensor) -> None:
        """Mirror ``TaskManager.decode_action_product_sequencing`` on pool progress."""
        maybe_release_unready_next_product(self._working)
        if product_sequencing.sum() == 0:
            self._refresh_masks()
            return
        if self.progress.get("next_product") is not None:
            return
        product_type_index = int(product_sequencing.nonzero()[0][0].item())
        product_type = list(CfgProductProcess.keys())[product_type_index]
        for material_state in self.material.values():
            key_variables = material_state["key_variables"]
            if (
                key_variables["type_name"] == product_type
                and material_state["finished_task"] == "none"
                and material_state["ongoing_task_record_index"] is None
            ):
                next_task = next_gallery_task_name(product_type, material_state["finished_task"])
                if not task_required_materials_ready(
                    self._working, product_type, key_variables["idx"], next_task
                ):
                    continue
                self.progress["next_product"] = product_type
                self.progress["next_product_index"] = key_variables["idx"]
                break
        self._refresh_masks()

    def _ongoing_product_indices(self) -> set[int]:
        return self._env_ongoing_keys | self.dispatched_product_indices

    def compute_b_eligible_mask(self) -> torch.Tensor:
        """Eligible B slots: no ongoing task on that product batch."""
        mask = torch.zeros(self.parallel_producing_limit + 1, dtype=torch.int32, device=self.device)
        ongoing = self._ongoing_product_indices()
        producing_indexs = self.progress["producing_indexs"]
        for i in range(len(producing_indexs)):
            if producing_indexs[i] not in ongoing and i not in self.served_slots:
                mask[i] = 1
        producing = self.progress["producing"]
        if (
            self.progress.get("next_product") is not None
            and len(producing) < wip_cap(self.progress, self.parallel_producing_limit)
            and self.staging_slot not in self.served_slots
        ):
            mask[self.staging_slot] = 1
        return mask

    def _refresh_masks(self) -> None:
        self._masker.generate_agents_mask(self._working)

    def get_c_mask_for_slot(self, slot_index: int) -> torch.Tensor:
        self._refresh_masks()
        b_mask = self.compute_b_eligible_mask()
        if b_mask[slot_index].item() == 0:
            c_dim = self._working["agent_action_mask"]["agent_C_process_task_planner"].shape[1]
            return torch.zeros(c_dim, dtype=torch.int32, device=self.device)

        c_2d = self._working["agent_action_mask"]["agent_C_process_task_planner"]
        row = c_2d[slot_index].clone()
        product_index = self._slot_to_product_index(slot_index)
        if product_index is not None:
            material_row = self._working["agent_action_mask"]["material"]["task_availability_mask"][product_index]
            human_t = self._working["agent_action_mask"]["human"]["task_availability_mask"]
            robot_t = self._working["agent_action_mask"]["robot"]["task_availability_mask"]
            machine_t = self._working["agent_action_mask"]["machine"]["task_availability_mask"]
            combined = human_t & robot_t & machine_t & material_row
            row = row & combined
        return row

    def _slot_to_product_index(self, slot_index: int) -> int | None:
        if slot_index == self.staging_slot:
            return self.progress.get("next_product_index")
        if slot_index < len(self.progress["producing_indexs"]):
            return self.progress["producing_indexs"][slot_index]
        return None

    def get_d_masks(self) -> dict[str, torch.Tensor]:
        self._refresh_masks()
        human_mask = self._working["agent_action_mask"]["human"]["self_availability_mask"].clone()
        robot_mask = self._working["agent_action_mask"]["robot"]["self_availability_mask"].clone()
        for idx in self.used_human_indices:
            if idx < human_mask.shape[0]:
                human_mask[idx] = 0
        for idx in self.used_robot_indices:
            if idx < robot_mask.shape[0]:
                robot_mask[idx] = 0
        return {"human": human_mask, "robot": robot_mask}

    def consume_dispatch(
        self,
        slot_index: int,
        task_name: str,
        human_index: int | None,
        robot_index: int | None,
    ) -> None:
        """Update shadow state after a successful C→D pair (mirrors env resource占用)."""
        product_index = self._slot_to_product_index(slot_index)
        if product_index is None:
            return
        product_type = (
            self.progress["next_product"]
            if slot_index == self.staging_slot
            else self.progress["producing"][slot_index]
        )
        task_type = CfgProcessTaskGalleryDetailedClassified[product_type][task_name]["task_type"]
        task_cfg = CfgProcessTaskGalleryDetailedClassified[product_type][task_name]
        target_machine = task_cfg["target_machine"]

        states = self.machine[target_machine]["state"]
        ws_index = find_workstation_index_for_task(states, task_type, task_name)
        if ws_index is not None:
            self.machine[target_machine]["state"][ws_index] = f"working_{task_name}"
            self.machine[target_machine]["ongoing_task_record_index"][ws_index] = product_index

        if task_type == "logistic":
            gantry_states = self.machine["num07_gantry_group"]["state"]
            gantry_idx = find_free_gantry_index(gantry_states)
            if gantry_idx is not None:
                gantry_states[gantry_idx] = f"working_{task_name}"
                self.machine["num07_gantry_group"]["ongoing_task_record_index"][gantry_idx] = product_index

        if human_index is not None:
            self.used_human_indices.add(human_index)
            human_keys = list(self.human.keys())
            if human_index < len(human_keys):
                hname = human_keys[human_index]
                self.human[hname]["state"] = f"working_{task_name}"
                self.human[hname]["ongoing_task_record_index"] = product_index

        if robot_index is not None:
            self.used_robot_indices.add(robot_index)
            robot_keys = list(self.robot.keys())
            if robot_index < len(robot_keys):
                rname = robot_keys[robot_index]
                self.robot[rname]["state"] = f"working_{task_name}"
                self.robot[rname]["ongoing_task_record_index"] = product_index

        material_name = f"num_{product_index:02d}_{product_type}"
        self.material[material_name]["ongoing_task_record_index"] = product_index
        self.dispatched_product_indices.add(product_index)
        self.served_slots.add(slot_index)

        if slot_index == self.staging_slot:
            self.progress["producing"].append(product_type)
            self.progress["producing_indexs"].append(product_index)
            self.progress["next_product"] = None
            self.progress["next_product_index"] = None
            self.progress["not_started"][product_type] -= 1

        self._refresh_masks()

    @staticmethod
    def task_name_from_planning(process_task_planning: torch.Tensor, inverse_index_to_task_name: dict) -> str:
        if process_task_planning.sum() == 0:
            return "none"
        idx = int(process_task_planning.nonzero()[0][0].item())
        return inverse_index_to_task_name[idx]

    @staticmethod
    def allocation_indices(human_robot_allocation: dict) -> tuple[int | None, int | None]:
        human_idx = None
        robot_idx = None
        h = human_robot_allocation.get("human")
        if h is not None and h.sum() == 1:
            human_idx = int(h.nonzero()[0][0].item())
        r = human_robot_allocation.get("robot")
        if r is not None and r.sum() == 1:
            robot_idx = int(r.nonzero()[0][0].item())
        return human_idx, robot_idx
