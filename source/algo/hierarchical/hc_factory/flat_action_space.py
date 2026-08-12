"""Fixed Cartesian joint action space for Flat TPA (joint index + joint mask).

Each joint index maps to ``(a_slot, b_slot, c_slot, h_slot, r_slot)`` and decodes to the
same env action dict as hierarchical A→B→C→D. Validity follows hierarchical mask semantics:
layer masks, B→C row coupling, and C→D allocation rules.
"""

from __future__ import annotations

import torch


class FlatJointActionSpace:
    """Enumerate legal (A,B,C,H,R) tuples as a fixed joint discrete space."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._ready = False
        self.num_a = 0
        self.b_dim = 0
        self.c_dim = 0
        self.h_dim = 0
        self.r_dim = 0
        self.a_slots = 0
        self.b_slots = 0
        self.c_slots = 0
        self.h_slots = 0
        self.r_slots = 0
        self.joint_dim = 0

    def _ensure_dims(self, env_state_action_dict: dict) -> None:
        if self._ready:
            return
        aam = env_state_action_dict["agent_action_mask"]
        self.num_a = int(aam["agent_A_product_sequencer"].shape[0])
        self.b_dim = int(aam["agent_B_product_selector"].shape[0])
        self.c_dim = int(aam["agent_C_process_task_planner"].shape[1])
        self.h_dim = int(aam["human"]["self_availability_mask"].shape[0])
        self.r_dim = int(aam["robot"]["self_availability_mask"].shape[0])
        self.a_slots = self.num_a + 1  # 0 = skip A
        self.b_slots = self.b_dim + 1  # 0 = skip B
        self.c_slots = self.c_dim
        self.h_slots = self.h_dim + 1
        self.r_slots = self.r_dim + 1
        self.joint_dim = self.a_slots * self.b_slots * self.c_slots * self.h_slots * self.r_slots
        self._ready = True
        print(
            f"[Flat] joint action dim={self.joint_dim} "
            f"(A={self.num_a} B={self.b_dim} C={self.c_dim} H={self.h_dim} R={self.r_dim})"
        )

    def _decode_slots(self, joint_idx: int) -> tuple[int, int, int, int, int]:
        r_slot = joint_idx % self.r_slots
        joint_idx //= self.r_slots
        h_slot = joint_idx % self.h_slots
        joint_idx //= self.h_slots
        c_slot = joint_idx % self.c_slots
        joint_idx //= self.c_slots
        b_slot = joint_idx % self.b_slots
        joint_idx //= self.b_slots
        a_slot = joint_idx
        return a_slot, b_slot, c_slot, h_slot, r_slot

    @staticmethod
    def _slot_to_one_hot(slot: int, dim: int, device: torch.device) -> torch.Tensor:
        out = torch.zeros(dim, dtype=torch.int32, device=device)
        if slot > 0:
            out[slot - 1] = 1
        return out

    def _is_valid(
        self,
        a_slot: int,
        b_slot: int,
        c_slot: int,
        h_slot: int,
        r_slot: int,
        aam: dict,
    ) -> bool:
        a_mask = aam["agent_A_product_sequencer"]
        b_mask = aam["agent_B_product_selector"]
        c_mask_2d = aam["agent_C_process_task_planner"]
        h_mask = aam["human"]["self_availability_mask"]
        r_mask = aam["robot"]["self_availability_mask"]

        if a_slot == 0:
            if int(a_mask.sum().item()) > 0:
                return False
        else:
            a_idx = a_slot - 1
            if a_idx >= self.num_a or int(a_mask[a_idx].item()) == 0:
                return False

        if b_slot == 0:
            if int(b_mask.sum().item()) > 0:
                return False
            return c_slot == 0 and h_slot == 0 and r_slot == 0

        b_idx = b_slot - 1
        if b_idx >= self.b_dim or int(b_mask[b_idx].item()) == 0:
            return False
        if int(c_mask_2d[b_idx, c_slot].item()) == 0:
            return False

        if c_slot == 0:
            return h_slot == 0 and r_slot == 0

        if h_slot == 0:
            if int(h_mask.sum().item()) > 0:
                return False
        else:
            h_idx = h_slot - 1
            if h_idx >= self.h_dim or int(h_mask[h_idx].item()) == 0:
                return False

        if r_slot == 0:
            if int(r_mask.sum().item()) > 0:
                return False
        else:
            r_idx = r_slot - 1
            if r_idx >= self.r_dim or int(r_mask[r_idx].item()) == 0:
                return False
        return True

    def build_joint_mask(self, env_state_action_dict: dict) -> torch.Tensor:
        """Return float mask of shape ``(joint_dim,)`` with 1.0 for legal joint indices."""
        self._ensure_dims(env_state_action_dict)
        aam = env_state_action_dict["agent_action_mask"]
        mask = torch.zeros(self.joint_dim, dtype=torch.float32, device=self.device)
        for joint_idx in range(self.joint_dim):
            slots = self._decode_slots(joint_idx)
            if self._is_valid(*slots, aam):
                mask[joint_idx] = 1.0
        if mask.sum() == 0:
            # Fallback: full no-op should always exist when B mask is empty.
            mask[0] = 1.0
        return mask

    def decode(self, joint_idx: int) -> dict:
        """Joint index → env action dict (same schema as hierarchical TPA)."""
        if not self._ready:
            raise RuntimeError("FlatJointActionSpace.decode called before dims initialized")
        a_slot, b_slot, c_slot, h_slot, r_slot = self._decode_slots(int(joint_idx))
        c = torch.zeros(self.c_dim, dtype=torch.int32, device=self.device)
        c[c_slot] = 1
        return {
            "product_sequencing": self._slot_to_one_hot(a_slot, self.num_a, self.device),
            "product_selection": self._slot_to_one_hot(b_slot, self.b_dim, self.device),
            "process_task_planning": c,
            "human_robot_allocation": {
                "human": self._slot_to_one_hot(h_slot, self.h_dim, self.device),
                "robot": self._slot_to_one_hot(r_slot, self.r_dim, self.device),
            },
        }

    def num_valid_actions(self, env_state_action_dict: dict) -> int:
        return int(self.build_joint_mask(env_state_action_dict).sum().item())
