from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Transition:
    """Replay entry: either preprocessed env dict (hier) or flat obs tensor (legacy)."""

    action: int
    reward: float
    mask: torch.Tensor
    next_mask: torch.Tensor
    done: bool
    # SMDP bootstrap multiplier (gamma ** elapsed simulation steps).
    discount: float | None = None
    # Parent action conditioning this hierarchical head (A->B, B->C, C->D).
    context: torch.Tensor | None = None
    pre: dict | None = None
    next_pre: dict | None = None
    obs: torch.Tensor | None = None
    next_obs: torch.Tensor | None = None


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def shrink_to(self, target_size: int) -> None:
        """Randomly drop entries until ``len <= target_size`` (keeps relative diversity)."""
        target = max(0, int(target_size))
        if len(self.buffer) <= target:
            return
        if target == 0:
            self.buffer.clear()
            return
        kept = random.sample(list(self.buffer), target)
        self.buffer = deque(kept, maxlen=self.capacity)
