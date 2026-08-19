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
