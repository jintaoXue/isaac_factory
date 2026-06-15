from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    reward: float
    next_obs: torch.Tensor
    mask: torch.Tensor
    next_mask: torch.Tensor
    done: bool


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
