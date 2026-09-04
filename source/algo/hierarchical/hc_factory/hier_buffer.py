from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
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


class PrioritizedReplayBuffer:
    """Proportional prioritization (Schaul et al.) over a fixed ring of Transitions.

    Sampling returns ``(batch, indices, importance_weights)``. Call
    ``update_priorities`` after computing TD errors.
    """

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        eps: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = max(1, int(beta_frames))
        self.eps = float(eps)
        self._data: list[Transition | None] = [None] * self.capacity
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0
        self._frame = 0

    def __len__(self) -> int:
        return self._size

    @property
    def beta(self) -> float:
        # Anneal β from beta_start → 1.0 over beta_frames sample calls.
        t = min(1.0, self._frame / float(self.beta_frames))
        return self.beta_start + t * (1.0 - self.beta_start)

    def push(self, transition: Transition) -> None:
        self._data[self._pos] = transition
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        assert self._size > 0
        batch_size = min(int(batch_size), self._size)
        prios = self._priorities[: self._size]
        probs = prios**self.alpha
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0:
            probs = np.ones(self._size, dtype=np.float64) / float(self._size)
        else:
            probs = probs / probs_sum
        indices = np.random.choice(self._size, size=batch_size, replace=False, p=probs)
        batch = [self._data[int(i)] for i in indices]
        assert all(t is not None for t in batch)
        beta = self.beta
        self._frame += 1
        weights = (self._size * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        return batch, indices.astype(np.int64), weights.astype(np.float32)  # type: ignore[return-value]

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        td = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        for idx, p in zip(indices, td):
            i = int(idx)
            self._priorities[i] = p
            if p > self._max_priority:
                self._max_priority = float(p)

    def shrink_to(self, target_size: int) -> None:
        """Drop lowest-priority entries until ``len <= target_size``."""
        target = max(0, int(target_size))
        if self._size <= target:
            return
        if target == 0:
            self._data = [None] * self.capacity
            self._priorities[:] = 0.0
            self._pos = 0
            self._size = 0
            return
        order = np.argsort(-self._priorities[: self._size])  # high priority first
        keep = order[:target]
        new_data: list[Transition | None] = [None] * self.capacity
        new_prios = np.zeros(self.capacity, dtype=np.float64)
        for j, src in enumerate(keep):
            new_data[j] = self._data[int(src)]
            new_prios[j] = self._priorities[int(src)]
        self._data = new_data
        self._priorities = new_prios
        self._size = target
        self._pos = target % self.capacity
        self._max_priority = float(new_prios[:target].max()) if target > 0 else 1.0
