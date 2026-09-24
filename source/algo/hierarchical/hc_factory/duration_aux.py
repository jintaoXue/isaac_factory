"""Completed-assignment supervision; observations are captured before dispatch."""
from __future__ import annotations

import copy
import math
import random
from collections import deque

import torch
from torch.nn import functional as F

from .hier_utils import detach_pre_to_cpu, pre_to_device


class DurationAuxReplay:
    def __init__(self, *, weight=0.05, scale=1000.0, capacity=2048, batch_size=32):
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("duration_aux requires finite weight>=0 and scale>0")
        self.weight, self.scale = float(weight), float(scale)
        self.batch_size = int(batch_size)
        self.samples = deque(maxlen=capacity)
        self.pending = {}
        self.completed = self.discarded = self.unmatched = self.updates = 0
        self.losses, self.errors = deque(maxlen=100), deque(maxlen=100)

    def observe(self, env_id, pre, events, *, done=False, restored=False):
        if restored:
            self._clear_env(env_id)
            return
        snapshot = None
        for event in events:
            key = (env_id, int(event['product']), int(event['task']), int(event['human']), int(event['start']))
            if event['kind'] == 'start':
                if snapshot is None:
                    # clone even on CPU: env/replay mutation must not change the label input.
                    snapshot = copy.deepcopy(detach_pre_to_cpu(pre))
                self.pending[key] = (snapshot, key[2], key[3])
            elif event['kind'] == 'complete':
                sample = self.pending.pop(key, None)
                if sample is None:
                    self.unmatched += 1  # e.g. task already running at a restored snapshot
                    continue
                duration = float(event['duration'])
                if math.isfinite(duration) and duration > 0:
                    self.samples.append((*sample, duration))
                    self.completed += 1
                else:
                    self.discarded += 1
        # Preserve this step's valid completions; discard unfinished/censored tasks.
        if done:
            self._clear_env(env_id)

    def _clear_env(self, env_id):
        keys = [key for key in self.pending if key[0] == env_id]
        self.discarded += len(keys)
        for key in keys:
            del self.pending[key]

    def compute_loss(self, allocator):
        if self.weight == 0 or len(self.samples) < self.batch_size:
            return None
        batch = random.sample(list(self.samples), self.batch_size)
        obs, actions, durations = [], [], []
        for pre, task, human, duration in batch:
            task_action = F.one_hot(torch.tensor(task, device=allocator.device), 13).float()
            obs.append(allocator.encode_human_obs(pre_to_device(pre, allocator.device), task_action))
            actions.append(human)
            durations.append(duration)
        predictions = allocator.human_dqn.q_net.predict_duration(torch.stack(obs))
        indices = torch.tensor(actions, device=allocator.device).unsqueeze(1)
        predicted = predictions.gather(1, indices).squeeze(1)
        actual = torch.tensor(durations, device=allocator.device, dtype=torch.float32)
        loss = F.smooth_l1_loss(predicted, torch.log1p(actual / self.scale))
        self.updates += 1
        self.losses.append(float(loss.detach()))
        self.errors.append(float((torch.expm1(predicted.detach().clamp(max=20)) * self.scale - actual).abs().mean()))
        return self.weight * loss

    def metrics(self):
        out = {"MetricAux/duration_samples": self.completed,
               "MetricAux/duration_buffer": len(self.samples),
               "MetricAux/duration_pending": len(self.pending),
               "MetricAux/duration_discarded": self.discarded,
               "MetricAux/duration_unmatched": self.unmatched,
               "MetricAux/duration_updates": self.updates}
        if self.losses:
            out["MetricAux/duration_loss"] = sum(self.losses) / len(self.losses)
            out["MetricAux/duration_mae_steps"] = sum(self.errors) / len(self.errors)
        return out
