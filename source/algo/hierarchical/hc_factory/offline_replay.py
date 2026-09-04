"""Offline replay dump / load for Hier4TPA ORU (T1/T2).

Layout under catalog root::

    offline_replay/
      meta.json
      A.pt
      B.pt
      C.pt
      D_human.pt
      D_robot.pt

Each ``*.pt`` is a ``torch.save`` of ``list[Transition]``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .hier_buffer import PrioritizedReplayBuffer, ReplayBuffer, Transition

HEADS = ("A", "B", "C", "D_human", "D_robot")


def _buffer_transitions(
    buf: ReplayBuffer | PrioritizedReplayBuffer | list[Transition] | None,
) -> list[Transition]:
    if buf is None:
        return []
    if isinstance(buf, ReplayBuffer):
        return list(buf.buffer)
    if isinstance(buf, PrioritizedReplayBuffer):
        return [t for t in buf._data[: buf._size] if t is not None]
    return list(buf)


def offline_replay_dir(catalog_root: str | Path) -> Path:
    return Path(catalog_root) / "offline_replay"


def default_warmup_updates(
    n_transitions: int,
    *,
    batch_size: int = 64,
    epochs: float = 5.0,
    min_updates: int = 2000,
    max_updates: int = 15000,
) -> int:
    """Size ORU warmup from offline corpus.

    Rule of thumb: ``epochs`` passes over the largest head, clipped to
    ``[min_updates, max_updates]``. With explore≈20 ep and buffer≤1e5,
    this typically lands near 5k–15k updates (a few minutes of GPU, not hours).
    """
    if n_transitions <= 0 or batch_size <= 0:
        return 0
    raw = int(math.ceil(epochs * float(n_transitions) / float(batch_size)))
    return int(min(max_updates, max(min_updates, raw)))


def _transition_to_state(t: Transition) -> dict[str, Any]:
    return {
        "action": int(t.action),
        "reward": float(t.reward),
        "mask": t.mask,
        "next_mask": t.next_mask,
        "done": bool(t.done),
        "discount": t.discount,
        "context": t.context,
        "pre": t.pre,
        "next_pre": t.next_pre,
        "obs": t.obs,
        "next_obs": t.next_obs,
    }


def _transition_from_state(d: dict[str, Any]) -> Transition:
    return Transition(
        action=int(d["action"]),
        reward=float(d["reward"]),
        mask=d["mask"],
        next_mask=d["next_mask"],
        done=bool(d["done"]),
        discount=d.get("discount"),
        context=d.get("context"),
        pre=d.get("pre"),
        next_pre=d.get("next_pre"),
        obs=d.get("obs"),
        next_obs=d.get("next_obs"),
    )


def save_offline_replay(
    root: str | Path,
    buffers: dict[str, ReplayBuffer | PrioritizedReplayBuffer | list[Transition] | None],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write per-head transition lists under ``root/offline_replay``."""
    out = offline_replay_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for head in HEADS:
        transitions = _buffer_transitions(buffers.get(head))
        path = out / f"{head}.pt"
        payload = [_transition_to_state(t) for t in transitions]
        torch.save(payload, path)
        counts[head] = len(payload)
    meta_payload = {
        "heads": HEADS,
        "counts": counts,
        "n_total": int(sum(counts.values())),
        **(meta or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(
        f"[ORU] saved offline replay → {out} "
        + " ".join(f"{h}={counts[h]}" for h in HEADS)
    )
    return out


def load_offline_replay(
    root: str | Path,
    *,
    capacity: int | None = None,
) -> dict[str, ReplayBuffer]:
    """Load per-head offline buffers from catalog root (or the offline_replay dir)."""
    root = Path(root)
    out_dir = root if (root / "meta.json").is_file() else offline_replay_dir(root)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"offline replay not found: {out_dir}")
    buffers: dict[str, ReplayBuffer] = {}
    for head in HEADS:
        path = out_dir / f"{head}.pt"
        transitions: list[Transition] = []
        if path.is_file():
            raw = torch.load(path, map_location="cpu", weights_only=False)
            transitions = [_transition_from_state(d) for d in raw]
        cap = int(capacity) if capacity is not None else max(len(transitions), 1)
        buf = ReplayBuffer(cap)
        for t in transitions:
            buf.push(t)
        buffers[head] = buf
    meta_path = out_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(
            f"[ORU] loaded offline replay ← {out_dir} "
            + " ".join(f"{h}={len(buffers[h])}" for h in HEADS)
            + f" meta_total={meta.get('n_total')}"
        )
    else:
        print(f"[ORU] loaded offline replay ← {out_dir}")
    return buffers


class ORUController:
    """Warmup + decaying offline mix; optionally shrinks offline buffers with mix."""

    def __init__(
        self,
        offline: dict[str, ReplayBuffer],
        *,
        enabled: bool,
        warmup_updates: int,
        mix_start: float = 1.0,
        mix_decay_env_steps: int = 300_000,
        shrink_offline: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and any(len(b) > 0 for b in offline.values())
        self.offline = offline
        self.initial_sizes = {h: len(b) for h, b in offline.items()}
        self.warmup_updates = max(0, int(warmup_updates)) if self.enabled else 0
        self.mix_start = float(max(0.0, min(1.0, mix_start)))
        self.mix_decay_env_steps = max(1, int(mix_decay_env_steps))
        self.shrink_offline = bool(shrink_offline)
        self.warmup_done = 0
        self._online_started = False
        self._online_start_env_step = 0
        self._mix_ratio = self.mix_start if self.enabled else 0.0
        self._last_shrink_mix = self._mix_ratio
        self._finished = not self.enabled

    @property
    def in_warmup(self) -> bool:
        return self.enabled and self.warmup_done < self.warmup_updates

    @property
    def mix_ratio(self) -> float:
        if not self.enabled or self._finished:
            return 0.0
        if self.in_warmup:
            return 1.0
        return float(self._mix_ratio)

    def mark_online_started(self, env_step: int = 0) -> None:
        self._online_started = True
        self._online_start_env_step = int(env_step)
        self._mix_ratio = self.mix_start
        self._last_shrink_mix = self._mix_ratio

    def on_env_step(self, env_step: int) -> float:
        """Update mix from env steps since online start; shrink offline buffers."""
        if not self.enabled or self._finished or self.in_warmup:
            return self.mix_ratio
        if not self._online_started:
            self.mark_online_started(env_step)
        elapsed = max(0, int(env_step) - int(self._online_start_env_step))
        t = min(1.0, float(elapsed) / float(self.mix_decay_env_steps))
        mix = self.mix_start * (1.0 - t)
        if mix <= 1e-8:
            mix = 0.0
            self._finished = True
            for buf in self.offline.values():
                buf.buffer.clear()
            print("[ORU] offline mix reached 0; offline buffer cleared")
        self._mix_ratio = mix
        # Shrink only when mix drops by ≥1% (avoid resampling every sim step).
        if self.shrink_offline and not self._finished and (self._last_shrink_mix - mix) >= 0.01:
            for head, buf in self.offline.items():
                target = int(math.floor(self.initial_sizes.get(head, 0) * mix))
                buf.shrink_to(target)
            self._last_shrink_mix = mix
        return mix

    def note_warmup_step(self) -> None:
        self.warmup_done += 1

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "MetricORU/mix_ratio": float(self.mix_ratio),
            "MetricORU/warmup_done": int(self.warmup_done),
            "MetricORU/warmup_total": int(self.warmup_updates),
            "MetricORU/offline_A": int(len(self.offline.get("A", []))),
            "MetricORU/offline_B": int(len(self.offline.get("B", []))),
            "MetricORU/offline_C": int(len(self.offline.get("C", []))),
            "MetricORU/offline_D_human": int(len(self.offline.get("D_human", []))),
            "MetricORU/offline_D_robot": int(len(self.offline.get("D_robot", []))),
        }
