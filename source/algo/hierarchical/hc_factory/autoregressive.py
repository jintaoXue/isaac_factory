# -*- coding: utf-8 -*-
"""Autoregressive decision helpers for E5/E6 (+A).

Protocol (docs/experiment_protocol.md):
  - Layered epsilon: per-layer scales on the shared base ε.
  - In-step candidate sampling: softmax-sample a few mask-valid actions,
    then pick the highest-Q among them (online act only; no replay rewrite).
  - Eval stays pure greedy (no candidate sampling) so seeds 43–52 stay comparable.

Scheduled sampling / teacher-forcing are explicitly deferred (would require
re-collecting real transitions for alternate upstream actions).
"""
from __future__ import annotations

import math
import random
from typing import Any

import torch


def layered_epsilon(base_eps: float, scale: float) -> float:
    """Clamp scaled ε into [0, 1]."""
    eps = float(base_eps) * float(scale)
    if eps <= 0.0:
        return 0.0
    if eps >= 1.0:
        return 1.0
    return eps


def candidate_select_action(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_candidates: int = 4,
    temperature: float = 1.0,
    stats: dict[str, int] | None = None,
) -> int | None:
    """Softmax-sample up to ``n_candidates`` valid actions, then pick max-Q among them.

    Returns ``None`` when no valid action. Updates optional ``stats`` with
    ``decisions`` / ``reselect`` (picked ≠ global masked argmax).
    """
    if mask.sum() == 0:
        return None

    valid = (mask > 0).nonzero(as_tuple=True)[0]
    n_valid = int(valid.numel())
    if n_valid == 0:
        return None

    # Global greedy (for reselect metric + trivial n_cand>=n_valid / n_cand<=1).
    q_valid = q_values[valid]
    greedy_local = int(torch.argmax(q_valid).item())
    greedy_idx = int(valid[greedy_local].item())

    k = max(1, int(n_candidates))
    if k <= 1 or n_valid == 1:
        if stats is not None:
            stats["decisions"] = int(stats.get("decisions", 0)) + 1
            # reselect stays 0
        return greedy_idx

    k = min(k, n_valid)
    tau = max(1e-6, float(temperature))
    logits = (q_valid / tau).detach().float().cpu()
    # Numerically stable softmax
    logits = logits - logits.max()
    probs = torch.exp(logits)
    probs = probs / probs.sum().clamp_min(1e-12)

    # Sample k distinct indices without replacement (weighted).
    # Fall back to uniform if probs degenerate.
    p = probs.tolist()
    if not all(math.isfinite(x) and x >= 0.0 for x in p) or sum(p) <= 0.0:
        chosen_local = random.sample(range(n_valid), k)
    else:
        # sequential weighted sample without replacement
        chosen_local = []
        remaining = list(range(n_valid))
        weights = list(p)
        for _ in range(k):
            total = sum(weights[i] for i in remaining)
            if total <= 0.0:
                pick = random.choice(remaining)
            else:
                r = random.random() * total
                acc = 0.0
                pick = remaining[-1]
                for j in remaining:
                    acc += weights[j]
                    if r <= acc:
                        pick = j
                        break
            chosen_local.append(pick)
            remaining.remove(pick)

    best_local = max(chosen_local, key=lambda i: float(q_valid[i].item()))
    picked_idx = int(valid[best_local].item())

    if stats is not None:
        stats["decisions"] = int(stats.get("decisions", 0)) + 1
        if picked_idx != greedy_idx:
            stats["reselect"] = int(stats.get("reselect", 0)) + 1
    return picked_idx


def ar_reselect_rate(stats: dict[str, int] | None) -> float | None:
    if not stats:
        return None
    n = int(stats.get("decisions", 0) or 0)
    if n <= 0:
        return None
    return float(stats.get("reselect", 0) or 0) / float(n)


def default_ar_eps_scales(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Upper layers explore less by default; B matches historical b_score ×0.5."""
    cfg = config or {}
    return {
        "A": float(cfg.get("ar_eps_scale_A", 0.5)),
        "B": float(cfg.get("ar_eps_scale_B", 0.5)),
        "C": float(cfg.get("ar_eps_scale_C", 1.0)),
        "D": float(cfg.get("ar_eps_scale_D", 1.0)),
    }
