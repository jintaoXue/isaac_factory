"""Prediction heads and temporal pooling for the PyTorch baselines."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """Content-weighted historical GRU pooling, initialized to the exact mean."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bth,h->bt", history, self.query) / math.sqrt(self.query.numel())
        weights = scores.softmax(dim=1)
        # This is a weighted mean. The residual form preserves the control's
        # floating-point mean exactly at initialization, without consuming RNG.
        correction = weights - torch.full_like(weights, 1.0 / history.shape[1])
        return history.mean(dim=1) + (correction[:, :, None] * history).sum(dim=1)


class FactoryPredictionHeads(nn.Module):
    def __init__(
        self,
        node_hidden_dim: int,
        global_dim: int,
        num_nodes: int,
        prediction_horizon: float,
        max_remain_windows: int,
        num_causes: int,
        event_context: bool = False,
        event_head: str = "binary",
        event_precursor_dim: int = 0,
        event_onset_aux: bool = False,
    ) -> None:
        super().__init__()
        if event_head not in {"binary", "three_class"}:
            raise ValueError("event_head must be binary or three_class")
        self.event_head = event_head
        if event_onset_aux and event_head != "binary":
            raise ValueError("Onset auxiliary supervision requires the binary event control")
        if event_precursor_dim not in {0, 23}:
            raise ValueError("event_precursor_dim must be zero or 23")
        self.event_precursor_dim = event_precursor_dim
        self.node_hidden_dim = node_hidden_dim
        self.global_dim = global_dim
        self.num_nodes = num_nodes
        del prediction_horizon
        self.max_remain_windows = max_remain_windows
        graph_dim = node_hidden_dim + global_dim
        self.remain_time_embedding = nn.Embedding(max_remain_windows, node_hidden_dim)
        self.remain_score_head = nn.Sequential(
            nn.Linear(node_hidden_dim * 2, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, 1),
        )
        self.remain_hot_head = nn.Sequential(
            nn.Linear(node_hidden_dim * 2, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, 1),
        )
        self.event_will_head = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, 1),
        )
        nn.init.constant_(self.event_will_head[-1].bias, -1.5)
        self.event_start_head = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, max_remain_windows),
        )
        self.event_duration_head = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, 1),
            nn.Softplus(),
        )
        self.remain_len_head = nn.Sequential(
            nn.Linear(graph_dim + 2, node_hidden_dim),
            nn.GELU(),
            nn.Linear(node_hidden_dim, 1),
            nn.Softplus(),
        )
        self.cause_head = nn.Linear(graph_dim, num_causes)
        self.event_context_projection = (
            nn.Sequential(
                nn.Linear(node_hidden_dim + graph_dim + 2, node_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(node_hidden_dim),
            )
            if event_context else None
        )
        if event_head == "three_class":
            # Match all common initial weights and the RNG stream across the two arms.
            with torch.random.fork_rng(devices=[]):
                self.event_will_head[-1] = nn.Linear(node_hidden_dim, 3)
            with torch.no_grad():
                self.event_will_head[-1].bias[0] = 0.0
                self.event_will_head[-1].bias[1:] = -1.5 - math.log(2.0)
        self.precursor_projection = None
        if event_precursor_dim:
            # Preserve every existing parameter and the training RNG stream at initialization.
            with torch.random.fork_rng(devices=[]):
                self.precursor_projection = nn.Sequential(
                    nn.Linear(event_precursor_dim, node_hidden_dim), nn.GELU(),
                    nn.Linear(node_hidden_dim, node_hidden_dim),
                )
                nn.init.zeros_(self.precursor_projection[-1].weight)
                nn.init.zeros_(self.precursor_projection[-1].bias)
        # Independent parameters, identical initialization, no RNG consumption.
        # This head supplies training gradients only; it never changes event decoding.
        self.event_onset_head = deepcopy(self.event_will_head) if event_onset_aux else None

    def forward(
        self,
        node_hidden: torch.Tensor,
        node_mask: torch.Tensor,
        target_node_mask: torch.Tensor,
        global_features: torch.Tensor,
        jobs_remaining: torch.Tensor,
        jobs_total: torch.Tensor,
        event_precursor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, node_count, hidden_dim = node_hidden.shape
        if node_count != self.num_nodes or hidden_dim != self.node_hidden_dim:
            raise ValueError(
                "Unexpected node hidden shape: "
                f"expected (*, {self.num_nodes}, {self.node_hidden_dim}), "
                f"got {tuple(node_hidden.shape)}"
            )
        node_hidden = node_hidden * node_mask[:, :, None].to(node_hidden.dtype)
        del target_node_mask
        mask_float = node_mask[:, :, None].to(node_hidden.dtype)
        graph_embedding = (node_hidden * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        graph_context = (
            torch.cat((graph_embedding, global_features[:, -1]), dim=-1)
            if self.global_dim
            else graph_embedding
        )
        future_steps = torch.arange(self.max_remain_windows, device=node_hidden.device)
        future_time = self.remain_time_embedding(future_steps)
        future_nodes = node_hidden[:, None].expand(-1, self.max_remain_windows, -1, -1)
        future_time = future_time[None, :, None].expand(batch_size, -1, node_count, -1)
        future_context = torch.cat((future_nodes, future_time), dim=-1)
        jobs_context = torch.stack((jobs_remaining, jobs_total), dim=-1)
        event_hidden = node_hidden
        if self.event_context_projection is not None:
            context = torch.cat((graph_context, torch.log1p(jobs_context.clamp_min(0))), dim=-1)
            context = context[:, None].expand(-1, node_count, -1)
            event_hidden = node_hidden + self.event_context_projection(
                torch.cat((node_hidden, context), dim=-1)
            )
            event_hidden = event_hidden * mask_float
        if self.precursor_projection is not None:
            if event_precursor is None or event_precursor.shape != (batch_size, node_count, self.event_precursor_dim):
                raise ValueError("Expected explicit (batch,node,23) precursor features")
            event_hidden = (event_hidden + self.precursor_projection(event_precursor)) * mask_float
        elif event_precursor is not None:
            raise ValueError("Precursor supplied to a model without its registered projection")
        remain_score = self.remain_score_head(future_context)
        remain_hot = self.remain_hot_head(future_context).squeeze(-1)
        event_logits = self.event_will_head(event_hidden)
        result = {
            "remain_score": remain_score,
            "remain_hot_logit": remain_hot,
            "event_will_logit": (
                torch.logsumexp(event_logits[..., 1:], dim=-1) - event_logits[..., 0]
                if self.event_head == "three_class" else event_logits.squeeze(-1)
            ),
            "event_start_logit": self.event_start_head(event_hidden),
            "event_duration": self.event_duration_head(event_hidden).squeeze(-1),
            "remain_len": self.remain_len_head(
                torch.cat((graph_context, jobs_context), dim=-1)
            ).squeeze(-1),
            "cause_logits": self.cause_head(graph_context),
            "node_hidden": node_hidden,
        }
        if self.event_head == "three_class":
            result["event_kind_logits"] = event_logits
        if self.event_onset_head is not None:
            result["event_onset_logit"] = self.event_onset_head(event_hidden).squeeze(-1)
        return result
