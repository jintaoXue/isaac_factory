"""Prediction heads shared by the B3-B5 PyTorch baselines."""

from __future__ import annotations

import torch
from torch import nn


class FactoryPredictionHeads(nn.Module):
    def __init__(
        self,
        node_hidden_dim: int,
        global_dim: int,
        num_nodes: int,
        prediction_horizon: float,
        max_remain_windows: int,
        num_causes: int,
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        node_hidden: torch.Tensor,
        node_mask: torch.Tensor,
        target_node_mask: torch.Tensor,
        global_features: torch.Tensor,
        jobs_remaining: torch.Tensor,
        jobs_total: torch.Tensor,
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
        return {
            "remain_score": self.remain_score_head(future_context),
            "remain_hot_logit": self.remain_hot_head(future_context).squeeze(-1),
            "event_will_logit": self.event_will_head(node_hidden).squeeze(-1),
            "event_start_logit": self.event_start_head(node_hidden),
            "event_duration": self.event_duration_head(node_hidden).squeeze(-1),
            "remain_len": self.remain_len_head(
                torch.cat((graph_context, jobs_context), dim=-1)
            ).squeeze(-1),
            "cause_logits": self.cause_head(graph_context),
            "node_hidden": node_hidden,
        }
