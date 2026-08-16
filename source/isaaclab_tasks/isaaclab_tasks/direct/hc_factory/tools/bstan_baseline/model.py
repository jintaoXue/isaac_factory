"""Dense GAT-GRU model for the BSTAN-style baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BstanModelConfig:
    input_dim: int
    global_dim: int
    num_nodes: int
    gat_hidden: int = 64
    gat_heads: int = 4
    gru_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.2
    prediction_horizon: float = 180.0
    max_remain_windows: int = 512
    num_causes: int = 10

    def __post_init__(self) -> None:
        if self.gat_hidden % self.gat_heads != 0:
            raise ValueError("gat_hidden must be divisible by gat_heads")
        for name in (
            "input_dim",
            "num_nodes",
            "gat_hidden",
            "gat_heads",
            "gru_hidden",
            "gru_layers",
            "prediction_horizon",
            "max_remain_windows",
            "num_causes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.global_dim < 0:
            raise ValueError("global_dim must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BstanModelConfig":
        return cls(**values)


class DenseGraphAttention(nn.Module):
    """Multi-head graph attention over a dense boolean adjacency matrix."""

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int,
        concat: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.concat = concat
        self.projection = nn.Linear(input_dim, head_dim * num_heads, bias=False)
        self.attention_source = nn.Parameter(torch.empty(num_heads, head_dim))
        self.attention_target = nn.Parameter(torch.empty(num_heads, head_dim))
        output_dim = head_dim * num_heads if concat else head_dim
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = x.shape
        projected = self.projection(x).view(
            batch_size, node_count, self.num_heads, self.head_dim
        )
        source_scores = (projected * self.attention_source).sum(dim=-1)
        target_scores = (projected * self.attention_target).sum(dim=-1)
        logits = F.leaky_relu(
            source_scores[:, :, None, :] + target_scores[:, None, :, :],
            negative_slope=0.2,
        )
        valid_edges = (
            adjacency.bool()[:, :, :, None]
            & node_mask.bool()[:, :, None, None]
            & node_mask.bool()[:, None, :, None]
        )
        logits = logits.masked_fill(~valid_edges, -1.0e9)
        attention = torch.softmax(logits, dim=2)
        attention = self.dropout(attention)
        output = torch.einsum("bijh,bjhd->bihd", attention, projected)
        if self.concat:
            output = output.reshape(batch_size, node_count, -1)
        else:
            output = output.mean(dim=2)
        output = output + self.bias
        return output * node_mask[:, :, None].to(output.dtype)


class BstanGatGru(nn.Module):
    """Two spatial GAT layers followed by a shared node-wise GRU."""

    def __init__(self, config: BstanModelConfig) -> None:
        super().__init__()
        self.config = config
        first_head_dim = config.gat_hidden // config.gat_heads
        self.gat1 = DenseGraphAttention(
            config.input_dim,
            first_head_dim,
            config.gat_heads,
            concat=True,
            dropout=config.dropout,
        )
        self.gat2 = DenseGraphAttention(
            config.gat_hidden,
            config.gat_hidden,
            config.gat_heads,
            concat=False,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=config.gat_hidden,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        graph_dim = config.gru_hidden + config.global_dim
        self.node_head = nn.Linear(config.gru_hidden, 1)
        self.occurrence_head = nn.Linear(graph_dim, 1)
        self.time_to_start_head = nn.Linear(graph_dim, 1)
        self.remain_time_embedding = nn.Embedding(
            config.max_remain_windows, config.gru_hidden
        )
        self.remain_score_head = nn.Sequential(
            nn.Linear(config.gru_hidden * 2, config.gru_hidden),
            nn.GELU(),
            nn.Linear(config.gru_hidden, 1),
        )
        self.remain_hot_head = nn.Sequential(
            nn.Linear(config.gru_hidden * 2, config.gru_hidden),
            nn.GELU(),
            nn.Linear(config.gru_hidden, 1),
        )
        self.remain_len_head = nn.Sequential(
            nn.Linear(graph_dim + 2, config.gru_hidden),
            nn.GELU(),
            nn.Linear(config.gru_hidden, 1),
            nn.Softplus(),
        )
        self.cause_head = nn.Linear(graph_dim, config.num_causes)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        target_node_mask: torch.Tensor,
        global_features: torch.Tensor,
        jobs_remaining: torch.Tensor,
        jobs_total: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, time_steps, node_count, _ = x.shape
        if node_count != self.config.num_nodes:
            raise ValueError(
                f"Expected {self.config.num_nodes} nodes, received {node_count}"
            )
        spatial_x = x.reshape(batch_size * time_steps, node_count, -1)
        spatial_adjacency = adjacency[:, None].expand(
            batch_size, time_steps, node_count, node_count
        )
        spatial_adjacency = spatial_adjacency.reshape(
            batch_size * time_steps, node_count, node_count
        )
        spatial_mask = node_mask[:, None].expand(batch_size, time_steps, node_count)
        spatial_mask = spatial_mask.reshape(batch_size * time_steps, node_count)

        spatial = self.gat1(spatial_x, spatial_adjacency, spatial_mask)
        spatial = self.dropout(F.elu(spatial))
        spatial = self.gat2(spatial, spatial_adjacency, spatial_mask)
        spatial = F.elu(spatial)
        spatial = spatial.view(batch_size, time_steps, node_count, -1)

        temporal_input = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * node_count, time_steps, self.config.gat_hidden
        )
        temporal_output, _ = self.gru(temporal_input)
        node_hidden = temporal_output[:, -1].view(
            batch_size, node_count, self.config.gru_hidden
        )
        node_hidden = node_hidden * node_mask[:, :, None].to(node_hidden.dtype)

        node_logits = self.node_head(node_hidden).squeeze(-1)
        node_logits = node_logits.masked_fill(~target_node_mask.bool(), -1.0e9)
        mask_float = node_mask[:, :, None].to(node_hidden.dtype)
        graph_embedding = (node_hidden * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        graph_context = (
            torch.cat((graph_embedding, global_features[:, -1]), dim=-1)
            if self.config.global_dim
            else graph_embedding
        )
        future_steps = torch.arange(
            self.config.max_remain_windows, device=x.device
        )
        future_time = self.remain_time_embedding(future_steps)
        future_nodes = node_hidden[:, None].expand(
            -1, self.config.max_remain_windows, -1, -1
        )
        future_time = future_time[None, :, None].expand(
            batch_size, -1, node_count, -1
        )
        future_context = torch.cat((future_nodes, future_time), dim=-1)
        remain_score = self.remain_score_head(future_context)
        remain_hot_logit = self.remain_hot_head(future_context).squeeze(-1)
        jobs_context = torch.stack(
            (jobs_remaining, jobs_total),
            dim=-1,
        )

        return {
            "occurrence_logit": self.occurrence_head(graph_context).squeeze(-1),
            "node_logits": node_logits,
            "time_to_start": torch.sigmoid(
                self.time_to_start_head(graph_context).squeeze(-1)
            )
            * self.config.prediction_horizon,
            "remain_score": remain_score,
            "remain_hot_logit": remain_hot_logit,
            "remain_len": self.remain_len_head(
                torch.cat((graph_context, jobs_context), dim=-1)
            ).squeeze(-1),
            "cause_logits": self.cause_head(graph_context),
            "node_hidden": node_hidden,
        }
