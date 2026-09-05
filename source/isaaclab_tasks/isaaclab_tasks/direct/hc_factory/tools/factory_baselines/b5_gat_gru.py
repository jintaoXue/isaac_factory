"""Dense GAT-GRU model for the BSTAN-style baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .torch_heads import FactoryPredictionHeads


@dataclass
class B5ModelConfig:
    input_dim: int
    global_dim: int
    num_nodes: int
    gat_hidden: int = 64
    gat_heads: int = 4
    gru_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.2
    event_context: bool = False
    prediction_horizon: float = 180.0
    max_remain_windows: int = 15
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
    def from_dict(cls, values: dict[str, Any]) -> "B5ModelConfig":
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


class B5GatGru(nn.Module):
    """Two spatial GAT layers followed by a shared node-wise GRU."""

    def __init__(self, config: B5ModelConfig) -> None:
        super().__init__()
        self.config = config
        first_head_dim = config.gat_hidden // config.gat_heads
        self.input_projection = nn.Linear(
            config.input_dim, config.gat_hidden, bias=False
        )
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
        self.gat1_norm = nn.LayerNorm(config.gat_hidden)
        self.gat2_norm = nn.LayerNorm(config.gat_hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=config.gat_hidden,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        self.heads = FactoryPredictionHeads(
            node_hidden_dim=config.gru_hidden,
            global_dim=config.global_dim,
            num_nodes=config.num_nodes,
            prediction_horizon=config.prediction_horizon,
            max_remain_windows=config.max_remain_windows,
            num_causes=config.num_causes,
            event_context=config.event_context,
        )

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
        valid_features = spatial_mask[:, :, None].to(spatial_x.dtype)

        input_residual = self.input_projection(spatial_x) * valid_features
        spatial = self.gat1(spatial_x, spatial_adjacency, spatial_mask)
        spatial = (
            self.dropout(F.elu(self.gat1_norm(spatial + input_residual)))
            * valid_features
        )
        graph_update = self.gat2(spatial, spatial_adjacency, spatial_mask)
        spatial = F.elu(self.gat2_norm(graph_update + spatial)) * valid_features
        spatial = spatial.view(batch_size, time_steps, node_count, -1)

        temporal_input = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * node_count, time_steps, self.config.gat_hidden
        )
        temporal_output, _ = self.gru(temporal_input)
        node_hidden = temporal_output[:, -1].view(
            batch_size, node_count, self.config.gru_hidden
        )
        return self.heads(
            node_hidden,
            node_mask,
            target_node_mask,
            global_features,
            jobs_remaining,
            jobs_total,
        )
