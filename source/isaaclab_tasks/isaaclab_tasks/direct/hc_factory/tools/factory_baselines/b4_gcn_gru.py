"""B4 dense GCN-GRU baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .torch_heads import FactoryPredictionHeads


@dataclass
class B4ModelConfig:
    input_dim: int
    global_dim: int
    num_nodes: int
    gcn_hidden: int = 64
    gru_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.2
    event_context: bool = False
    prediction_horizon: float = 180.0
    max_remain_windows: int = 15
    num_causes: int = 10

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "num_nodes",
            "gcn_hidden",
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
    def from_dict(cls, values: dict[str, Any]) -> "B4ModelConfig":
        return cls(**values)


class DenseGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = node_mask.bool()
        edges = adjacency.to(x.dtype)
        edges = edges * valid[:, :, None] * valid[:, None, :]
        identity = torch.eye(edges.shape[1], device=x.device, dtype=x.dtype)[None]
        edges = torch.maximum(edges, identity * valid[:, :, None])
        degree = edges.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt[:, :, None] * edges * inv_sqrt[:, None, :]
        output = self.linear(torch.bmm(normalized, x))
        return output * valid[:, :, None].to(output.dtype)


class B4GcnGru(nn.Module):
    def __init__(self, config: B4ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            config.input_dim, config.gcn_hidden, bias=False
        )
        self.gcn1 = DenseGraphConvolution(config.input_dim, config.gcn_hidden)
        self.gcn2 = DenseGraphConvolution(config.gcn_hidden, config.gcn_hidden)
        self.gcn1_norm = nn.LayerNorm(config.gcn_hidden)
        self.gcn2_norm = nn.LayerNorm(config.gcn_hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=config.gcn_hidden,
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
        spatial = self.gcn1(spatial_x, spatial_adjacency, spatial_mask)
        spatial = (
            self.dropout(F.relu(self.gcn1_norm(spatial + input_residual)))
            * valid_features
        )
        graph_update = self.gcn2(spatial, spatial_adjacency, spatial_mask)
        spatial = F.relu(self.gcn2_norm(graph_update + spatial)) * valid_features
        spatial = spatial.view(batch_size, time_steps, node_count, -1)
        temporal = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * node_count, time_steps, self.config.gcn_hidden
        )
        temporal_output, _ = self.gru(temporal)
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
