"""B3 non-graph LSTM baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .torch_heads import FactoryPredictionHeads


@dataclass
class B3ModelConfig:
    input_dim: int
    global_dim: int
    num_nodes: int
    lstm_hidden: int = 128
    lstm_layers: int = 1
    node_hidden: int = 128
    node_embedding: int = 32
    dropout: float = 0.2
    event_context: bool = False
    prediction_horizon: float = 180.0
    max_remain_windows: int = 15
    num_causes: int = 10

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "num_nodes",
            "lstm_hidden",
            "lstm_layers",
            "node_hidden",
            "node_embedding",
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
    def from_dict(cls, values: dict[str, Any]) -> "B3ModelConfig":
        return cls(**values)


class B3Lstm(nn.Module):
    """Flatten fixed-order nodes at each step and model only temporal dependence."""

    def __init__(self, config: B3ModelConfig) -> None:
        super().__init__()
        self.config = config
        sequence_dim = config.num_nodes * config.input_dim + config.global_dim
        self.lstm = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.node_embedding = nn.Embedding(config.num_nodes, config.node_embedding)
        self.node_projection = nn.Sequential(
            nn.Linear(config.lstm_hidden + config.node_embedding, config.node_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.heads = FactoryPredictionHeads(
            node_hidden_dim=config.node_hidden,
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
        del adjacency
        batch_size, time_steps, node_count, feature_count = x.shape
        if (
            node_count != self.config.num_nodes
            or feature_count != self.config.input_dim
        ):
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")
        masked_x = x * node_mask[:, None, :, None].to(x.dtype)
        sequence = masked_x.reshape(batch_size, time_steps, -1)
        if self.config.global_dim:
            sequence = torch.cat((sequence, global_features), dim=-1)
        output, _ = self.lstm(sequence)
        graph_hidden = output[:, -1]
        node_ids = torch.arange(node_count, device=x.device)
        node_embedding = self.node_embedding(node_ids)[None].expand(batch_size, -1, -1)
        node_hidden = self.node_projection(
            torch.cat(
                (graph_hidden[:, None].expand(-1, node_count, -1), node_embedding),
                dim=-1,
            )
        )
        return self.heads(
            node_hidden,
            node_mask,
            target_node_mask,
            global_features,
            jobs_remaining,
            jobs_total,
        )
