"""Shared implementations for B2-B5 factory bottleneck baselines."""

from .b5_gat_gru import B5GatGru, B5ModelConfig
from .b3_lstm import B3Lstm, B3ModelConfig
from .b4_gcn_gru import B4GcnGru, B4ModelConfig
from .torch_losses import MultiTaskLossConfig, compute_multitask_loss
from .torch_trainer import (
    TorchTrainConfig,
    evaluate_torch_checkpoint,
    train_torch_baseline,
)
from .dataset import (
    FactoryBaselineTensorDataset,
    build_factory_baseline_dataset,
    load_shared_dataset,
)
from .b2_xgboost import B2XGBoostConfig, train_b2_xgboost

__all__ = [
    "B5GatGru",
    "B5ModelConfig",
    "B3Lstm",
    "B3ModelConfig",
    "B4GcnGru",
    "B4ModelConfig",
    "MultiTaskLossConfig",
    "TorchTrainConfig",
    "FactoryBaselineTensorDataset",
    "B2XGBoostConfig",
    "build_factory_baseline_dataset",
    "compute_multitask_loss",
    "evaluate_torch_checkpoint",
    "load_shared_dataset",
    "train_torch_baseline",
    "train_b2_xgboost",
]
