"""Training-only event sampling preserves evaluation populations and update budgets."""

import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from factory_baselines import torch_trainer as trainer
from factory_baselines.torch_losses import MultiTaskLossConfig
from run_staged_baseline import sampling_configuration
from test_baseline_warm_start import stage_parents


class EventDataset(Dataset):
    def __init__(self, payload, indices):
        self.payload, self.indices = payload, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        self.payload["reads"].append(source)
        return {"sample_index": torch.tensor(source), **self.payload["rows"][source]}


def event_payload():
    rows = []
    for will, start, mask in (
        ([0, 0], [0, 0], [1, 1]),
        ([1, 0], [0, 0], [1, 1]),
        ([0, 1], [0, 2], [1, 1]),
        ([0, 1], [0, 2], [1, 0]),
        ([0, 0], [1, 2], [1, 1]),
        ([1, 1], [0, 2], [1, 1]),
    ):
        rows.append({"event_will": torch.tensor(will), "event_start": torch.tensor(start),
                     "occ_node_mask": torch.tensor(mask)})
    return {"rows": rows, "reads": [], "split_indices": {
        "train": torch.tensor([0, 1, 2, 3]),
        "validation": torch.tensor([4]), "test": torch.tensor([5]),
    }}


@pytest.mark.parametrize("target,expected", [
    ("any_event", [1, 4, 4, 1]), ("upcoming", [1, 1, 4, 1]),
])
def test_only_valid_positive_train_windows_are_upweighted(target, expected):
    payload = event_payload()
    config = trainer.TorchTrainConfig(event_oversample_factor=4, event_oversample_target=target,
                                      batch_size=2, device="cpu")
    with patch.object(trainer, "FactoryBaselineTensorDataset", EventDataset):
        loaders = trainer._loaders(payload, config)
    assert payload["reads"] == [0, 1, 2, 3]
    sampler = loaders["train"].sampler
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.weights.tolist() == expected
    assert sampler.replacement and sampler.num_samples == 4
    assert len(loaders["train"]) == 2
    assert [b["sample_index"].tolist() for b in loaders["validation"]] == [[4]]
    assert [b["sample_index"].tolist() for b in loaders["test"]] == [[5]]
    meta = trainer._training_sampling_metadata(loaders["train"], config)
    assert meta["draws_per_epoch"] == meta["population_windows"] == 4
    assert meta["expected_eligible_draw_fraction"] == pytest.approx(
        sum(w for w in expected if w > 1) / sum(expected)
    )
    assert meta["importance_corrected"] is False


def test_uniform_control_preserves_exact_old_loader_order():
    payload = event_payload()
    config = trainer.TorchTrainConfig(batch_size=2, device="cpu")
    with patch.object(trainer, "FactoryBaselineTensorDataset", EventDataset):
        loaders = trainer._loaders(payload, config)
    assert payload["reads"] == []
    reference = DataLoader(EventDataset(payload, [0, 1, 2, 3]), batch_size=2, shuffle=True,
                           generator=torch.Generator().manual_seed(config.seed))
    assert [b["sample_index"].tolist() for b in loaders["train"]] == [
        b["sample_index"].tolist() for b in reference
    ]
    assert trainer._training_sampling_metadata(loaders["train"], config)["method"] == "uniform_without_replacement"


def test_sampling_seed_is_repeatable_and_holds_epoch_length():
    config = trainer.TorchTrainConfig(event_oversample_factor=4, device="cpu")
    with patch.object(trainer, "FactoryBaselineTensorDataset", EventDataset):
        a = trainer._loaders(event_payload(), config)["train"]
        b = trainer._loaders(event_payload(), config)["train"]
    for _ in range(3):
        first, second = list(a.sampler), list(b.sampler)
        assert first == second and len(first) == 4
        assert set(first) <= set(range(4))


def test_absent_train_events_cannot_be_replaced_with_holdout_events():
    payload = event_payload()
    for row in payload["rows"][:4]:
        row["event_will"].zero_()
    config = trainer.TorchTrainConfig(event_oversample_factor=4, device="cpu")
    with patch.object(trainer, "FactoryBaselineTensorDataset", EventDataset):
        with pytest.raises(ValueError, match="No eligible training windows"):
            trainer._loaders(payload, config)
    assert payload["reads"] == [0, 1, 2, 3]


@pytest.mark.parametrize("factor", [0, .5, float("nan"), float("inf")])
def test_invalid_factor_rejected(factor):
    with pytest.raises(ValueError, match="event_oversample_factor"):
        trainer.TorchTrainConfig(event_oversample_factor=factor)


def test_preregistered_arms_only_change_sampling():
    parent = {"model": {"temporal_readout": "last_mean", "node_embedding": 0, "event_context": False},
              "training": {"max_epochs": 60, "evaluate_test": False, "learning_rate": .0003},
              "loss": MultiTaskLossConfig().to_dict()}
    original = copy.deepcopy(parent)
    configurations = []
    for arm, target, factor in (("uniform_control", "any_event", 1),
                                ("event4", "any_event", 4), ("upcoming4", "upcoming", 4)):
        c = sampling_configuration(parent, arm, "same", "cpu", 42)
        assert c["training"].pop("event_oversample_factor") == factor
        assert c["training"].pop("event_oversample_target") == target
        assert c["training"]["max_epochs"] == 60
        configurations.append(c)
    assert configurations[0] == configurations[1] == configurations[2]
    assert parent == original
    with pytest.raises(ValueError, match="event_oversample_target"):
        trainer.TorchTrainConfig(event_oversample_target="test")


@pytest.mark.parametrize("kind", ("b4_gcn_gru", "b5_gat_gru"))
def test_real_training_records_sampling_without_test_evaluation(stage_parents, tmp_path, kind):
    _, dataset, result, models = stage_parents
    out = tmp_path / kind
    summary = trainer.train_torch_baseline(
        kind, dataset, out, model_overrides=models[kind],
        train_config=trainer.TorchTrainConfig(evaluate_test=False, batch_size=8, max_epochs=1,
            min_epochs=1, patience=1, device="cpu", event_oversample_factor=4),
    )
    assert summary["status"] == "validation_completed"
    meta = json.loads((out / "config.json").read_text())["metadata"]["training_sampling"]
    assert meta["draws_per_epoch"] == len(result["payload"]["split_indices"]["train"])
    assert meta["method"] == "weighted_with_replacement"
    assert not (out / "metrics_test.json").exists()
