"""A precursor ablation adds causal inputs without changing baseline decoding."""

import csv
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))
from audit_baseline_precursor_history import load_reference, history_summary
from factory_baselines.precursor import pack_history_precursor, attach_precursor
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig
from factory_baselines.torch_trainer import _model_inputs, load_checkpoint
from train_dense_baseline_control import dense_configuration
import test_b5_gat_gru as model_fixture


class TestPrecursorModels(unittest.TestCase):
    def test_matches_pinned_definition_and_is_independent_of_unobserved_time_ranges(self):
        reference, _ = load_reference(TOOLS.parents[5])
        features = np.random.default_rng(12).normal(0, 50, (90, 2, 27)).astype(np.float32)
        for mode in ("near", "near_far"):
            for t in (30, 40, 75):
                with self.subTest(mode=mode, t=t):
                    actual = pack_history_precursor(features, t, mode)
                    np.testing.assert_array_equal(actual, history_summary(reference, features, t, include_far=mode == "near_far"))
                    altered = features.copy()
                    altered[t:] = 9000
                    altered[:max(0, t-(60 if mode == "near_far" else 30))] = -9000
                    np.testing.assert_array_equal(actual, pack_history_precursor(altered, t, mode))

    def test_models_start_exactly_at_original_outputs_and_only_event_heads_use_summary(self):
        fixture = model_fixture.TestB5GatGru()
        fixture.setUp()
        inputs = fixture._inputs(fixture._batch())
        precursor = torch.randn(4, 5, 23)
        for cls, config_cls, hidden_name in ((B4GcnGru, B4ModelConfig, "gcn_hidden"),
                                             (B5GatGru, B5ModelConfig, "gat_hidden")):
            with self.subTest(model=cls.__name__):
                values = dict(input_dim=6, global_dim=2, num_nodes=5, gru_hidden=8,
                              dropout=0., temporal_readout="last_mean", **{hidden_name: 8})
                torch.manual_seed(25)
                original = cls(config_cls(**values)).eval()
                rng = torch.get_rng_state().clone()
                torch.manual_seed(25)
                candidate = cls(config_cls(**values, event_precursor="near")).eval()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)
                old = original(**inputs)
                new = candidate(**inputs, event_precursor=precursor)
                for key in old:
                    torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
                new["event_will_logit"].sum().backward()
                self.assertGreater(candidate.heads.precursor_projection[-1].weight.grad.abs().sum(), 0)
                with torch.no_grad():
                    candidate.heads.precursor_projection[-1].weight.fill_(.1)
                changed = candidate(**inputs, event_precursor=precursor)
                self.assertFalse(torch.equal(changed["event_will_logit"], old["event_will_logit"]))
                for key in ("remain_hot_logit", "remain_score", "remain_len", "cause_logits"):
                    torch.testing.assert_close(changed[key], old[key], rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    candidate(**inputs)
                with self.assertRaises(ValueError):
                    original(**inputs, event_precursor=precursor)
                saved = io.BytesIO()
                torch.save({"model_kind": "b4_gcn_gru" if hidden_name == "gcn_hidden" else "b5_gat_gru",
                            "model_config": candidate.config.to_dict(), "model_state_dict": candidate.state_dict()}, saved)
                saved.seek(0)
                restored, _ = load_checkpoint(saved, torch.device("cpu"))
                restored.eval()
                torch.testing.assert_close(restored(**inputs, event_precursor=precursor)["event_will_logit"], changed["event_will_logit"], rtol=0, atol=0)

    def test_attach_builds_only_requested_split_and_rejects_contract_mismatch(self):
        payload = {"x": torch.ones(3, 30, 2, 27), "node_mask": torch.tensor([[1, 0]] * 3),
                   "observation_mask": torch.ones(3, 30, 2, dtype=torch.bool),
                   "target_start_position": torch.tensor([30, 30, 30]),
                   "split_indices": {"train": torch.tensor([0]), "validation": torch.tensor([1]), "test": torch.tensor([2])}}
        manifest = {"input_windows": 30, "window_size_s": 60, "node_ids": ["n0", "n1"],
                    "shared_bundle_alignment": {"bundle_sha256": "same"},
                    "source_episodes": [{"group_id": k, "main_episode_name": k} for k in ("train", "validation", "test")]}
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["split", "group_id", "sample_index", "input_window_indices", "anchor_time_s"])
        writer.writeheader()
        for i, split in enumerate(("train", "validation", "test")):
            writer.writerow(dict(split=split, group_id=split, sample_index=i,
                                 input_window_indices=str(list(range(30))), anchor_time_s=29*60))
        csv_text = stream.getvalue()
        class Bundle(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __getitem__(self, key):
                if key.startswith("test_"): raise AssertionError("Test arrays must not be loaded")
                return super().__getitem__(key)
        bundle = Bundle(resource_ids=np.array(["n0", "n1"]))
        for split in ("train", "validation"):
            bundle[split + "_features"] = np.ones((40, 2, 27), dtype=np.float32)
            bundle[split + "_windows"] = np.arange(40)
            bundle[split + "_window_start_s"] = np.arange(40) * 60
        real_open = Path.open
        def open_index(path, *args, **kwargs):
            if path.name == "model_sample_index.csv": return io.StringIO(csv_text)
            if path.name == "normalization.json": return io.StringIO(json.dumps({"feature_mean": [0.] * 21, "feature_std": [1.] * 21}))
            return real_open(path, *args, **kwargs)
        with patch("factory_baselines.precursor.file_hash", return_value="same"), \
             patch("factory_baselines.precursor.np.load", return_value=bundle), \
             patch.object(Path, "open", autospec=True, side_effect=open_index):
            attached, contract = attach_precursor(payload, manifest, Path("/unused"), "near", ("train", "validation"))
            self.assertNotIn("event_precursor", payload)
            self.assertEqual(attached["event_precursor_valid"].tolist(), [True, True, False])
            self.assertTrue((attached["event_precursor"][:, 1] == 0).all())
            self.assertTrue((attached["event_precursor"][..., 18:] == 0).all())
            self.assertEqual(contract["extra_history_windows_max"], 0)
            with self.assertRaises(ValueError):
                attach_precursor(payload, manifest, Path("/unused"), "near_far", ("train",), contract)

    def test_candidate_changes_only_the_declared_model_input_configuration(self):
        for model in ("B4", "B5"):
            control, architecture, loss = dense_configuration(model, "history_control", 42, "cpu")
            near, near_arch, near_loss = dense_configuration(model, "near_precursor", 42, "cpu")
            near.training_profile = control.training_profile
            self.assertEqual(near, control)
            self.assertEqual(near_loss, loss)
            self.assertEqual(near_arch.pop("event_precursor"), "near")
            self.assertEqual(near_arch, architecture)


if __name__ == "__main__":
    unittest.main()
