"""Smoke-test HC_EMB_DEBUG probe path (no GPU required)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


class EmbDebugProbeTest(unittest.TestCase):
    def test_log_emb_debug_writes_line(self):
        from source.algo.hierarchical.hc_factory import hier_networks as hn

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emb_debug.log"
            env = {
                "HC_EMB_DEBUG": "1",
                "HC_EMB_DEBUG_EVERY": "1",
                "HC_EMB_DEBUG_NAMES": "next_logistic_id",
                "HC_EMB_DEBUG_PATH": str(path),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                hn._EMB_DEBUG_COUNT = 0
                emb = nn.Embedding(32, 8, padding_idx=0)
                t = torch.tensor([[0, 3, 31]], dtype=torch.int64)
                clamped = t.clamp(0, emb.num_embeddings - 1)
                hn._log_emb_debug("next_logistic_id", t, clamped, emb)
                text = path.read_text(encoding="utf-8")
            self.assertIn("name=next_logistic_id", text)
            self.assertIn("num_emb=32", text)
            self.assertIn("c_max=31", text)

    def test_encode_ongoing_respects_debug_off(self):
        from source.algo.hierarchical.hc_factory.hier_networks import StateEncoder

        enc = StateEncoder(out_dim=64, emb_dim=8, hidden=32, max_ongoing=2, max_subtasks=4)
        m = 2
        ot = {
            "mask": torch.ones(m),
            "task_id": torch.zeros(m, dtype=torch.long),
            "task_type_id": torch.zeros(m, dtype=torch.long),
            "product_id": torch.zeros(m, dtype=torch.long),
            "machine_id": torch.zeros(m, dtype=torch.long),
            "logistic_machine_id": torch.zeros(m, dtype=torch.long),
            "logistic_submat_id": torch.zeros(m, dtype=torch.long),
            "next_task_id": torch.zeros(m, dtype=torch.long),
            "next_logistic_id": torch.zeros(m, dtype=torch.long),
            "task_done": torch.zeros(m),
            "is_final": torch.zeros(m),
            "age_norm": torch.zeros(m),
            "ongoing_index": torch.zeros(m),
            "num_subtasks_n": torch.zeros(m),
            "human_slot": torch.full((m,), -1.0),
            "robot_slot": torch.full((m,), -1.0),
            "workstation_i": torch.full((m,), -1.0),
            "start_area_ids": torch.zeros(m, 3, dtype=torch.long),
            "goal_area_ids": torch.zeros(m, 3, dtype=torch.long),
            "subtask_seq": torch.zeros(m, 4, 4, dtype=torch.long),
            "subtask_mask": torch.zeros(m, 4),
            "mat_state_seq": torch.zeros(m, 4, dtype=torch.long),
        }
        with mock.patch.dict(os.environ, {"HC_EMB_DEBUG": "0", "HC_CUDA_SYNC_ENCODE": "0"}, clear=False):
            out = enc._encode_ongoing(ot)
        self.assertEqual(out.shape[0], 1)
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
