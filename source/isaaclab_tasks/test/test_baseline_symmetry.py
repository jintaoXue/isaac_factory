"""Exact-symmetry diagnostics must check graph, history, and masks together."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from diagnose_baseline_symmetry import indistinguishable_pair


class TestSymmetry(unittest.TestCase):
    def test_graph_history_and_validity_are_required(self):
        x = torch.zeros(4, 3, 3, 2)
        graph = torch.ones(4, 3, 3, dtype=torch.bool)
        mask = torch.ones(4, 3, dtype=torch.bool)
        x[1, 0, 1, 0] = 1
        graph[2, 0, 2] = False
        mask[3, 1] = False
        self.assertEqual(indistinguishable_pair(x, graph, mask, 0, 1).tolist(),
                         [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
