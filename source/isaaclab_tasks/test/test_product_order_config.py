"""Pure-Python tests for configurable factory order load."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


CFG_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_material_product.py"
)


def _load_config():
    spec = importlib.util.spec_from_file_location("cfg_material_product_order_test", CFG_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestProductOrderConfig(unittest.TestCase):
    def setUp(self):
        self.config = _load_config()

    def test_override_changes_order_without_reducing_registered_capacity(self):
        result = self.config.configure_product_order_count(15)

        self.assertEqual(result["ProductWaterPipe"], 15)
        self.assertEqual(self.config.CfgRegistrationInfos["ProductWaterPipe"], 18)

    def test_low_load_is_supported(self):
        self.config.configure_product_order_count(10)
        self.assertEqual(self.config.CfgProductOrder["ProductWaterPipe"], 10)

    def test_non_positive_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.config.configure_product_order_count(0)

    def test_order_above_registered_capacity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds registered max"):
            self.config.configure_product_order_count(19)


if __name__ == "__main__":
    unittest.main()
