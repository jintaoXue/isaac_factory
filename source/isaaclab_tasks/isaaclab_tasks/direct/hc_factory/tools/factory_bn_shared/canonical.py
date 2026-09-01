"""Import helpers for the canonical factory_bn experiment package."""

from __future__ import annotations

import sys
from pathlib import Path


PDFORMER_ROOT = Path(__file__).resolve().parents[2] / "PDFormer"
if not (PDFORMER_ROOT / "factory_bn").is_dir():
    raise RuntimeError(f"Canonical factory_bn package is missing: {PDFORMER_ROOT}")
if str(PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(PDFORMER_ROOT))
