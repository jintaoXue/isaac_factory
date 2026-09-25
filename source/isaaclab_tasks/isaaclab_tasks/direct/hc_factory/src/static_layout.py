"""Authored local poses exported once from the source USD, not a physics state."""
import json
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def layout():
    return json.loads((Path(__file__).parents[1] / "env_asset_cfg/static_layout.json").read_text())["poses"]


def local_pose(path, device):
    key = "/obj/" + path.split("/obj/", 1)[1]
    if key not in layout():
        raise KeyError(f"Static layout missing {key}; export the matching USD layout")
    p = layout()[key]
    return (torch.tensor([p["position"]], dtype=torch.float32, device=device),
            torch.tensor([p["orientation"]], dtype=torch.float32, device=device))
