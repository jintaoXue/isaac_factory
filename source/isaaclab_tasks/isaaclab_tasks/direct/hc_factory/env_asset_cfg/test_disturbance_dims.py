"""Multi-dim parsing and L1/L2 merge (no Isaac)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parent / "cfg_disturbance.py"
    spec = importlib.util.spec_from_file_location("cfg_disturbance_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_and_label() -> None:
    d = _load()
    assert d.parse_disturbance_dims("human") == ["human"]
    assert d.parse_disturbance_dims("human,logistics") == ["human", "logistics"]
    assert d.parse_disturbance_dims("logistics+human") == ["human", "logistics"]
    assert d.dim_label(["human", "logistics"]) == "human+logistics"
    assert d.parse_disturbance_dims("none") == ["none"]


def test_single_human_l1_unchanged() -> None:
    d = _load()
    d.configure_disturbance_from_cli("human", 1.0)
    assert d.RuntimeDisturbanceCfg["dim"] == "human"
    assert d.RuntimeDisturbanceCfg["dims"] == ["human"]
    assert abs(d.RuntimeDisturbanceCfg["human_time_scale"] - 1.55) < 1e-9
    assert abs(d.RuntimeDisturbanceCfg["machine_success_rate"] - 0.95) < 1e-9


def test_mixed_human_logistics_stacks_l1() -> None:
    d = _load()
    d.configure_disturbance_from_cli("human,logistics", 1.0)
    assert d.RuntimeDisturbanceCfg["dim"] == "human+logistics"
    assert d.RuntimeDisturbanceCfg["dims"] == ["human", "logistics"]
    assert abs(d.RuntimeDisturbanceCfg["human_time_scale"] - 1.55) < 1e-9
    assert abs(d.RuntimeDisturbanceCfg["gantry_time_scale"] - 1.65) < 1e-9


def test_mixed_l2_has_both_dims() -> None:
    d = _load()
    d.configure_disturbance_from_cli("human,logistics", 1.0)
    d.RuntimeDisturbanceCfg["applied"] = {
        "human_count": 3,
        "active_gantry_indices": [0, 1, 2, 3],
        "agv_count": 2,
    }
    ev = d.episode_l2_schedule("human+logistics", 1.0, seed=42, env_id=0, episode_id=0)
    dims = {str(e.get("dim")) for e in ev}
    assert "human" in dims and "logistics" in dims
    solo = d.episode_l2_schedule("human", 1.0, seed=42, env_id=0, episode_id=0)
    assert all(e.get("dim") == "human" for e in solo)
    assert len(solo) == 2


def test_all_ood_mix_parse() -> None:
    d = _load()
    specs = (
        ("machine,human", ["machine", "human"], "machine+human"),
        ("machine,logistics", ["machine", "logistics"], "machine+logistics"),
        ("machine,material", ["machine", "material"], "machine+material"),
        ("human,logistics", ["human", "logistics"], "human+logistics"),
        ("human,material", ["human", "material"], "human+material"),
        ("logistics,material", ["logistics", "material"], "logistics+material"),
        ("machine,human,logistics", ["machine", "human", "logistics"], "machine+human+logistics"),
        ("machine,human,material", ["machine", "human", "material"], "machine+human+material"),
        ("machine,logistics,material", ["machine", "logistics", "material"], "machine+logistics+material"),
        ("human,logistics,material", ["human", "logistics", "material"], "human+logistics+material"),
        (
            "machine,human,logistics,material",
            ["machine", "human", "logistics", "material"],
            "machine+human+logistics+material",
        ),
        ("material+human+machine", ["machine", "human", "material"], "machine+human+material"),
    )
    for raw, dims, label in specs:
        assert d.parse_disturbance_dims(raw) == dims, raw
        assert d.dim_label(dims) == label, raw


def test_mixed_triple_and_quad_l1_l2() -> None:
    d = _load()
    d.configure_disturbance_from_cli("machine,logistics,material", 1.0)
    assert d.RuntimeDisturbanceCfg["dim"] == "machine+logistics+material"
    assert abs(d.RuntimeDisturbanceCfg["gantry_time_scale"] - 1.65) < 1e-9
    assert d.RuntimeDisturbanceCfg["machine_success_rate"] < 0.9
    assert int(d.RuntimeDisturbanceCfg["material_l1_hide_count"] or 0) > 0

    d.configure_disturbance_from_cli("machine,human,logistics,material", 1.0)
    assert d.RuntimeDisturbanceCfg["dims"] == ["machine", "human", "logistics", "material"]
    assert abs(d.RuntimeDisturbanceCfg["human_time_scale"] - 1.55) < 1e-9
    assert abs(d.RuntimeDisturbanceCfg["gantry_time_scale"] - 1.65) < 1e-9
    d.RuntimeDisturbanceCfg["applied"] = {
        "human_count": 3,
        "active_gantry_indices": [0, 1, 2, 3],
        "agv_count": 2,
    }
    ev = d.episode_l2_schedule("machine+human+logistics+material", 1.0, seed=42, env_id=0, episode_id=0)
    dims = {str(e.get("dim")) for e in ev}
    assert dims == {"machine", "human", "logistics", "material"}


if __name__ == "__main__":
    test_parse_and_label()
    test_single_human_l1_unchanged()
    test_mixed_human_logistics_stacks_l1()
    test_mixed_l2_has_both_dims()
    test_all_ood_mix_parse()
    test_mixed_triple_and_quad_l1_l2()
    print("ok")

