"""Export dense_i1 plus selected 2/3/4-dim mix runs into raw_data/dense_i1_mix."""

from __future__ import annotations

from pathlib import Path

from factory_bn.export_dataset import export_runs


BN_ROOT = (
    Path(__file__).resolve().parents[2] / "output" / "bottleneck_dataset"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "raw_data" / "dense_i1_mix"

DENSE_RUNS = [
    "machine20",
    "human20",
    "logistics20",
    "material20",
    "extra_machine",
    "extra_human",
    "extra_logistics",
    "extra_material",
    "unsup_n10_i1/n10_machine1.0",
    "unsup_n10_i1/n10_human1.0",
    "unsup_n10_i1/n10_logistics1.0",
    "unsup_n10_i1/n10_material1.0",
]

# All complete 10/10, no deadlock. Keep every combo so pair/triple/quad
# interactions are represented; aliases avoid single-dim name collisions.
MIX_RUNS = [
    ("human+log", "n10_mix_hl1.0"),
    ("human+mach", "n10_mix_mh1.0"),
    ("human+mat", "n10_mix_hm1.0"),
    ("mach+log", "n10_mix_ml1.0"),
    ("mach+mat", "n10_mix_mm1.0"),
    ("mat+log", "n10_mix_lm1.0"),
    ("human+log+mat", "n10_mix_hlm1.0"),
    ("mach+human+log", "n10_mix_mhl1.0"),
    ("mach+human+mat", "n10_mix_mhm1.0"),
    ("mach+log+mat", "n10_mix_mlm1.0"),
    ("four_dim", "n10_mix_all1.0"),
]


def main() -> None:
    run_dirs: list[Path] = []
    run_names: list[str] = []
    for rel in DENSE_RUNS:
        path = BN_ROOT / rel
        run_dirs.append(path)
        run_names.append(path.name)
    for folder, alias in MIX_RUNS:
        run_dirs.append(BN_ROOT / folder)
        run_names.append(alias)
    missing = [p for p in run_dirs if not (p / "derived").is_dir()]
    if missing:
        raise SystemExit("missing derived/: " + ", ".join(str(p) for p in missing))
    export_runs(
        run_dirs,
        OUT_DIR,
        window_size=60.0,
        write_atomic=True,
        require_complete=10,
        skip_deadlock=True,
        run_names=run_names,
    )


if __name__ == "__main__":
    main()
