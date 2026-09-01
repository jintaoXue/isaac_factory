"""Runtime disturbance config — set from CLI before env construction.

Usage (train.py)::

    --disturbance_dim machine|human|logistics|material|none
    --disturbance_dim human,logistics     # mixed OOD (also human+logistics)
    --disturbance_intensity 1.0
    [--disturbance_human_count N]
    [--disturbance_agv_count N]
    [--disturbance_gantry_count N]
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

DISTURBANCE_DIMS = ("none", "material", "human", "logistics", "machine")
# Apply order when several dims are on: L0/L1 union, then L2 queues concatenated.
CANONICAL_ACTIVE_DIMS = ("machine", "human", "logistics", "material")


def parse_disturbance_dims(raw: str | None) -> list[str]:
    """Parse ``human`` or ``human,logistics`` / ``human+logistics``. ``none`` wins if alone."""
    text = str(raw or "none").lower().strip()
    if not text or text == "none":
        return ["none"]
    parts = [p.strip() for p in text.replace("+", ",").replace("|", ",").split(",") if p.strip()]
    seen: list[str] = []
    for part in parts:
        if part == "none":
            continue
        if part not in CANONICAL_ACTIVE_DIMS:
            raise ValueError(
                f"disturbance_dim must be none or one/more of {CANONICAL_ACTIVE_DIMS}, got {raw!r}"
            )
        if part not in seen:
            seen.append(part)
    ordered = [d for d in CANONICAL_ACTIVE_DIMS if d in seen]
    return ordered or ["none"]


def dim_label(dims: list[str] | None) -> str:
    dims = list(dims or ["none"])
    if not dims or dims == ["none"]:
        return "none"
    return "+".join(dims)


def active_dims() -> list[str]:
    stored = RuntimeDisturbanceCfg.get("dims")
    if isinstance(stored, (list, tuple)) and stored:
        return [str(x) for x in stored]
    return parse_disturbance_dims(str(RuntimeDisturbanceCfg.get("dim") or "none"))

# Snapshot of defaults restored when re-applying (multi-run in one process).
_DEFAULT_SNAPSHOT: dict[str, Any] | None = None

RuntimeDisturbanceCfg: dict[str, Any] = {
    "dim": "none",
    "dims": ["none"],
    "intensity": 1.0,
    # Optional CLI overrides (None = use intensity-derived defaults).
    "human_count": None,
    "agv_count": None,
    "gantry_count": None,
    # Derived / applied values (filled by apply_disturbance_to_cfgs).
    "applied": {},
    # L1 noise / multipliers read by managers at runtime.
    "machine_process_noise_std": 0.0,
    "machine_success_rate": 1.0,
    "human_subtask_noise_std": 2.0,  # matches current SubtaskTimeNoiseStdSteps default
    "human_time_scale": 1.0,
    # Per-human skill multipliers (idx 0 = fastest). Empty → all 1.0.
    "human_skill_scales": [],
    "gantry_animation_noise_std": 2.0,
    "gantry_time_scale": 1.0,
    # Material: L1 opening wave hides this many idle kit pieces, then restores.
    "material_shortage_frac": 0.0,
    "material_l1_hide_count": 0,
    "material_l1_duration": 0,
    "material_l1_hide_count_range": [],
    "material_l1_duration_range": [],
    "material_l1_sku": "product_00_flange",
    "material_l2_hide_count_range": [],
    # L2 event schedule (logic steps). end < 0 means disabled.
    "event_start_step": -1,
    "event_duration_steps": 0,
    "event_target": None,  # e.g. machine type name or human idx
    # If non-empty and event_schedule_mode == "fixed", injector uses this list.
    # Default mode resamples a new list every episode (see sample_l2_schedule).
    "event_schedule": [],
    "event_schedule_mode": "resample_per_episode",
    "tool_wear_per_1k_steps": 0.0,
    "disabled_workstations": [],
    "qc_holds": [],
}

L2_MACHINE_TARGETS = (
    "num02_rollerbedCNCPipeIntersectionCuttingMachine",
    "num04_groovingMachineLarge",
    "num01_weldingRobot",
    "num00_rotaryPipeAutomaticWeldingMachine",
    "num08_workbench",
)

L2_MATERIAL_TARGETS = (
    "product_00_flange",
    "product_00_elbow",
    "product_00_pipe_raw",
)

# Dual-station machines: close ws1 so jobs queue on ws0.
HALF_WS_TARGETS = (
    ("num08_workbench", 1),
    ("num00_rotaryPipeAutomaticWeldingMachine", 1),
)

QC_HOLD_TARGETS = (
    "num01_weldingRobot",
    "num00_rotaryPipeAutomaticWeldingMachine",
    "num08_workbench",
)
QC_HOLD_TASKS = frozenset(
    {"arc_welding_root", "MIG_welding_surface", "batch_spot_welding"}
)

# Short-order pack: 10 pipes, WIP cap 5. Pulses must still span several 60s
# windows, but 2×30 min L2 would eat a ~1.5–3 h episode. Still < stall 5000.
# Material L2 is a mid-episode kitting throttle; the long gate is L1 kit hide.
_L2_BASE_DURATION = {"machine": 700.0, "human": 800.0, "logistics": 650.0, "material": 800.0}
_L2_HORIZON_LO = 400
_L2_HORIZON_HI = 8000
# Must land before typical 10-pipe makespan (~8–12k with L0/L2).
_L2_MATERIAL_HORIZON_HI = 8000
_L2_MIN_GAP = 360
_L2_COUNT_CAP = 12
# Align with occupancy y: hot_min_windows=8. 6 min pulses were dropped after
# mid-window alignment; 10 min (600 steps) survives the 8-min filter.
_L2_MIN_DURATION = 600
_MATERIAL_ORDER_N = 10
# Kit SKUs consumed at batch_spot_welding (not pipe_raw — hiding pipe only idles the head).
_MATERIAL_KIT_SKUS = ("product_00_flange", "product_00_elbow")
_MATERIAL_L1_SKU = "product_00_flange"


def _l2_count_range(intensity: float) -> tuple[int, int]:
    """Event-count band vs intensity. Short-order episodes fit fewer pulses.

    1.0 → 2,  2.0 → 2–3,  3.0 → 3–4
    """
    if intensity < 1.5:
        lo, hi = 2, 2
    elif intensity < 2.5:
        lo, hi = 2, 3
    else:
        lo, hi = 3, 4
    hi = min(_L2_COUNT_CAP, max(lo, hi))
    return lo, hi


def _l2_duration_cap(intensity: float, dim: str = "") -> int:
    """Longer failures at higher intensity, always below stall_timeout_steps=5000.

    I=1.0≈900 (15 min), I=2.0≈1200, I=3.0≈1400. Floor is ``_L2_MIN_DURATION`` (10 min).
    """
    del dim
    return int(min(1400, 500 + 400 * max(intensity, 0.5)))


def material_l1_keep_visible(intensity: float) -> int:
    """Flange/elbow pieces left in the warehouse so kitting still crawls (watchdog).

    Order=10. I=1 → 3, I=2 → 2, I=3 → 2. Never drop below 2.
    """
    return max(2, int(round(4.0 - float(intensity))))


def material_l1_hide_count_range(intensity: float) -> tuple[int, int]:
    """Hide most idle flange *or* elbow (one SKU per episode). Never all 10.

    I=1 → 6–7 (keep 3), I=2 → 7–8 (keep 2), I=3 → 7–8 (keep 2).
    """
    keep = material_l1_keep_visible(intensity)
    hi = min(_MATERIAL_ORDER_N - keep, _MATERIAL_ORDER_N - 2)
    lo = max(5, hi - 1)
    if hi < lo:
        hi = lo
    return lo, hi


def material_l1_start_range(_intensity: float = 1.0) -> tuple[int, int]:
    """Start once the first pipes are approaching kitting, not at t=0."""
    del _intensity
    return 400, 1000


def material_l1_duration_range(intensity: float) -> tuple[int, int]:
    """Hold the kit SKU while pipes occupy the workbench (WAITING materialReadyFor_).

    Long enough for grooving to back up behind 2 workbench slots, short enough
    that leftover visible pieces keep some weld/gantry motion (< stall 5000).
    I=1 → 1400–2200 (~23–37 min).
    """
    i = max(1.0, float(intensity))
    lo = int(min(2500, 1400 + 600 * (i - 1.0)))
    hi = int(min(3200, 2200 + 500 * (i - 1.0)))
    if hi < lo:
        hi = lo
    return lo, hi


def material_l2_count_range(intensity: float) -> tuple[int, int]:
    """Pulses of the *other* kit SKU after L1 restores."""
    if intensity < 1.5:
        return 1, 2
    if intensity < 2.5:
        return 2, 3
    return 3, 4


def material_l2_hide_count_range(intensity: float) -> tuple[int, int]:
    """Idle kit pieces to hide per L2 pulse. Leave a few so kitting crawls.

    Order=10. I=1 → 4–6; never hide the last 3.
    """
    cap = _MATERIAL_ORDER_N - 3
    lo = max(3, int(round(4.0 * float(intensity))))
    hi = min(cap, int(round(6.0 * float(intensity))))
    lo = min(lo, cap)
    if hi < lo:
        hi = lo
    return lo, hi


def material_l1_hide_count(intensity: float) -> int:
    """Midpoint of the hide-count band (for applied-cfg logging)."""
    lo, hi = material_l1_hide_count_range(intensity)
    return (lo + hi) // 2


def material_l1_duration(intensity: float) -> int:
    """Midpoint of the duration band (for applied-cfg logging)."""
    lo, hi = material_l1_duration_range(intensity)
    return (lo + hi) // 2


def l2_schedule_rng(seed: int, env_id: int, episode_id: int) -> random.Random:
    """Private RNG so L2 sampling does not consume the simulation RNG."""
    mixed = (int(seed) ^ 0x9E3779B9) + 0x85EBCA6B * int(episode_id) + 0xC2B2AE35 * int(env_id)
    return random.Random(mixed & 0xFFFFFFFFFFFFFFFF)


def _sample_starts(
    rng: random.Random,
    n: int,
    durs: list[int],
    lo: int,
    hi: int,
    min_gap: int,
) -> list[int]:
    for _ in range(80):
        starts = sorted(rng.randint(lo, hi) for _ in range(n))
        ok = True
        for i in range(n - 1):
            if starts[i + 1] < starts[i] + durs[i] + min_gap:
                ok = False
                break
        if ok and starts[-1] + durs[-1] <= hi + 800:
            return starts
    t = lo
    starts = []
    slack = max(0, (hi - lo - sum(durs) - min_gap * max(n - 1, 0)) // max(n, 1))
    for i in range(n):
        starts.append(t + rng.randint(0, max(0, slack)))
        t = starts[-1] + durs[i] + min_gap
    return starts


def human_skill_range(intensity: float) -> tuple[float, float]:
    """Skill multiplier band. I=1.0 → (0.8, 1.4); I=2.0 → (0.7, 1.6); I=3.0 → (0.6, 1.8)."""
    lo = max(0.5, 0.9 - 0.1 * float(intensity))
    hi = min(2.0, 1.2 + 0.2 * float(intensity))
    return lo, hi


def sample_human_skill_scales(n: int, intensity: float) -> list[float]:
    """Even ladder by human idx: human_0 fastest, human_{n-1} slowest."""
    n = max(0, int(n))
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    lo, hi = human_skill_range(intensity)
    return [round(lo + (hi - lo) * i / (n - 1), 4) for i in range(n)]


def logistics_default_gantry_count(intensity: float, n_nominal: int = 4) -> int:
    """Keep every zone crane. Cutting gantries 2/3 orphans grooving / yellow storage.

    Capacity cut is AGV count + slowdown + L2 freeze, not dropping a zone.
    """
    del intensity
    return max(1, int(n_nominal))


def logistics_default_agv_count(intensity: float, n_nominal: int = 4) -> int:
    """I=1.0: half the fleet (4→2). I≥2.0: one AGV left."""
    n_nominal = max(1, int(n_nominal))
    if float(intensity) < 2.0:
        return max(1, n_nominal // 2)
    return 1


def disabled_workstations_for_intensity(intensity: float) -> list[tuple[str, int]]:
    """L0: close workbench ws1; I≥1.0 also rotary-weld ws1 (serial bottleneck)."""
    del intensity
    return list(HALF_WS_TARGETS)


def sample_qc_holds(intensity: float, rng: random.Random) -> list[dict[str, Any]]:
    """Time windows when finishing a weld/kitting job is held for extra QC steps.

    Independent of the exclusive L2 DOWN queue (can overlap a machine failure).
    """
    intensity = max(0.0, float(intensity))
    if intensity <= 0.0:
        return []
    if intensity < 1.5:
        n = rng.randint(1, 2)
    elif intensity < 2.5:
        n = rng.randint(2, 3)
    else:
        n = rng.randint(3, 4)
    window_durs = [int(rng.uniform(400, 900)) for _ in range(n)]
    hold_cap = int(min(240, 40 + 50 * intensity))
    hold_steps = [
        int(max(40, min(hold_cap, rng.uniform(0.7, 1.2) * 50.0 * intensity)))
        for _ in range(n)
    ]
    starts = _sample_starts(rng, n, window_durs, 500, 7500, 400)
    out: list[dict[str, Any]] = []
    last = None
    for i in range(n):
        pool = list(QC_HOLD_TARGETS)
        if last and len(pool) > 1:
            pool = [t for t in pool if t != last]
        target = rng.choice(pool)
        last = target
        out.append(
            {
                "start": int(starts[i]),
                "duration": int(window_durs[i]),
                "hold_steps": int(hold_steps[i]),
                "target": target,
                "kind": "qc_hold",
                "dim": "machine",
            }
        )
    out.sort(key=lambda e: e["start"])
    return out


def episode_qc_holds(intensity: float, seed: int, env_id: int, episode_id: int) -> list[dict[str, Any]]:
    rng = l2_schedule_rng(int(seed) ^ 0xA5A5A5A5, env_id, episode_id)
    return sample_qc_holds(intensity, rng)


def _ensure_logistics_agv_freeze(
    chosen: list[str],
    targets: list[str],
    rng: random.Random,
) -> None:
    """I=1.0 logistics: keep one gantry freeze, but every episode must freeze an AGV.

    Occupancy y no longer counts driving delay, so AGV positives only come from
    STOP / inbound wait. Random draw among 4 gantries + 2 AGVs left AGV empty.
    """
    if not chosen:
        return
    agv_pool = [t for t in targets if str(t).startswith("agv_")]
    if not agv_pool:
        return
    if any(str(t).startswith("agv_") for t in chosen):
        return
    idx = next(
        (i for i, t in enumerate(chosen) if str(t).startswith("gantry_")),
        len(chosen) - 1,
    )
    prev = chosen[idx - 1] if idx > 0 else None
    nxt = chosen[idx + 1] if idx + 1 < len(chosen) else None
    pool = [a for a in agv_pool if a != prev and a != nxt] or agv_pool
    chosen[idx] = rng.choice(pool)


def sample_l2_schedule(
    dim: str,
    intensity: float,
    rng: random.Random,
    *,
    human_count: int = 5,
    gantry_indices: list[int] | None = None,
    agv_count: int = 2,
) -> list[dict[str, Any]]:
    """Draw a non-overlapping L2 queue for one episode.

    Count, start, duration and target are random. Intensity (intended 1.0–3.0)
    scales both how many events fire and how long each lasts. Logistics always
    freezes at least one AGV when ``agv_count`` > 0. Each pulse is at least
    ``_L2_MIN_DURATION`` steps (10 min at I=1.0).
    """
    dim = (dim or "none").lower().strip()
    intensity = max(0.0, float(intensity))
    if dim == "none" or intensity <= 0.0:
        return []

    opening: list[dict[str, Any]] = []
    horizon_lo = _L2_HORIZON_LO
    horizon_hi = _L2_HORIZON_HI
    l1_kit_sku = _MATERIAL_L1_SKU
    if dim == "material":
        n_lo, n_hi = material_l1_hide_count_range(intensity)
        d_lo, d_hi = material_l1_duration_range(intensity)
        s_lo, s_hi = material_l1_start_range(intensity)
        n_hide = rng.randint(n_lo, n_hi)
        dur_l1 = rng.randint(d_lo, d_hi)
        l1_kit_sku = rng.choice(list(_MATERIAL_KIT_SKUS))
        if n_hide > 0 and dur_l1 > 0:
            start0 = rng.randint(s_lo, s_hi)
            opening.append(
                {
                    "start": start0,
                    "duration": dur_l1,
                    "target": l1_kit_sku,
                    "max_units": n_hide,
                    "dim": "material",
                    "wave": "l1",
                }
            )
            horizon_lo = max(_L2_HORIZON_LO, start0 + dur_l1 + _L2_MIN_GAP)
        horizon_hi = _L2_MATERIAL_HORIZON_HI

    if dim == "machine":
        targets = list(L2_MACHINE_TARGETS)
    elif dim == "human":
        targets = [f"human_{i}" for i in range(max(1, int(human_count)))]
    elif dim == "logistics":
        idxs = list(gantry_indices or [0, 1])
        targets = [f"gantry_{i}" for i in idxs]
        n_agv = max(0, int(agv_count))
        targets.extend(f"agv_{i}" for i in range(n_agv))
        if not targets:
            targets = ["gantry_0"]
    elif dim == "material":
        # Other kit SKU first so L2 is the complementary shortage (elbow if L1 was flange).
        targets = [s for s in _MATERIAL_KIT_SKUS if s != l1_kit_sku] or list(_MATERIAL_KIT_SKUS)
        if intensity >= 2.5:
            targets = list(_MATERIAL_KIT_SKUS)
    else:
        return []

    if dim == "material":
        n_lo, n_hi = material_l2_count_range(intensity)
    else:
        n_lo, n_hi = _l2_count_range(intensity)
    n = rng.randint(n_lo, n_hi)
    base_dur = _L2_BASE_DURATION.get(dim, 120.0)
    dur_cap = _l2_duration_cap(intensity, dim)
    durs = [
        int(
            max(
                _L2_MIN_DURATION,
                min(dur_cap, rng.uniform(0.7, 1.3) * base_dur * max(intensity, 0.5)),
            )
        )
        for _ in range(n)
    ]
    min_gap = max(_L2_MIN_GAP, _L2_MIN_GAP - 10 * n)
    starts = _sample_starts(rng, n, durs, horizon_lo, horizon_hi, min_gap)

    chosen: list[str] = []
    for _ in range(n):
        pool = targets
        if chosen and len(targets) > 1:
            alt = [t for t in targets if t != chosen[-1]]
            if alt:
                pool = alt
        chosen.append(rng.choice(pool))
    if dim == "logistics":
        _ensure_logistics_agv_freeze(chosen, targets, rng)

    mu_lo, mu_hi = material_l2_hide_count_range(intensity) if dim == "material" else (0, 0)
    events: list[dict[str, Any]] = []
    for i in range(n):
        ev: dict[str, Any] = {
            "start": int(starts[i]),
            "duration": int(durs[i]),
            "target": chosen[i],
            "dim": dim,
        }
        if dim == "material":
            ev["max_units"] = int(rng.randint(mu_lo, mu_hi))
            ev["wave"] = "l2"
        events.append(ev)
    events.sort(key=lambda e: e["start"])
    return opening + events


def episode_l2_schedule(dim: str, intensity: float, seed: int, env_id: int, episode_id: int) -> list[dict[str, Any]]:
    """Deterministic per-(seed, env, episode) sample used by injector and episode_config.

    One dim keeps the historical RNG. Mixed dims concatenate per-dim queues
    (injector still runs one L2 at a time; L0/L1 already overlap).
    """
    applied = RuntimeDisturbanceCfg.get("applied") or {}
    dims = parse_disturbance_dims(dim)
    kwargs = dict(
        human_count=int(applied.get("human_count") or 5),
        gantry_indices=applied.get("active_gantry_indices"),
        agv_count=int(applied.get("agv_count") or 2),
    )
    if dims == ["none"]:
        return []
    if len(dims) == 1:
        rng = l2_schedule_rng(seed, env_id, episode_id)
        return sample_l2_schedule(dims[0], intensity, rng, **kwargs)
    events: list[dict[str, Any]] = []
    salt = {"machine": 1, "human": 2, "logistics": 3, "material": 4}
    for d in dims:
        rng = l2_schedule_rng(int(seed) ^ (0x9E3779B9 * salt.get(d, 9)), env_id, episode_id)
        events.extend(sample_l2_schedule(d, intensity, rng, **kwargs))
    events.sort(key=lambda e: (int(e.get("start") or 0), str(e.get("dim") or "")))
    return events


def configure_disturbance_from_cli(
    dim: str = "none",
    intensity: float = 1.0,
    human_count: int | None = None,
    agv_count: int | None = None,
    gantry_count: int | None = None,
) -> dict[str, Any]:
    """Fill RuntimeDisturbanceCfg from CLI; call before gym.make / apply.

    ``dim`` may be one name or ``human,logistics`` / ``human+logistics``.
    One-dim L1 numbers stay identical to the exclusive-dim days.
    """
    dims = parse_disturbance_dims(dim)
    label = dim_label(dims)
    intensity = max(0.0, float(intensity))

    RuntimeDisturbanceCfg["dim"] = label
    RuntimeDisturbanceCfg["dims"] = list(dims)
    RuntimeDisturbanceCfg["intensity"] = intensity
    RuntimeDisturbanceCfg["human_count"] = human_count
    RuntimeDisturbanceCfg["agv_count"] = agv_count
    RuntimeDisturbanceCfg["gantry_count"] = gantry_count
    RuntimeDisturbanceCfg["applied"] = {}

    # Reset derived fields to safe baseline, then specialize by dim.
    RuntimeDisturbanceCfg["machine_process_noise_std"] = 0.0
    RuntimeDisturbanceCfg["machine_success_rate"] = 1.0
    RuntimeDisturbanceCfg["human_subtask_noise_std"] = 2.0
    RuntimeDisturbanceCfg["human_time_scale"] = 1.0
    RuntimeDisturbanceCfg["human_skill_scales"] = []
    RuntimeDisturbanceCfg["gantry_animation_noise_std"] = 2.0
    RuntimeDisturbanceCfg["gantry_time_scale"] = 1.0
    RuntimeDisturbanceCfg["material_shortage_frac"] = 0.0
    RuntimeDisturbanceCfg["material_l1_hide_count"] = 0
    RuntimeDisturbanceCfg["material_l1_duration"] = 0
    RuntimeDisturbanceCfg["material_l1_hide_count_range"] = []
    RuntimeDisturbanceCfg["material_l1_duration_range"] = []
    RuntimeDisturbanceCfg["material_l1_sku"] = _MATERIAL_L1_SKU
    RuntimeDisturbanceCfg["material_l2_hide_count_range"] = []
    RuntimeDisturbanceCfg["event_start_step"] = -1
    RuntimeDisturbanceCfg["event_duration_steps"] = 0
    RuntimeDisturbanceCfg["event_target"] = None
    RuntimeDisturbanceCfg["event_schedule"] = []
    RuntimeDisturbanceCfg["event_schedule_mode"] = "resample_per_episode"
    RuntimeDisturbanceCfg["tool_wear_per_1k_steps"] = 0.0
    RuntimeDisturbanceCfg["disabled_workstations"] = []
    RuntimeDisturbanceCfg["qc_holds"] = []

    if dims == ["none"] or intensity <= 0.0:
        RuntimeDisturbanceCfg["dim"] = "none"
        RuntimeDisturbanceCfg["dims"] = ["none"]
        RuntimeDisturbanceCfg["event_schedule_mode"] = "none"
        return RuntimeDisturbanceCfg

    noise = 0.0
    success = 1.0
    wear = 0.0
    if "machine" in dims:
        noise = max(noise, 5.0 * intensity)
        success = min(success, max(0.55, 1.0 - 0.12 * intensity))
        wear = max(wear, 15.0 * intensity)
        RuntimeDisturbanceCfg["disabled_workstations"] = [
            {"machine": m, "ws": w} for m, w in disabled_workstations_for_intensity(intensity)
        ]
    if "human" in dims:
        RuntimeDisturbanceCfg["human_subtask_noise_std"] = 2.0 + 10.0 * intensity
        RuntimeDisturbanceCfg["human_time_scale"] = 1.0 + 0.55 * intensity
        noise = max(noise, 2.0 * intensity)
        success = min(success, max(0.82, 1.0 - 0.05 * intensity))
        wear = max(wear, 6.0 * intensity)
    if "logistics" in dims:
        RuntimeDisturbanceCfg["gantry_animation_noise_std"] = 2.0 + 8.0 * intensity
        RuntimeDisturbanceCfg["gantry_time_scale"] = 1.0 + 0.65 * intensity
        noise = max(noise, 2.0 * intensity)
        success = min(success, max(0.82, 1.0 - 0.05 * intensity))
        wear = max(wear, 6.0 * intensity)
    if "material" in dims:
        n_lo, n_hi = material_l1_hide_count_range(intensity)
        d_lo, d_hi = material_l1_duration_range(intensity)
        l2_lo, l2_hi = material_l2_hide_count_range(intensity)
        RuntimeDisturbanceCfg["material_shortage_frac"] = 0.0
        RuntimeDisturbanceCfg["material_l1_hide_count"] = material_l1_hide_count(intensity)
        RuntimeDisturbanceCfg["material_l1_duration"] = material_l1_duration(intensity)
        RuntimeDisturbanceCfg["material_l1_hide_count_range"] = [n_lo, n_hi]
        RuntimeDisturbanceCfg["material_l1_duration_range"] = [d_lo, d_hi]
        RuntimeDisturbanceCfg["material_l1_sku"] = "|".join(_MATERIAL_KIT_SKUS)
        RuntimeDisturbanceCfg["material_l2_hide_count_range"] = [l2_lo, l2_hi]
        # Yield stays 1.0 unless another dim already lowered it.

    RuntimeDisturbanceCfg["machine_process_noise_std"] = noise
    RuntimeDisturbanceCfg["machine_success_rate"] = success
    RuntimeDisturbanceCfg["tool_wear_per_1k_steps"] = wear

    return RuntimeDisturbanceCfg


def _ensure_default_snapshot() -> None:
    global _DEFAULT_SNAPSHOT
    if _DEFAULT_SNAPSHOT is not None:
        return
    from .cfg_human import CfgHumanRegistrationInfos
    from .cfg_robot import CfgRobotRegistrationInfos
    from .cfg_machine import CfgMachine
    from . import cfg_process_subtask_gallery as subtask_mod

    _DEFAULT_SNAPSHOT = {
        "human": deepcopy(CfgHumanRegistrationInfos),
        "robot": deepcopy(CfgRobotRegistrationInfos),
        "active_gantry_indices": list(CfgMachine["num07_gantry_group"]["active_gantry_indices"]),
        "gantry_move_speed": float(
            CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]["move_speed"]
        ),
        "gantry_move_speed_noise_std": float(
            CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"].get(
                "move_speed_noise_std", 0.0
            )
        ),
        "subtask_noise_std": float(subtask_mod.SubtaskTimeNoiseStdSteps),
        "machine_animation_times": {
            mtype: {
                part: info.get("animation_time")
                for part, info in cfg.get("registration_infos", {}).items()
            }
            for mtype, cfg in CfgMachine.items()
            if mtype != "num07_gantry_group"
        },
        "machine_animation_noise": {
            mtype: {
                part: info.get("animation_time_noise_std", 0.0)
                for part, info in cfg.get("registration_infos", {}).items()
            }
            for mtype, cfg in CfgMachine.items()
            if mtype != "num07_gantry_group"
        },
        "machine_reset_states": {
            mtype: list(cfg.get("reset_state", {}).get("state") or [])
            for mtype, cfg in CfgMachine.items()
        },
    }


def apply_disturbance_to_cfgs() -> dict[str, Any]:
    """Mutate global registration / timing cfgs. Must run before managers construct."""
    _ensure_default_snapshot()
    assert _DEFAULT_SNAPSHOT is not None

    from .cfg_human import CfgHumanRegistrationInfos
    from .cfg_robot import CfgRobotRegistrationInfos
    from .cfg_machine import CfgMachine
    from . import cfg_process_subtask_gallery as subtask_mod

    # Restore defaults first (idempotent re-apply).
    CfgHumanRegistrationInfos.clear()
    CfgHumanRegistrationInfos.update(deepcopy(_DEFAULT_SNAPSHOT["human"]))
    CfgRobotRegistrationInfos.clear()
    CfgRobotRegistrationInfos.update(deepcopy(_DEFAULT_SNAPSHOT["robot"]))
    CfgMachine["num07_gantry_group"]["active_gantry_indices"].clear()
    CfgMachine["num07_gantry_group"]["active_gantry_indices"].extend(
        _DEFAULT_SNAPSHOT["active_gantry_indices"]
    )
    gantry_info = CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]
    gantry_info["move_speed"] = _DEFAULT_SNAPSHOT["gantry_move_speed"]
    gantry_info["move_speed_noise_std"] = _DEFAULT_SNAPSHOT["gantry_move_speed_noise_std"]
    subtask_mod.SubtaskTimeNoiseStdSteps = _DEFAULT_SNAPSHOT["subtask_noise_std"]
    for mtype, parts in _DEFAULT_SNAPSHOT["machine_animation_times"].items():
        for part, t in parts.items():
            if part in CfgMachine[mtype]["registration_infos"]:
                CfgMachine[mtype]["registration_infos"][part]["animation_time"] = t
                CfgMachine[mtype]["registration_infos"][part]["animation_time_noise_std"] = (
                    _DEFAULT_SNAPSHOT["machine_animation_noise"][mtype].get(part, 0.0)
                )
    for mtype, states in (_DEFAULT_SNAPSHOT.get("machine_reset_states") or {}).items():
        if mtype in CfgMachine and "reset_state" in CfgMachine[mtype]:
            CfgMachine[mtype]["reset_state"]["state"] = list(states)

    dim = RuntimeDisturbanceCfg["dim"]
    dims = active_dims()
    intensity = float(RuntimeDisturbanceCfg["intensity"])
    applied: dict[str, Any] = {"dim": dim, "dims": list(dims), "intensity": intensity}

    if dims == ["none"] or intensity <= 0.0:
        RuntimeDisturbanceCfg["applied"] = applied
        return applied

    # Always push L1 noise knobs into modules that managers read.
    subtask_mod.SubtaskTimeNoiseStdSteps = float(RuntimeDisturbanceCfg["human_subtask_noise_std"])
    applied["human_subtask_noise_std"] = subtask_mod.SubtaskTimeNoiseStdSteps
    applied["human_time_scale"] = RuntimeDisturbanceCfg["human_time_scale"]
    applied["human_skill_scales"] = list(RuntimeDisturbanceCfg.get("human_skill_scales") or [])
    applied["machine_process_noise_std"] = RuntimeDisturbanceCfg["machine_process_noise_std"]
    applied["machine_success_rate"] = RuntimeDisturbanceCfg["machine_success_rate"]
    applied["material_shortage_frac"] = RuntimeDisturbanceCfg["material_shortage_frac"]
    applied["material_l1_hide_count"] = int(RuntimeDisturbanceCfg.get("material_l1_hide_count") or 0)
    applied["material_l1_duration"] = int(RuntimeDisturbanceCfg.get("material_l1_duration") or 0)
    applied["material_l1_hide_count_range"] = list(
        RuntimeDisturbanceCfg.get("material_l1_hide_count_range") or []
    )
    applied["material_l1_duration_range"] = list(
        RuntimeDisturbanceCfg.get("material_l1_duration_range") or []
    )
    applied["material_l1_sku"] = str(RuntimeDisturbanceCfg.get("material_l1_sku") or _MATERIAL_L1_SKU)
    applied["material_l2_hide_count_range"] = list(
        RuntimeDisturbanceCfg.get("material_l2_hide_count_range") or []
    )
    applied["tool_wear_per_1k_steps"] = float(RuntimeDisturbanceCfg.get("tool_wear_per_1k_steps", 0.0) or 0.0)
    applied["disabled_workstations"] = list(RuntimeDisturbanceCfg.get("disabled_workstations") or [])

    if "machine" in dims:
        for item in applied["disabled_workstations"]:
            mtype = item["machine"]
            ws = int(item["ws"])
            states = CfgMachine[mtype]["reset_state"]["state"]
            if 0 <= ws < len(states):
                states[ws] = "invalid"

    if "human" in dims:
        default_n = int(_DEFAULT_SNAPSHOT["human"].get("NormalHuman", 5))
        n = RuntimeDisturbanceCfg["human_count"]
        if n is None:
            # I=1.0 → 3 people (was 4); enough to shift the weld/workbench constraint.
            n = max(1, int(round(default_n - (1.0 + intensity))))
        n = max(1, min(int(n), default_n))
        CfgHumanRegistrationInfos["NormalHuman"] = n
        applied["human_count"] = n
        skills = sample_human_skill_scales(n, intensity)
        RuntimeDisturbanceCfg["human_skill_scales"] = skills
        applied["human_skill_scales"] = skills
        base = float(RuntimeDisturbanceCfg["human_time_scale"])
        applied["human_time_scales"] = [round(base * s, 4) for s in skills]

    if "logistics" in dims:
        default_agv = int(_DEFAULT_SNAPSHOT["robot"].get("AGV", 2))
        n_agv = RuntimeDisturbanceCfg["agv_count"]
        if n_agv is None:
            n_agv = logistics_default_agv_count(intensity, n_nominal=default_agv)
        n_agv = max(1, min(int(n_agv), default_agv))
        CfgRobotRegistrationInfos["AGV"] = n_agv
        applied["agv_count"] = n_agv

        default_gantry = list(_DEFAULT_SNAPSHOT["active_gantry_indices"])
        n_g = RuntimeDisturbanceCfg["gantry_count"]
        if n_g is None:
            n_g = logistics_default_gantry_count(intensity, n_nominal=len(default_gantry))
        n_g = max(1, min(int(n_g), max(len(default_gantry), 4)))
        gantry_indices = CfgMachine["num07_gantry_group"]["active_gantry_indices"]
        gantry_indices.clear()
        gantry_indices.extend(range(n_g))
        applied["gantry_count"] = n_g
        applied["active_gantry_indices"] = list(gantry_indices)

        scale = float(RuntimeDisturbanceCfg["gantry_time_scale"])
        # Old API stretched animation_time; new API slows move_speed by the same factor.
        base_speed = float(_DEFAULT_SNAPSHOT["gantry_move_speed"])
        gantry_info["move_speed"] = max(1e-4, base_speed / max(scale, 1e-6))
        # Old noise was in animation steps (~20). Map relative noise onto speed units.
        rel_noise = float(RuntimeDisturbanceCfg["gantry_animation_noise_std"]) / 20.0
        gantry_info["move_speed_noise_std"] = max(0.0, gantry_info["move_speed"] * rel_noise)
        applied["gantry_move_speed"] = gantry_info["move_speed"]
        applied["gantry_move_speed_noise_std"] = gantry_info["move_speed_noise_std"]

    if "material" in dims:
        applied["material_l1_sku"] = str(RuntimeDisturbanceCfg.get("material_l1_sku") or _MATERIAL_L1_SKU)
        applied["material_l1_hide_count_range"] = list(
            RuntimeDisturbanceCfg.get("material_l1_hide_count_range") or []
        )
        applied["material_l1_duration_range"] = list(
            RuntimeDisturbanceCfg.get("material_l1_duration_range") or []
        )
        applied["material_l2_hide_count_range"] = list(
            RuntimeDisturbanceCfg.get("material_l2_hide_count_range") or []
        )
        applied["machine_success_rate"] = RuntimeDisturbanceCfg["machine_success_rate"]

    std = float(RuntimeDisturbanceCfg["machine_process_noise_std"])
    if std > 0.0:
        for mtype, cfg in CfgMachine.items():
            if mtype == "num07_gantry_group":
                continue
            for part, info in cfg.get("registration_infos", {}).items():
                if "animation_time" in info:
                    info["animation_time_noise_std"] = std
        applied["machine_animation_time_noise_std"] = std

    applied["event_start_step"] = RuntimeDisturbanceCfg["event_start_step"]
    applied["event_duration_steps"] = RuntimeDisturbanceCfg["event_duration_steps"]
    applied["event_target"] = RuntimeDisturbanceCfg["event_target"]
    applied["event_schedule_mode"] = RuntimeDisturbanceCfg.get("event_schedule_mode", "resample_per_episode")
    # Per-episode queues are sampled at reset; do not freeze a clock here.
    applied["event_schedule"] = list(RuntimeDisturbanceCfg.get("event_schedule") or [])
    RuntimeDisturbanceCfg["applied"] = applied

    print(f"[Disturbance] dim={dim} intensity={intensity} applied={applied}")
    return applied
