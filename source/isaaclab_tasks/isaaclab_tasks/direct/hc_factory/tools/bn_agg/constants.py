"""Graph layout, score weights, and event-detection thresholds for Stage-C."""

from __future__ import annotations

ACTIVE_STATES = frozenset({"PROCESSING"})
BLOCKED_STATES = frozenset({"BLOCKED"})
STARVED_STATES = frozenset({"STARVED", "WAITING"})
STOP_STATES = frozenset({"STOP"})

# Dense PDFormer target = process congestion (queue / stall / coupling), not
# injected downtime. STOP/unavailable stays an *input* feature; L2 pulses are
# environment context via disturbance_active_s, not score y.
# Stall/active/up/down are already 0–1; queue/wait/duration are min-max in-window.
W_QUEUE = 0.15
W_WAIT = 0.10
W_STALL = 0.20
W_ACTIVE = 0.10
W_ACTIVE_DUR = 0.05
W_UPSTREAM = 0.10
W_DOWNSTREAM = 0.10
W_STOP = 0.0

# Keep in sync with PDFormer/factory_bn/graph.py (serial flow; siblings are parallel).
PROCESS_CHAIN: list[str] = [
    "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
    "num04_groovingMachineLarge_ws0",
    "num08_workbench_ws0",
    "num08_workbench_ws1",
    "num01_weldingRobot_ws0",
    "num00_rotaryPipeAutomaticWeldingMachine_ws0",
    "num00_rotaryPipeAutomaticWeldingMachine_ws1",
]
BUFFER_MACHINE_AFFINITY: dict[str, list[str]] = {
    "BlackStorage": [
        "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        "num04_groovingMachineLarge_ws0",
    ],
    "YellowStorage": [
        "num08_workbench_ws0",
        "num08_workbench_ws1",
        "num01_weldingRobot_ws0",
        "num00_rotaryPipeAutomaticWeldingMachine_ws0",
        "num00_rotaryPipeAutomaticWeldingMachine_ws1",
    ],
    "GroundStorage": [
        "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        "num04_groovingMachineLarge_ws0",
        "num08_workbench_ws0",
    ],
}
MATERIAL_CONSUMER: dict[str, str] = {
    "product_00_pipe_raw": "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
    "product_00_flange": "num08_workbench_ws0",
    "product_00_elbow": "num08_workbench_ws0",
}

# Parallel workstations share the same up/down neighbors (Lai 2021 TPM).
PROCESS_NEIGHBORS: dict[str, dict[str, list[str]]] = {
    "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0": {
        "up": [],
        "down": ["num04_groovingMachineLarge_ws0"],
    },
    "num04_groovingMachineLarge_ws0": {
        "up": ["num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0"],
        "down": ["num08_workbench_ws0", "num08_workbench_ws1"],
    },
    "num08_workbench_ws0": {
        "up": ["num04_groovingMachineLarge_ws0"],
        "down": ["num01_weldingRobot_ws0"],
    },
    "num08_workbench_ws1": {
        "up": ["num04_groovingMachineLarge_ws0"],
        "down": ["num01_weldingRobot_ws0"],
    },
    "num01_weldingRobot_ws0": {
        "up": ["num08_workbench_ws0", "num08_workbench_ws1"],
        "down": [
            "num00_rotaryPipeAutomaticWeldingMachine_ws0",
            "num00_rotaryPipeAutomaticWeldingMachine_ws1",
        ],
    },
    "num00_rotaryPipeAutomaticWeldingMachine_ws0": {
        "up": ["num01_weldingRobot_ws0"],
        "down": [],
    },
    "num00_rotaryPipeAutomaticWeldingMachine_ws1": {
        "up": ["num01_weldingRobot_ws0"],
        "down": [],
    },
}

DEFAULT_SCORE_THRESHOLD = 0.55
# Isolated turning-points are valid STGNPP onsets (do not require a 2-window run).
DEFAULT_MIN_EVENT_WINDOWS = 1
# Busy processing is a PDFormer feature, not an STGNPP event trigger.
HOT_ACTIVE_PCT = 0.70
# is_window_peak (PDFormer input only): relative score peak inside a window.
SCORE_PEAK_FLOOR = 0.20
SCORE_PEAK_RATIO = 0.95

# L2 interval types that fill the *input* column disturbance_active_s.
# They are environment configuration, not bottleneck events / will labels.
# quality_hold is a planned window logged at episode start — not an L2 pulse.
# Config-only rows (machine_config / …) are ignored.
DISTURBANCE_L2_TYPES = frozenset(
    {
        "transport_delay",  # logistics: gantry or AGV down
        "machine_failure",  # machine: station invalid
        "human_unavailable",  # human: worker leave
        "material_shortage",  # material: hide idle warehouse stock
    }
)
