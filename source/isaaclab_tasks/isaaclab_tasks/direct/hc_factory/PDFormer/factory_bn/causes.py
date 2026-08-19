"""A3 bottleneck root-cause classes.

Must stay in sync with ``tools/bn_agg/labels.py`` (``root_cause_reason`` strings).
Stage-C now emits process heuristics only (queue / stall / utilization).
The first four names are kept for old npz/checkpoints; new labels do not use
them — injected L2 is environment context, not a bottleneck class.
Empty / unknown → ``-1`` (ignored by the classification loss).
"""

from __future__ import annotations

ROOT_CAUSE_CLASSES: tuple[str, ...] = (
    "machine_failure",
    "human_unavailable",
    "transport_delay",
    "material_shortage",
    "blocked_downstream",
    "starved_upstream",
    "unavailable",
    "high_utilization",
    "queue_buildup",
    "score_threshold",
)

ROOT_CAUSE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ROOT_CAUSE_CLASSES)}


def encode_root_cause(reason: str | None) -> int:
    """Map a ``root_cause_reason`` string to a class id; unlabeled → -1."""
    key = (reason or "").strip()
    if not key:
        return -1
    return ROOT_CAUSE_TO_ID.get(key, -1)


def decode_root_cause(class_id: int) -> str:
    if class_id < 0 or class_id >= len(ROOT_CAUSE_CLASSES):
        return ""
    return ROOT_CAUSE_CLASSES[class_id]
