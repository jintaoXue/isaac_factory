"""A3 bottleneck root-cause classes.

Must stay in sync with ``tools/bn_agg/labels.py`` (``root_cause_reason`` strings).
Stage-C emits process heuristics from window features (shortage flag, inbound
wait, queue, blocked/starved). It does not copy ``disturbance_log`` types.
``machine_failure`` / ``human_unavailable`` / ``unavailable`` stay in the
tuple for old npz; new labels do not write those three.
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
