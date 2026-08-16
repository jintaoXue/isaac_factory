"""Root-cause classes shared with dev_tyx PDFormer A.3."""

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

ROOT_CAUSE_TO_ID = {name: index for index, name in enumerate(ROOT_CAUSE_CLASSES)}


def encode_root_cause(reason: str | None) -> int:
    return ROOT_CAUSE_TO_ID.get((reason or "").strip(), -1)


def decode_root_cause(class_id: int) -> str:
    if 0 <= class_id < len(ROOT_CAUSE_CLASSES):
        return ROOT_CAUSE_CLASSES[class_id]
    return ""
