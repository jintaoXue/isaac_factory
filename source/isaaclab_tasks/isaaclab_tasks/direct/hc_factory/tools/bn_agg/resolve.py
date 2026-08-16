"""Map disturbance / station ids onto window_feature_table resource_id."""

from __future__ import annotations

from .constants import MATERIAL_CONSUMER

def resolve_feature_resource_id(rid: str, known_ids) -> str:
    """Map machine-group / material disturbance ids onto window_feature_table resource_id.

    Feature rows use workstation ids (``...CuttingMachine_ws0``). Disturbance
    logs may still write the machine group name without ``_wsN``. Material L2
    ids ``material_{sku}`` are attached to the starved process node.
    """
    rid = (rid or "").strip()
    if not rid:
        return rid
    known = list(known_ids)
    known_set = set(known)
    if rid in known_set:
        return rid
    sku = rid[len("material_") :] if rid.startswith("material_") else rid
    material_node = MATERIAL_CONSUMER.get(sku)
    if material_node and material_node in known_set:
        return material_node
    workstations = [k for k in known if k.startswith(rid + "_ws")]
    if workstations:
        preferred = f"{rid}_ws0"
        return preferred if preferred in known_set else sorted(workstations)[0]
    return rid


def wait_resource_id(row: dict[str, str], known_ids) -> str:
    """Attach queue waits to the downstream station, not the finishing one.

    ``queue_enter`` is logged on the current ``station_id`` while the job waits
    to leave for ``output_buffer_id`` (next machine type, e.g. grooving).
    """
    station = (row.get("station_id") or "unknown").strip()
    out = (row.get("output_buffer_id") or "").strip()
    if out.startswith("num"):
        mapped = resolve_feature_resource_id(out, known_ids)
        if mapped:
            return mapped
    return resolve_feature_resource_id(station, known_ids) or station
