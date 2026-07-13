"""Camera poses, sensor intrinsics, and ground-plane visibility helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

_CFG_DIR = Path(__file__).resolve().parent
CAMERA_GROUND_FOOTPRINTS_JSON = _CFG_DIR / "camera_ground_footprints.json"

_WORLD_UP = (0.0, 0.0, 1.0)
_DEFAULT_GROUND_Z = 1.0

_CAMERA_SENSOR = {
    "width": 1080,
    "height": 720,
    "update_period": 0.0,
    "data_types": ["rgb"],
    "spawn": {
        "focal_length": 22.0,
        "focus_distance": 400.0,
        "horizontal_aperture": 30.0,
        "clipping_range": (0.1, 1.0e5),
    },
}

_CAMERA_RESET_STATE = {
    "key_variables": {"type_name": None, "idx": None, "machine_name": None},
    "rgb": None,
    "is_initialized": False,
}

# machine_name -> camera_name -> {eye, lookat}
CAMERA_POSES = {
    "highrise_for_env": {
        "camera_num00_highrise_for_env": {
            "eye": (-3, -2.83729, 10),
            "lookat": (14, 8.76496, 1.0),
        },
        "camera_num01_highrise_for_env": {
            "eye": (10, -2.83729, 10),
            "lookat": (-17, 7.76346, 1.0),
        },
    },
    "num00_rotaryPipeAutomaticWeldingMachine": {
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
            "eye": (40, 20, 12),
            "lookat": (40, 14.5, 1.0),
        },
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
            "eye": (38, 20, 12),
            "lookat": (38, 14.5, 1.0),
        },
    },
    "num01_weldingRobot": {
        "camera_num01_weldingRobot_part02_robot_arm_and_base": {
            "eye": (24.75003, 20, 12),
            "lookat": (24.75003, 14.5, 1.0),
        },
    },
    "num02_rollerbedCNCPipeIntersectionCuttingMachine": {
        "camera_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
            "eye": (10, 20, 12),
            "lookat": (10, 14.5, 1.0),
        },
    },
    "num04_groovingMachineLarge": {
        "camera_num04_groovingMachineLarge_part01_large_fixed_base": {
            "eye": (-7.65483, 20, 12),
            "lookat": (-7.65483, 14.5, 1.0),
        },
    },
    "num08_workbench": {
        "camera_num08_workbench": {
            "eye": (26, -3, 12),
            "lookat": (26, -1, 1.0),
        },
    },
}


# ---------------------------------------------------------------------------
# Geometry: pinhole frustum intersected with ground plane z = ground_z
# ---------------------------------------------------------------------------


def _vec3_sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec3_add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec3_mul(a: Sequence[float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec3_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec3_cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec3_length(v: Sequence[float]) -> float:
    return math.sqrt(_vec3_dot(v, v))


def _vec3_normalize(v: Sequence[float]) -> tuple[float, float, float]:
    length = _vec3_length(v)
    if length < 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return _vec3_mul(v, 1.0 / length)


def compute_camera_frame(
    eye: Sequence[float],
    lookat: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return (forward, right, up) in env-local world frame."""
    forward = _vec3_normalize(_vec3_sub(lookat, eye))
    right = _vec3_cross(forward, _WORLD_UP)
    if _vec3_length(right) < 1e-9:
        right = (1.0, 0.0, 0.0)
    else:
        right = _vec3_normalize(right)
    up = _vec3_normalize(_vec3_cross(right, forward))
    return forward, right, up


def fov_from_spawn(
    spawn: dict,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Return horizontal and vertical FOV (radians) from Isaac pinhole spawn params."""
    focal_length = float(spawn["focal_length"])
    horizontal_aperture = float(spawn["horizontal_aperture"])
    vertical_aperture = float(spawn.get("vertical_aperture", horizontal_aperture * height / width))
    h_fov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
    v_fov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
    return h_fov, v_fov


def _corner_ray(
    forward: Sequence[float],
    right: Sequence[float],
    up: Sequence[float],
    h_fov: float,
    v_fov: float,
    u: float,
    v: float,
) -> tuple[float, float, float]:
    """Image corner ray; u,v in [0,1], origin top-left."""
    tan_h = math.tan(h_fov * 0.5)
    tan_v = math.tan(v_fov * 0.5)
    ndc_x = (2.0 * u - 1.0) * tan_h
    ndc_y = (1.0 - 2.0 * v) * tan_v
    direction = _vec3_add(
        _vec3_add(forward, _vec3_mul(right, ndc_x)),
        _vec3_mul(up, ndc_y),
    )
    return _vec3_normalize(direction)


def intersect_ground_plane(
    eye: Sequence[float],
    direction: Sequence[float],
    ground_z: float,
) -> tuple[float, float, float] | None:
    """Intersect a ray with horizontal plane z = ground_z; None if parallel or behind camera."""
    if abs(direction[2]) < 1e-9:
        return None
    t = (ground_z - eye[2]) / direction[2]
    if t <= 0.0:
        return None
    return (
        eye[0] + t * direction[0],
        eye[1] + t * direction[1],
        ground_z,
    )


def compute_ground_footprint_xy(
    eye: Sequence[float],
    lookat: Sequence[float],
    spawn: dict | None = None,
    width: int | None = None,
    height: int | None = None,
    ground_z: float | None = None,
) -> list[list[float]]:
    """Project image corners onto ground plane; order: top-left, top-right, bottom-right, bottom-left.

    Uses the frustum cross-section on z=ground_z anchored at the forward-axis ground hit.
    This stays stable for steep/overhead cameras where top corner rays may miss the floor.
    """
    spawn = spawn or _CAMERA_SENSOR["spawn"]
    width = width or int(_CAMERA_SENSOR["width"])
    height = height or int(_CAMERA_SENSOR["height"])
    if ground_z is None:
        ground_z = float(lookat[2])

    forward, right, up = compute_camera_frame(eye, lookat)
    h_fov, v_fov = fov_from_spawn(spawn, width, height)

    axis_hit = intersect_ground_plane(eye, forward, ground_z)
    if axis_hit is None:
        raise ValueError(f"Camera forward axis does not hit ground z={ground_z} for eye={eye}")

    t_axis = (ground_z - eye[2]) / forward[2]
    half_w = abs(t_axis) * math.tan(h_fov * 0.5)
    half_h = abs(t_axis) * math.tan(v_fov * 0.5)
    cx, cy = axis_hit[0], axis_hit[1]

    footprint: list[list[float]] = []
    for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        sx = -1.0 if u < 0.5 else 1.0
        sy = 1.0 if v < 0.5 else -1.0
        x = cx + sx * half_w * right[0] + sy * half_h * up[0]
        y = cy + sx * half_w * right[1] + sy * half_h * up[1]
        footprint.append([round(x, 6), round(y, 6)])
    return footprint


def point_in_polygon(x: float, y: float, polygon_xy: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test for a simple quadrilateral or polygon."""
    n = len(polygon_xy)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        denom = yj - yi
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (denom if abs(denom) > 1e-12 else 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def visible_human_ids_in_camera(
    human_xy_by_id: dict[str, tuple[float, float]],
    ground_footprint_xy: Sequence[Sequence[float]],
) -> list[str]:
    """Return human ids whose (x, y) lies inside the camera ground footprint."""
    visible: list[str] = []
    for human_id, (x, y) in human_xy_by_id.items():
        if point_in_polygon(x, y, ground_footprint_xy):
            visible.append(human_id)
    return sorted(visible)


# ---------------------------------------------------------------------------
# Camera registry
# ---------------------------------------------------------------------------


def _iter_camera_pose_entries(
    camera_poses: dict,
) -> list[tuple[str, dict[str, tuple[float, float, float]], str]]:
    entries: list[tuple[str, dict[str, tuple[float, float, float]], str]] = []
    for machine_name, cameras in camera_poses.items():
        for camera_name, pose in cameras.items():
            entries.append((camera_name, pose, machine_name))
    return entries


def _make_machine_camera(
    camera_name: str,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    machine_name: str,
) -> dict:
    return {
        "type_name": camera_name,
        "machine_name": machine_name,
        "meta_registeration_info": {
            "prim_paths_expr": f"/World/envs/env_{{i}}/obj/HC_factory/{camera_name}",
            "name": camera_name,
        },
        "eye": eye,
        "lookat": lookat,
        "camera_sensor": _CAMERA_SENSOR,
        "reset_state": _CAMERA_RESET_STATE,
        "debug_save_frames": True,
        "debug_max_frames": 50,
    }


CfgCamera = {
    camera_name: _make_machine_camera(camera_name, pose["eye"], pose["lookat"], machine_name)
    for camera_name, pose, machine_name in _iter_camera_pose_entries(CAMERA_POSES)
}

CfgCameraRegistrationInfos = {camera_name: 1 for camera_name in CfgCamera}


def has_registered_cameras() -> bool:
    return any(count > 0 for count in CfgCameraRegistrationInfos.values())


def build_camera_ground_footprints_document(
    ground_z: float = _DEFAULT_GROUND_Z,
) -> dict:
    """Build JSON-serializable ground footprints for all registered cameras."""
    cameras: dict[str, dict] = {}
    for camera_name, cfg in CfgCamera.items():
        eye = cfg["eye"]
        lookat = cfg["lookat"]
        sensor = cfg["camera_sensor"]
        spawn = sensor["spawn"]
        width = int(sensor["width"])
        height = int(sensor["height"])
        h_fov, v_fov = fov_from_spawn(spawn, width, height)
        footprint = compute_ground_footprint_xy(
            eye, lookat, spawn=spawn, width=width, height=height, ground_z=ground_z,
        )
        cameras[camera_name] = {
            "machine_name": cfg["machine_name"],
            "state_key": f"num_00_{camera_name}",
            "eye": list(eye),
            "lookat": list(lookat),
            "resolution": [width, height],
            "horizontal_fov_deg": round(math.degrees(h_fov), 4),
            "vertical_fov_deg": round(math.degrees(v_fov), 4),
            "ground_z": ground_z,
            "ground_footprint_xy": footprint,
            "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        }
    return {
        "schema": "hc_factory.camera_ground_footprints.v1",
        "ground_z": ground_z,
        "cameras": cameras,
    }


def export_camera_ground_footprints(
    output_path: str | Path | None = None,
    ground_z: float = _DEFAULT_GROUND_Z,
) -> Path:
    output_path = Path(output_path or CAMERA_GROUND_FOOTPRINTS_JSON)
    document = build_camera_ground_footprints_document(ground_z=ground_z)
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_camera_ground_footprints(path: str | Path | None = None) -> dict:
    json_path = Path(path or CAMERA_GROUND_FOOTPRINTS_JSON)
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Camera ground footprints not found: {json_path}. "
            f"Run: python {Path(__file__).resolve()} --output {json_path}"
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export camera ground footprints to JSON.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(CAMERA_GROUND_FOOTPRINTS_JSON),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--ground-z",
        type=float,
        default=_DEFAULT_GROUND_Z,
        help="Ground plane height used for frustum intersection.",
    )
    args = parser.parse_args()
    out = export_camera_ground_footprints(output_path=args.output, ground_z=args.ground_z)
    doc = json.loads(out.read_text(encoding="utf-8"))
    print(f"[cfg_camera] Wrote {out} ({len(doc['cameras'])} cameras, ground_z={args.ground_z})")


if __name__ == "__main__":
    main()
