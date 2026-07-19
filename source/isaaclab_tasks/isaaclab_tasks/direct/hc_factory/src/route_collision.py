"""Geometry, overlap tests, and occupancy-map helpers for route collision avoidance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CoordinateMapping:
    """Bilinear mapping between PNG image coords and Isaac Sim world xy."""

    u0: float
    v0: float
    u1: float
    v1: float
    x_at_u0: float
    x_at_u1: float
    y_at_v0: float
    y_at_v1: float

    @classmethod
    def from_cfg(cls, png_coords: dict, isaac_coords: dict) -> CoordinateMapping:
        tl = png_coords["top_left"]
        br = png_coords["bottom_right"]
        stl = isaac_coords["top_left"]
        sbr = isaac_coords["bottom_right"]
        return cls(
            u0=float(tl[0]),
            v0=float(tl[1]),
            u1=float(br[0]),
            v1=float(br[1]),
            x_at_u0=float(stl[0]),
            x_at_u1=float(sbr[0]),
            y_at_v0=float(stl[1]),
            y_at_v1=float(sbr[1]),
        )

    def isaac_xy_to_image(self, x_sim: float, y_sim: float) -> tuple[float, float]:
        du = self.u1 - self.u0
        dv = self.v1 - self.v0
        dx = self.x_at_u1 - self.x_at_u0
        dy = self.y_at_v1 - self.y_at_v0
        u = self.u0 if abs(dx) < 1e-12 else self.u0 + (float(x_sim) - self.x_at_u0) * du / dx
        v = self.v0 if abs(dy) < 1e-12 else self.v0 + (float(y_sim) - self.y_at_v0) * dv / dy
        return u, v


def circle_footprint(cx: float, cy: float, diameter: float) -> dict:
    return {"type": "circle", "cx": float(cx), "cy": float(cy), "r": float(diameter) * 0.5}


def rect_corners_from_pose(
    x: float,
    y: float,
    yaw: float,
    local_bounds: dict[str, float],
) -> list[tuple[float, float]]:
    """Build world-frame rectangle corners from local footprint bounds and yaw."""
    min_x = float(local_bounds["min_x"])
    max_x = float(local_bounds["max_x"])
    min_y = float(local_bounds["min_y"])
    max_y = float(local_bounds["max_y"])
    local_corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners: list[tuple[float, float]] = []
    for lx, ly in local_corners:
        wx = x + cos_yaw * lx - sin_yaw * ly
        wy = y + sin_yaw * lx + cos_yaw * ly
        corners.append((wx, wy))
    return corners


def rect_footprint(x: float, y: float, yaw: float, local_bounds: dict[str, float]) -> dict:
    return {
        "type": "rect",
        "corners": rect_corners_from_pose(x, y, yaw, local_bounds),
    }


def _project_polygon(axis: tuple[float, float], corners: list[tuple[float, float]]) -> tuple[float, float]:
    ax, ay = axis
    dots = [px * ax + py * ay for px, py in corners]
    return min(dots), max(dots)


def _rect_axes(corners: list[tuple[float, float]]) -> list[tuple[float, float]]:
    axes: list[tuple[float, float]] = []
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        edge_x = x1 - x0
        edge_y = y1 - y0
        axis = (-edge_y, edge_x)
        length = math.hypot(axis[0], axis[1])
        if length < 1e-9:
            continue
        axes.append((axis[0] / length, axis[1] / length))
    return axes


def _circle_rect_overlap(cx: float, cy: float, radius: float, corners: list[tuple[float, float]]) -> bool:
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        edge_x = x1 - x0
        edge_y = y1 - y0
        if abs(edge_x) < 1e-9 and abs(edge_y) < 1e-9:
            continue
        t = ((cx - x0) * edge_x + (cy - y0) * edge_y) / (edge_x * edge_x + edge_y * edge_y)
        t = max(0.0, min(1.0, t))
        closest_x = x0 + t * edge_x
        closest_y = y0 + t * edge_y
        if (cx - closest_x) ** 2 + (cy - closest_y) ** 2 <= radius * radius:
            return True
    for px, py in corners:
        if (cx - px) ** 2 + (cy - py) ** 2 <= radius * radius:
            return True
    return False


def _rect_rect_overlap(corners_a: list[tuple[float, float]], corners_b: list[tuple[float, float]]) -> bool:
    for corners in (corners_a, corners_b):
        for axis in _rect_axes(corners):
            min_a, max_a = _project_polygon(axis, corners_a)
            min_b, max_b = _project_polygon(axis, corners_b)
            if max_a < min_b or max_b < min_a:
                return False
    return True


def primitives_overlap(primitive_a: dict, primitive_b: dict) -> bool:
    type_a, type_b = primitive_a["type"], primitive_b["type"]
    if type_a == "rect" and type_b == "circle":
        primitive_a, primitive_b = primitive_b, primitive_a
        type_a, type_b = type_b, type_a

    if type_a == "circle" and type_b == "circle":
        dx = primitive_a["cx"] - primitive_b["cx"]
        dy = primitive_a["cy"] - primitive_b["cy"]
        return dx * dx + dy * dy <= (primitive_a["r"] + primitive_b["r"]) ** 2

    if type_a == "circle" and type_b == "rect":
        return _circle_rect_overlap(
            primitive_a["cx"], primitive_a["cy"], primitive_a["r"], primitive_b["corners"]
        )

    if type_a == "rect" and type_b == "rect":
        return _rect_rect_overlap(primitive_a["corners"], primitive_b["corners"])

    raise ValueError(f"Unsupported footprint primitive types: {type_a}, {type_b}")


def passing_ranges_overlap(range_a: dict, range_b: dict) -> bool:
    for prim_a in range_a.get("primitives", []):
        for prim_b in range_b.get("primitives", []):
            if primitives_overlap(prim_a, prim_b):
                return True
    return False


def _sample_points_in_circle(cx: float, cy: float, radius: float, step: float) -> list[tuple[float, float]]:
    points = [(cx, cy)]
    if radius <= 1e-6:
        return points
    angle_steps = max(8, int(math.ceil(2.0 * math.pi * radius / step)))
    for i in range(angle_steps):
        angle = 2.0 * math.pi * i / angle_steps
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    ring_r = radius * 0.5
    for i in range(angle_steps):
        angle = 2.0 * math.pi * i / angle_steps
        points.append((cx + ring_r * math.cos(angle), cy + ring_r * math.sin(angle)))
    return points


def _sample_points_in_rect(corners: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    points: list[tuple[float, float]] = []
    x = min_x
    while x <= max_x + 1e-9:
        y = min_y
        while y <= max_y + 1e-9:
            points.append((x, y))
            y += step
        x += step
    for corner in corners:
        points.append(corner)
    return points


def sample_passing_range_points(passing_range: dict, step: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for primitive in passing_range.get("primitives", []):
        if primitive["type"] == "circle":
            points.extend(
                _sample_points_in_circle(primitive["cx"], primitive["cy"], primitive["r"], step)
            )
        elif primitive["type"] == "rect":
            points.extend(_sample_points_in_rect(primitive["corners"], step))
    return points


class OccupancyMap:
    """Grayscale occupancy image: bright pixels are free, dark pixels are occupied."""

    def __init__(
        self,
        image_path: str,
        png_coords: dict,
        isaac_coords: dict,
        free_threshold: int,
    ):
        path = Path(image_path).expanduser()
        self._image = np.array(Image.open(path).convert("L"))
        self._height, self._width = self._image.shape
        self._mapping = CoordinateMapping.from_cfg(png_coords, isaac_coords)
        self._free_threshold = int(free_threshold)

    def is_free_at_isaac_xy(self, x: float, y: float) -> bool:
        u, v = self._mapping.isaac_xy_to_image(x, y)
        col = int(round(u))
        row = int(round(v))
        if col < 0 or row < 0 or col >= self._width or row >= self._height:
            return False
        return int(self._image[row, col]) >= self._free_threshold

    def passing_range_collides(self, passing_range: dict, sample_step: float = 0.25) -> bool:
        for x, y in sample_passing_range_points(passing_range, sample_step):
            if not self.is_free_at_isaac_xy(x, y):
                return True
        return False


def densify_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    step: float,
) -> list[tuple[float, float]]:
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-9:
        return [(x0, y0)]
    count = max(1, int(math.ceil(dist / step)))
    points: list[tuple[float, float]] = []
    for i in range(count + 1):
        t = i / count
        points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return points


def yaw_from_xy_delta(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)


def clamp_yaw_step(previous_yaw: float, target_yaw: float, max_step: float) -> float:
    """Limit per-waypoint yaw change for smoother routes."""
    delta = target_yaw - previous_yaw
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    delta = max(-max_step, min(max_step, delta))
    return previous_yaw + delta


def agent_priority_key(agent_type: str, agent_key: str) -> tuple[int, int]:
    """Lower tuple = higher priority. Larger agent index waits within the same type."""
    parts = agent_key.split("_")
    agent_index = int(parts[1]) if len(parts) >= 2 else 0
    return (0 if agent_type == "human" else 1, agent_index)


def should_agent_wait(agent_a: dict, agent_b: dict) -> bool:
    """Return True if agent_a should wait for agent_b."""
    return agent_priority_key(agent_a["agent_type"], agent_a["agent_key"]) > agent_priority_key(
        agent_b["agent_type"], agent_b["agent_key"]
    )
