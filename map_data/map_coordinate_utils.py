"""
PNG 图像坐标（列 x、行 y，与 matplotlib imshow 一致）与 Isaac Sim 世界坐标的仿射映射。

配置见 map_data/config.json：
  - png_image_coordinates（或 png_image_coordinates_robot）
  - isaac_sim_coordinates（或 isaac_sim_coordinates_robot）

矩形四角线性插值：图像左上角 -> isaac top_left，右下角 -> isaac bottom_right。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


COORDINATE_FRAME_ISAAC = "isaac_sim"
COORDINATE_FRAME_IMAGE = "png_image"


@dataclass(frozen=True)
class CoordinateMapping:
    """图像 u（向右）, v（向下）与 Sim x, y 的双线性边界映射（各轴独立线性）。"""

    u0: float
    v0: float
    u1: float
    v1: float
    # Sim：与 (u0,v0)、(u1,v1) 对齐的角点（与 config 中 top_left / bottom_right 一致）
    x_at_u0: float
    x_at_u1: float
    y_at_v0: float
    y_at_v1: float

    @property
    def du_den(self) -> float:
        return self.u1 - self.u0

    @property
    def dv_den(self) -> float:
        return self.v1 - self.v0

    @property
    def dxd_u(self) -> float:
        if abs(self.du_den) < 1e-12:
            return 0.0
        return (self.x_at_u1 - self.x_at_u0) / self.du_den

    @property
    def dyd_v(self) -> float:
        if abs(self.dv_den) < 1e-12:
            return 0.0
        return (self.y_at_v1 - self.y_at_v0) / self.dv_den

    def image_xy_to_isaac(self, x_img: float, y_img: float) -> tuple[float, float]:
        u, v = float(x_img), float(y_img)
        if abs(self.du_den) < 1e-12:
            x = self.x_at_u0
        else:
            x = self.x_at_u0 + (u - self.u0) * (self.x_at_u1 - self.x_at_u0) / self.du_den
        if abs(self.dv_den) < 1e-12:
            y = self.y_at_v0
        else:
            y = self.y_at_v0 + (v - self.v0) * (self.y_at_v1 - self.y_at_v0) / self.dv_den
        return x, y

    def isaac_xy_to_image(self, x_sim: float, y_sim: float) -> tuple[float, float]:
        sx = self.dxd_u
        sy = self.dyd_v
        if abs(sx) < 1e-12:
            u = self.u0
        else:
            u = self.u0 + (float(x_sim) - self.x_at_u0) / sx
        if abs(sy) < 1e-12:
            v = self.v0
        else:
            v = self.v0 + (float(y_sim) - self.y_at_v0) / sy
        return u, v

    def yaw_image_delta_to_isaac(self, du: float, dv: float) -> float:
        """图像平面上的方向向量 (du, dv) 映射到 Sim 平面上的朝向（弧度）。"""
        dx = du * self.dxd_u
        dy = dv * self.dyd_v
        return math.atan2(dy, dx)


def _corner_mapping(png: Mapping[str, Any], isaac: Mapping[str, Any]) -> CoordinateMapping:
    tl = png["top_left"]
    br = png["bottom_right"]
    stl = isaac["top_left"]
    sbr = isaac["bottom_right"]
    u0, v0 = float(tl[0]), float(tl[1])
    u1, v1 = float(br[0]), float(br[1])
    x0, y0 = float(stl[0]), float(stl[1])
    x1, y1 = float(sbr[0]), float(sbr[1])
    return CoordinateMapping(
        u0=u0,
        v0=v0,
        u1=u1,
        v1=v1,
        x_at_u0=x0,
        x_at_u1=x1,
        y_at_v0=y0,
        y_at_v1=y1,
    )


def load_mapping_from_embedded_points_json(data: Mapping[str, Any]) -> CoordinateMapping | None:
    """路网点 JSON 内嵌的 png/isaac 角点（与 config.json 相同字段名）。"""
    return load_mapping_from_config_dict(data, for_robot=False)


def load_mapping_from_config_dict(cfg: Mapping[str, Any], *, for_robot: bool = False) -> CoordinateMapping | None:
    """
    从已解析的 config.json 字典构建映射。
    若存在 robot 专用字段则用于 for_robot=True，否则回退到 png_image_coordinates / isaac_sim_coordinates。
    """
    if for_robot:
        png = cfg.get("png_image_coordinates_robot")
        sim = cfg.get("isaac_sim_coordinates_robot")
        if isinstance(png, dict) and isinstance(sim, dict):
            return _corner_mapping(png, sim)
    png = cfg.get("png_image_coordinates")
    sim = cfg.get("isaac_sim_coordinates")
    if not isinstance(png, dict) or not isinstance(sim, dict):
        return None
    return _corner_mapping(png, sim)


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def resolve_for_robot_flag(cfg: Mapping[str, Any], map_png_path: Path | None) -> bool:
    """根据 config 中的 map_path / map_path_robot 与当前占据图路径判断是否使用 robot 坐标块。"""
    if map_png_path is None:
        return False
    try:
        m = map_png_path.expanduser().resolve()
    except Exception:
        m = map_png_path.expanduser()
    robot_s = cfg.get("map_path_robot")
    human_s = cfg.get("map_path")
    if isinstance(robot_s, str):
        try:
            if Path(robot_s).expanduser().resolve() == m:
                return True
        except Exception:
            if Path(robot_s).expanduser() == m:
                return True
    if isinstance(human_s, str):
        try:
            if Path(human_s).expanduser().resolve() == m:
                return False
        except Exception:
            if Path(human_s).expanduser() == m:
                return False
    # 无法匹配时：若存在 robot 专用 png 坐标且路径名含 robot，则偏向 robot
    name = map_png_path.name.lower()
    if "robot" in name and cfg.get("png_image_coordinates_robot"):
        return True
    return False


def world_edge_length(
    mapping: CoordinateMapping,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
) -> float:
    """图像平面两点对应 Sim 平面欧氏距离（米或 Sim 单位）。"""
    x0, y0 = mapping.image_xy_to_isaac(u0, v0)
    x1, y1 = mapping.image_xy_to_isaac(u1, v1)
    return math.hypot(x1 - x0, y1 - y0)


def read_coordinate_frame_from_points_json(data: Mapping[str, Any]) -> str:
    """缺省视为旧版 png 像素坐标。"""
    cf = data.get("coordinate_frame")
    if cf == COORDINATE_FRAME_ISAAC:
        return COORDINATE_FRAME_ISAAC
    return COORDINATE_FRAME_IMAGE
