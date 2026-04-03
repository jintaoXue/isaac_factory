from __future__ import annotations

"""
根据占据栅格地图和人工标注的路网点，预计算点对之间的最短路径。

输入：
  - 占据图：例如 occupancy_map_asset.png
  - 路网点：例如 roadmap_points_human.json（由 roadmap_generation.py 生成）

步骤：
  1. 读取占据图，将较暗区域视为障碍，较亮区域视为可行走区域。
  2. 为每个路网点，在“无碰撞”的情况下与若干近邻点连边，得到图结构。
  3. 对每个起点运行 Dijkstra，得到到其他所有点的最短路（欧氏距离加权）。
  4. 将所有点对 (i, j) 的最短路径（节点序列 + 总距离）保存到 JSON 文件中。

示例用法：
  python3 map_data/roadmap_process.py \\
      --map map_data/occupancy_map_asset.png \\
      --points map_data/roadmap_points_human.json \\
      --out map_data/roadmap_routes_human.json
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

import heapq

import matplotlib.pyplot as plt
import numpy as np

from map_coordinate_utils import (
    COORDINATE_FRAME_ISAAC,
    CoordinateMapping,
    load_config_file,
    load_mapping_from_config_dict,
    read_coordinate_frame_from_points_json,
    resolve_for_robot_flag,
    world_edge_length,
)


@dataclass
class RoadmapPoint:
    id: int
    x: float
    y: float


def load_occupancy_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    img = plt.imread(str(path))
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    # 统一到灰度 [0, 1]
    if img.ndim == 3:
        # RGB 或 RGBA
        img = img[..., :3]
        img = img.mean(axis=-1)
    return img.astype(np.float32)


def load_points(
    path: Path,
    mapping: CoordinateMapping | None,
) -> Tuple[List[RoadmapPoint], str]:
    """
    读取路网点；若 JSON 为 Isaac 坐标且提供 mapping，则转为 PNG 像素坐标供占据图使用。
    返回 (像素坐标路网点, 原始文件中的 coordinate_frame)。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = data.get("points", data)
    if not isinstance(pts, list):
        raise ValueError("路网点 JSON 格式不正确：需要 list 或包含 points:list")
    frame = read_coordinate_frame_from_points_json(data)
    out: List[RoadmapPoint] = []
    for i, p in enumerate(pts):
        if not isinstance(p, dict):
            raise ValueError(f"点格式错误 index={i}: {p!r}")
        pid = p.get("id", i)
        x = float(p["x"])
        y = float(p["y"])
        if frame == COORDINATE_FRAME_ISAAC:
            if mapping is None:
                raise ValueError(
                    "路网点为 isaac_sim 坐标，但未提供坐标映射（请传 --config 且包含 png/isaac 角点）。"
                )
            x, y = mapping.isaac_xy_to_image(x, y)
        out.append(RoadmapPoint(int(pid), x, y))
    return out, frame


def build_occupancy_mask(gray_img: np.ndarray, obstacle_threshold: float = 0.4) -> np.ndarray:
    """
    根据灰度图构建占据栅格：
      True  表示该像素为障碍
      False 表示该像素为可通行
    默认认为颜色越暗越可能是障碍（例如黑色障碍物）。
    """
    # 灰度越小越接近 0，越暗 => 障碍
    occ = gray_img < obstacle_threshold
    return occ


def is_segment_collision_free(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    occ: np.ndarray,
    safety_margin: int = 0,
) -> bool:
    """
    检查从 p1 到 p2 的线段是否与障碍物碰撞（基于占据栅格）。
    p1, p2: (x, y) 像素坐标，x->列，y->行。
    """
    h, w = occ.shape
    x1, y1 = p1
    x2, y2 = p2
    num_samples = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    if num_samples <= 1:
        num_samples = 2
    xs = np.linspace(x1, x2, num_samples)
    ys = np.linspace(y1, y2, num_samples)
    for x, y in zip(xs, ys):
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            # 出界视为碰撞
            return False
        # 带安全边界检查邻域
        x_min = max(ix - safety_margin, 0)
        x_max = min(ix + safety_margin, w - 1)
        y_min = max(iy - safety_margin, 0)
        y_max = min(iy + safety_margin, h - 1)
        if occ[y_min : y_max + 1, x_min : x_max + 1].any():
            return False
    return True


def build_graph(
    points: List[RoadmapPoint],
    occ: np.ndarray,
    k_neighbors: int = 8,
    max_neighbor_dist: float = 250.0,
    safety_margin: int = 1,
    mapping: CoordinateMapping | None = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    基于“最近邻 + 碰撞检测”构建无向图。
    返回：
      graph[node_id] = [(neighbor_id, cost), ...]
    """
    # 预先构建坐标数组便于计算距离
    coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    ids = [p.id for p in points]
    id_to_index = {pid: i for i, pid in enumerate(ids)}
    n = len(points)
    graph: Dict[int, List[Tuple[int, float]]] = {pid: [] for pid in ids}

    for i in range(n):
        pi = coords[i]
        # 所有点到当前点的距离
        diff = coords - pi
        dist = np.sqrt((diff**2).sum(axis=1))
        order = np.argsort(dist)
        neighbors_added = 0
        for j in order[1:]:  # 跳过自己
            if neighbors_added >= k_neighbors:
                break
            d = float(dist[j])
            if d > max_neighbor_dist:
                break
            pid_i = ids[i]
            pid_j = ids[j]
            p1 = (float(pi[0]), float(pi[1]))
            p2 = (float(coords[j, 0]), float(coords[j, 1]))
            if not is_segment_collision_free(p1, p2, occ, safety_margin=safety_margin):
                continue
            # 无碰撞，则添加双向边；若存在 Sim 映射，Dijkstra 边权使用 Sim 平面欧氏距离
            w = (
                world_edge_length(mapping, p1[0], p1[1], p2[0], p2[1])
                if mapping is not None
                else d
            )
            graph[pid_i].append((pid_j, w))
            graph[pid_j].append((pid_i, w))
            neighbors_added += 1

    return graph


def build_edge_list(
    points: List[RoadmapPoint],
    graph: Dict[int, List[Tuple[int, float]]],
    edge_interp_step: float | None = None,
    mapping: CoordinateMapping | None = None,
) -> List[Dict[str, Any]]:
    """将邻接表压缩为无向边列表，只记录必要的相邻路网点之间的路径。

    每条边只保存一次（u < v），并记录：
      - u, v: 两端节点 id
      - length: 欧氏距离
      - yaw: 从 u 指向 v 的朝向角（弧度）
      - samples: 可选，仅当 edge_interp_step>0 时存在，
                 为该边的插值路径点列表 [{x,y,yaw}, ...]

    对于 v->u 的方向，可以在使用时通过 yaw + pi 推断；如需要 v->u 的 samples，
    可以在运行时将 samples 反向并对 yaw 加 pi。
    """
    points_by_id: Dict[int, Tuple[float, float]] = {p.id: (p.x, p.y) for p in points}
    edges: List[Dict[str, Any]] = []

    for u, neighbors in graph.items():
        for v, _ in neighbors:
            if u >= v:
                continue  # 只保留 u < v，避免重复
            if u not in points_by_id or v not in points_by_id:
                continue
            x0, y0 = points_by_id[u]
            x1, y1 = points_by_id[v]
            dx = x1 - x0
            dy = y1 - y0
            length_px = math.hypot(dx, dy)
            if length_px <= 1e-6:
                continue
            if mapping is not None:
                length = world_edge_length(mapping, x0, y0, x1, y1)
                yaw = mapping.yaw_image_delta_to_isaac(dx, dy)
            else:
                length = length_px
                yaw = math.atan2(dy, dx)
            edge_entry: Dict[str, Any] = {
                "u": int(u),
                "v": int(v),
                "length": float(length),
                "yaw": float(yaw),
            }

            # 如指定 edge_interp_step，则为这条“最短边”（两个路网点之间）预先生成插值路径
            if edge_interp_step is not None and edge_interp_step > 0.0:
                step = float(edge_interp_step)
                n_steps = max(1, int(length_px / step))
                samples: List[Dict[str, float]] = []
                for k in range(n_steps):
                    t = k / n_steps
                    sx = x0 + dx * t
                    sy = y0 + dy * t
                    if mapping is not None:
                        ix, iy = mapping.image_xy_to_isaac(sx, sy)
                        samples.append({"x": float(ix), "y": float(iy), "yaw": float(yaw)})
                    else:
                        samples.append({"x": float(sx), "y": float(sy), "yaw": float(yaw)})
                # 确保终点也包含在 samples 中
                if mapping is not None:
                    ex, ey = mapping.image_xy_to_isaac(x1, y1)
                    samples.append({"x": float(ex), "y": float(ey), "yaw": float(yaw)})
                else:
                    samples.append({"x": float(x1), "y": float(y1), "yaw": float(yaw)})
                edge_entry["samples"] = samples

            edges.append(edge_entry)
    return edges


def dijkstra_all_targets(
    source: int, graph: Dict[int, List[Tuple[int, float]]]
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """从 source 出发运行 Dijkstra，返回到所有节点的最短距离和前驱。"""
    dist: Dict[int, float] = {node: math.inf for node in graph.keys()}
    prev: Dict[int, int] = {}
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(source: int, target: int, prev: Dict[int, int]) -> List[int]:
    """根据前驱字典，从 source 到 target 重建节点序列。"""
    if source == target:
        return [source]
    if target not in prev:
        return []
    path = [target]
    cur = target
    while cur != source:
        cur = prev.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


def precompute_all_pairs_shortest_paths(
    graph: Dict[int, List[Tuple[int, float]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """对图中所有节点两两之间的最短路进行预计算（仅保存节点序列与最短距离）。"""
    nodes = sorted(graph.keys())
    all_paths: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for src in nodes:
        dist, prev = dijkstra_all_targets(src, graph)
        src_dict: Dict[str, Dict[str, Any]] = {}
        for dst in nodes:
            if math.isinf(dist.get(dst, math.inf)):
                continue
            path_nodes = reconstruct_path(src, dst, prev)
            if not path_nodes:
                continue
            src_dict[str(dst)] = {"cost": float(dist[dst]), "nodes": path_nodes}
        all_paths[str(src)] = src_dict

    return all_paths

def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据占据图和路网点预计算所有点对之间的最短路径，并保存到 JSON 文件。"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选：map_data/config.json（含 png_image_coordinates / isaac_sim_coordinates），"
        "用于输出 Isaac Sim 坐标及 Sim 距离权重的最短路；默认与脚本同目录的 config.json",
    )
    parser.add_argument(
        "--map",
        required=True,
        type=str,
        help="占据地图图片路径，例如 map_data/occupancy_map_asset.png",
    )
    parser.add_argument(
        "--points",
        required=True,
        type=str,
        help="路网点 JSON 文件路径，例如 map_data/roadmap_points_human.json",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="输出最短路 JSON 文件路径，例如 map_data/roadmap_routes_human.json",
    )
    parser.add_argument(
        "--obstacle-threshold",
        type=float,
        default=0.4,
        help="灰度阈值，小于该值认为是障碍（0-1），默认 0.4",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="每个点最多连接的近邻数目（用于构建图），默认 8",
    )
    parser.add_argument(
        "--max-neighbor-dist",
        type=float,
        default=250.0,
        help="近邻连接的最大距离（像素），超过则不连边，默认 250",
    )
    parser.add_argument(
        "--safety-margin",
        type=int,
        default=1,
        help="碰撞检测时的安全像素边界大小，默认 1 像素",
    )
    parser.add_argument(
        "--interp-step",
        type=float,
        default=5.0,
        help="（可选）为每条相邻边预先生成插值路径时的步长（像素）；<=0 表示不生成 samples。",
    )

    args = parser.parse_args()
    map_path = Path(args.map).expanduser()
    points_path = Path(args.points).expanduser()
    out_path = Path(args.out).expanduser()

    cfg_path = (
        Path(args.config).expanduser()
        if args.config
        else Path(__file__).resolve().with_name("config.json")
    )
    cfg_data = load_config_file(cfg_path)
    for_robot = resolve_for_robot_flag(cfg_data, map_path)
    mapping = load_mapping_from_config_dict(cfg_data, for_robot=for_robot)
    if mapping is None:
        print(
            "[WARN] 未从 config 加载到 png/isaac 坐标映射：将使用像素坐标输出（与旧版一致）。"
        )

    print(f"[INFO] 读取占据图: {map_path}")
    gray = load_occupancy_image(map_path)
    occ = build_occupancy_mask(gray, obstacle_threshold=float(args.obstacle_threshold))

    print(f"[INFO] 读取路网点: {points_path}")
    points, _points_frame = load_points(points_path, mapping)
    print(f"[INFO] 路网点数量: {len(points)}")

    print("[INFO] 构建图（邻接关系 + 碰撞检测）...")
    graph = build_graph(
        points,
        occ,
        k_neighbors=int(args.k_neighbors),
        max_neighbor_dist=float(args.max_neighbor_dist),
        safety_margin=int(args.safety_margin),
        mapping=mapping,
    )

    # 简单统计一下每个点的度数
    degrees = {nid: len(neis) for nid, neis in graph.items()}
    print(
        f"[INFO] 图构建完成：总节点 {len(graph)}, "
        f"平均度数 {np.mean(list(degrees.values())):.2f}, "
        f"最大度数 {max(degrees.values()) if degrees else 0}"
    )

    print("[INFO] 压缩存储相邻路网点之间的边信息（无向图，每条边只记录一次）...")
    edge_interp_step = float(args.interp_step)
    if edge_interp_step <= 0.0:
        edges = build_edge_list(points, graph, edge_interp_step=None, mapping=mapping)
        print("[INFO] 未为边生成插值 samples（interp-step<=0）。")
    else:
        edges = build_edge_list(points, graph, edge_interp_step=edge_interp_step, mapping=mapping)
        print(f"[INFO] 为每条边预生成插值 samples，步长={edge_interp_step:.2f} 像素。")
    print(f"[INFO] 边数量: {len(edges)}")

    print("[INFO] 预计算所有点对的最短路径（Dijkstra all-pairs，仅保存节点序列与最短距离）...")
    all_paths = precompute_all_pairs_shortest_paths(graph)

    def _path_to_tilde(p: Path) -> str:
        """将位于当前用户 HOME 下的绝对路径收敛为 ~/...，便于跨机器复用 JSON。"""
        s = str(p)
        home = os.path.expanduser("~").rstrip("/")
        if s.startswith(home + "/"):
            return "~" + s[len(home) :]
        return s

    coord_frame = COORDINATE_FRAME_ISAAC if mapping is not None else "png_image"
    result: Dict[str, Any] = {
        "coordinate_frame": coord_frame,
        "map_path": _path_to_tilde(map_path),
        "points_path": _path_to_tilde(points_path),
        "num_nodes": len(graph),
        "num_edges": len(edges),
        "obstacle_threshold": float(args.obstacle_threshold),
        "k_neighbors": int(args.k_neighbors),
        "max_neighbor_dist": float(args.max_neighbor_dist),
        "safety_margin": int(args.safety_margin),
        "edges": edges,
        "paths": all_paths,
    }
    if mapping is not None:
        result["path_cost_units"] = "isaac_sim_euclidean"
        used_robot_block = for_robot and isinstance(
            cfg_data.get("png_image_coordinates_robot"), dict
        )
        png_k = (
            "png_image_coordinates_robot" if used_robot_block else "png_image_coordinates"
        )
        sim_k = (
            "isaac_sim_coordinates_robot" if used_robot_block else "isaac_sim_coordinates"
        )
        if isinstance(cfg_data.get(png_k), dict):
            result["png_image_coordinates"] = cfg_data[png_k]
        if isinstance(cfg_data.get(sim_k), dict):
            result["isaac_sim_coordinates"] = cfg_data[sim_k]
    else:
        result["path_cost_units"] = "png_pixels"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] 已保存最短路结果到: {out_path}")


if __name__ == "__main__":
    main()

