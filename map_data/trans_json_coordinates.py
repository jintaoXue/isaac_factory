from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from map_coordinate_utils import (
    COORDINATE_FRAME_IMAGE,
    COORDINATE_FRAME_ISAAC,
    CoordinateMapping,
    load_config_file,
    load_mapping_from_config_dict,
    read_coordinate_frame_from_points_json,
    resolve_for_robot_flag,
    world_edge_length,
)


def _path_to_tilde(p: Path) -> str:
    s = str(p)
    home = os.path.expanduser("~").rstrip("/")
    if s.startswith(home + "/"):
        return "~" + s[len(home) :]
    return s


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是 dict：{path}")
    return data


def _dump_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _choose_output_path(src: Path, *, out_dir: Path | None, in_place: bool) -> Path:
    if in_place:
        return src
    if out_dir is None:
        return src.with_name(src.stem + ".isaac.json")
    return out_dir / src.name


def _embed_bounds(payload: Dict[str, Any], cfg: Mapping[str, Any], *, for_robot: bool) -> None:
    used_robot_block = for_robot and isinstance(cfg.get("png_image_coordinates_robot"), dict)
    png_k = "png_image_coordinates_robot" if used_robot_block else "png_image_coordinates"
    sim_k = "isaac_sim_coordinates_robot" if used_robot_block else "isaac_sim_coordinates"
    if isinstance(cfg.get(png_k), dict):
        payload["png_image_coordinates"] = cfg[png_k]
    if isinstance(cfg.get(sim_k), dict):
        payload["isaac_sim_coordinates"] = cfg[sim_k]


def convert_points_json(
    src_path: Path,
    *,
    mapping: CoordinateMapping,
    cfg: Mapping[str, Any],
    for_robot: bool,
) -> Dict[str, Any]:
    data = _load_json(src_path)
    frame = read_coordinate_frame_from_points_json(data)
    if frame == COORDINATE_FRAME_ISAAC:
        return data
    pts = data.get("points", data.get("data", None))
    if pts is None:
        pts = data.get("points")
    if pts is None:
        pts = data.get("points", [])
    if not isinstance(pts, list):
        raise ValueError(f"points 字段不是 list：{src_path}")

    out_pts: List[Dict[str, Any]] = []
    for i, p in enumerate(pts):
        if not isinstance(p, dict):
            raise ValueError(f"点格式错误 index={i}: {p!r}")
        pid = int(p.get("id", i))
        x_img = float(p["x"])
        y_img = float(p["y"])
        x_sim, y_sim = mapping.image_xy_to_isaac(x_img, y_img)
        out_pts.append({"id": pid, "x": float(x_sim), "y": float(y_sim)})

    payload: Dict[str, Any] = dict(data)
    payload["coordinate_frame"] = COORDINATE_FRAME_ISAAC
    payload["points"] = out_pts
    _embed_bounds(payload, cfg, for_robot=for_robot)
    return payload


def _recompute_path_costs_from_edges(
    *,
    paths: Mapping[str, Any],
    edges: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    # Build undirected length table from edges (u<v convention)
    length_uv: Dict[Tuple[int, int], float] = {}
    for e in edges:
        try:
            u = int(e["u"])
            v = int(e["v"])
            length = float(e["length"])
        except Exception:
            continue
        a, b = (u, v) if u < v else (v, u)
        if length > 0:
            length_uv[(a, b)] = length

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for s_key, dsts in paths.items():
        if not isinstance(dsts, dict):
            continue
        out_dsts: Dict[str, Dict[str, Any]] = {}
        for t_key, info in dsts.items():
            if not isinstance(info, dict):
                continue
            nodes = info.get("nodes")
            if not isinstance(nodes, list) or len(nodes) < 1:
                continue
            cost = 0.0
            ok = True
            for i in range(len(nodes) - 1):
                a = int(nodes[i])
                b = int(nodes[i + 1])
                k = (a, b) if a < b else (b, a)
                seg = length_uv.get(k)
                if seg is None:
                    ok = False
                    break
                cost += float(seg)
            new_info = dict(info)
            if ok:
                new_info["cost"] = float(cost)
            out_dsts[str(t_key)] = new_info
        out[str(s_key)] = out_dsts
    return out


def convert_routes_json(
    src_path: Path,
    *,
    mapping: CoordinateMapping,
    cfg: Mapping[str, Any],
    for_robot: bool,
) -> Dict[str, Any]:
    data = _load_json(src_path)
    frame = data.get("coordinate_frame")
    if frame == COORDINATE_FRAME_ISAAC:
        return data

    edges = data.get("edges")
    if not isinstance(edges, list):
        raise ValueError(f"routes JSON 缺少 edges:list：{src_path}")

    out_edges: List[Dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        u = int(e.get("u"))
        v = int(e.get("v"))
        # old may have pixel yaw/length; we recompute in sim space
        # get endpoints from samples if available, else skip
        samples = e.get("samples")
        out_e: Dict[str, Any] = {"u": u, "v": v}
        if isinstance(samples, list) and samples:
            out_samples: List[Dict[str, Any]] = []
            prev_img: Tuple[float, float] | None = None
            for s in samples:
                if not isinstance(s, dict):
                    continue
                x_img = float(s["x"])
                y_img = float(s["y"])
                x_sim, y_sim = mapping.image_xy_to_isaac(x_img, y_img)
                out_samples.append({"x": float(x_sim), "y": float(y_sim), "yaw": float(s.get("yaw", 0.0))})
                prev_img = (x_img, y_img)
            out_e["samples"] = out_samples

            # Recompute length from endpoints
            x0_img = float(samples[0]["x"])
            y0_img = float(samples[0]["y"])
            x1_img = float(samples[-1]["x"])
            y1_img = float(samples[-1]["y"])
            out_e["length"] = float(world_edge_length(mapping, x0_img, y0_img, x1_img, y1_img))
            dx = x1_img - x0_img
            dy = y1_img - y0_img
            out_e["yaw"] = float(mapping.yaw_image_delta_to_isaac(dx, dy))
        else:
            # No samples: cannot convert reliably; fall back to mapping endpoints unknown -> keep old
            # but still mark frame to avoid silent misuse.
            out_e.update({k: v for k, v in e.items() if k not in {"samples"}})
        out_edges.append(out_e)

    payload: Dict[str, Any] = dict(data)
    payload["coordinate_frame"] = COORDINATE_FRAME_ISAAC
    payload["path_cost_units"] = "isaac_sim_euclidean"
    payload["edges"] = out_edges
    _embed_bounds(payload, cfg, for_robot=for_robot)

    # Recompute paths cost using converted edges length
    paths = payload.get("paths")
    if isinstance(paths, dict):
        payload["paths"] = _recompute_path_costs_from_edges(paths=paths, edges=out_edges)
    return payload


@dataclass(frozen=True)
class Job:
    kind: str  # points|routes
    src: Path
    for_robot: bool


def _collect_jobs(cfg: Mapping[str, Any]) -> List[Job]:
    jobs: List[Job] = []
    for kind, key, is_robot in (
        ("points", "points_path", False),
        ("routes", "routes_path", False),
        ("points", "points_path_robot", True),
        ("routes", "routes_path_robot", True),
    ):
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            jobs.append(Job(kind=kind, src=Path(v).expanduser(), for_robot=is_robot))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(
        description="批量将 config.json 中的 points/routes JSON 从 PNG 像素坐标迁移为 Isaac Sim 坐标。"
    )
    p.add_argument("--config", required=True, type=str, help="map_data/config.json 路径")
    p.add_argument("--in-place", action="store_true", help="原地覆盖写回（默认：生成新文件）")
    p.add_argument("--out-dir", type=str, default=None, help="输出目录（默认：同目录生成 *.isaac.json）")
    p.add_argument("--only", choices=["points", "routes", "all"], default="all", help="只转换 points 或 routes")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = load_config_file(cfg_path)
    if not cfg:
        raise ValueError(f"config.json 解析失败或为空：{cfg_path}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    jobs = _collect_jobs(cfg)
    if args.only != "all":
        jobs = [j for j in jobs if j.kind == args.only]

    if not jobs:
        print("[WARN] config.json 中未找到 points_path/routes_path（含 *_robot）。")
        return

    for j in jobs:
        if not j.src.exists():
            print(f"[WARN] 跳过不存在的文件: {j.src}")
            continue
        # Decide mapping block based on configured map path (human/robot)
        map_key = "map_path_robot" if j.for_robot else "map_path"
        map_path = Path(cfg.get(map_key, "")).expanduser() if isinstance(cfg.get(map_key), str) else None
        # resolve_for_robot_flag uses actual map png; use the configured one
        for_robot = resolve_for_robot_flag(cfg, map_path) if map_path is not None else j.for_robot
        mapping = load_mapping_from_config_dict(cfg, for_robot=for_robot)
        if mapping is None:
            raise ValueError(
                f"无法从 config.json 加载坐标映射（缺少 png_image_coordinates/isaac_sim_coordinates 或 robot 块）：{cfg_path}"
            )

        dst = _choose_output_path(j.src, out_dir=out_dir, in_place=bool(args.in_place))
        if j.kind == "points":
            payload = convert_points_json(j.src, mapping=mapping, cfg=cfg, for_robot=for_robot)
        else:
            payload = convert_routes_json(j.src, mapping=mapping, cfg=cfg, for_robot=for_robot)
        _dump_json(dst, payload)
        print(f"[OK] {j.kind}: {j.src} -> {dst}")


if __name__ == "__main__":
    main()

