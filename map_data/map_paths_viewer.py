from __future__ import annotations

"""
基于 map_paths_human.json + 占据图，对 **已有的预计算路径** 做插值测试和可视化。

特点：
- 不再在线做 Dijkstra，只读取 map_paths_human.json 里预存的 paths[src][dst]
- 使用 map_points_human.json 做路径插值，生成 [x, y, yaw] 序列并统计耗时
- 显示两幅图，并且两个窗口默认尽量全屏：
  1) 原始占据图（建议用 occupancy_map_with_points.png），叠加所有路网点和当前路径
  2) 白色背景的放大图，只显示当前路径，自动缩放以尽量填满视图
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_image(path: Path) -> np.ndarray:
    img = plt.imread(str(path))
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    return img


def load_paths(paths_file: Path) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], Dict[str, object]]:
    """读取 map_paths_human.json，返回预计算路径表 paths 及元信息。"""
    data: Dict[str, object] = json.loads(paths_file.read_text(encoding="utf-8"))
    paths = data.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("JSON 中缺少 'paths' 字段，或格式不是 dict。")
    return paths, data


def load_points(points_file: Path) -> Tuple[Dict[int, Tuple[float, float]], List[int]]:
    pts_data = json.loads(points_file.read_text(encoding="utf-8"))
    pts = pts_data.get("points", pts_data)
    id_to_xy: Dict[int, Tuple[float, float]] = {}
    ids: List[int] = []
    for p in pts:
        pid = int(p["id"])
        x = float(p["x"])
        y = float(p["y"])
        id_to_xy[pid] = (x, y)
        ids.append(pid)
    return id_to_xy, ids


def _try_fullscreen(fig: plt.Figure) -> None:
    """尽可能将 matplotlib 窗口全屏显示（根据不同后端做 best-effort 处理）。"""
    try:
        manager = fig.canvas.manager  # type: ignore[attr-defined]
    except Exception:
        return

    # 常见后端专用调用（Qt / Tk / WX 等）
    for attr in ("window", "canvas", "frame"):
        win = getattr(manager, attr, None)
        if win is None:
            continue
        if hasattr(win, "showMaximized"):
            try:
                win.showMaximized()
                return
            except Exception:
                pass
    # 退而求其次：matplotlib 自带的 full_screen_toggle
    try:
        manager.full_screen_toggle()  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于 map_paths_human.json 的预计算路径，对任意两点做插值并可视化（双窗口）。"
    )
    parser.add_argument("--paths", required=True, type=str, help="预计算路径 JSON 文件，例如 map_paths_human.json")
    parser.add_argument(
        "--map",
        required=False,
        type=str,
        help="用于显示的底图，例如 occupancy_map_with_points.png；不填则尝试使用 JSON 中的 map_path。",
    )
    parser.add_argument("--src", type=int, default=None, help="起点节点 id")
    parser.add_argument("--dst", type=int, default=None, help="终点节点 id")
    parser.add_argument(
        "--interp-step",
        type=float,
        default=5.0,
        help="路径插值的步长（像素），越小越密集，默认 5 像素",
    )
    args = parser.parse_args()

    paths_file = Path(args.paths).expanduser()
    paths, meta = load_paths(paths_file)

    # 底图：优先使用命令行传入的 occupancy_map_with_points，否则回退到 JSON 中的 map_path
    if args.map:
        map_path = Path(args.map).expanduser()
    else:
        map_path_str = meta.get("map_path")
        if not isinstance(map_path_str, str):
            raise ValueError("既未通过 --map 指定底图，也未在 JSON 中找到有效的 map_path。")
        map_path = Path(map_path_str).expanduser()

    img = _load_image(map_path)

    # 推断 points 文件路径（仅为了获得坐标）
    id_to_xy: Dict[int, Tuple[float, float]] = {}
    all_point_ids: List[int] = []
    points_path_str = meta.get("points_path")
    points_candidate = None
    if isinstance(points_path_str, str):
        points_candidate = Path(points_path_str).expanduser()
    elif "paths_" in paths_file.name:
        points_candidate = paths_file.with_name(paths_file.name.replace("paths_", "points_", 1))

    if points_candidate is not None and points_candidate.exists():
        id_to_xy, all_point_ids = load_points(points_candidate)
    else:
        print(f"[WARN] 未找到对应的 points 文件: {points_candidate}")

    # 从 JSON 中构建边到 samples 的索引：edge_samples[(u,v)] = [{x,y,yaw}, ...]
    edge_samples: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    edges_meta = meta.get("edges")
    if isinstance(edges_meta, list):
        for e in edges_meta:
            if not isinstance(e, dict):
                continue
            try:
                u = int(e["u"])
                v = int(e["v"])
            except Exception:
                continue
            samples = e.get("samples")
            if isinstance(samples, list) and samples:
                edge_samples[(u, v)] = samples  # 只存 u->v 方向，v->u 运行时反向拼接

    # 主图：占据图 + 所有路网点与编号
    plt.ion()
    fig_main, ax_main = plt.subplots()
    ax_main.imshow(img)
    ax_main.set_title("Map view（所有路网点 + 当前路径）")
    ax_main.set_aspect("equal")
    _try_fullscreen(fig_main)

    if id_to_xy:
        xs_all = [id_to_xy[i][0] for i in all_point_ids]
        ys_all = [id_to_xy[i][1] for i in all_point_ids]
        ax_main.scatter(xs_all, ys_all, s=15, c="cyan", edgecolors="black", linewidths=0.3, alpha=0.9)
        for pid in all_point_ids:
            x, y = id_to_xy[pid]
            # 点编号改用蓝色，与 overlay 图保持一致
            ax_main.text(x + 3, y + 3, str(pid), color="#0066ff", fontsize=6)
    else:
        print("[WARN] 没有节点坐标信息，窗口中不会显示点和编号。")

    # 放大图：白色背景，只显示当前路径
    fig_zoom, ax_zoom = plt.subplots()
    fig_zoom.patch.set_facecolor("white")
    ax_zoom.set_facecolor("white")
    ax_zoom.set_aspect("equal")
    ax_zoom.set_title("Zoomed path view")
    _try_fullscreen(fig_zoom)

    current_line_main = None

    def run_once(src: int, dst: int) -> None:
        nonlocal current_line_main

        src_key = str(src)
        dst_key = str(dst)
        if src_key not in paths or dst_key not in paths[src_key]:
            print(f"[WARN] 找不到预计算路径：{src} -> {dst}")
            return

        path_info = paths[src_key][dst_key]
        node_ids_raw = path_info.get("nodes")
        if not isinstance(node_ids_raw, list):
            print(f"[WARN] 预计算路径格式错误：paths['{src_key}']['{dst_key}'] 无有效 nodes。")
            return
        node_ids = [int(n) for n in node_ids_raw]
        path_cost = float(path_info.get("cost", 0.0))

        print(
            f"[INFO] {src} -> {dst} | 预计算最短路长度={path_cost:.2f} | 节点数={len(node_ids)}"
        )

        if not id_to_xy:
            print("[WARN] 没有节点坐标信息，只能在终端打印节点序列。")
            print("路径节点序列:", node_ids)
            return

        # 仅使用已预生成的边 samples 做“路径拼接”，不再重新插值，并统计耗时
        t_interp0 = time.perf_counter()
        xs_interp: List[float] = []
        ys_interp: List[float] = []
        yaws_interp: List[float] = []
        step = float(args.interp_step)

        # 沿着节点序列，把每一段 (nid0, nid1) 对应的边 samples 拼接起来
        for i in range(len(node_ids) - 1):
            nid0 = node_ids[i]
            nid1 = node_ids[i + 1]
            key_fwd = (nid0, nid1)
            key_rev = (nid1, nid0)

            seg_samples: List[Dict[str, object]] | None = None
            forward = True
            if key_fwd in edge_samples:
                seg_samples = edge_samples[key_fwd]
                forward = True
            elif key_rev in edge_samples:
                seg_samples = edge_samples[key_rev]
                forward = False

            # 如果没有预生成 samples，则跳过这一段（也可以考虑回退到在线插值）
            if not seg_samples:
                print(f"[WARN] 边 {nid0}->{nid1} 没有预生成 samples，已跳过该段。")
                continue

            # 避免段与段之间重复拼接端点：从第二段开始跳过 samples 的首点
            first_seg = len(xs_interp) == 0

            if forward:
                iterable = seg_samples
            else:
                iterable = list(reversed(seg_samples))

            for j, s in enumerate(iterable):
                if not first_seg and j == 0:
                    continue
                try:
                    x_s = float(s["x"])
                    y_s = float(s["y"])
                    yaw_s = float(s.get("yaw", 0.0))
                except Exception:
                    continue

                if not forward:
                    # 反向时 yaw 需要加 pi
                    yaw_s += math.pi
                    if yaw_s > math.pi:
                        yaw_s -= 2 * math.pi
                    elif yaw_s < -math.pi:
                        yaw_s += 2 * math.pi

                xs_interp.append(x_s)
                ys_interp.append(y_s)
                yaws_interp.append(yaw_s)

        t_interp1 = time.perf_counter()
        elapsed_ms = (t_interp1 - t_interp0) * 1000.0
        # 计算真实平均步长（相邻样本点的平均距离）
        avg_step = 0.0
        if len(xs_interp) > 1:
            xs_arr_full = np.array(xs_interp)
            ys_arr_full = np.array(ys_interp)
            dists = np.hypot(np.diff(xs_arr_full), np.diff(ys_arr_full))
            if dists.size > 0:
                avg_step = float(dists.mean())

        print(
            f"[TIME] 拼接路径耗时: {elapsed_ms:.3f} ms | 路径点数: {len(xs_interp)} "
            f"| 平均间距约: {avg_step:.1f} 像素"
        )
        if xs_interp and ys_interp and yaws_interp:
            print(
                f"[DEBUG] 首点 [x,y,yaw]: "
                f"{xs_interp[0]:.1f}, {ys_interp[0]:.1f}, {yaws_interp[0]:.3f} rad"
            )

        # ===== 仅用于可视化的降采样（保留首尾点） =====
        if len(xs_interp) > 2:
            max_vis_points = 200  # 可视化最多显示点数
            stride = max(1, len(xs_interp) // max_vis_points)
            idxs = list(range(0, len(xs_interp), stride))
            if idxs[-1] != len(xs_interp) - 1:
                idxs.append(len(xs_interp) - 1)
            xs_vis = [xs_interp[i] for i in idxs]
            ys_vis = [ys_interp[i] for i in idxs]
            yaws_vis = [yaws_interp[i] for i in idxs]
        else:
            xs_vis, ys_vis, yaws_vis = xs_interp, ys_interp, yaws_interp

        # 更新主图上的路径（使用降采样后的点）
        if current_line_main is not None:
            current_line_main.remove()
            current_line_main = None
        if xs_vis and ys_vis:
            (current_line_main,) = ax_main.plot(
                xs_vis, ys_vis, "-o", color="red", linewidth=1.5, markersize=2.5
            )
            fig_main.canvas.draw_idle()

        # 更新放大图：白底 + 箭头表示朝向（只显示路网点，不显示插值点）
        ax_zoom.clear()
        ax_zoom.set_facecolor("white")
        ax_zoom.set_aspect("equal")

        # 使用路径上的路网点坐标，而不是插值点
        if node_ids and id_to_xy:
            xs_nodes: List[float] = []
            ys_nodes: List[float] = []
            for nid in node_ids:
                if nid in id_to_xy:
                    x_n, y_n = id_to_xy[nid]
                    xs_nodes.append(x_n)
                    ys_nodes.append(y_n)

            if len(xs_nodes) < 2:
                plt.pause(0.001)
                return

            # 对路网点本身做一次降采样（保留首尾）
            if len(xs_nodes) > 200:
                max_nodes = 200
                stride_n = max(1, len(xs_nodes) // max_nodes)
                idxs_n = list(range(0, len(xs_nodes), stride_n))
                if idxs_n[-1] != len(xs_nodes) - 1:
                    idxs_n.append(len(xs_nodes) - 1)
                xs_nodes = [xs_nodes[i] for i in idxs_n]
                ys_nodes = [ys_nodes[i] for i in idxs_n]

            xs_arr = np.array(xs_nodes)
            ys_arr = np.array(ys_nodes)
            x_min, x_max = float(xs_arr.min()), float(xs_arr.max())
            y_min, y_max = float(ys_arr.min()), float(ys_arr.max())
            width = max(x_max - x_min, 1.0)
            height = max(y_max - y_min, 1.0)
            margin = 0.1 * max(width, height)

            # 为每个路网点计算朝向（使用相邻点连线）
            yaws_nodes: List[float] = []
            for i in range(len(xs_nodes) - 1):
                dx_n = xs_nodes[i + 1] - xs_nodes[i]
                dy_n = ys_nodes[i + 1] - ys_nodes[i]
                if dx_n == 0 and dy_n == 0:
                    yaw_i = 0.0
                else:
                    yaw_i = math.atan2(dy_n, dx_n)
                yaws_nodes.append(yaw_i)
            # 最后一个点复用前一个点的朝向
            yaws_nodes.append(yaws_nodes[-1])

            # 使用 arrow 为每个路网点画箭头（稍微短一点，避免遮挡过多），并交替使用两种颜色和样式
            arrow_len = 0.04 * max(width, height)
            colors = ["red", "blue"]
            linestyles = ["solid", "dashed"]

            for idx, (x_n, y_n, yaw_n) in enumerate(zip(xs_nodes, ys_nodes, yaws_nodes)):
                dx_n = arrow_len * math.cos(yaw_n)
                dy_n = arrow_len * math.sin(yaw_n)
                ax_zoom.arrow(
                    x_n,
                    y_n,
                    dx_n,
                    dy_n,
                    width=arrow_len * 0.02,
                    head_width=arrow_len * 0.20,
                    head_length=arrow_len * 0.28,
                    length_includes_head=True,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    alpha=0.9,
                )
            # 标记起点和终点（路网点）
            ax_zoom.scatter([xs_nodes[0]], [ys_nodes[0]], c="green", s=40, label="start")
            ax_zoom.scatter([xs_nodes[-1]], [ys_nodes[-1]], c="blue", s=40, label="end")
            ax_zoom.legend(loc="best", fontsize=8)

            ax_zoom.set_xlim(x_min - margin, x_max + margin)
            ax_zoom.set_ylim(y_min - margin, y_max + margin)
            ax_zoom.set_title(f"Zoomed path: {src} -> {dst} (len={path_cost:.1f})")
            fig_zoom.canvas.draw_idle()

        plt.pause(0.001)

    plt.show(block=False)

    if args.src is not None and args.dst is not None:
        run_once(int(args.src), int(args.dst))
        return

    print("进入交互模式：输入起点/终点 id 可视化插值路径（输入 q 退出）")
    while True:
        s = input("起点 id (或 q 退出): ").strip()
        if s.lower() in {"q", "quit", "exit"}:
            break
        t = input("终点 id: ").strip()
        try:
            src = int(s)
            dst = int(t)
        except ValueError:
            print("请输入整数 id。")
            continue
        run_once(src, dst)


if __name__ == "__main__":
    main()

