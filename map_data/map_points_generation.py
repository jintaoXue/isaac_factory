from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np


RoadmapFormat = Literal["json", "csv"]

try:
    from PIL import Image, ImageDraw  # type: ignore

    _PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    _PIL_AVAILABLE = False


def _infer_format(path: Path) -> RoadmapFormat:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("json", "csv"):
        return suffix  # type: ignore[return-value]
    raise ValueError(f"不支持的输出格式: {path.suffix}（仅支持 .json / .csv）")


def _load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    img = plt.imread(str(path))
    # Normalize to HxW or HxWxC numpy array
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    return img


def _load_points(path: Path) -> list[dict[str, Any]]:
    fmt = _infer_format(path)
    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        pts = data.get("points", data)
        if not isinstance(pts, list):
            raise ValueError("JSON 路网文件格式不正确：需要 list 或包含 points:list")
        out: list[dict[str, Any]] = []
        for i, p in enumerate(pts):
            if isinstance(p, dict):
                x = p.get("x")
                y = p.get("y")
                pid = p.get("id", i)
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x, y = p[0], p[1]
                pid = i
            else:
                raise ValueError(f"JSON 点格式不正确（index={i}）: {p!r}")
            out.append({"id": int(pid), "x": float(x), "y": float(y)})
        return out

    # csv
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV 路网文件为空或无表头")
        # allow id optional
        out2: list[dict[str, Any]] = []
        for i, row in enumerate(reader):
            if "x" not in row or "y" not in row:
                raise ValueError("CSV 需要包含 x,y 列（可选 id 列）")
            pid = row.get("id")
            out2.append(
                {
                    "id": int(pid) if pid not in (None, "") else i,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )
        return out2


def _try_fullscreen(fig: plt.Figure) -> None:
    """尽可能将 matplotlib 窗口全屏显示（best-effort）。"""
    try:
        manager = fig.canvas.manager  # type: ignore[attr-defined]
    except Exception:
        return

    # 常见后端（Qt / Tk / WX 等）的窗口对象
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

    # 退而求其次：使用 matplotlib 自带的 full_screen_toggle
    try:
        manager.full_screen_toggle()  # type: ignore[attr-defined]
    except Exception:
        pass


def _export_overlay_image(
    *,
    map_path: Path,
    points_path: Path,
    out_image_path: Path,
    radius_px: int = 4,
    draw_labels: bool = True,
) -> None:
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    points = _load_points(points_path)

    if _PIL_AVAILABLE:
        img = Image.open(map_path).convert("RGBA")  # type: ignore[union-attr]
        draw = ImageDraw.Draw(img)  # type: ignore[union-attr]
        for idx, p in enumerate(points):
            x = float(p["x"])
            y = float(p["y"])
            r = int(radius_px)
            bbox = (x - r, y - r, x + r, y + r)
            draw.ellipse(bbox, fill=(255, 0, 0, 255), outline=(255, 255, 255, 255), width=1)
            if draw_labels:
                # 默认字体即可；避免依赖外部字体文件
                draw.text((x + r + 2, y - r - 2), str(idx), fill=(255, 255, 0, 255))
        img.save(out_image_path)
        return

    # fallback: matplotlib（尽量保持原始分辨率）
    img_arr = _load_image(map_path)
    h, w = int(img_arr.shape[0]), int(img_arr.shape[1])
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img_arr)
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    ax.scatter(xs, ys, s=(radius_px * radius_px * 6), c="red", edgecolors="white", linewidths=0.8)
    if draw_labels:
        for idx, p in enumerate(points):
            ax.text(
                float(p["x"]) + radius_px + 2,
                float(p["y"]) - radius_px - 2,
                str(idx),
                color="yellow",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 0.8},
            )
    fig.savefig(out_image_path, dpi=dpi)
    plt.close(fig)


def _save_points(
    path: Path,
    *,
    points: list[dict[str, Any]],
    map_path: str | None,
    image_shape: tuple[int, ...] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _infer_format(path)

    if fmt == "json":
        payload: dict[str, Any] = {
            "map_path": map_path,
            "image_shape": list(image_shape) if image_shape is not None else None,
            "points": points,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    # csv
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x", "y"])
        writer.writeheader()
        for p in points:
            writer.writerow({"id": int(p["id"]), "x": float(p["x"]), "y": float(p["y"])})


@dataclass
class EditorConfig:
    map_path: Path
    out_path: Path
    load_path: Path | None = None
    marker_size: float = 30.0
    show_labels: bool = True
    delete_radius_px: float = 12.0


class RoadmapPointEditor:
    def __init__(self, cfg: EditorConfig):
        self.cfg = cfg
        self.img = _load_image(cfg.map_path)

        self.points: list[dict[str, Any]] = []
        if cfg.load_path is not None and cfg.load_path.exists():
            self.points = _load_points(cfg.load_path)

        self._fig, self._ax = plt.subplots()
        self._ax.set_title("路网点编辑器（点击添加，按 h 查看快捷键）")
        self._ax.imshow(self.img)
        self._ax.set_aspect("equal")
        self._scatter = None
        self._labels: list[Any] = []
        self._info_text = self._ax.text(
            0.01,
            0.99,
            "",
            transform=self._ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        # 默认尽量全屏显示窗口
        _try_fullscreen(self._fig)

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render()
        self._update_info()

    def _update_info(self, extra: str | None = None) -> None:
        n = len(self.points)
        msg = f"点数: {n} | 输出: {self.cfg.out_path.name}"
        if self.cfg.load_path is not None:
            msg += f" | 载入: {self.cfg.load_path.name}"
        msg += " | h:帮助"
        if extra:
            msg += f"\n{extra}"
        self._info_text.set_text(msg)
        self._fig.canvas.draw_idle()

    def _render(self) -> None:
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None
        for t in self._labels:
            try:
                t.remove()
            except Exception:
                pass
        self._labels = []

        if len(self.points) == 0:
            self._fig.canvas.draw_idle()
            return

        xs = [p["x"] for p in self.points]
        ys = [p["y"] for p in self.points]
        self._scatter = self._ax.scatter(
            xs,
            ys,
            s=self.cfg.marker_size,
            c="red",
            marker="o",
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            zorder=3,
        )
        if self.cfg.show_labels:
            for i, p in enumerate(self.points):
                self._labels.append(
                    self._ax.text(
                        float(p["x"]) + 4.0,
                        float(p["y"]) - 4.0,
                        str(i),
                        color="yellow",
                        fontsize=9,
                        zorder=4,
                        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 0.8},
                    )
                )
        self._fig.canvas.draw_idle()

    def _next_id(self) -> int:
        if not self.points:
            return 0
        return int(max(p["id"] for p in self.points)) + 1

    def _add_point(self, x: float, y: float) -> None:
        self.points.append({"id": self._next_id(), "x": float(x), "y": float(y)})
        self._render()
        self._update_info(f"已添加点: x={x:.1f}, y={y:.1f}")

    def _nearest_index(self, x: float, y: float) -> tuple[int | None, float]:
        if not self.points:
            return None, float("inf")
        xs = np.array([p["x"] for p in self.points], dtype=float)
        ys = np.array([p["y"] for p in self.points], dtype=float)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    def _delete_nearest(self, x: float, y: float) -> None:
        idx, dist = self._nearest_index(x, y)
        if idx is None or dist > self.cfg.delete_radius_px:
            self._update_info(f"未删除：最近点距离 {dist:.1f}px（阈值 {self.cfg.delete_radius_px}px）")
            return
        p = self.points.pop(idx)
        self._render()
        self._update_info(f"已删除点(index={idx}, id={p['id']}): x={p['x']:.1f}, y={p['y']:.1f}")

    def _save(self) -> None:
        _save_points(
            self.cfg.out_path,
            points=self.points,
            map_path=str(self.cfg.map_path),
            image_shape=tuple(self.img.shape) if isinstance(self.img, np.ndarray) else None,
        )
        self._update_info(f"已保存: {self.cfg.out_path}（点数 {len(self.points)}）")

    def _print_help(self) -> None:
        help_msg = (
            "鼠标左键: 添加点\n"
            "鼠标右键: 删除最近点（需在阈值内）\n"
            "u: 撤销（删除最后一个点）\n"
            "d: 删除鼠标附近最近点\n"
            "s: 保存到输出文件\n"
            "l: 重新从 --load 文件加载（覆盖当前点）\n"
            "c: 清空所有点\n"
            "t: 切换显示序号\n"
            "q 或 ESC: 退出（不自动保存）"
        )
        self._update_info(help_msg)

    def _on_click(self, event) -> None:
        if event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if event.button == 1:
            self._add_point(x, y)
        elif event.button == 3:
            self._delete_nearest(x, y)

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(self._fig)
            return
        if key == "h":
            self._print_help()
            return
        if key == "s":
            self._save()
            return
        if key == "u":
            if self.points:
                p = self.points.pop()
                self._render()
                self._update_info(f"撤销：删除最后点 id={p['id']}")
            else:
                self._update_info("没有可撤销的点")
            return
        if key == "c":
            self.points = []
            self._render()
            self._update_info("已清空所有点（记得按 s 保存）")
            return
        if key == "t":
            self.cfg.show_labels = not self.cfg.show_labels
            self._render()
            self._update_info(f"显示序号: {self.cfg.show_labels}")
            return
        if key in ("d",):
            if event.xdata is None or event.ydata is None:
                self._update_info("请将鼠标放在图内再按 d 删除")
                return
            self._delete_nearest(float(event.xdata), float(event.ydata))
            return
        if key == "l":
            if self.cfg.load_path is None:
                self._update_info("未指定 --load，无法重新加载")
                return
            if not self.cfg.load_path.exists():
                self._update_info(f"载入文件不存在: {self.cfg.load_path}")
                return
            self.points = _load_points(self.cfg.load_path)
            self._render()
            self._update_info(f"已重新加载: {self.cfg.load_path}（点数 {len(self.points)}）")

    def show(self) -> None:
        plt.show()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="根据占据栅格图交互式生成路网点（鼠标点选添加/删除，保存为 JSON 或 CSV）"
    )
    p.add_argument(
        "--map",
        required=True,
        type=str,
        help="占据栅格地图图片路径（png/jpg等），例如 map_data/occupancy_map_hall_asset.png",
    )
    p.add_argument(
        "--out",
        required=True,
        type=str,
        help="输出路网点文件（.json 或 .csv），例如 map_data/roadmap_points.json",
    )
    p.add_argument(
        "--load",
        default=None,
        type=str,
        help="可选：加载已有路网点文件（.json 或 .csv）并叠加显示后继续编辑",
    )
    p.add_argument("--marker-size", type=float, default=30.0, help="点的显示大小（matplotlib scatter size）")
    p.add_argument("--no-labels", action="store_true", help="不显示点序号")
    p.add_argument("--delete-radius", type=float, default=12.0, help="删除最近点的半径阈值（像素）")

    p.add_argument(
        "--overlay-map",
        default=None,
        type=str,
        help="可选：导出“带点的地图图片”时使用的底图（例如 map_data/occupancy_map_asset.png）",
    )
    p.add_argument(
        "--overlay-points",
        default=None,
        type=str,
        help="可选：导出叠加图时使用的点文件（.json/.csv）；不填则优先用 --load，否则用 --out",
    )
    p.add_argument(
        "--overlay-out",
        default=None,
        type=str,
        help="可选：导出叠加图的输出图片路径（例如 map_data/occupancy_map_with_points.png）",
    )
    p.add_argument("--overlay-radius", type=int, default=4, help="导出叠加图时点的半径（像素）")
    p.add_argument("--overlay-no-labels", action="store_true", help="导出叠加图时不显示点序号")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    # 导出叠加图模式：只要用户提供了 --overlay-out，就直接导出并退出
    if args.overlay_out is not None:
        overlay_map = Path(args.overlay_map).expanduser() if args.overlay_map else Path(args.map).expanduser()
        if args.overlay_points:
            overlay_points = Path(args.overlay_points).expanduser()
        elif args.load:
            overlay_points = Path(args.load).expanduser()
        else:
            overlay_points = Path(args.out).expanduser()
        _export_overlay_image(
            map_path=overlay_map,
            points_path=overlay_points,
            out_image_path=Path(args.overlay_out).expanduser(),
            radius_px=int(args.overlay_radius),
            draw_labels=not bool(args.overlay_no_labels),
        )
        return

    cfg = EditorConfig(
        map_path=Path(args.map).expanduser(),
        out_path=Path(args.out).expanduser(),
        load_path=Path(args.load).expanduser() if args.load else None,
        marker_size=float(args.marker_size),
        show_labels=not bool(args.no_labels),
        delete_radius_px=float(args.delete_radius),
    )
    editor = RoadmapPointEditor(cfg)
    editor.show()


if __name__ == "__main__":
    main()
