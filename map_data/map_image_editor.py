from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


DrawMode = Literal["free", "occ"]


@dataclass
class ImageEditorConfig:
    map_path: Path
    out_path: Path
    brush_radius: int = 5  # 像素半径
    init_mode: DrawMode = "free"  # free: 画白，可通行；occ: 画黑，占据


def _try_fullscreen(fig: plt.Figure) -> None:
    """尽可能将 matplotlib 窗口全屏显示（best-effort）。"""
    try:
        manager = fig.canvas.manager  # type: ignore[attr-defined]
    except Exception:
        return

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

    try:
        manager.full_screen_toggle()  # type: ignore[attr-defined]
    except Exception:
        pass


class OccupancyImageEditor:
    """
    简单的占据栅格图编辑器：
    - 鼠标左键：第一次点击确定起点，第二次点击确定终点，并在两点之间画线段
    - 鼠标右键：同样是两次点击画线段，但颜色/模式与左键相反（相当于橡皮擦）
    - s: 保存到输出文件
    - m: 在“画白(可通行)”和“画黑(占据)”之间切换
    - + / -: 调整画笔半径
    - q / ESC: 退出（不自动保存）
    """

    def __init__(self, cfg: ImageEditorConfig):
        self.cfg = cfg

        img = plt.imread(str(cfg.map_path))
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        # 统一转为 [0,1] 灰度或 RGB
        if img.ndim == 2:
            self.img = img.astype(float)
        elif img.ndim == 3:
            # 只取亮度通道
            self.img = img.mean(axis=2).astype(float)
        else:
            raise ValueError(f"不支持的图像维度: {img.shape}")

        # 归一化到 [0,1]
        if self.img.max() > 1.0:
            self.img = self.img / 255.0
        self.img = np.clip(self.img, 0.0, 1.0)

        # 强制二值化：只保留纯黑 / 纯白，不使用灰度
        # 阈值 0.5：>0.5 视为白色（可通行），否则为黑色（占据）
        self.img = (self.img > 0.5).astype(float)

        self.mode: DrawMode = cfg.init_mode
        self.brush_radius: int = int(cfg.brush_radius)
        # 线段绘制：记录当前是否已经点下起点
        self._pending_point: tuple[float, float] | None = None
        self._pending_button: int | None = None

        self._fig, self._ax = plt.subplots()
        _try_fullscreen(self._fig)
        self._im = self._ax.imshow(self.img, cmap="gray", vmin=0.0, vmax=1.0)
        self._ax.set_axis_off()
        self._update_title()

        self._cid_press = self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_key = self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._fig.tight_layout()

    # ================= 交互逻辑 =================
    def _update_title(self) -> None:
        mode_str = "画白(可通行)" if self.mode == "free" else "画黑(占据)"
        self._fig.suptitle(
            f"占据图编辑器 - {mode_str} | 半径={self.brush_radius}px | "
            f"s:保存  m:切换模式  +/-:调节半径  q/ESC:退出",
            fontsize=11,
        )
        self._fig.canvas.draw_idle()

    def _apply_brush(self, x: float, y: float, button: int) -> None:
        if x is None or y is None:
            return
        h, w = self.img.shape[:2]
        cx = int(round(x))
        cy = int(round(y))
        r = self.brush_radius
        if r <= 0:
            r = 1

        x_min = max(cx - r, 0)
        x_max = min(cx + r, w - 1)
        y_min = max(cy - r, 0)
        y_max = min(cy + r, h - 1)

        if x_max < x_min or y_max < y_min:
            return

        yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

        # 左键：使用当前模式；右键：反向模式
        if button == 1:
            mode = self.mode
        else:
            mode = "occ" if self.mode == "free" else "free"

        if mode == "free":
            value = 1.0  # 白色，表示可通行
        else:
            value = 0.0  # 黑色，表示占据

        sub = self.img[y_min : y_max + 1, x_min : x_max + 1]
        sub[mask] = value
        self.img[y_min : y_max + 1, x_min : x_max + 1] = sub

        self._im.set_data(self.img)
        self._fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes != self._ax:
            return
        if event.button not in (1, 3):
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        # 如果还没有起点，则记录为起点
        if self._pending_point is None:
            self._pending_point = (x, y)
            self._pending_button = event.button
            # 在起点位置画一个小圆，便于可视化
            self._apply_brush(x, y, event.button)
            self._fig.canvas.draw_idle()
            return

        # 已经有起点，再次点击则画一条线段并清空 pending
        x0, y0 = self._pending_point
        button0 = self._pending_button or event.button

        # 使用简单的插值，在两点之间均匀采样若干点，再用画笔涂抹
        num_steps = max(
            int(np.hypot(x - x0, y - y0) / max(self.brush_radius, 1) * 2), 1
        )
        for t in np.linspace(0.0, 1.0, num_steps + 1):
            xx = x0 + (x - x0) * t
            yy = y0 + (y - y0) * t
            self._apply_brush(xx, yy, button0)

        self._pending_point = None
        self._pending_button = None

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(self._fig)
            return
        if key == "s":
            self._save()
            return
        if key == "m":
            self.mode = "occ" if self.mode == "free" else "free"
            self._update_title()
            return
        if key in ("+", "=", "]"):
            self.brush_radius = min(self.brush_radius + 1, 200)
            self._update_title()
            return
        if key in ("-", "_", "["):
            self.brush_radius = max(self.brush_radius - 1, 1)
            self._update_title()
            return

    # ================= 存盘 =================
    def _save(self) -> None:
        out = self.cfg.out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.clip(self.img * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        plt.imsave(str(out), arr, cmap="gray", vmin=0, vmax=255)
        print(f"[INFO] 已保存占据图到: {out}")
        self._fig.suptitle(f"已保存到: {out}", fontsize=11)
        self._fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="简单的占据栅格图编辑器（黑=占据，白=可通行）")
    p.add_argument(
        "--map",
        required=True,
        type=str,
        help="要编辑的占据图 PNG，例如 map_data/occupancy_map_hall_asset.png",
    )
    p.add_argument(
        "--out",
        default=None,
        type=str,
        help="输出路径，不填则覆盖原图",
    )
    p.add_argument(
        "--brush-radius",
        type=int,
        default=5,
        help="画笔半径（像素，默认 5）",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="free",
        choices=["free", "occ"],
        help="初始模式：free=画白(可通行)，occ=画黑(占据)。默认 free。",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    map_path = Path(args.map).expanduser()
    if not map_path.exists():
        raise FileNotFoundError(str(map_path))
    out_path = Path(args.out).expanduser() if args.out else map_path

    cfg = ImageEditorConfig(
        map_path=map_path,
        out_path=out_path,
        brush_radius=int(args.brush_radius),
        init_mode="free" if args.mode == "free" else "occ",
    )
    editor = OccupancyImageEditor(cfg)
    editor.show()


if __name__ == "__main__":
    main()

