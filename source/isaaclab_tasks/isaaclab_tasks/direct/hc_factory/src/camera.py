"""HRTPA camera manager using isaacsim.sensors.camera.Camera (Isaac Sim native API)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import carb
import numpy as np
import torch
from PIL import Image

from ..env_asset_cfg.perception.cfg_camera import CfgCamera, CfgCameraRegistrationInfos

_DEBUG_CAMERA_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "debug_camera"


def _eye_lookat_arrays(
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Eye/lookat in env-local frame (relative to /World/envs/env_{i})."""
    return np.asarray(eye, dtype=np.float64), np.asarray(lookat, dtype=np.float64)


def _apply_spawn_intrinsics(sensor: Any, spawn_cfg: dict | None) -> None:
    """Apply pinhole intrinsics from cfg camera_sensor.spawn (controls FOV)."""
    if not spawn_cfg:
        return
    if "focal_length" in spawn_cfg:
        sensor.set_focal_length(float(spawn_cfg["focal_length"]))
    if "horizontal_aperture" in spawn_cfg:
        sensor.set_horizontal_aperture(float(spawn_cfg["horizontal_aperture"]))
    if "vertical_aperture" in spawn_cfg:
        sensor.set_vertical_aperture(float(spawn_cfg["vertical_aperture"]))
    if "focus_distance" in spawn_cfg:
        sensor.set_focus_distance(float(spawn_cfg["focus_distance"]))
    if "clipping_range" in spawn_cfg:
        near, far = spawn_cfg["clipping_range"]
        sensor.set_clipping_range(float(near), float(far))


def _rgb_from_frame(frame: dict) -> np.ndarray | None:
    """Extract RGB uint8 array from Isaac Sim Camera.get_current_frame() dict.

    Isaac Sim 5.x attaches the ``rgb`` annotator, so the frame key is ``rgb``
    (often HxWx4 RGBA). Older builds may still expose ``rgba``.
    """
    rgba = frame.get("rgb")
    if rgba is None:
        rgba = frame.get("rgba")
    if rgba is None:
        return None
    if isinstance(rgba, torch.Tensor):
        rgba_np = rgba.detach().cpu().numpy()
    else:
        rgba_np = np.asarray(rgba)
    if rgba_np.size == 0:
        return None
    if rgba_np.ndim == 4:
        rgba_np = rgba_np[0]
    rgb_np = rgba_np[..., :3]
    if rgb_np.dtype != np.uint8:
        if rgb_np.max() <= 1.0 + 1e-3:
            rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
    return rgb_np


def _enable_rtx_sensors_flag() -> None:
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)


class CameraManager:
    def __init__(self, env_id: int, cuda_device: torch.device):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg_camera = CfgCamera
        self.cfg_registration_infos = CfgCameraRegistrationInfos
        self.camera_list: list[Camera] = []
        self._register_camera_list()

    def reset(self, env_state_action_dict: dict) -> dict:
        for camera in self.camera_list:
            camera.reset(env_state_action_dict)
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        for camera in self.camera_list:
            camera.step(env_state_action_dict)
        return env_state_action_dict

    def _register_camera_list(self) -> None:
        if not carb.settings.get_settings().get("/isaaclab/cameras_enabled"):
            return
        for cfg_key, count in self.cfg_registration_infos.items():
            if count <= 0:
                continue
            for idx in range(count):
                self.camera_list.append(Camera(idx, self.cfg_camera[cfg_key], self.env_id, self.cuda_device))
        if any(camera.cfg.get("debug_save_frames", False) for camera in self.camera_list):
            print(
                f"[INFO] Debug camera frames (env_{self.env_id:02d}) -> "
                f"{_DEBUG_CAMERA_OUTPUT_DIR.resolve()}/env_{self.env_id:02d}_num_XX_<CameraName>/"
            )
        for camera in self.camera_list:
            camera._register_sensor()


class Camera:
    """Thin wrapper around isaacsim.sensors.camera.Camera for HRTPA env state dict."""

    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        self.idx = idx
        self.cfg = copy.deepcopy(cfg)
        self.type_name = cfg["type_name"]
        self.meta_registeration_info = cfg["meta_registeration_info"]
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.reset_state = copy.deepcopy(cfg["reset_state"])
        self.reset_state["key_variables"] = self._iter_key_variables()
        self.state: dict = {}
        self._sensor: Any | None = None
        self._saved_frame_counter = 0

    def _iter_key_variables(self) -> dict:
        return {
            "type_name": self.type_name,
            "idx": self.idx,
            "machine_name": self.cfg.get("machine_name", self.type_name),
        }

    def _register_sensor(self) -> None:
        """Create Camera, set pose, and initialize once (Isaac Sim standalone pattern)."""
        if self._sensor is not None:
            return

        meta = self.meta_registeration_info
        sensor_cfg = self.cfg["camera_sensor"]
        prim_path = meta["prim_paths_expr"].format(i=self.env_id)

        update_period = float(sensor_cfg.get("update_period", 0.0))
        frequency = None if update_period <= 0.0 else max(1, int(round(1.0 / update_period)))

        eye, target = _eye_lookat_arrays(self.cfg["eye"], self.cfg["lookat"])
        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.camera")
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera as IsaacSimCamera

        _enable_rtx_sensors_flag()
        self._sensor = IsaacSimCamera(
            prim_path=prim_path,
            name=self.state_key,
            position=position,
            orientation=orientation,
            resolution=(sensor_cfg["width"], sensor_cfg["height"]),
            frequency=frequency,
        )
        set_camera_view(
            eye=eye.tolist(),
            target=target.tolist(),
            camera_prim_path=prim_path,
        )

        self._sensor.initialize()
        _apply_spawn_intrinsics(self._sensor, sensor_cfg.get("spawn"))
        print(
            f"[INFO] {self.state_key} initialized @ {self._sensor.prim_path} "
            f"local_eye={eye.tolist()} local_lookat={target.tolist()}"
        )

    @property
    def state_key(self) -> str:
        return f"{self.type_name}"

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state = copy.deepcopy(self.reset_state)
        self._saved_frame_counter = 0
        env_state_action_dict["camera"][self.state_key] = self.state
        if self._sensor is not None:
            self._sensor.post_reset()
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        self.capture_frame(env_state_action_dict)
        return env_state_action_dict

    def capture_frame(self, env_state_action_dict: dict) -> dict:
        """Read RGB via get_current_frame()"""
        if self._sensor is None:
            self.state["is_initialized"] = False
            return env_state_action_dict

        frame = self._sensor.get_current_frame(clone=False)
        rgb_np = _rgb_from_frame(frame)
        if rgb_np is None:
            self.state["is_initialized"] = False
            return env_state_action_dict

        self.state["rgb"] = torch.as_tensor(rgb_np, dtype=torch.uint8, device=self.cuda_device)
        self._save_frame(rgb_np)
        self.state["is_initialized"] = True
        env_state_action_dict["camera"][self.state_key] = self.state
        return env_state_action_dict

    def _save_frame(self, rgb_np: np.ndarray) -> None:
        if not self.cfg.get("debug_save_frames", False):
            return
        max_frames = self.cfg.get("debug_max_frames")
        if max_frames is not None and self._saved_frame_counter >= max_frames:
            return
        if rgb_np.size == 0:
            return

        out_dir = _DEBUG_CAMERA_OUTPUT_DIR / f"env_{self.env_id:02d}_{self.state_key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._saved_frame_counter == 0:
            print(
                f"[INFO] Saving {self.state_key} frames to: {out_dir.resolve()} "
                f"(mean_rgb={float(rgb_np.mean()):.1f}, unique_colors={len(np.unique(rgb_np.reshape(-1, 3), axis=0))})"
            )
        out_path = out_dir / f"{self._saved_frame_counter:06d}.jpg"
        Image.fromarray(rgb_np).save(out_path, format="JPEG", quality=95)
        self._saved_frame_counter += 1
