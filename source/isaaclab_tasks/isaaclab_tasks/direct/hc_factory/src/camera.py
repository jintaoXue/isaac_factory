"""HRTPA camera manager using isaacsim.sensors.camera.Camera (Isaac Sim native API)."""

from __future__ import annotations

import copy
from pathlib import Path

import carb
import isaacsim.core.utils.stage as stage_utils
import numpy as np
import torch
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaacsim.core.prims import XFormPrim
from isaacsim.sensors.camera import Camera as IsaacSimCamera
from PIL import Image

from ..env_asset_cfg.cfg_camera import CfgCamera, CfgCameraRegistrationInfos

_DEBUG_CAMERA_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "debug_camera"


def _eye_lookat_arrays(
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Eye/lookat in env-local frame (relative to /World/envs/env_{i})."""
    return np.asarray(eye, dtype=np.float64), np.asarray(lookat, dtype=np.float64)


def _pose_from_eye_lookat(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenGL/USD local pose from eye/lookat (+ nadir handling from viewport utils)."""
    up_axis = stage_utils.get_stage_up_axis()

    if np.allclose(eye[:2], target[:2], atol=1e-4):
        position = eye.astype(np.float32)
        if eye[2] > target[2]:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            orientation = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return position, orientation

    eyes_t = torch.tensor(eye, dtype=torch.float32).unsqueeze(0)
    targets_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    rot_mat = create_rotation_matrix_from_view(eyes_t, targets_t, up_axis=up_axis, device="cpu")
    orientation = quat_from_matrix(rot_mat)[0].numpy().astype(np.float32)
    return eye.astype(np.float32), orientation


def _set_prim_local_pose(prim_path: str, translation: np.ndarray, orientation_wxyz: np.ndarray, cuda_device: torch.device) -> None:
    """Set pose relative to env parent prim (vector-env safe; follows env clone offset)."""
    XFormPrim(prim_path).set_local_poses(
        translations=torch.tensor(translation, dtype=torch.float32, device=cuda_device).unsqueeze(0),
        orientations=torch.tensor(orientation_wxyz, dtype=torch.float32, device=cuda_device).unsqueeze(0),
    )


def _apply_spawn_intrinsics(sensor: IsaacSimCamera, spawn_cfg: dict | None) -> None:
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
        return env_state_action_dict

    def prepare_sensors(self) -> bool:
        """Initialize sensors and apply poses. Returns True if any pose changed."""
        pose_dirty = False
        for camera in self.camera_list:
            if camera.prepare_sensor():
                pose_dirty = True
        return pose_dirty

    def capture_sensors(self, dt: float, env_state_action_dict: dict) -> dict:
        for camera in self.camera_list:
            camera.capture_sensor(dt, env_state_action_dict)
        return env_state_action_dict

    def update_sensors(self, dt: float, env_state_action_dict: dict) -> dict:
        self.prepare_sensors()
        return self.capture_sensors(dt, env_state_action_dict)

    def create_sensors(self) -> None:
        for camera in self.camera_list:
            camera.create_sensor()

    def _register_camera_list(self) -> None:
        for type_name, count in self.cfg_registration_infos.items():
            if count <= 0:
                continue
            cls = globals()[type_name]
            for idx in range(count):
                self.camera_list.append(cls(idx, self.cfg_camera[type_name], self.env_id, self.cuda_device))
        if any(camera.cfg.get("debug_save_frames", False) for camera in self.camera_list):
            print(
                f"[INFO] Debug camera frames (env_{self.env_id:02d}) -> "
                f"{_DEBUG_CAMERA_OUTPUT_DIR.resolve()}/env_{self.env_id:02d}_num_XX_<CameraName>/"
            )


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
        self._sensor: IsaacSimCamera | None = None
        self._saved_frame_counter = 0
        self._initialized = False
        self._pose_dirty = False

    def _iter_key_variables(self) -> dict:
        return {"type_name": self.type_name, "idx": self.idx}

    def create_sensor(self) -> None:
        if self._sensor is not None:
            return

        meta = self.meta_registeration_info
        sensor_cfg = self.cfg["camera_sensor"]
        prim_path = meta["prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}")

        update_period = float(sensor_cfg.get("update_period", 0.0))
        frequency = None if update_period <= 0.0 else max(1, int(round(1.0 / update_period)))

        _enable_rtx_sensors_flag()
        self._sensor = IsaacSimCamera(
            prim_path=prim_path,
            name=self.state_key,
            resolution=(sensor_cfg["width"], sensor_cfg["height"]),
            frequency=frequency,
        )

    @property
    def state_key(self) -> str:
        return f"num_{self.idx:02d}_{self.type_name}"

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state = copy.deepcopy(self.reset_state)
        self._saved_frame_counter = 0
        env_state_action_dict["camera"][self.state_key] = self.state
        if self._sensor is not None and self._initialized:
            self._sensor.post_reset()
            self._apply_local_pose()
        return env_state_action_dict

    def _apply_local_pose(self) -> None:
        if self._sensor is None:
            return
        eye, target = _eye_lookat_arrays(self.cfg["eye"], self.cfg["lookat"])
        translation, orientation = _pose_from_eye_lookat(eye, target)
        _set_prim_local_pose(self._sensor.prim_path, translation, orientation, self.cuda_device)
        self._pose_dirty = True

    def prepare_sensor(self) -> bool:
        """Initialize sensor and apply pose. Returns True if pose changed this step."""
        if self._sensor is None:
            return False

        pose_dirty = False
        sensor_cfg = self.cfg["camera_sensor"]
        if not self._initialized:
            self._sensor.initialize()
            _apply_spawn_intrinsics(self._sensor, sensor_cfg.get("spawn"))
            self._initialized = True
            eye, target = _eye_lookat_arrays(self.cfg["eye"], self.cfg["lookat"])
            print(
                f"[INFO] {self.state_key} initialized @ {self._sensor.prim_path} "
                f"local_eye={eye.tolist()} local_lookat={target.tolist()}"
            )
            pose_dirty = True

        if pose_dirty or self._pose_dirty:
            self._apply_local_pose()
            return True
        return False

    def capture_sensor(self, dt: float, env_state_action_dict: dict) -> dict:
        if self._sensor is None or not self._initialized:
            self.state["is_initialized"] = False
            return env_state_action_dict

        self._pose_dirty = False
        rgba = self._sensor.get_rgba()
        if rgba is None or (hasattr(rgba, "size") and rgba.size == 0):
            self.state["is_initialized"] = False
            return env_state_action_dict

        rgb_np = np.asarray(rgba)[..., :3]
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
        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        if rgb_np.shape[-1] > 3:
            rgb_np = rgb_np[..., :3]
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0 + 1e-3:
                rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
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


class TestCamera(Camera):
    pass


class DetailCamera(Camera):
    pass
