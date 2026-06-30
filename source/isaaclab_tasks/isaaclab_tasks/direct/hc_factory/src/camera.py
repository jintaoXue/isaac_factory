import copy
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.sensors import Camera as IsaacLabCameraSensor
from isaaclab.sensors.camera import CameraCfg
from PIL import Image

from ..env_asset_cfg.cfg_camera import CfgCamera, CfgCameraRegistrationInfos

_DEBUG_CAMERA_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "debug_camera"


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

    def update_sensors(self, dt: float, env_state_action_dict: dict) -> dict:
        for camera in self.camera_list:
            camera.update_sensor(dt, env_state_action_dict)
        return env_state_action_dict

    def create_sensors(self, env_origin: torch.Tensor | None = None) -> None:
        for camera in self.camera_list:
            camera.create_sensor(env_origin=env_origin)

    def _register_camera_list(self) -> None:
        for type_name, n in self.cfg_registration_infos.items():
            cls = globals()[type_name]
            for idx in range(n):
                self.camera_list.append(cls(idx, self.cfg_camera[type_name], self.env_id, self.cuda_device))
        if any(camera.cfg.get("debug_save_frames", False) for camera in self.camera_list):
            print(
                f"[INFO] Debug camera frames (env_{self.env_id:02d}) -> "
                f"{_DEBUG_CAMERA_OUTPUT_DIR.resolve()}/env_{self.env_id:02d}_num_XX_<CameraName>/"
            )


class Camera:
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
        self._sensor: IsaacLabCameraSensor | None = None
        self._saved_frame_counter = 0
        self._env_origin: torch.Tensor | None = None
        self._pose_applied = False

    def _iter_key_variables(self) -> dict:
        return {
            "type_name": self.type_name,
            "idx": self.idx,
        }

    def create_sensor(self, env_origin: torch.Tensor | None = None) -> None:
        self._env_origin = env_origin
        if self._sensor is None:
            self._register_sensor()

    def _build_sensor_cfg(self) -> CameraCfg:
        meta = self.meta_registeration_info
        sensor_cfg = self.cfg["camera_sensor"]
        spawn_cfg = sensor_cfg["spawn"]
        prim_path = meta["prim_paths_expr"].format(i=self.env_id, idx=f"{self.idx:02d}")

        return CameraCfg(
            prim_path=prim_path,
            width=sensor_cfg["width"],
            height=sensor_cfg["height"],
            update_period=sensor_cfg["update_period"],
            data_types=sensor_cfg["data_types"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=spawn_cfg["focal_length"],
                focus_distance=spawn_cfg["focus_distance"],
                horizontal_aperture=spawn_cfg["horizontal_aperture"],
                clipping_range=spawn_cfg["clipping_range"],
            ),
        )

    def _register_sensor(self) -> None:
        self._sensor = IsaacLabCameraSensor(self._build_sensor_cfg())

    def _apply_world_pose(self) -> None:
        if self._pose_applied or self._sensor is None or not self._sensor.is_initialized:
            return

        device = self._sensor.device
        origin = (
            self._env_origin.to(device=device, dtype=torch.float32)
            if self._env_origin is not None
            else torch.zeros(3, device=device)
        )
        eye = torch.tensor(self.cfg["eye"], dtype=torch.float32, device=device) + origin
        target = torch.tensor(self.cfg["lookat"], dtype=torch.float32, device=device) + origin
        self._sensor.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
        self._pose_applied = True
        print(f"[INFO] {self.state_key} world pose eye={eye.tolist()} lookat={target.tolist()}")

    @property
    def state_key(self) -> str:
        return f"num_{self.idx:02d}_{self.type_name}"

    def reset(self, env_state_action_dict: dict) -> dict:
        self.state = copy.deepcopy(self.reset_state)
        self._saved_frame_counter = 0
        self._pose_applied = False
        env_state_action_dict["camera"][self.state_key] = self.state
        if self._sensor is not None and self._sensor.is_initialized:
            self._sensor.reset()
            self._apply_world_pose()
        return env_state_action_dict

    def update_sensor(self, dt: float, env_state_action_dict: dict) -> dict:
        if self._sensor is None or not self._sensor.is_initialized:
            self.state["is_initialized"] = False
            return env_state_action_dict

        self._apply_world_pose()
        self._sensor.update(dt, force_recompute=True)
        output = self._sensor.data.output
        rgb = output.get("rgb")
        if rgb is not None:
            self.state["rgb"] = rgb.to(self.cuda_device)
            self._save_frame(rgb)
        self.state["is_initialized"] = True
        env_state_action_dict["camera"][self.state_key] = self.state
        return env_state_action_dict

    def _save_frame(self, rgb: torch.Tensor) -> None:
        if not self.cfg.get("debug_save_frames", False):
            return
        max_frames = self.cfg.get("debug_max_frames")
        if max_frames is not None and self._saved_frame_counter >= max_frames:
            return
        rgb_np = rgb.detach().cpu().numpy()
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
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)


class DetailCamera(Camera):
    def __init__(self, idx: int, cfg: dict, env_id: int, cuda_device: torch.device):
        super().__init__(idx, cfg, env_id, cuda_device)
