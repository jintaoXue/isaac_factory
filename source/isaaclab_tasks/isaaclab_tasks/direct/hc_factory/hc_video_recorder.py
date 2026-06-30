from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio  # type: ignore[no-redef]


class HcVideoRecorder:
    """Record mp4 from DirectRLEnv.render() for HRTPA vec-env (custom reset/step API)."""

    def __init__(
        self,
        env,
        video_folder: str | Path,
        step_trigger: Callable[[int], bool],
        video_length: int,
        fps: int | None = None,
    ):
        self.env = env
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.step_trigger = step_trigger
        self.video_length = video_length
        self._fps = fps
        self.global_step = 0
        self._recording = False
        self._frames: list[np.ndarray] = []

    def _gym_env(self):
        return self.env._hc_env()

    def _get_fps(self) -> int:
        if self._fps is not None:
            return self._fps
        gym_env = self._gym_env()     
        dt = float(gym_env.cfg.sim.dt) * int(gym_env.cfg.decimation)
        return max(1, int(round(1.0 / dt)))

    def _capture_frame(self) -> None:
        if not self._recording:
            return
        gym_env = self._gym_env()
        if gym_env.render_mode != "rgb_array":
            return
        gym_env.sim.render()
        frame = gym_env.render()
        if frame is None or frame.size == 0:
            return
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        self._frames.append(np.asarray(frame, dtype=np.uint8))
        if len(self._frames) >= self.video_length:
            self._save_video()

    def _save_video(self) -> None:
        if not self._frames:
            self._recording = False
            return
        out_path = self.video_folder / f"rl-video-step-{self.global_step:08d}.mp4"
        imageio.mimsave(out_path, self._frames, fps=self._get_fps())
        print(f"[INFO] Saved training video: {out_path}")
        self._frames = []
        self._recording = False

    def _maybe_start_recording(self) -> None:
        if not self._recording and self.step_trigger(self.global_step):
            self._recording = True
            self._frames = []

    def reset(self, num_worker=None, num_robot=None, evaluate=False):
        if self._recording:
            self._save_video()
        obs = self.env.reset(num_worker=num_worker, num_robot=num_robot, evaluate=evaluate)
        self._maybe_start_recording()
        self._capture_frame()
        return obs

    def step(self, actions, action_extra=None):
        obs = self.env.step(actions, action_extra)
        self.global_step += 1
        self._maybe_start_recording()
        self._capture_frame()
        return obs

    def get_env_info(self):
        return self.env.get_env_info()

    @property
    def unwrapped(self):
        inner = self.env
        if hasattr(inner, "_hc_env"):
            return inner._hc_env()
        return getattr(inner, "unwrapped", inner)

    def close(self):
        if self._recording:
            self._save_video()
        if hasattr(self.env, "close"):
            self.env.close()
