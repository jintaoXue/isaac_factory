import math
import torch
from ..env_asset_cfg.cfg_machine import CfgMachine

def yaw_to_quaternion_wxyz(yaw: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Z-axis yaw (rad) -> unit quaternion [w, x, y, z]."""
    half = 0.5 * float(yaw)
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=dtype, device=device)


def quat_multiply_wxyz(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions. Supports shape (..., 4)."""
    w1, x1, y1, z1 = quat_a.unbind(dim=-1)
    w2, x2, y2, z2 = quat_b.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quaternion_wxyz_to_yaw(quat: torch.Tensor) -> float:
    """Extract Z-axis yaw (rad) from a wxyz quaternion."""
    w, _, _, z = quat.unbind(dim=-1)
    return float(torch.atan2(2.0 * w * z, 1.0 - 2.0 * z * z).item())


class PoseAnimation:
    def __init__(self, start_pose: torch.Tensor, end_pose: torch.Tensor, animation_time: int, device: torch.device):
        self.animation_time = animation_time
        self.device = device
        self.initialize(start_pose.to(device), end_pose.to(device))

    def step_next_pose(self):
        self.done = self.is_done()
        if self.done:
            return self.end_pose
        self.step_time += 1.0
        self.step_time = min(self.step_time, self.animation_time)
        next_pose = self.start_pose + (self.end_pose - self.start_pose) * self.step_time / self.animation_time
        return next_pose

    def is_done(self):
        return self.step_time >= self.animation_time

    def set_target_pose(self, target_pose: torch.Tensor):
        self.start_pose = self.end_pose
        self.end_pose = target_pose.to(self.device)
        self.step_time = 0
        self.done = self.is_done()
    
    def initialize(self, start_pose: torch.Tensor, end_pose: torch.Tensor):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.step_time = 0
        self.done = self.is_done()


class GantryGroupAnimation(PoseAnimation):
    def __init__(self, start_pose: torch.Tensor, end_pose: torch.Tensor, animation_time: int, device: torch.device, num_gantrys: int):
        self.animation_time = animation_time
        self.device = device
        self.num_gantrys = num_gantrys
        gantry_cfg = CfgMachine["num07_gantry_group"]["registration_infos"]["num07_gantry_group"]
        self.gantry_indexs = gantry_cfg["gantry_indexs"].to(device)
        self.initialize(start_pose.to(device), end_pose.to(device))

    def initialize(self, start_pose: torch.Tensor, end_pose: torch.Tensor):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.step_time = [0.0] * self.num_gantrys
        self.done = self.is_done()

    def _gantry_mask(self, gantry_index: int) -> torch.Tensor:
        if gantry_index < 0 or gantry_index >= self.num_gantrys:
            raise ValueError(f"gantry_index must be in [0, {self.num_gantrys}), got {gantry_index}")
        return self.gantry_indexs == gantry_index

    def step_next_pose(self):
        self.done = self.is_done()
        if all(self.done):
            return self.end_pose
        next_pose = self.start_pose.clone()
        for gantry_index in range(self.num_gantrys):
            gantry_mask = self._gantry_mask(gantry_index)
            if not self.done[gantry_index]:
                self.step_time[gantry_index] += 1.0
                self.step_time[gantry_index] = min(self.step_time[gantry_index], self.animation_time)
                t = self.step_time[gantry_index] / self.animation_time
                next_pose[gantry_mask] = self.start_pose[gantry_mask] + (
                    self.end_pose[gantry_mask] - self.start_pose[gantry_mask]
                ) * t
            else:
                next_pose[gantry_mask] = self.end_pose[gantry_mask]
        return next_pose

    def is_done(self, gantry_index: int | None = None):
        done_list = [step_time >= self.animation_time for step_time in self.step_time]
        if gantry_index is None:
            return done_list
        return done_list[gantry_index]

    def set_target_pose(self, target_pose: torch.Tensor, gantry_index: int) -> None:
        """Start animation for one gantry. target_pose is the full 8-d joint vector."""
        target_pose = target_pose.to(self.device)
        gantry_mask = self._gantry_mask(gantry_index)
        self.start_pose[gantry_mask] = self.end_pose[gantry_mask]
        self.end_pose[gantry_mask] = target_pose[gantry_mask]
        self.step_time[gantry_index] = 0.0
        self.done[gantry_index] = False

    def sync_gantry_pose(self, pose: torch.Tensor, gantry_index: int) -> None:
        """Snap one gantry to pose immediately (no animation)."""
        pose = pose.to(self.device)
        gantry_mask = self._gantry_mask(gantry_index)
        self.start_pose[gantry_mask] = pose[gantry_mask]
        self.end_pose[gantry_mask] = pose[gantry_mask]
        self.step_time[gantry_index] = float(self.animation_time)
        self.done[gantry_index] = True