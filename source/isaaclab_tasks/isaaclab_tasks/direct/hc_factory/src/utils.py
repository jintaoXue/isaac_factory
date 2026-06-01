import math
import torch


def yaw_to_quaternion_wxyz(yaw: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Z-axis yaw (rad) -> unit quaternion [w, x, y, z]."""
    half = 0.5 * float(yaw)
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=dtype, device=device)


class PoseAnimation:
    def __init__(self, start_pose: torch.Tensor, end_pose: torch.Tensor, animation_time: int, device: torch.device):
        self.animation_time = animation_time
        self.device = device
        self.reinitialize(start_pose.to(device), end_pose.to(device))

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
        self.end_pose = target_pose
        self.step_time = 0
        self.done = self.is_done()
    
    def reinitialize(self, start_pose: torch.Tensor, end_pose: torch.Tensor):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.step_time = 0
        self.done = self.is_done()