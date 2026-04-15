import torch


class PoseAnimation:
    def __init__(self, start_pose: torch.Tensor, end_pose: torch.Tensor, time: int):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.time = time
        self.step_time = 0
        self.done = False

    def step_next_pose(self):
        if self.done:
            return self.end_pose
        self.step_time += 1.0
        self.step_time = min(self.step_time, self.time)
        next_pose = self.start_pose + (self.end_pose - self.start_pose) * self.step_time / self.time
        return next_pose

    def is_done(self):
        dis = torch.norm(self.start_pose - self.end_pose)
        return dis < 0.01

    def reset(self, target_pose: torch.Tensor):
        self.start_pose = self.end_pose
        self.end_pose = target_pose
        self.step_time = 0