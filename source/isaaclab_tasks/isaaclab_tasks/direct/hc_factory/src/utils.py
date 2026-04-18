import torch


class PoseAnimation:
    def __init__(self, start_pose: torch.Tensor, end_pose: torch.Tensor, animation_time: int):
        self.animation_time = animation_time
        self.start_pose = None
        self.end_pose = None
        self.step_time = None
        self.done = None
        self.reinitialize(start_pose, end_pose)

    def step_next_pose(self):
        self.done = self.is_done()
        if self.done:
            return self.end_pose
        self.step_time += 1.0
        self.step_time = min(self.step_time, self.animation_time)
        next_pose = self.start_pose + (self.end_pose - self.start_pose) * self.step_time / self.animation_time
        return next_pose

    def is_done(self):
        dis = torch.norm(self.start_pose - self.end_pose)
        return dis < 0.01

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