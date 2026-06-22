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

    def initialize(self, start_pose: torch.Tensor, end_pose: torch.Tensor):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.step_time = 0.0
        self.done = False

    def step_next_pose(self):
        if self.done:
            return self.end_pose
        self.step_time += 1.0
        self.step_time = min(self.step_time, self.animation_time)
        t = self.step_time / self.animation_time
        next_pose = self.start_pose + (self.end_pose - self.start_pose) * t
        self.done = self.step_time >= self.animation_time
        return next_pose

    def is_done(self):
        return self.done

    def set_target_pose(self, target_pose: torch.Tensor) -> None:
        target_pose = target_pose.to(self.device)
        self.start_pose = self.end_pose.clone()
        self.end_pose = target_pose
        self.step_time = 0.0
        self.done = False


class GantryGroupAnimation(PoseAnimation):
    def __init__(
        self,
        start_pose: torch.Tensor,
        end_pose: torch.Tensor,
        animation_time: int,
        device: torch.device,
        num_gantrys: int,
    ):
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
        self.is_yield_move = [False] * self.num_gantrys
        self.done = self.is_done()

    def _gantry_mask(self, gantry_index: int) -> torch.Tensor:
        if gantry_index < 0 or gantry_index >= self.num_gantrys:
            raise ValueError(f"gantry_index must be in [0, {self.num_gantrys}), got {gantry_index}")
        return self.gantry_indexs == gantry_index

    def _lerp_gantry_pose(self, gantry_index: int, t: float) -> torch.Tensor:
        gantry_mask = self._gantry_mask(gantry_index)
        return self.start_pose[gantry_mask] + (self.end_pose[gantry_mask] - self.start_pose[gantry_mask]) * t

    def step_next_pose(
        self,
        joint_position: torch.Tensor,
        active_indices: list[int],
        safe_x_gap: float,
        world_x_fn,
        priority_fn,
    ):
        """Advance active gantries; enforce safe_x_gap on x axis for all active gantries."""
        next_pose = joint_position.clone()
        move_order = sorted(
            active_indices,
            key=lambda gantry_index: (0 if self.is_yield_move[gantry_index] else 1, priority_fn(gantry_index)),
        )

        committed_world_x = {
            gantry_index: world_x_fn(gantry_index, next_pose) for gantry_index in active_indices
        }

        for gantry_index in active_indices:
            gantry_mask = self._gantry_mask(gantry_index)
            if self.done[gantry_index]:
                next_pose[gantry_mask] = self.end_pose[gantry_mask]
                committed_world_x[gantry_index] = world_x_fn(gantry_index, next_pose)

        for gantry_index in move_order:
            gantry_mask = self._gantry_mask(gantry_index)
            if self.done[gantry_index]:
                continue

            t_next = min(self.step_time[gantry_index] + 1.0, self.animation_time) / self.animation_time
            proposed = self._lerp_gantry_pose(gantry_index, t_next)
            proposed_pose = next_pose.clone()
            proposed_pose[gantry_mask] = proposed
            proposed_x = world_x_fn(gantry_index, proposed_pose)

            blocked = False
            for other_index, other_x in committed_world_x.items():
                if other_index == gantry_index:
                    continue
                if abs(proposed_x - other_x) < safe_x_gap:
                    blocked = True
                    break

            if blocked:
                t_current = self.step_time[gantry_index] / self.animation_time
                next_pose[gantry_mask] = self._lerp_gantry_pose(gantry_index, t_current)
            else:
                self.step_time[gantry_index] += 1.0
                self.step_time[gantry_index] = min(self.step_time[gantry_index], self.animation_time)
                next_pose[gantry_mask] = proposed
                if self.step_time[gantry_index] >= self.animation_time:
                    self.done[gantry_index] = True

            committed_world_x[gantry_index] = world_x_fn(gantry_index, next_pose)

        return next_pose

    def is_done(self, gantry_index: int | None = None):
        done_list = [step_time >= self.animation_time for step_time in self.step_time]
        if gantry_index is None:
            return done_list
        return done_list[gantry_index]

    def _begin_gantry_move(
        self,
        target_pose: torch.Tensor,
        gantry_index: int,
        current_joint_position: torch.Tensor,
        is_yield: bool,
    ) -> None:
        target_pose = target_pose.to(self.device)
        current_joint_position = current_joint_position.to(self.device)
        gantry_mask = self._gantry_mask(gantry_index)
        self.start_pose[gantry_mask] = current_joint_position[gantry_mask]
        self.end_pose[gantry_mask] = target_pose[gantry_mask]
        self.step_time[gantry_index] = 0.0
        self.is_yield_move[gantry_index] = is_yield
        self.done[gantry_index] = False

    def set_target_pose(
        self, target_pose: torch.Tensor, gantry_index: int, current_joint_position: torch.Tensor
    ) -> None:
        """Start animation for one gantry from its current articulation pose."""
        self._begin_gantry_move(target_pose, gantry_index, current_joint_position, is_yield=False)

    def set_yield_target_pose(
        self, target_pose: torch.Tensor, gantry_index: int, current_joint_position: torch.Tensor
    ) -> None:
        """Sidestep move for a lower-priority gantry; does not change the stored task target."""
        self._begin_gantry_move(target_pose, gantry_index, current_joint_position, is_yield=True)

    def sync_gantry_pose(self, pose: torch.Tensor, gantry_index: int) -> None:
        """Snap one gantry to pose immediately (no animation)."""
        pose = pose.to(self.device)
        gantry_mask = self._gantry_mask(gantry_index)
        self.start_pose[gantry_mask] = pose[gantry_mask]
        self.end_pose[gantry_mask] = pose[gantry_mask]
        self.step_time[gantry_index] = float(self.animation_time)
        self.is_yield_move[gantry_index] = False
        self.done[gantry_index] = True


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp_angle(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rotation(
    rot_a: tuple[float, float, float], rot_b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    return (
        _lerp_angle(rot_a[0], rot_b[0], t),
        _lerp_angle(rot_a[1], rot_b[1], t),
        _lerp_angle(rot_a[2], rot_b[2], t),
    )


def _compute_walk_pose(phase: float, params: dict) -> dict[str, tuple[float, float, float]]:
    """Procedural walk: sine gait with 180° left/right phase offset."""
    theta = 2.0 * math.pi * phase
    s_left = math.sin(theta)
    s_right = -s_left

    thigh_swing = float(params.get("thigh_swing_deg", 30.0))
    knee_lift = float(params.get("knee_lift_deg", 35.0))
    knee_stance = float(params.get("knee_stance_deg", 8.0))
    foot_lift = float(params.get("foot_lift_deg", 15.0))
    foot_push = float(params.get("foot_push_deg", 8.0))
    arm_swing = float(params.get("arm_swing_deg", 22.0))
    forearm_follow = float(params.get("forearm_follow_deg", 8.0))
    arm_hang_z_l = float(params.get("arm_hang_z_l", -70.0))
    arm_hang_z_r = float(params.get("arm_hang_z_r", 70.0))
    torso_lean = float(params.get("torso_lean_deg", 5.0))

    def forward_weight(s: float) -> float:
        return max(0.0, s)

    def backward_weight(s: float) -> float:
        return max(0.0, -s)

    return {
        "waist": (torso_lean * 0.3, 0.0, 0.0),
        "spine01": (torso_lean, 0.0, 0.0),
        "spine02": (0.0, 0.0, 0.0),
        # 下肢：髋-膝-踝交替迈步，左右腿相位差 180°
        "thigh_l": (thigh_swing * s_left, 0.0, 0.0),
        "thigh_r": (thigh_swing * s_right, 0.0, 0.0),
        "calf_l": (
            -knee_lift * forward_weight(s_left) - knee_stance * backward_weight(s_left),
            0.0,
            0.0,
        ),
        "calf_r": (
            -knee_lift * forward_weight(s_right) - knee_stance * backward_weight(s_right),
            0.0,
            0.0,
        ),
        "foot_l": (
            foot_lift * forward_weight(s_left) + foot_push * backward_weight(s_left),
            0.0,
            0.0,
        ),
        "foot_r": (
            foot_lift * forward_weight(s_right) + foot_push * backward_weight(s_right),
            0.0,
            0.0,
        ),
        # 上肢：沿身体前后轴（X）摆臂，Z 锁定下垂姿态，禁止 Y 轴横摆
        "upperarm_l": (-arm_swing * s_left, 0.0, arm_hang_z_l),
        "upperarm_r": (arm_swing * s_left, 0.0, arm_hang_z_r),
        "forearm_l": (0.0, 0.0, -10.0 - forearm_follow * abs(s_left)),
        "forearm_r": (0.0, 0.0, 10.0 + forearm_follow * abs(s_left)),
    }


class HumanAnimation:
    """Keyframe-based full-body human animation with smooth interpolation."""

    def __init__(self, animation_cfg: dict):
        self._animations: dict = animation_cfg.get("animations", {})
        self._current_name: str = "idle"
        self._frame_index: float = 0.0

    @property
    def current_animation(self) -> str:
        return self._current_name

    def set_animation(self, name: str) -> None:
        name = name.lower()
        if name not in self._animations:
            name = "idle"
        if name != self._current_name:
            self._current_name = name
            self._frame_index = 0.0

    def reset(self) -> None:
        self._current_name = "idle"
        self._frame_index = 0.0

    def step(self) -> dict[str, tuple[float, float, float]]:
        anim = self._animations.get(self._current_name)
        if anim is None:
            return {}
        cycle_length: int = int(anim.get("cycle_length", 1))
        if cycle_length <= 0:
            return {}

        self._frame_index = (self._frame_index + 1.0) % float(cycle_length)
        phase = self._frame_index / float(cycle_length)

        anim_type = anim.get("type", "keyframes")
        if anim_type == "procedural_sine":
            return _compute_walk_pose(phase, anim.get("params", {}))

        keyframes: list[dict] = anim.get("keyframes", [])
        if not keyframes:
            return {}
        return self._interpolate_keyframes(keyframes, phase)

    def get_pose_at_phase(self, name: str, phase: float) -> dict[str, tuple[float, float, float]]:
        anim = self._animations.get(name.lower())
        if anim is None:
            return {}
        if anim.get("type") == "procedural_sine":
            return _compute_walk_pose(phase % 1.0, anim.get("params", {}))
        return self._interpolate_keyframes(anim.get("keyframes", []), phase % 1.0)

    def _interpolate_keyframes(
        self, keyframes: list[dict], phase: float
    ) -> dict[str, tuple[float, float, float]]:
        num_keyframes = len(keyframes)
        if num_keyframes == 1:
            return {k: tuple(v) for k, v in keyframes[0].items()}

        segment = phase * num_keyframes
        idx0 = int(segment) % num_keyframes
        idx1 = (idx0 + 1) % num_keyframes
        t = _smoothstep(segment - int(segment))

        pose0 = keyframes[idx0]
        pose1 = keyframes[idx1]
        all_joints = set(pose0.keys()) | set(pose1.keys())
        result: dict[str, tuple[float, float, float]] = {}
        for joint_key in all_joints:
            rot0 = tuple(pose0.get(joint_key, (0.0, 0.0, 0.0)))
            rot1 = tuple(pose1.get(joint_key, rot0))
            result[joint_key] = _lerp_rotation(rot0, rot1, t)
        return result
