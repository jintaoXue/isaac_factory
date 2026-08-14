    # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
# from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path, set_prim_visibility
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import RigidPrim, Articulation
from isaacsim.core.api.world import World

import torch
import copy
# from abc import abstractmethod
# import numpy as np
# from .cfgs.hc_env_cfg import PoseAnimation
from abc import abstractmethod
from .src.machine import MachineManager
from .src.material import ProductMaterialManager
from .src.human import HumanManager
from .src.robot import RobotManager
from .src.camera import CameraManager
from .src.perception import PerceptionManager
from .src.storage import StorageManager
from .src.route import RouteManagerVectorEnv
from .env_asset_cfg.cfg_hc_env import SingleEnvStateActionDictTemplate, HcVectorEnvCfg
# from .env_asset_cfg.cfg_perception import CfgPerception
from .env_asset_cfg.cfg_bottleneck_data import CfgBottleneckData
from .src.bottleneck_data import BottleneckDataCollector # added: for bottleneck data collection
from .src.disturbance import DisturbanceInjector
from .env_asset_cfg.perception.cfg_perception import CfgPerception
from .src.algo_hierarchical_masker import AlgoHierarchicalMasker
from .src.task_progress_manager import TaskManager
import time

class HcSingleEnvBase():
    def __init__(
        self,
        env_id: int,
        route_manager: RouteManagerVectorEnv,
        cuda_device: torch.device,
        max_episodes: int | None = None,
    ):
        self.env_id : int = env_id
        self.env_id_str : str = f"env_{env_id}"
        self.cuda_device = cuda_device
        self.max_episodes = max_episodes
        self.reward_buf = torch.zeros(1, dtype=torch.float32, device=self.cuda_device)
        # 每个 env 持有独立的 state dict，避免多 env 共享引用导致状态串扰
        self.env_state_action_dict = copy.deepcopy(SingleEnvStateActionDictTemplate)
        self.route_manager = route_manager
        self.episode_num = 0
        # Episode-level stall watchdog (logic steps with no progress signature change).
        self.stall_timeout_steps = 5000
        self._stall_steps = 0
        self._last_progress_sig = None
        self.register_env_assets()
    
    def register_env_assets(self):
        self.storage_manager = StorageManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.product_material_manager = ProductMaterialManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.machine_manager = MachineManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.human_manager = HumanManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.robot_manager = RobotManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.camera_manager = CameraManager(env_id=self.env_id, cuda_device=self.cuda_device)
        self.perception_manager = PerceptionManager(
            env_id=self.env_id, cuda_device=self.cuda_device, cfg=CfgPerception
        )
        self.bottleneck_collector = BottleneckDataCollector(
            env_id=self.env_id, cfg=CfgBottleneckData # added: for bottleneck data collection
        )
        self.disturbance_injector = DisturbanceInjector(
            env_id=self.env_id, collector=self.bottleneck_collector
        )
        self.algo_hierarchical_masker = AlgoHierarchicalMasker(self.cuda_device)
        self.task_manager = TaskManager(
            self.cuda_device,
            max_episodic_steps=int(HcVectorEnvCfg().max_episodic_steps),
            step_penalty=float(HcVectorEnvCfg().rl_step_penalty),
            finish_bonus=float(HcVectorEnvCfg().rl_finish_bonus),
            task_bonus=float(HcVectorEnvCfg().rl_task_bonus),
            success_bonus=float(HcVectorEnvCfg().rl_success_bonus),
        )
        # self.route_manager = RouteManagerVectorEnv(cuda_device=self.cuda_device)

    def iter_managers(self):
        return (
            self.storage_manager,
            self.product_material_manager,
            self.human_manager,
            self.robot_manager,
            self.camera_manager,
            self.route_manager,
            self.machine_manager,
            self.task_manager,
            self.algo_hierarchical_masker,
        )
    
    # def update_task_availability_mask(self):
    #     self.machine_manager.update_task_availability_mask(self.env_state_action_dict)
    #     self.product_material_manager.update_task_availability_mask(self.env_state_action_dict)
    #     self.human_manager.update_task_availability_mask(self.env_state_action_dict)
    #     self.robot_manager.update_task_availability_mask(self.env_state_action_dict)
    
    # def update_self_availability_mask(self):
    #     self.human_manager.update_self_availability_mask(self.env_state_action_dict)
    #     self.robot_manager.update_self_availability_mask(self.env_state_action_dict)

    def reset_env(self):
        for m in self.iter_managers():
            m.reset(self.env_state_action_dict)
        self.env_state_action_dict["time_step"] = 0
        self.env_state_action_dict["episode_num"] = self.episode_num
        self.env_state_action_dict["run_done"] = False
        self.episode_num += 1
        self._stall_steps = 0
        self._last_progress_sig = None
        self.perception_manager.reset(self.env_state_action_dict)
        self.bottleneck_collector.reset(self.env_state_action_dict)
        self.disturbance_injector.reset(self.env_state_action_dict)
        return self.env_state_action_dict

    def apply_data_to_sim(self) -> None:
        #articulations
        articulations : dict = self.env_state_action_dict["articulations"]
        for name, data in articulations.items():
            obj : Articulation = data["object"]
            obj.set_joint_positions(data["joint_position"])
            # Currently, joint velocities are set to zero.
            joint_velocities = torch.zeros_like(data["joint_position"], device=self.cuda_device)
            obj.set_joint_velocities(joint_velocities)
        #rigid prims
        rigid_prims : dict = self.env_state_action_dict["rigid_prims"]
        for name, data in rigid_prims.items():
            rigid_prim : RigidPrim = data["object"]
            rigid_prim.set_local_poses(translations=data["position"], orientations=data["orientation"])
            rigid_prim.set_velocities(torch.zeros((1,6), device=self.cuda_device))

    def _round_xy(self, pos) -> tuple | None:
        if pos is None:
            return None
        try:
            if hasattr(pos, "detach"):
                flat = pos.detach().reshape(-1)
                if flat.numel() < 2:
                    return (round(float(flat[0].item()), 2),)
                return (round(float(flat[0].item()), 2), round(float(flat[1].item()), 2))
            seq = list(pos)
            if len(seq) >= 2:
                return (round(float(seq[0]), 2), round(float(seq[1]), 2))
        except Exception:
            return None
        return None

    def _progress_signature(self) -> tuple:
        """Compact fingerprint of production progress for stall detection.

        Must include motion. After real-path ``go_to_*=None`` a gantry/human/AGV
        can walk for well over 5000 steps without changing task/subtask flags;
        omitting travel distance made the watchdog treat long hauls as deadlock.
        """
        progress = self.env_state_action_dict.get("progress", {})
        finished = tuple(
            (k, tuple(v)) for k, v in sorted((progress.get("finished") or {}).items())
        )
        ongoing = []
        for jid, tr in sorted((progress.get("ongoing_task_records") or {}).items()):
            sd = tr.get("subtasks_dict") or {}
            finished_flags = sd.get("finished") or []
            ongoing.append(
                (
                    jid,
                    tr.get("task"),
                    sd.get("ongoing_index"),
                    tuple(bool(x) for x in finished_flags),
                    tr.get("chosen_gantry_index"),
                    tr.get("chosen_workstation_index"),
                )
            )
        machines = []
        for name, mstate in sorted((self.env_state_action_dict.get("machine") or {}).items()):
            machines.append((name, tuple(mstate.get("state") or [])))

        gantry_motion = ()
        gantry = getattr(self.machine_manager, "num07_gantry_group", None)
        anim = getattr(gantry, "animation_num07_gantry_group", None) if gantry is not None else None
        if anim is not None:
            gantry_motion = tuple(
                round(float(d), 2) for d in (anim.distance_traveled or [])
            )

        movers = []
        prims = self.env_state_action_dict.get("rigid_prims") or {}
        for group in ("human", "robot"):
            for name, state in sorted((self.env_state_action_dict.get(group) or {}).items()):
                pos = (prims.get(name) or {}).get("position")
                movers.append(
                    (
                        name,
                        state.get("current_area_id"),
                        state.get("target_area_id"),
                        self._round_xy(pos),
                    )
                )
        return (finished, tuple(ongoing), tuple(machines), gantry_motion, tuple(movers))

    def _maybe_watchdog_reset(self) -> bool:
        """Reset episode if logic state stops progressing for stall_timeout_steps."""
        if self.env_state_action_dict.get("progress", {}).get("production_done"):
            self._stall_steps = 0
            return False
        sig = self._progress_signature()
        if sig == self._last_progress_sig:
            self._stall_steps += 1
        else:
            self._last_progress_sig = sig
            self._stall_steps = 0
            return False
        if self._stall_steps < self.stall_timeout_steps:
            return False

        t = self.env_state_action_dict.get("time_step", 0)
        print(
            f"[DeadlockWatchdog] env_{self.env_id} stall={self._stall_steps} "
            f"at t={t}; forcing episode reset"
        )
        if self.bottleneck_collector is not None:
            try:
                self.bottleneck_collector.log_disturbance(
                    {
                        "disturbance_id": "deadlock_watchdog",
                        "disturbance_type": "deadlock_reset",
                        "target_resource_id": "episode",
                        "target_resource_type": "system",
                        "start_time_step": t,
                        "end_time_step": t,
                        "intensity": 1.0,
                        "parameter_before": "stalled",
                        "parameter_after": "reset",
                        "notes": f"no progress for {self._stall_steps} steps",
                    }
                )
            except Exception:
                pass
        self.reset_env()
        return True

    def step_env_logic(self, action: dict | None = None, action_extra: list[dict] | None = None) -> None:
        # time_start = time.time()
        self.env_state_action_dict['action'] = action
        # Apply action first (task assignment), then inject L2 disturbance on leftover
        # free resources, then refresh masks so the next action sees DOWN / absent.
        managers = self.iter_managers()
        for m in managers[:-1]:
            m.step(self.env_state_action_dict)
        self.disturbance_injector.step(self.env_state_action_dict)
        managers[-1].step(self.env_state_action_dict)  # algo_hierarchical_masker
        self.env_state_action_dict["time_step"] += 1
        self.perception_manager.step(self.env_state_action_dict)
        self.bottleneck_collector.step(self.env_state_action_dict)

        if self._maybe_watchdog_reset():
            return

        # Episode end: ENV resets here. Snapshot ``rl`` so callers can still read
        # this step's reward/done after reset (DQN bootstrap uses done flag).
        rl = self.env_state_action_dict.get("rl") or {}
        if bool(rl.get("done")):
            rl_snapshot = copy.deepcopy(rl)
            if self.max_episodes is not None and self.episode_num >= self.max_episodes:
                self.env_state_action_dict["run_done"] = True
            else:
                self.reset_env()
                self.env_state_action_dict["rl"] = rl_snapshot
        return

