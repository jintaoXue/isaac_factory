"""Bottleneck raw-data collector (read-only, no simulation logic changes)."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..env_asset_cfg.cfg_human import CfgHumanRegistrationInfos
from ..env_asset_cfg.cfg_machine import CfgMachine
from ..env_asset_cfg.cfg_material_product import CfgProductOrder
from ..env_asset_cfg.cfg_process_task_gallery import CfgProductProcessGallery
from ..env_asset_cfg.cfg_robot import CfgRobotRegistrationInfos

AGENT_COL_HUMAN = 0
AGENT_COL_GANTRY = 1

_WALK_SUBTASKS = frozenset(
    {"go_to_material", "go_to_goal_area", "go_to_processing_machine"}
)
_OPERATE_SUBTASKS = frozenset(
    {
        "control_gantry",
        "control_machine",
        "material_on_gantry",
        "material_on_robot",
        "material_on_goal_area",
        "carry_to_robot",
        "carry_to_goal_area",
        "go_to_processing_machine",
        "move_to_goal_area",
        "process",
    }
)


def to_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, set):
        return sorted(to_serializable(v) for v in obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return str(obj)


class BottleneckRunContext:
    """Shared metadata for one train.py run (set once from vector env)."""

    run_id: str | None = None
    sim_dt: float = 1.0 / 120.0
    decimation: int = 1
    seed: int = 0
    algo: str = "unknown"
    num_envs: int = 1
    env_yaml_path: str = ""
    agent_yaml_path: str = ""
    train_env_len_setting: list | None = None
    _initialized: bool = False

    @classmethod
    def ensure_initialized(cls, env_cfg: Any) -> None:
        if cls._initialized:
            return
        train_cfg = getattr(env_cfg, "train_cfg", None) or {}
        params = train_cfg.get("params", {})
        config = params.get("config", {})
        seed = params.get("seed", 0)
        algo = params.get("algo", {}).get("name", "unknown")
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cls.run_id = f"{time_str}_seed{seed}"
        cls.sim_dt = float(getattr(env_cfg.sim, "dt", 1.0 / 120.0))
        cls.decimation = int(getattr(env_cfg, "decimation", 1))
        cls.seed = int(seed)
        cls.algo = str(algo)
        cls.num_envs = int(env_cfg.scene.num_envs)
        cls.train_env_len_setting = getattr(env_cfg, "train_env_len_setting", None)
        train_dir = config.get("train_dir", "")
        exp_name = config.get("full_experiment_name", "")
        if train_dir and exp_name:
            base = os.path.join(train_dir, exp_name, "params")
            cls.env_yaml_path = os.path.join(base, "env.yaml")
            cls.agent_yaml_path = os.path.join(base, "agent.yaml")
        cls._initialized = True


def map_machine_state(raw_state: str) -> tuple[str, str | None]:
    if raw_state == "free":
        return "IDLE", None
    if raw_state == "invalid":
        return "STOP", "invalid_workstation"
    if raw_state.startswith("working_"):
        return "PROCESSING", f"task={raw_state.split('_', 1)[1]}"
    if raw_state.startswith("waiting_"):
        return "BLOCKED", "downstream_not_ready"
    if raw_state.startswith("materialReadyFor_"):
        return "WAITING", "material_ready"
    return "IDLE", f"unknown_state={raw_state}"


def map_human_robot_state(
    raw_state: str,
    subtask_name: str | None,
) -> tuple[str, str | None]:
    if raw_state == "free":
        return "IDLE", None
    if not raw_state.startswith("working_"):
        return "IDLE", f"unknown_state={raw_state}"
    if subtask_name == "wait":
        return "STARVED", "waiting_for_gantry_or_partner"
    if subtask_name in _WALK_SUBTASKS:
        return "PROCESSING", "subtask=walk"
    if subtask_name in _OPERATE_SUBTASKS or subtask_name is not None:
        return "PROCESSING", f"subtask={subtask_name}"
    return "PROCESSING", f"task={raw_state.split('_', 1)[1]}"


def _human_subtask_name(env: dict, human_state: dict) -> str | None:
    rid = human_state.get("ongoing_task_record_index")
    if rid is None:
        return None
    tr = env["progress"]["ongoing_task_records"].get(rid)
    if not tr:
        return None
    sd = tr.get("subtasks_dict") or {}
    ongoing = sd.get("ongoing")
    if not ongoing:
        return None
    return ongoing[AGENT_COL_HUMAN] if len(ongoing) > AGENT_COL_HUMAN else None


def _gantry_subtask_name(env: dict, job_id: int | None) -> str | None:
    if job_id is None:
        return None
    tr = env["progress"]["ongoing_task_records"].get(job_id)
    if not tr:
        return None
    sd = tr.get("subtasks_dict") or {}
    ongoing = sd.get("ongoing")
    if not ongoing or len(ongoing) <= AGENT_COL_GANTRY:
        return None
    return ongoing[AGENT_COL_GANTRY]


def _sim_time_s(time_step: int) -> float:
    return time_step * BottleneckRunContext.sim_dt


def _task_id(job_id: int | None, task: str | None) -> str | None:
    if job_id is None or task is None:
        return None
    return f"{job_id}_{task}"


def _station_id_for_task(tr: dict) -> str:
    if tr.get("task_type") == "logistic":
        gidx = tr.get("chosen_gantry_index")
        if gidx is not None:
            return f"gantry_{gidx}"
        return f"logistic_{tr.get('task', 'unknown')}"
    machine = tr.get("target_machine")
    ws = tr.get("chosen_workstation_index")
    if machine is not None and ws is not None:
        return f"{machine}_ws{ws}"
    return str(machine or "unknown")


class _CsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._initialized = False

    def write_header_only(self) -> None:
        if self._initialized:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        self._initialized = True

    def write_row(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._initialized else "w"
        with self.path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow({k: row.get(k) for k in self.fieldnames})


class BottleneckDataCollector:
    """Append-only logger; reads env_state_action_dict without mutating it."""

    def __init__(self, env_id: int, cfg: dict):
        self.env_id = env_id
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.episode_id = -1
        self._out_dir: Path | None = None
        self._steps_logged = 0

        self._prev_resource: dict[str, str] = {}
        self._prev_buffer: dict[str, tuple[int, str]] = {}
        self._prev_tasks: dict[int, dict] = {}
        self._prev_finished: dict[str, list] = {}
        self._seen_job_selected: set[int] = set()
        self._transport_open: dict[str, dict] = {}

        self._episode_config_writer: _CsvWriter | None = None
        self._job_trace_writer: _CsvWriter | None = None
        self._buffer_writer: _CsvWriter | None = None
        self._transport_writer: _CsvWriter | None = None
        self._material_writer: _CsvWriter | None = None
        self._resource_log_path: Path | None = None
        self._disturbance_writer: _CsvWriter | None = None

    def reset(self, env: dict) -> None:
        if not self.enabled:
            return
        self.episode_id += 1
        self._steps_logged = 0
        self._clear_episode_state()
        self._setup_output_dir()
        self._write_episode_config(env)
        if self._disturbance_writer:
            self._disturbance_writer.write_header_only()
        self._snapshot_all_resources(env, initial=True)
        self._snapshot_buffers(env, initial=True)
        self._snapshot_tasks(env, initial=True)
        self._snapshot_materials(env, initial=True)

    def step(self, env: dict) -> None:
        if not self.enabled or self._out_dir is None:
            return
        max_steps = self.cfg.get("max_steps_per_episode")
        if max_steps is not None and self._steps_logged >= max_steps:
            return

        time_step = int(env["time_step"])
        if self.cfg.get("log_resource_events", True):
            self._log_resource_events(env, time_step)
        if self.cfg.get("log_buffer_events", True):
            self._log_buffer_events(env, time_step)
        if self.cfg.get("log_job_trace", True):
            self._log_job_trace(env, time_step)
        if self.cfg.get("log_transport_tasks", True):
            self._log_transport_tasks(env, time_step)
        if self.cfg.get("log_material_inventory", True):
            self._log_material_inventory(env, time_step)

        self._steps_logged += 1

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_output_dir(self) -> None:
        run_id = BottleneckRunContext.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = Path(self.cfg["output_dir"]) / run_id / f"env_{self.env_id:02d}"
        base.mkdir(parents=True, exist_ok=True)
        self._out_dir = base

        if self.env_id == 0 and self.episode_id == 0:
            manifest = {
                "run_id": run_id,
                "collector_version": self.cfg.get("collector_version", "v0.1"),
                "tables": [
                    "episode_config.csv",
                    "disturbance_log.csv",
                    "resource_event_log.jsonl",
                    "job_trace.csv",
                    "buffer_event_log.csv",
                    "route_transport_task.csv",
                    "material_inventory_log.csv",
                ],
            }
            manifest_path = Path(self.cfg["output_dir"]) / run_id / "run_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self._episode_config_writer = _CsvWriter(
            base / "episode_config.csv",
            [
                "run_id", "episode_id", "env_id", "seed", "algo", "num_envs",
                "sim_dt", "decimation", "duration_limit_steps", "product_order",
                "product_mix", "process_time_config", "buffer_capacity_config",
                "human_config", "robot_config", "gantry_config", "arrival_rate",
                "env_yaml_path", "agent_yaml_path", "collector_version",
            ],
        )
        self._job_trace_writer = _CsvWriter(
            base / "job_trace.csv",
            [
                "run_id", "env_id", "episode_id", "job_id", "product_type", "task",
                "task_type", "station_id", "input_buffer_id", "output_buffer_id",
                "event", "time_step", "sim_time_s",
                "queue_enter_time_step", "queue_leave_time_step",
                "process_start_time_step", "process_end_time_step", "departure_time_step",
            ],
        )
        self._buffer_writer = _CsvWriter(
            base / "buffer_event_log.csv",
            [
                "run_id", "env_id", "episode_id", "time_step", "sim_time_s",
                "buffer_id", "buffer_type", "occupancy", "capacity", "occupancy_ratio",
                "state", "material_type", "event", "enqueue_job_id", "dequeue_job_id",
                "supporting_materials",
            ],
        )
        self._transport_writer = _CsvWriter(
            base / "route_transport_task.csv",
            [
                "run_id", "env_id", "episode_id", "task_id", "job_id", "task",
                "carrier_type", "carrier_id", "from_node", "to_node", "route_id",
                "request_time_step", "pickup_time_step", "transport_start_time_step",
                "transport_end_time_step", "dropoff_time_step", "delay_reason",
                "chosen_gantry_index", "human_key", "robot_key", "status",
            ],
        )
        self._material_writer = _CsvWriter(
            base / "material_inventory_log.csv",
            [
                "run_id", "env_id", "episode_id", "time_step", "sim_time_s",
                "material_id", "material_type", "job_id", "inventory_level",
                "reserved_quantity", "consume_quantity", "replenish_quantity",
                "storage_location", "shortage_flag", "finished_task", "event",
            ],
        )
        self._resource_log_path = base / "resource_event_log.jsonl"
        self._disturbance_writer = _CsvWriter(
            base / "disturbance_log.csv",
            [
                "run_id", "env_id", "disturbance_id", "disturbance_type",
                "target_resource_id", "target_resource_type",
                "start_time_step", "end_time_step", "start_sim_time_s", "end_sim_time_s",
                "severity", "parameter_before", "parameter_after", "notes",
            ],
        )

    def _write_episode_config(self, env: dict) -> None:
        if not self._episode_config_writer:
            return
        buffer_cap = {}
        for key, st in env.get("storage", {}).items():
            kv = st.get("key_variables", {})
            buffer_cap[key] = kv.get("capacity", 0)

        process_times = {}
        for product, gallery in CfgProductProcessGallery.items():
            process_times[product] = {
                step: {
                    "process_time": info["process_time"],
                    "gaussian_random_time": info.get("gaussian_random_time", 0),
                }
                for step, info in gallery.get("process_steps", {}).items()
            }

        gantry_cfg = CfgMachine.get("num07_gantry_group", {})
        row = {
            "run_id": BottleneckRunContext.run_id,
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "seed": BottleneckRunContext.seed,
            "algo": BottleneckRunContext.algo,
            "num_envs": BottleneckRunContext.num_envs,
            "sim_dt": BottleneckRunContext.sim_dt,
            "decimation": BottleneckRunContext.decimation,
            "duration_limit_steps": json.dumps(BottleneckRunContext.train_env_len_setting),
            "product_order": json.dumps(CfgProductOrder),
            "product_mix": json.dumps(CfgProductOrder),
            "process_time_config": json.dumps(process_times),
            "buffer_capacity_config": json.dumps(buffer_cap),
            "human_config": json.dumps(CfgHumanRegistrationInfos),
            "robot_config": json.dumps(CfgRobotRegistrationInfos),
            "gantry_config": json.dumps({"active_gantry_indices": gantry_cfg.get("active_gantry_indices", [])}),
            "arrival_rate": "",
            "env_yaml_path": BottleneckRunContext.env_yaml_path,
            "agent_yaml_path": BottleneckRunContext.agent_yaml_path,
            "collector_version": self.cfg.get("collector_version", "v0.1"),
        }
        self._episode_config_writer.write_row(row)

    def _clear_episode_state(self) -> None:
        self._prev_resource.clear()
        self._prev_buffer.clear()
        self._prev_tasks.clear()
        self._prev_finished.clear()
        self._seen_job_selected.clear()
        self._transport_open.clear()

    # ------------------------------------------------------------------
    # Resource events
    # ------------------------------------------------------------------

    def _iter_resource_snapshots(self, env: dict):
        machines = env.get("machine", {})
        gantry_type = "num07_gantry_group"
        active_gantry = CfgMachine.get(gantry_type, {}).get("active_gantry_indices", [0, 1])

        for machine_type, mstate in machines.items():
            states = mstate.get("state", [])
            ongoing_list = mstate.get("ongoing_task_record_index", [])
            if machine_type == gantry_type:
                for gantry_idx in active_gantry:
                    if gantry_idx >= len(states):
                        continue
                    raw = states[gantry_idx]
                    job_id = ongoing_list[gantry_idx] if gantry_idx < len(ongoing_list) else None
                    task = None
                    if job_id is not None:
                        tr = env["progress"]["ongoing_task_records"].get(job_id)
                        task = tr.get("task") if tr else None
                    norm, reason = map_machine_state(raw)
                    yield {
                        "resource_id": f"gantry_{gantry_idx}",
                        "resource_type": "transport_robot",
                        "raw_state": raw,
                        "norm_state": norm,
                        "reason": reason,
                        "job_id": job_id,
                        "task": task,
                        "workstation_idx": gantry_idx,
                        "machine_type": gantry_type,
                    }
                continue

            for ws_idx, raw in enumerate(states):
                job_id = ongoing_list[ws_idx] if ws_idx < len(ongoing_list) else None
                task = None
                if job_id is not None:
                    tr = env["progress"]["ongoing_task_records"].get(job_id)
                    task = tr.get("task") if tr else None
                norm, reason = map_machine_state(raw)
                yield {
                    "resource_id": f"{machine_type}_ws{ws_idx}",
                    "resource_type": "machine",
                    "raw_state": raw,
                    "norm_state": norm,
                    "reason": reason,
                    "job_id": job_id,
                    "task": task,
                    "workstation_idx": ws_idx,
                    "machine_type": machine_type,
                }

        for _hk, hs in env.get("human", {}).items():
            idx = hs.get("key_variables", {}).get("idx", 0)
            subtask = _human_subtask_name(env, hs)
            norm, reason = map_human_robot_state(hs.get("state", "free"), subtask)
            job_id = hs.get("ongoing_task_record_index")
            task = None
            if job_id is not None:
                tr = env["progress"]["ongoing_task_records"].get(job_id)
                task = tr.get("task") if tr else None
            yield {
                "resource_id": f"human_{idx}",
                "resource_type": "human",
                "raw_state": hs.get("state", "free"),
                "norm_state": norm,
                "reason": reason,
                "job_id": job_id,
                "task": task,
                "workstation_idx": None,
                "machine_type": None,
            }

        for _rk, rs in env.get("robot", {}).items():
            idx = rs.get("key_variables", {}).get("idx", 0)
            job_id = rs.get("ongoing_task_record_index")
            subtask = None
            if job_id is not None:
                tr = env["progress"]["ongoing_task_records"].get(job_id)
                if tr:
                    ongoing = (tr.get("subtasks_dict") or {}).get("ongoing") or []
                    subtask = ongoing[3] if len(ongoing) > 3 else None
            norm, reason = map_human_robot_state(rs.get("state", "free"), subtask)
            task = None
            if job_id is not None:
                tr = env["progress"]["ongoing_task_records"].get(job_id)
                task = tr.get("task") if tr else None
            yield {
                "resource_id": f"robot_{idx}",
                "resource_type": "transport_robot",
                "raw_state": rs.get("state", "free"),
                "norm_state": norm,
                "reason": reason,
                "job_id": job_id,
                "task": task,
                "workstation_idx": None,
                "machine_type": None,
            }

    def _snapshot_all_resources(self, env: dict, initial: bool = False) -> None:
        for snap in self._iter_resource_snapshots(env):
            rid = snap["resource_id"]
            self._prev_resource[rid] = snap["raw_state"]
            if initial and self.cfg.get("log_resource_events", True):
                self._append_resource_event(
                    env, snap, "INIT", snap["norm_state"], time_step=0, raw_from_state=None
                )

    def _log_resource_events(self, env: dict, time_step: int) -> None:
        for snap in self._iter_resource_snapshots(env):
            rid = snap["resource_id"]
            raw = snap["raw_state"]
            prev_raw = self._prev_resource.get(rid)
            if prev_raw == raw:
                continue
            if prev_raw is None:
                prev_norm = "INIT"
            elif snap["resource_type"] == "machine" or rid.startswith("gantry_"):
                prev_norm, _ = map_machine_state(prev_raw)
            else:
                prev_norm, _ = map_human_robot_state(prev_raw, None)
            self._append_resource_event(env, snap, prev_norm, snap["norm_state"], time_step, prev_raw)
            self._prev_resource[rid] = raw

    def _append_resource_event(
        self,
        env: dict,
        snap: dict,
        from_state: str,
        to_state: str,
        time_step: int,
        raw_from_state: str | None = None,
    ) -> None:
        if not self._resource_log_path:
            return
        record = {
            "run_id": BottleneckRunContext.run_id,
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "time_step": time_step,
            "sim_time_s": _sim_time_s(time_step),
            "resource_id": snap["resource_id"],
            "resource_type": snap["resource_type"],
            "from_state": from_state,
            "to_state": to_state,
            "raw_from_state": raw_from_state,
            "raw_to_state": snap["raw_state"],
            "job_id": snap["job_id"],
            "task_id": _task_id(snap["job_id"], snap["task"]),
            "task": snap["task"],
            "reason": snap["reason"],
            "workstation_idx": snap["workstation_idx"],
            "machine_type": snap["machine_type"],
        }
        self._resource_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._resource_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_serializable(record), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Buffer events
    # ------------------------------------------------------------------

    def _snapshot_buffers(self, env: dict, initial: bool = False) -> None:
        for key, st in env.get("storage", {}).items():
            occ = int(st.get("num_material", 0))
            state = st.get("state", "empty")
            self._prev_buffer[f"storage_{key}"] = (occ, state)

    def _log_buffer_events(self, env: dict, time_step: int) -> None:
        interval = int(self.cfg.get("buffer_snapshot_interval", 0) or 0)
        for key, st in env.get("storage", {}).items():
            buffer_id = f"storage_{key}"
            occ = int(st.get("num_material", 0))
            state = st.get("state", "empty")
            prev = self._prev_buffer.get(buffer_id)
            changed = prev is None or prev != (occ, state)
            periodic = interval > 0 and time_step % interval == 0
            if not changed and not periodic:
                continue
            kv = st.get("key_variables", {})
            capacity = int(kv.get("capacity", 1) or 1)
            event = "snapshot" if periodic and not changed else "state_change"
            if prev and occ > prev[0]:
                event = "enqueue"
            elif prev and occ < prev[0]:
                event = "dequeue"
            row = {
                "run_id": BottleneckRunContext.run_id,
                "env_id": self.env_id,
                "episode_id": self.episode_id,
                "time_step": time_step,
                "sim_time_s": _sim_time_s(time_step),
                "buffer_id": buffer_id,
                "buffer_type": kv.get("class_name", key.split("_")[0]),
                "occupancy": occ,
                "capacity": capacity,
                "occupancy_ratio": occ / capacity if capacity else 0.0,
                "state": state,
                "material_type": st.get("material_type"),
                "event": event,
                "enqueue_job_id": None,
                "dequeue_job_id": None,
                "supporting_materials": json.dumps(to_serializable(kv.get("supporting_materials", []))),
            }
            if self._buffer_writer:
                self._buffer_writer.write_row(row)
            self._prev_buffer[buffer_id] = (occ, state)

    # ------------------------------------------------------------------
    # Job trace
    # ------------------------------------------------------------------

    def _snapshot_tasks(self, env: dict, initial: bool = False) -> None:
        ongoing = env["progress"].get("ongoing_task_records", {})
        self._prev_tasks = {int(k): dict(v) for k, v in ongoing.items()}
        self._prev_finished = {
            k: list(v) for k, v in env["progress"].get("finished", {}).items()
        }

    def _log_job_trace(self, env: dict, time_step: int) -> None:
        if not self._job_trace_writer:
            return
        ongoing = env["progress"].get("ongoing_task_records", {})
        base = {
            "run_id": BottleneckRunContext.run_id,
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "time_step": time_step,
            "sim_time_s": _sim_time_s(time_step),
        }

        for job_id, tr in ongoing.items():
            job_id = int(job_id)
            prev = self._prev_tasks.get(job_id)
            sd = tr.get("subtasks_dict") or {}

            if job_id not in self._seen_job_selected:
                self._seen_job_selected.add(job_id)
                self._job_trace_writer.write_row({
                    **base,
                    "job_id": job_id,
                    "product_type": tr.get("product"),
                    "task": tr.get("task"),
                    "task_type": tr.get("task_type"),
                    "station_id": _station_id_for_task(tr),
                    "input_buffer_id": sd.get("material_start_area"),
                    "output_buffer_id": sd.get("material_goal_area"),
                    "event": "job_selected",
                    "queue_enter_time_step": None,
                    "queue_leave_time_step": None,
                    "process_start_time_step": tr.get("task_start_time_step"),
                    "process_end_time_step": None,
                    "departure_time_step": None,
                })

            if prev is None:
                continue

            if not prev.get("task_start_time_step") and tr.get("task_start_time_step"):
                self._job_trace_writer.write_row({
                    **base,
                    "job_id": job_id,
                    "product_type": tr.get("product"),
                    "task": tr.get("task"),
                    "task_type": tr.get("task_type"),
                    "station_id": _station_id_for_task(tr),
                    "input_buffer_id": sd.get("material_start_area"),
                    "output_buffer_id": sd.get("material_goal_area"),
                    "event": "process_start",
                    "queue_enter_time_step": None,
                    "queue_leave_time_step": None,
                    "process_start_time_step": tr.get("task_start_time_step"),
                    "process_end_time_step": None,
                    "departure_time_step": None,
                })

            if not prev.get("task_done") and tr.get("task_done"):
                self._job_trace_writer.write_row({
                    **base,
                    "job_id": job_id,
                    "product_type": tr.get("product"),
                    "task": tr.get("task"),
                    "task_type": tr.get("task_type"),
                    "station_id": _station_id_for_task(tr),
                    "input_buffer_id": sd.get("material_start_area"),
                    "output_buffer_id": sd.get("material_goal_area"),
                    "event": "process_end",
                    "queue_enter_time_step": None,
                    "queue_leave_time_step": None,
                    "process_start_time_step": tr.get("task_start_time_step"),
                    "process_end_time_step": time_step,
                    "departure_time_step": time_step,
                })

        finished = env["progress"].get("finished", {})
        for product_type, indices in finished.items():
            prev_indices = self._prev_finished.get(product_type, [])
            for idx in indices:
                if idx not in prev_indices:
                    self._job_trace_writer.write_row({
                        **base,
                        "job_id": idx,
                        "product_type": product_type,
                        "task": "paint_rust_proof",
                        "task_type": "processing",
                        "station_id": "finished",
                        "input_buffer_id": None,
                        "output_buffer_id": None,
                        "event": "stage_complete",
                        "queue_enter_time_step": None,
                        "queue_leave_time_step": None,
                        "process_start_time_step": None,
                        "process_end_time_step": None,
                        "departure_time_step": time_step,
                    })

        self._prev_tasks = {int(k): dict(v) for k, v in ongoing.items()}
        self._prev_finished = {k: list(v) for k, v in finished.items()}

    # ------------------------------------------------------------------
    # Transport tasks
    # ------------------------------------------------------------------

    def _log_transport_tasks(self, env: dict, time_step: int) -> None:
        if not self._transport_writer:
            return
        ongoing = env["progress"].get("ongoing_task_records", {})
        seen_ids: set[str] = set()

        for job_id, tr in ongoing.items():
            if tr.get("task_type") != "logistic":
                continue
            task = tr.get("task")
            tid = _task_id(int(job_id), task)
            if not tid:
                continue
            seen_ids.add(tid)
            sd = tr.get("subtasks_dict") or {}
            ongoing_row = sd.get("ongoing") or []
            finished = sd.get("finished") or []
            delay_reason = None
            if len(ongoing_row) > AGENT_COL_GANTRY and ongoing_row[AGENT_COL_GANTRY] == "finding_free_gantry":
                if len(finished) <= AGENT_COL_GANTRY or not finished[AGENT_COL_GANTRY]:
                    delay_reason = "finding_free_gantry"

            gidx = tr.get("chosen_gantry_index")
            if tid not in self._transport_open:
                meta = {"request_time_step": time_step, "logged_start": False}
                self._transport_open[tid] = meta
                self._write_transport_row(tr, tid, int(job_id), sd, gidx, delay_reason, meta, time_step, "requested")
            else:
                meta = self._transport_open[tid]

            if tr.get("task_start_time_step") and not meta.get("logged_start"):
                meta["pickup_time_step"] = tr["task_start_time_step"]
                meta["logged_start"] = True
                self._write_transport_row(tr, tid, int(job_id), sd, gidx, delay_reason, meta, time_step, "in_progress")

            if delay_reason and meta.get("last_delay") != delay_reason:
                meta["last_delay"] = delay_reason
                self._write_transport_row(tr, tid, int(job_id), sd, gidx, delay_reason, meta, time_step, "delayed")

        done_ids = [tid for tid in list(self._transport_open) if tid not in seen_ids]
        for tid in done_ids:
            meta = self._transport_open.pop(tid)
            job_id_str, _, task = tid.partition("_")
            self._transport_writer.write_row({
                "run_id": BottleneckRunContext.run_id,
                "env_id": self.env_id,
                "episode_id": self.episode_id,
                "task_id": tid,
                "job_id": int(job_id_str),
                "task": task,
                "carrier_type": "gantry",
                "carrier_id": None,
                "from_node": meta.get("from_node"),
                "to_node": meta.get("to_node"),
                "route_id": None,
                "request_time_step": meta.get("request_time_step"),
                "pickup_time_step": meta.get("pickup_time_step"),
                "transport_start_time_step": meta.get("pickup_time_step"),
                "transport_end_time_step": time_step,
                "dropoff_time_step": time_step,
                "delay_reason": meta.get("last_delay"),
                "chosen_gantry_index": meta.get("gantry_index"),
                "human_key": None,
                "robot_key": None,
                "status": "completed",
            })

    def _write_transport_row(
        self,
        tr: dict,
        tid: str,
        job_id: int,
        sd: dict,
        gidx: int | None,
        delay_reason: str | None,
        meta: dict,
        time_step: int,
        status: str,
    ) -> None:
        meta["from_node"] = sd.get("material_start_area")
        meta["to_node"] = sd.get("material_goal_area")
        meta["gantry_index"] = gidx
        if not self._transport_writer:
            return
        self._transport_writer.write_row({
            "run_id": BottleneckRunContext.run_id,
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "task_id": tid,
            "job_id": job_id,
            "task": tr.get("task"),
            "carrier_type": "gantry",
            "carrier_id": f"gantry_{gidx}" if gidx is not None else None,
            "from_node": sd.get("material_start_area"),
            "to_node": sd.get("material_goal_area"),
            "route_id": None,
            "request_time_step": meta.get("request_time_step"),
            "pickup_time_step": meta.get("pickup_time_step"),
            "transport_start_time_step": tr.get("task_start_time_step"),
            "transport_end_time_step": None,
            "dropoff_time_step": None,
            "delay_reason": delay_reason,
            "chosen_gantry_index": gidx,
            "human_key": tr.get("human"),
            "robot_key": tr.get("robot"),
            "status": status,
        })

    # ------------------------------------------------------------------
    # Material inventory
    # ------------------------------------------------------------------

    def _snapshot_materials(self, env: dict, initial: bool = False) -> None:
        self._prev_material: dict[str, tuple] = {}
        for mat_key, ms in env.get("material", {}).items():
            for sub_name, sub_info in (ms.get("submaterials") or {}).items():
                mid = f"material_{sub_name}"
                storage = sub_info.get("storage_name")
                self._prev_material[mid] = (storage, ms.get("finished_task"))

    def _log_material_inventory(self, env: dict, time_step: int) -> None:
        if not self._material_writer:
            return
        if not hasattr(self, "_prev_material"):
            self._prev_material = {}

        interval = int(self.cfg.get("material_snapshot_interval", 0) or 0)
        base = {
            "run_id": BottleneckRunContext.run_id,
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "time_step": time_step,
            "sim_time_s": _sim_time_s(time_step),
        }

        for mat_key, ms in env.get("material", {}).items():
            try:
                job_id = int(mat_key.split("_")[1])
            except (IndexError, ValueError):
                job_id = None
            finished_task = ms.get("finished_task", "none")

            for sub_name, sub_info in (ms.get("submaterials") or {}).items():
                mid = f"material_{sub_name}"
                storage = sub_info.get("storage_name")
                prev = self._prev_material.get(mid)
                changed = prev != (storage, finished_task)
                periodic = interval > 0 and time_step % interval == 0
                if not changed and not periodic:
                    continue

                inventory_level = 1 if storage else 0
                prev_storage = prev[0] if prev else None
                consume = 1 if prev_storage and not storage else 0
                replenish = 1 if storage and not prev_storage else 0
                event = "snapshot" if periodic and not changed else "relocate"
                if consume:
                    event = "consume"
                elif replenish:
                    event = "replenish"

                shortage_flag = self._material_shortage_flag(ms, sub_name)

                self._material_writer.write_row({
                    **base,
                    "material_id": mid,
                    "material_type": sub_name,
                    "job_id": job_id,
                    "inventory_level": inventory_level,
                    "reserved_quantity": 1 if ms.get("ongoing_task_record_index") else 0,
                    "consume_quantity": consume,
                    "replenish_quantity": replenish,
                    "storage_location": storage,
                    "shortage_flag": shortage_flag,
                    "finished_task": finished_task,
                    "event": event,
                })
                self._prev_material[mid] = (storage, finished_task)

    @staticmethod
    def _material_shortage_flag(ms: dict, sub_name: str) -> int:
        """Heuristic: semi/product parts missing while job is in progress."""
        if ms.get("ongoing_task_record_index") is None:
            return 0
        finished = ms.get("finished_task", "none")
        if finished in ("none", "logistic_for_pipe_cutting", "pipe_cutting"):
            if sub_name in ("product_00_flange", "product_00_elbow") and not ms["submaterials"][sub_name].get("storage_name"):
                return 1
        return 0


def init_bottleneck_run_context(env_cfg: Any) -> None:
    """Called once from HcVectorEnvBase."""
    BottleneckRunContext.ensure_initialized(env_cfg)
