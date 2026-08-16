"""Pure-Python tests for Phase E0-E1 bottleneck data quality logic."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


HC_ROOT = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory"
sys.path.insert(0, str(HC_ROOT / "tools"))

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_module(
    "audit_bottleneck_data",
    HC_ROOT / "tools/audit_bottleneck_data.py",
)
DISTURBANCE_CFG = _load_module(
    "cfg_disturbance_quality_test",
    HC_ROOT / "env_asset_cfg/cfg_disturbance.py",
)

for package_name in ("quality_fake", "quality_fake.env_asset_cfg", "quality_fake.src"):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
sys.modules["quality_fake.env_asset_cfg.cfg_disturbance"] = DISTURBANCE_CFG
DISTURBANCE = _load_module(
    "quality_fake.src.disturbance",
    HC_ROOT / "src/disturbance.py",
)


class _Collector:
    def __init__(self):
        self.rows = []

    def log_disturbance(self, row):
        self.rows.append(dict(row))


class TestDisturbanceSchedule(unittest.TestCase):
    def test_schedule_is_reproducible_and_varies_by_episode(self):
        DISTURBANCE_CFG.configure_disturbance_from_cli(
            dim="machine", intensity=1.0, base_seed=42
        )
        first = DISTURBANCE_CFG.sample_episode_event_schedule(0, 0)[:2]
        repeated = DISTURBANCE_CFG.sample_episode_event_schedule(0, 0)[:2]
        second_episode = DISTURBANCE_CFG.sample_episode_event_schedule(0, 1)[:2]

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second_episode)
        self.assertGreaterEqual(first[0], 650)
        self.assertLessEqual(first[0], 1200)
        self.assertGreater(first[1], 0)

    def test_material_uses_recoverable_runtime_hold(self):
        config = DISTURBANCE_CFG.configure_disturbance_from_cli(
            dim="material", intensity=1.0, base_seed=42
        )
        self.assertEqual(config["material_shortage_frac"], 0.0)
        self.assertEqual(config["machine_success_rate"], 1.0)
        self.assertGreater(config["event_duration_steps"], 0)
        self.assertIsNotNone(config["event_start_range"])

    def test_material_hold_starts_and_recovers(self):
        DISTURBANCE_CFG.configure_disturbance_from_cli(
            dim="material", intensity=1.0, base_seed=42
        )
        collector = _Collector()
        injector = DISTURBANCE.DisturbanceInjector(env_id=0, collector=collector)
        material_state = {
            "key_variables": {"idx": 0},
            "ongoing_task_record_index": None,
            "submaterials": {
                "product_00_pipe": {"storage_name": "BlackStorage_00"},
                "product_00_flange": {"storage_name": "YellowStorage_00"},
            },
        }
        env = {
            "episode_num": 0,
            "time_step": 0,
            "material": {"num_00_ProductWaterPipe": material_state},
            "progress": {
                "finished": {},
                "producing_indexs": [0],
                "production_done": False,
            },
        }

        injector.reset(env)
        env["time_step"] = injector._planned_start
        injector.step(env)
        held_material = material_state.get("disturbance_material_hold")
        self.assertEqual(held_material, "product_00_flange")
        self.assertEqual(
            [row["event_phase"] for row in collector.rows], ["CONFIG", "START"]
        )

        env["time_step"] = injector._planned_start + injector._duration
        injector.step(env)
        self.assertNotIn("disturbance_material_hold", material_state)
        self.assertEqual(
            [row["event_phase"] for row in collector.rows],
            ["CONFIG", "START", "END"],
        )
        self.assertEqual(
            collector.rows[1]["actual_target_resource_id"],
            collector.rows[2]["actual_target_resource_id"],
        )

    def test_material_hold_waits_for_an_in_process_batch(self):
        injector = DISTURBANCE.DisturbanceInjector(env_id=0, collector=_Collector())
        material_state = {
            "key_variables": {"idx": 0},
            "ongoing_task_record_index": None,
            "submaterials": {
                "product_00_flange": {"storage_name": "YellowStorage_00"},
            },
        }
        env = {
            "material": {"num_00_ProductWaterPipe": material_state},
            "progress": {"finished": {}, "producing_indexs": []},
        }

        target = injector._activate_material_hold(env)

        self.assertIsNone(target)
        self.assertNotIn("disturbance_material_hold", material_state)

    def test_human_event_returns_graph_resource_id(self):
        injector = DISTURBANCE.DisturbanceInjector(env_id=0, collector=_Collector())
        human = {
            "key_variables": {"idx": 2},
            "state": "free",
            "ongoing_task_record_index": None,
        }
        env = {"human": {"num_02_NormalHuman": human}}

        target = injector._activate_human_absent(env, "human_2")

        self.assertEqual(target, "human_2")
        self.assertEqual(human["state"], "working_disturbance_absent")
        injector._restore_if_needed(env)
        self.assertEqual(human["state"], "free")


class TestRawDataAudit(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _make_episode(
        self,
        root: Path,
        *,
        completed_jobs: int = 2,
        include_event_end: bool = True,
    ) -> tuple[Path, Path]:
        run_dir = root / "run_seed42"
        env_dir = run_dir / "episode_00" / "env_00"
        env_dir.mkdir(parents=True)
        config = {
            "run_id": "run_seed42",
            "env_id": 0,
            "episode_id": 0,
            "collector_version": "v0.3",
            "disturbance_dim": "human",
            "disturbance_intensity": 1.0,
            "logic_dt": 1.0,
            "product_order": json.dumps({"ProductWaterPipe": 2}),
            "human_config": json.dumps({"NormalHuman": 3}),
            "robot_config": json.dumps({"Robot": 2}),
            "gantry_config": json.dumps({"active_gantry_indices": [0, 1]}),
        }
        self._write_csv(env_dir / "episode_config.csv", list(config), [config])

        event_rows = [
            {
                "disturbance_id": "human_cfg",
                "disturbance_type": "human_config",
                "target_resource_id": "episode",
                "start_time_step": 0,
                "end_time_step": "",
            },
            {
                "disturbance_id": "human_event_1",
                "disturbance_type": "human_unavailable",
                "target_resource_id": "human_1",
                "start_time_step": 700,
                "end_time_step": "",
            },
        ]
        if include_event_end:
            event_rows.append(
                {
                    **event_rows[-1],
                    "end_time_step": 850,
                }
            )
        self._write_csv(
            env_dir / "disturbance_log.csv", list(event_rows[0]), event_rows
        )
        (env_dir / "resource_event_log.jsonl").write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "resource_id": "machine_a_ws0",
                        "time_step": 100,
                        "resource_type": "machine",
                        "from_state": "IDLE",
                        "to_state": "PROCESSING",
                        "raw_from_state": "free",
                        "raw_to_state": "working_cutting",
                    },
                    {
                        "resource_id": "human_1",
                        "time_step": 700,
                        "resource_type": "human",
                        "from_state": "IDLE",
                        "to_state": "PROCESSING",
                        "raw_from_state": "free",
                        "raw_to_state": "working_disturbance_absent",
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        job_rows = []
        for job_id in range(2):
            job_rows.extend(
                [
                    {
                        "job_id": job_id,
                        "task": "cutting",
                        "event": "job_selected",
                        "time_step": 100 + job_id * 400,
                    },
                    {
                        "job_id": job_id,
                        "task": "cutting",
                        "event": "departure",
                        "time_step": 300 + job_id * 400,
                    },
                ]
            )
            if job_id < completed_jobs:
                job_rows.append(
                    {
                        "job_id": job_id,
                        "task": "paint_rust_proof",
                        "event": "stage_complete",
                        "time_step": 400 + job_id * 400,
                    }
                )
        self._write_csv(env_dir / "job_trace.csv", list(job_rows[0]), job_rows)
        self._write_csv(
            env_dir / "buffer_event_log.csv",
            ["time_step", "buffer_id", "capacity", "occupancy"],
            [
                {
                    "time_step": 30,
                    "buffer_id": "storage_A",
                    "capacity": 4,
                    "occupancy": 1,
                }
            ],
        )
        self._write_csv(
            env_dir / "route_transport_task.csv",
            ["task_id", "status", "request_time_step", "transport_end_time_step"],
            [
                {
                    "task_id": "0_move",
                    "status": "completed",
                    "request_time_step": 100,
                    "transport_end_time_step": 200,
                }
            ],
        )
        self._write_csv(
            env_dir / "material_inventory_log.csv",
            ["time_step", "material_id", "shortage_flag"],
            [{"time_step": 60, "material_id": "material_0_pipe", "shortage_flag": 0}],
        )
        return run_dir, env_dir

    def test_complete_episode_with_paired_event_is_trainable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, env_dir = self._make_episode(Path(temp_dir))
            row = AUDIT.audit_env_dir(run_dir, env_dir)
            self.assertTrue(row["trainable"])
            self.assertEqual(row["runtime_event_count"], 1)
            self.assertEqual(row["completed_jobs"], 2)
            self.assertEqual(row["lifecycle_event"], "PROVEN_COMPLETE")
            self.assertEqual(AUDIT.build_report([row])["status"], "passed")

    def test_unpaired_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, env_dir = self._make_episode(
                Path(temp_dir), include_event_end=False
            )
            row = AUDIT.audit_env_dir(run_dir, env_dir)
            self.assertFalse(row["trainable"])
            self.assertIn("Unpaired runtime disturbance: human_event_1", row["errors"])

    def test_incomplete_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, env_dir = self._make_episode(Path(temp_dir), completed_jobs=1)
            row = AUDIT.audit_env_dir(run_dir, env_dir)
            self.assertFalse(row["trainable"])
            self.assertIn("completed_jobs=1/2", row["errors"])

    def test_old_collector_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, env_dir = self._make_episode(Path(temp_dir))
            config_path = env_dir / "episode_config.csv"
            with config_path.open(newline="", encoding="utf-8") as stream:
                config = next(csv.DictReader(stream))
            config["collector_version"] = "v0.6"
            self._write_csv(config_path, list(config), [config])

            row = AUDIT.audit_env_dir(run_dir, env_dir)

            self.assertFalse(row["trainable"])
            self.assertIn("collector_version='v0.6', expected='v0.3'", row["errors"])

    def test_flat_env_layout_is_not_discovered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run_seed42"
            (run_dir / "env_00").mkdir(parents=True)

            self.assertEqual(AUDIT.discover_env_dirs([run_dir]), [])


if __name__ == "__main__":
    unittest.main()
