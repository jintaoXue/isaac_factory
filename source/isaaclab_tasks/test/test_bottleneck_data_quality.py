"""Pure-Python tests for Phase E0-E1 bottleneck data quality logic."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
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

    def test_open_event_is_right_censored_for_complete_episode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, env_dir = self._make_episode(
                Path(temp_dir), include_event_end=False
            )
            row = AUDIT.audit_env_dir(run_dir, env_dir)
            self.assertTrue(row["trainable"])
            self.assertEqual(row["runtime_event_count"], 1)
            self.assertEqual(row["runtime_events"][0]["end"], 800.0)
            self.assertTrue(row["runtime_events"][0]["right_censored"])
            self.assertIn(
                "runtime_disturbance_right_censored=human_event_1",
                row["warnings"],
            )

    def test_deadlock_reset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, env_dir = self._make_episode(Path(temp_dir))
            disturbance_path = env_dir / "disturbance_log.csv"
            with disturbance_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            rows.append(
                {
                    "disturbance_id": "deadlock_watchdog",
                    "disturbance_type": "deadlock_reset",
                    "target_resource_id": "episode",
                    "start_time_step": 900,
                    "end_time_step": 900,
                }
            )
            self._write_csv(disturbance_path, fieldnames, rows)

            row = AUDIT.audit_env_dir(run_dir, env_dir)

            self.assertFalse(row["trainable"])
            self.assertIn("deadlock_reset_detected", row["errors"])

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

    def test_symlinked_episode_preserves_logical_run_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, target_env = self._make_episode(root / "physical")
            logical_run = root / "unsup_n10_i1" / "n10_human1.0"
            logical_run.mkdir(parents=True)
            (logical_run / "episode_00").symlink_to(
                target_env.parent, target_is_directory=True
            )

            pairs = AUDIT.discover_env_dirs([logical_run])

            self.assertEqual(len(pairs), 1)
            run_dir, env_dir = pairs[0]
            self.assertEqual(run_dir, logical_run.absolute())
            self.assertEqual(
                env_dir,
                (logical_run / "episode_00/env_00").absolute(),
            )
            self.assertEqual(env_dir.relative_to(run_dir), Path("episode_00/env_00"))
            self.assertEqual(
                AUDIT.audit_env_dir(run_dir, env_dir)["run_dir"],
                str(logical_run.absolute()),
            )


if __name__ == "__main__":
    unittest.main()
