"""Pure-Python tests for the BSTAN Phase-B feature and label pipeline."""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks/direct/hc_factory/tools/build_bottleneck_features.py"
)
SPEC = importlib.util.spec_from_file_location("build_bottleneck_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestPhaseBFeatures(unittest.TestCase):
    @staticmethod
    def _write_csv(path, fieldnames, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_strided_windows_and_local_features(self):
        timelines = {
            "machine_a_ws0": MODULE.ResourceTimeline(
                "machine_a_ws0",
                "machine",
                [
                    MODULE.Interval(0.0, 15.0, "PROCESSING"),
                    MODULE.Interval(15.0, 75.0, "WAITING"),
                ],
            ),
            "machine_b_ws0": MODULE.ResourceTimeline(
                "machine_b_ws0", "machine", [MODULE.Interval(0.0, 75.0, "IDLE")]
            ),
        }
        job_rows = [
            {
                "job_id": "0",
                "task": "cut",
                "station_id": "machine_a_ws0",
                "event": "job_selected",
                "time_step": "0",
            },
            {
                "job_id": "0",
                "task": "cut",
                "station_id": "machine_a_ws0",
                "event": "queue_enter",
                "time_step": "0",
            },
            {
                "job_id": "0",
                "task": "cut",
                "station_id": "machine_a_ws0",
                "event": "departure",
                "time_step": "40",
            },
            {
                "job_id": "0",
                "task": "cut",
                "station_id": "machine_a_ws0",
                "event": "stage_complete",
                "time_step": "50",
            },
        ]
        buffer_rows = [
            {
                "buffer_id": "BlackStorage_00",
                "time_step": "0",
                "occupancy": "0",
                "occupancy_ratio": "0",
            },
            {
                "buffer_id": "BlackStorage_00",
                "time_step": "20",
                "occupancy": "1",
                "occupancy_ratio": "1",
            },
            {
                "buffer_id": "BlackStorage_00",
                "time_step": "40",
                "occupancy": "0",
                "occupancy_ratio": "0",
            },
        ]
        transport_rows = [
            {
                "status": "completed",
                "request_time_step": "0",
                "pickup_time_step": "20",
                "transport_end_time_step": "40",
                "carrier_id": "gantry_0",
                "from_node": "machine_a_ws0",
                "to_node": "BlackStorage_00",
            }
        ]
        material_rows = [
            {
                "time_step": "10",
                "material_type": "flange",
                "storage_location": "BlackStorage_00",
                "shortage_flag": "1",
            }
        ]
        episode_config = {
            "process_time_config": '{"Product":{"cut":{"machine":"machine_a","required_materials":["flange"]}}}',
            "buffer_capacity_config": '{"BlackStorage_00":{"capacity":2,"supporting_materials":["flange"]}}',
        }

        rows = MODULE.compute_window_features(
            timelines=timelines,
            job_rows=job_rows,
            buffer_rows=buffer_rows,
            transport_rows=transport_rows,
            material_rows=material_rows,
            disturbance_rows=[{"start_time_step": "10", "end_time_step": "30"}],
            episode_config=episode_config,
            window_size=30.0,
            stride=15.0,
            episode_end=75.0,
            run_id="run",
            env_id=0,
            episode_id=2,
            logic_dt=0.5,
        )

        machine_rows = [row for row in rows if row["resource_id"] == "machine_a_ws0"]
        self.assertEqual(
            [row["window_start_s"] for row in machine_rows], [0.0, 15.0, 30.0, 45.0]
        )
        first = machine_rows[0]
        self.assertEqual(first["window_end_step"], 60)
        self.assertAlmostEqual(first["active_pct_s"], 0.5)
        self.assertAlmostEqual(first["starved_ratio_s"], 0.5)
        self.assertAlmostEqual(first["output_rate_s"], 1 / 30, places=6)
        self.assertEqual(first["transport_waiting_time_s"], 10.0)
        self.assertEqual(first["route_delay_s"], 20.0)
        self.assertEqual(first["material_shortage_flag_s"], 1.0)
        self.assertEqual(first["total_WIP"], 1)
        self.assertEqual(first["disturbance_flag"], 1)

        unrelated = next(
            row
            for row in rows
            if row["resource_id"] == "machine_b_ws0" and row["window_index"] == 0
        )
        self.assertEqual(unrelated["route_delay_s"], 0.0)
        self.assertEqual(unrelated["material_shortage_flag_s"], 0.0)

        buffer = next(
            row
            for row in rows
            if row["resource_id"] == "storage_BlackStorage_00"
            and row["window_index"] == 0
        )
        self.assertAlmostEqual(buffer["occupancy_ratio_s"], 1 / 3, places=6)

    def test_label_anchor_and_tail_censoring(self):
        rows = []
        for window_index in range(5):
            hot = window_index in (2, 3)
            rows.append(
                {
                    "run_id": "run",
                    "env_id": 0,
                    "episode_id": 0,
                    "window_index": window_index,
                    "window_start_step": window_index * 30,
                    "window_end_step": (window_index + 1) * 30,
                    "window_start_s": window_index * 30.0,
                    "window_end_s": (window_index + 1) * 30.0,
                    "window_size_s": 30.0,
                    "stride_s": 30.0,
                    "resource_id": "machine_a_ws0",
                    "resource_type": "machine",
                    "bottleneck_score_s": 0.8 if hot else 0.1,
                    "blocked_time_s": 0.0,
                    "starved_time_s": 0.0,
                    "active_pct_s": 1.0 if hot else 0.0,
                    "queue_length_s": 0.0,
                    "avg_waiting_time_s": 0.0,
                }
            )

        labels, events = MODULE.build_labels_and_events(
            rows,
            horizon=30.0,
            score_threshold=0.55,
            min_event_windows=2,
            episode_end=150.0,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_s"], 60.0)
        self.assertEqual(events[0]["duration_observed"], 1)
        self.assertEqual(labels[0]["anchor_time_s"], 30.0)
        self.assertEqual(labels[0]["will_bottleneck"], 1)
        self.assertEqual(labels[0]["time_to_start"], 30.0)
        self.assertEqual(labels[1]["will_bottleneck"], 0)
        self.assertEqual(labels[-1]["label_observed"], 0)
        self.assertEqual(labels[-1]["will_bottleneck"], "")

        MODULE.validate_phase_b_outputs(
            [
                {
                    **row,
                    **{field: 0.0 for field in MODULE.MODEL_FEATURE_FIELDS},
                }
                for row in rows
            ],
            labels,
        )

    def test_process_prefers_lifecycle_end_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_dir = Path(temp_dir) / "episode_00" / "env_00"
            out_dir = Path(temp_dir) / "derived" / "episode_00" / "env_00"
            env_dir.mkdir(parents=True)
            (env_dir / "resource_event_log.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "env_id": 0,
                        "episode_id": 0,
                        "time_step": 0,
                        "logic_time_s": 0,
                        "resource_id": "machine_a_ws0",
                        "resource_type": "machine",
                        "from_state": "INIT",
                        "to_state": "IDLE",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self._write_csv(
                env_dir / "episode_config.csv",
                [
                    "run_id",
                    "env_id",
                    "episode_id",
                    "logic_dt",
                    "process_time_config",
                    "buffer_capacity_config",
                ],
                [
                    {
                        "run_id": "run",
                        "env_id": 0,
                        "episode_id": 0,
                        "logic_dt": 0.5,
                        "process_time_config": "{}",
                        "buffer_capacity_config": "{}",
                    }
                ],
            )
            self._write_csv(
                env_dir / "episode_lifecycle.csv",
                ["event", "time_step", "logic_time_s"],
                [
                    {"event": "START", "time_step": 0, "logic_time_s": 0},
                    {"event": "END", "time_step": 300, "logic_time_s": 150},
                ],
            )
            for filename in (
                "job_trace.csv",
                "buffer_event_log.csv",
                "route_transport_task.csv",
                "material_inventory_log.csv",
                "disturbance_log.csv",
            ):
                self._write_csv(env_dir / filename, ["time_step"], [])

            summary = MODULE.process_env_dir(
                env_dir=env_dir,
                out_dir=out_dir,
                window_sizes=[30.0],
                stride=30.0,
                horizon=60.0,
                score_threshold=0.55,
                min_event_windows=2,
            )

            self.assertEqual(summary["episode_end_s"], 150.0)
            self.assertEqual(summary["n_feature_rows"], 5)
            self.assertEqual(summary["observed_label_rows"], 3)
            self.assertEqual(summary["censored_label_rows"], 2)
            metadata = json.loads(
                (out_dir / "label_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["label_version"], "bstan_weak_v1")
            self.assertEqual(metadata["strides_s"], {"30.0": 30.0})


if __name__ == "__main__":
    unittest.main()
