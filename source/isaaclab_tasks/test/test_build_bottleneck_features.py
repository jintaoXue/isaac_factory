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

    @staticmethod
    def _system_row(
        window_index,
        *,
        resource_id="machine_a_ws0",
        resource_type="machine",
        point_wip=0,
        completed_operations=0,
        cycles=None,
        queue_growth=0.0,
        occupancy=0.0,
        propagation=0.0,
    ):
        return {
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
            "resource_id": resource_id,
            "resource_type": resource_type,
            "bottleneck_score_s": 0.8,
            "blocked_time_s": 0.0,
            "starved_time_s": 0.0,
            "active_pct_s": 1.0 if resource_type != "buffer" else 0.0,
            "queue_length_s": 1.0,
            "avg_waiting_time_s": 0.0,
            "queue_growth_rate_s": queue_growth,
            "occupancy_ratio_s": occupancy,
            "upstream_blocked_ratio_s": propagation,
            "downstream_starved_ratio_s": 0.0,
            "total_WIP": point_wip,
            "wip_at_window_end": point_wip,
            "operation_throughput_rolling": completed_operations / 30.0,
            "completed_operations_in_window": completed_operations,
            "completed_operation_cycle_times_s": json.dumps(cycles or []),
        }

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
                "task_type": "processing",
                "event": "job_selected",
                "time_step": "0",
            },
            {
                "job_id": "0",
                "task": "cut",
                "station_id": "machine_a_ws0",
                "task_type": "processing",
                "event": "queue_enter",
                "time_step": "0",
            },
            {
                "job_id": "0",
                "task": "cut",
                "station_id": "machine_a_ws0",
                "task_type": "processing",
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
            disturbance_rows=[
                {
                    "disturbance_id": "machine_event_1",
                    "event_phase": "START",
                    "actual_start_time_step": "10",
                    "actual_target_resource_id": "machine_a_ws0",
                    "disturbance_type": "machine_failure",
                },
                {
                    "disturbance_id": "machine_event_1",
                    "event_phase": "END",
                    "actual_end_time_step": "30",
                    "actual_target_resource_id": "machine_a_ws0",
                    "disturbance_type": "machine_failure",
                },
            ],
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
        self.assertEqual(first["total_WIP"], 0)
        self.assertEqual(first["wip_overlap_count"], 1)
        self.assertEqual(first["wip_at_window_end"], 0)
        self.assertEqual(first["completed_operations_in_window"], 1)
        self.assertEqual(json.loads(first["completed_operation_cycle_times_s"]), [20.0])
        self.assertEqual(first["runtime_disturbance_active"], 1)
        self.assertEqual(first["disturbance_flag"], 0)

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

    def test_runtime_disturbance_pair_excludes_config_row(self):
        rows = [
            {
                "disturbance_id": "human_cfg",
                "event_phase": "CONFIG",
                "start_logic_time_s": "0",
                "actual_target_resource_id": "episode",
            },
            {
                "disturbance_id": "human_event_1",
                "event_phase": "START",
                "actual_start_time_step": "10",
                "actual_target_resource_id": "human_2",
                "disturbance_type": "human_unavailable",
            },
            {
                "disturbance_id": "human_event_1",
                "event_phase": "END",
                "actual_end_time_step": "20",
                "actual_target_resource_id": "human_2",
                "disturbance_type": "human_unavailable",
            },
        ]

        events = MODULE._paired_disturbance_events(
            rows, logic_dt=1.0, episode_end=100.0
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start"], 10.0)
        self.assertEqual(events[0]["end"], 20.0)
        self.assertEqual(events[0]["target"], "human_2")

    def test_label_anchor_and_tail_censoring(self):
        rows = []
        for window_index in range(8):
            row = self._system_row(
                window_index,
                point_wip=0 if window_index < 5 else 1,
            )
            row["bottleneck_score_s"] = 0.8 if window_index in (5, 6) else 0.1
            rows.append(row)

        labels, events = MODULE.build_labels_and_events(
            rows,
            horizon=30.0,
            score_threshold=0.55,
            min_event_windows=2,
            episode_end=240.0,
            disturbance_rows=[
                {
                    "disturbance_id": "machine_event_1",
                    "event_phase": "START",
                    "actual_start_time_step": "120",
                    "disturbance_type": "machine_failure",
                    "actual_target_resource_id": "machine_a_ws0",
                },
                {
                    "disturbance_id": "machine_event_1",
                    "event_phase": "END",
                    "actual_end_time_step": "210",
                    "disturbance_type": "machine_failure",
                    "actual_target_resource_id": "machine_a_ws0",
                },
            ],
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_s"], 150.0)
        self.assertEqual(events[0]["duration_observed"], 1)
        self.assertEqual(events[0]["candidate_cause_type"], "machine_failure")
        self.assertEqual(events[0]["cause_target_resource_id"], "machine_a_ws0")
        self.assertEqual(events[0]["cause_label_confidence"], 1.0)
        self.assertEqual(labels[3]["anchor_time_s"], 120.0)
        self.assertEqual(labels[3]["will_bottleneck"], 1)
        self.assertEqual(labels[3]["time_to_start"], 30.0)
        self.assertEqual(labels[4]["will_bottleneck"], 0)
        self.assertEqual(labels[6]["system_impact_raw_flag_t"], 0)
        self.assertEqual(labels[6]["system_impact_flag_t"], 1)
        self.assertEqual(labels[6]["system_impact_age_windows_t"], 1)
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

    def test_high_score_without_system_impact_does_not_force_event(self):
        rows = []
        for window_index in range(4):
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
                    "resource_id": "storage_buffer_a",
                    "resource_type": "buffer",
                    "bottleneck_score_s": 0.9,
                    "blocked_time_s": 0.0,
                    "starved_time_s": 0.0,
                    "active_pct_s": 0.0,
                    "queue_length_s": 4.0,
                    "avg_waiting_time_s": 0.0,
                    "total_WIP": 5,
                    "operation_throughput_rolling": 0.0,
                    "completed_operations_in_window": 0,
                    "completed_operation_cycle_times_s": "[]",
                }
            )

        labels, events = MODULE.build_labels_and_events(rows, 30.0, 0.70, 2, 120.0)

        self.assertEqual(events, [])
        self.assertTrue(all(label["is_bottleneck_window"] == 0 for label in labels))

    def test_v2_2_uses_point_wip_after_warmup(self):
        point_wip = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
        rows = [
            self._system_row(index, point_wip=value)
            for index, value in enumerate(point_wip)
        ]

        labels, events = MODULE.build_labels_and_events(rows, 60.0, 0.65, 2, 300.0)

        self.assertEqual(labels[3]["warmup_gate_t"], 0)
        self.assertEqual(labels[5]["warmup_gate_t"], 1)
        self.assertEqual(labels[5]["system_impact_reason_t"], "wip_growth")
        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(events[0]["start_s"], MODULE.WARMUP_S)

    def test_v2_2_rejects_static_full_buffer(self):
        rows = [
            self._system_row(
                index,
                resource_id="storage_buffer_a",
                resource_type="buffer",
                point_wip=index,
                occupancy=1.0,
            )
            for index in range(10)
        ]

        labels, events = MODULE.build_labels_and_events(rows, 60.0, 0.65, 2, 300.0)

        self.assertEqual(events, [])
        self.assertTrue(all(label["buffer_pressure_gate_t"] == 0 for label in labels))

    def test_v2_2_uses_operation_throughput_and_cycle_growth(self):
        throughput_rows = [
            self._system_row(index, completed_operations=1 if index < 4 else 0)
            for index in range(10)
        ]
        throughput_labels, throughput_events = MODULE.build_labels_and_events(
            throughput_rows, 60.0, 0.65, 2, 300.0
        )

        self.assertEqual(len(throughput_events), 1)
        self.assertEqual(throughput_labels[6]["system_impact_flag_t"], 0)
        self.assertIn(
            "operation_throughput_drop",
            throughput_labels[7]["system_impact_reason_t"],
        )
        self.assertGreaterEqual(
            throughput_labels[7]["baseline_completed_operations_t"], 2
        )

        cycle_rows = [
            self._system_row(
                index,
                completed_operations=1,
                cycles=[10.0 if index < 4 else 13.0],
            )
            for index in range(10)
        ]
        cycle_labels, cycle_events = MODULE.build_labels_and_events(
            cycle_rows, 60.0, 0.65, 2, 300.0
        )

        self.assertEqual(len(cycle_events), 1)
        self.assertIn(
            "operation_cycle_time_growth",
            cycle_labels[7]["system_impact_reason_t"],
        )
        self.assertGreaterEqual(cycle_labels[7]["cycle_time_growth_ratio_t"], 0.2)

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
                    "collector_version",
                    "logic_dt",
                    "process_time_config",
                    "buffer_capacity_config",
                ],
                [
                    {
                        "run_id": "run",
                        "env_id": 0,
                        "episode_id": 0,
                        "collector_version": "v0.6",
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
            self.assertEqual(summary["n_resources"], 1)
            self.assertEqual(summary["n_resource_event_nodes"], 1)
            self.assertEqual(summary["n_buffer_nodes"], 0)
            self.assertEqual(summary["n_feature_rows"], 5)
            self.assertEqual(summary["observed_label_rows"], 3)
            self.assertEqual(summary["censored_label_rows"], 2)
            metadata = json.loads(
                (out_dir / "label_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["label_version"], "bstan_weak_v2_2")
            self.assertEqual(metadata["system_impact_config"]["impact_hold_windows"], 2)
            self.assertEqual(metadata["relative_score_margin"], 0.1)
            self.assertEqual(metadata["strides_s"], {"30.0": 30.0})

    def test_process_skips_aborted_episode_and_keeps_audit_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_dir = Path(temp_dir) / "episode_00" / "env_00"
            out_dir = Path(temp_dir) / "derived" / "episode_00" / "env_00"
            env_dir.mkdir(parents=True)
            (env_dir / "resource_event_log.jsonl").write_text("", encoding="utf-8")
            self._write_csv(
                env_dir / "episode_config.csv",
                ["run_id", "env_id", "episode_id", "collector_version", "logic_dt"],
                [
                    {
                        "run_id": "run",
                        "env_id": 0,
                        "episode_id": 0,
                        "collector_version": "v0.6",
                        "logic_dt": 0.5,
                    }
                ],
            )
            self._write_csv(
                env_dir / "episode_lifecycle.csv",
                [
                    "event",
                    "time_step",
                    "logic_time_s",
                    "completed_jobs",
                    "termination_reason",
                ],
                [
                    {
                        "event": "START",
                        "time_step": 0,
                        "logic_time_s": 0,
                        "completed_jobs": 0,
                        "termination_reason": "",
                    },
                    {
                        "event": "ABORTED",
                        "time_step": 5000,
                        "logic_time_s": 2500,
                        "completed_jobs": 2,
                        "termination_reason": "deadlock_watchdog",
                    },
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

            self.assertEqual(summary["status"], "skipped")
            self.assertEqual(summary["termination_reason"], "deadlock_watchdog")
            self.assertEqual(summary["abort_time_s"], 2500.0)
            self.assertEqual(summary["completed_jobs"], 2)
            self.assertTrue((out_dir / "pipeline_summary.json").is_file())
            self.assertFalse((out_dir / "window_feature_table.csv").exists())
            self.assertFalse((out_dir / "bottleneck_labels.csv").exists())


if __name__ == "__main__":
    unittest.main()
