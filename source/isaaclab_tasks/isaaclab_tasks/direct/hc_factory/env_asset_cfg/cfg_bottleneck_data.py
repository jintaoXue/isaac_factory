"""Bottleneck raw-data collection configuration."""

from pathlib import Path

_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "output" / "bottleneck_dataset"
)

CfgBottleneckData = {
    "enabled": True,
    "output_dir": str(_DEFAULT_OUTPUT_DIR),
    "collector_version": "v0.2",
    # Buffer / material periodic snapshots (0 = event-only).
    "buffer_snapshot_interval": 30,
    "material_snapshot_interval": 60,
    "log_resource_events": True,
    "log_resource_init_events": False,
    "log_job_trace": True,
    "log_buffer_events": True,
    "log_transport_tasks": True,
    "log_material_inventory": True,
    # Stop logging after this many logic steps per episode (None = unlimited).
    "max_steps_per_episode": None,
}
