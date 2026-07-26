"""Per-episode TPA makespan / idle-rate logging for ICCBEI paper experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .paper_exp_config import append_jsonl, write_json


class TpaMetricsLogger:
    def __init__(self, env_id: int = 0):
        self.env_id = env_id
        self.enabled = bool(os.environ.get("HC_TPA_METRICS_DIR"))
        self.out_dir = Path(os.environ["HC_TPA_METRICS_DIR"]) if self.enabled else None
        self._human_free = 0
        self._machine_free = 0
        self._human_slots = 0
        self._machine_slots = 0
        self._steps = 0
        self._episode_id: int | None = None

    def reset(self, env: dict) -> None:
        if not self.enabled:
            return
        assert self.out_dir is not None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._episode_id = int(env.get("episode_num", 0))
        self._human_free = 0
        self._machine_free = 0
        self._steps = 0
        humans = env.get("human", {}) or {}
        self._human_slots = max(1, len(humans))
        self._machine_slots = max(1, self._count_machine_slots(env))

    @staticmethod
    def _count_machine_slots(env: dict) -> int:
        n = 0
        for ms in (env.get("machine") or {}).values():
            if not isinstance(ms, dict):
                continue
            st = ms.get("state")
            if isinstance(st, list):
                n += len(st)
            elif isinstance(st, str):
                n += 1
        return n

    @staticmethod
    def _count_machine_free(env: dict) -> int:
        free = 0
        for ms in (env.get("machine") or {}).values():
            if not isinstance(ms, dict):
                continue
            st = ms.get("state")
            if isinstance(st, list):
                free += sum(1 for s in st if isinstance(s, str) and s == "free")
            elif isinstance(st, str) and st == "free":
                free += 1
        return free

    def step(self, env: dict) -> None:
        if not self.enabled:
            return
        self._steps += 1
        for hs in (env.get("human") or {}).values():
            if isinstance(hs, dict) and hs.get("state") == "free":
                self._human_free += 1
        self._machine_free += self._count_machine_free(env)

    def flush_episode(self, env: dict, *, success: bool) -> dict[str, Any] | None:
        """Write one episode record (call when production_done, before reset)."""
        if not self.enabled or self.out_dir is None:
            return None
        t = max(1, self._steps)
        nh = max(1, self._human_slots)
        nm = max(1, self._machine_slots)
        progress = env.get("progress") or {}
        record = {
            "env_id": self.env_id,
            "episode_id": self._episode_id,
            "success": bool(success),
            "makespan": int(self._steps),
            "idle_h": float(self._human_free) / float(nh * t),
            "idle_m": float(self._machine_free) / float(nm * t),
            "num_humans": nh,
            "product_order": dict(progress.get("product_order") or {}),
            "finished": {k: list(v) for k, v in (progress.get("finished") or {}).items()},
        }
        append_jsonl(self.out_dir / "episodes.jsonl", record)
        write_json(self.out_dir / f"episode_{int(self._episode_id or 0):06d}.json", record)
        print(
            f"[tpa_metrics] ep={record['episode_id']} success={record['success']} "
            f"makespan={record['makespan']} idle_h={record['idle_h']:.3f} idle_m={record['idle_m']:.3f}"
        )
        return record
