"""Collect station, cause, and remaining-time entity-ablation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
ARMS = ("full", "noinfo", "nocross", "machineonly")
STARTS = (5, 10, 15)


def _f(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--output",
        default="libcity/cache/model_cache/entity_ablation_summary",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in ARMS:
        for start in STARTS:
            name = (
                f"dense_i1_entity_{arm}_start{start}_min8_"
                f"cold_ep{args.epochs}_seed{args.seed}"
            )
            path = ROOT / "libcity/cache/model_cache" / name / "last_metrics.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            station = payload.get("test") or {}
            cause_block = payload.get("cause_selected") or {}
            cause = cause_block.get("test") or station
            remain_block = payload.get("remain_selected") or {}
            remain = remain_block.get("test") or station
            rows.append(
                {
                    "arm": arm,
                    "start_max_min": start,
                    "station_epoch": payload.get("best_epoch"),
                    "who_precision": _f(station, "who_precision"),
                    "who_recall": _f(station, "who_recall"),
                    "who_f1": _f(station, "who_f1"),
                    "report_f1": _f(station, "report_f1"),
                    "upcoming_report_recall": _f(
                        station, "report_recall_upcoming"
                    ),
                    "ongoing_station_recall": _f(
                        station, "who_recall_ongoing"
                    ),
                    "machine_who_f1": _f(station, "machine_who_f1"),
                    "machine_report_f1": _f(station, "machine_report_f1"),
                    "logistics_who_f1": _f(station, "logistics_who_f1"),
                    "logistics_report_f1": _f(
                        station, "logistics_report_f1"
                    ),
                    "cause_epoch": cause_block.get("best_epoch"),
                    "cause_accuracy": _f(cause, "cause_acc"),
                    "cause_macro_recall": _f(
                        cause, "cause_macro_recall"
                    ),
                    "cause_support": _f(cause, "cause_n"),
                    "remain_epoch": remain_block.get("best_epoch"),
                    "remain_mae_min": _f(remain, "remain_len_mae"),
                    "remain_primary_mae_min": _f(
                        remain, "remain_len_mae_primary"
                    ),
                }
            )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "missing": missing}, indent=2),
        encoding="utf-8",
    )
    if rows:
        with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[summary] rows={len(rows)} missing={len(missing)} -> {out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
