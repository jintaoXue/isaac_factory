#!/usr/bin/env python3
"""Sidecar monitor for long Isaac Lab / HC factory training runs.

Run in a separate tmux pane while training. Logs GPU / RAM / swap / CPU and
watches the training process + kernel log for NVIDIA Xid / lockup hints.
Each sample is fsync'd so the last line may survive a hard freeze.

Usage:
  python tools/monitor_training.py
  python tools/monitor_training.py --match "train.py.*hier" --interval 30
  python tools/monitor_training.py --pid 12345 --interval 15 --output-dir output/train_monitor
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MATCH = "train.py"
DEFAULT_INTERVAL = 30.0
DEFAULT_OUTPUT = "output/train_monitor"
DEFAULT_STALL_INTERVALS = 10  # no CPU progress for N samples → flag stall

KERNEL_PATTERNS = re.compile(
    r"Xid|NVRM.*error|soft lockup|hard LOCKUP|Out of memory|Killed process|"
    r"GPU has fallen|Resetting GPU|watchdog: BUG",
    re.IGNORECASE,
)


@dataclass
class MemSnapshot:
    mem_total_kb: int
    mem_available_kb: int
    swap_total_kb: int
    swap_free_kb: int

    @property
    def mem_used_pct(self) -> float:
        if self.mem_total_kb <= 0:
            return 0.0
        used = self.mem_total_kb - self.mem_available_kb
        return 100.0 * used / self.mem_total_kb

    @property
    def swap_used_kb(self) -> int:
        return max(0, self.swap_total_kb - self.swap_free_kb)


@dataclass
class ProcSnapshot:
    pid: int
    cmd: str
    rss_mb: float
    cpu_total_jiffies: int


@dataclass
class GpuSnapshot:
    index: int
    name: str
    temp_c: float | None
    gpu_util_pct: float | None
    mem_util_pct: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None
    power_w: float | None


@dataclass
class Sample:
    ts: str
    uptime_sec: float
    mem: MemSnapshot
    loadavg_1: float
    gpus: list[GpuSnapshot] = field(default_factory=list)
    procs: list[ProcSnapshot] = field(default_factory=list)
    kernel_hits: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_mem() -> MemSnapshot:
    data: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                data[parts[0].rstrip(":")] = int(parts[1])
    return MemSnapshot(
        mem_total_kb=data.get("MemTotal", 0),
        mem_available_kb=data.get("MemAvailable", data.get("MemFree", 0)),
        swap_total_kb=data.get("SwapTotal", 0),
        swap_free_kb=data.get("SwapFree", 0),
    )


def _read_loadavg() -> float:
    with open("/proc/loadavg", encoding="utf-8") as f:
        return float(f.read().split()[0])


def _read_uptime() -> float:
    with open("/proc/uptime", encoding="utf-8") as f:
        return float(f.read().split()[0])


def _run(cmd: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _parse_float(text: str) -> float | None:
    text = text.strip()
    if not text or text in {"[N/A]", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def query_gpus() -> list[GpuSnapshot]:
    proc = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        return []
    gpus: list[GpuSnapshot] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        gpus.append(
            GpuSnapshot(
                index=int(parts[0]),
                name=parts[1],
                temp_c=_parse_float(parts[2]),
                gpu_util_pct=_parse_float(parts[3]),
                mem_util_pct=_parse_float(parts[4]),
                mem_used_mb=_parse_float(parts[5]),
                mem_total_mb=_parse_float(parts[6]),
                power_w=_parse_float(parts[7]),
            )
        )
    return gpus


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _proc_cpu_jiffies(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # comm may contain spaces; field index 13/14 are utime/stime after ')'
    after = stat.rsplit(")", 1)[-1].split()
    if len(after) < 14:
        return None
    utime = int(after[11])
    stime = int(after[12])
    return utime + stime


def _proc_rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def find_pids(match: str, explicit_pid: int | None) -> list[int]:
    if explicit_pid is not None:
        return [explicit_pid] if Path(f"/proc/{explicit_pid}").exists() else []
    proc = _run(["pgrep", "-f", match])
    if proc.returncode != 0:
        return []
    pids = []
    for token in proc.stdout.split():
        if token.isdigit():
            pids.append(int(token))
    # Drop the monitor itself if pattern is too broad.
    pids = [p for p in pids if p != os.getpid()]
    return sorted(set(pids))


def snapshot_procs(pids: list[int]) -> list[ProcSnapshot]:
    out: list[ProcSnapshot] = []
    for pid in pids:
        cpu = _proc_cpu_jiffies(pid)
        if cpu is None:
            continue
        out.append(
            ProcSnapshot(
                pid=pid,
                cmd=_proc_cmdline(pid)[:240],
                rss_mb=round(_proc_rss_mb(pid), 1),
                cpu_total_jiffies=cpu,
            )
        )
    return out


class KernelTail:
    """Tail /var/log/kern.log (or dmesg fallback) for new GPU/OOM/lockup lines."""

    def __init__(self) -> None:
        self._path = Path("/var/log/kern.log")
        self._offset = 0
        self._inode: int | None = None
        if self._path.exists():
            st = self._path.stat()
            self._offset = st.st_size
            self._inode = st.st_ino

    def poll(self) -> list[str]:
        hits: list[str] = []
        if self._path.exists():
            st = self._path.stat()
            if self._inode != st.st_ino:
                self._offset = 0
                self._inode = st.st_ino
            with self._path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                if KERNEL_PATTERNS.search(line):
                    hits.append(line.strip())
            return hits

        proc = _run(["dmesg", "--ctime", "--level=err,warn"], timeout=10)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if KERNEL_PATTERNS.search(line):
                    hits.append(line.strip())
        return hits


def sample_to_dict(sample: Sample) -> dict[str, Any]:
    return {
        "ts": sample.ts,
        "uptime_sec": sample.uptime_sec,
        "mem_total_gb": round(sample.mem.mem_total_kb / 1024 / 1024, 2),
        "mem_used_pct": round(sample.mem.mem_used_pct, 1),
        "swap_used_gb": round(sample.mem.swap_used_kb / 1024 / 1024, 2),
        "loadavg_1": sample.loadavg_1,
        "gpus": [asdict(g) for g in sample.gpus],
        "procs": [asdict(p) for p in sample.procs],
        "kernel_hits": sample.kernel_hits,
        "flags": sample.flags,
        "notes": sample.notes,
    }


def format_line(sample: Sample) -> str:
    gpu_parts = []
    for g in sample.gpus:
        gpu_parts.append(
            f"GPU{g.index} {g.mem_used_mb}/{g.mem_total_mb}MB {g.gpu_util_pct}% "
            f"T={g.temp_c}C P={g.power_w}W"
        )
    proc_parts = [
        f"pid={p.pid} rss={p.rss_mb}MB cpu={p.cpu_total_jiffies}" for p in sample.procs
    ]
    flags = ",".join(sample.flags) if sample.flags else "ok"
    return (
        f"[{sample.ts}] flags={flags} mem={sample.mem.mem_used_pct:.1f}% "
        f"swap={sample.mem.swap_used_kb // 1024}MB load={sample.loadavg_1:.2f} | "
        + " | ".join(gpu_parts or ["no-gpu"])
        + " | "
        + " ; ".join(proc_parts or ["no-train-proc"])
        + (
            f" | kernel: {' ; '.join(sample.kernel_hits)}"
            if sample.kernel_hits
            else ""
        )
    )


class MonitorWriter:
    def __init__(self, output_dir: Path, run_tag: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = output_dir / f"monitor_{run_tag}.log"
        self.jsonl_path = output_dir / f"monitor_{run_tag}.jsonl"
        self.latest_path = output_dir / "latest.json"
        self._text = self.text_path.open("a", encoding="utf-8", buffering=1)
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8", buffering=1)

    def write(self, sample: Sample) -> None:
        line = format_line(sample)
        print(line, flush=True)
        self._text.write(line + "\n")
        self._text.flush()
        os.fsync(self._text.fileno())

        payload = sample_to_dict(sample)
        self._jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        os.fsync(self._jsonl.fileno())

        latest_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        with self.latest_path.open("w", encoding="utf-8") as latest_f:
            latest_f.write(latest_text)
            latest_f.flush()
            os.fsync(latest_f.fileno())

    def close(self) -> None:
        self._text.close()
        self._jsonl.close()


def evaluate_flags(
    sample: Sample,
    prev_cpu: dict[int, int],
    stall_intervals: int,
    stall_counts: dict[int, int],
    vram_history: list[float],
    mem_warn_pct: float,
    vram_warn_pct: float,
    swap_warn_gb: float,
) -> None:
    if sample.kernel_hits:
        sample.flags.append("KERNEL_ALERT")

    if sample.mem.mem_used_pct >= mem_warn_pct:
        sample.flags.append(f"MEM_HIGH_{sample.mem.mem_used_pct:.0f}pct")
    if sample.mem.swap_used_kb >= swap_warn_gb * 1024 * 1024:
        sample.flags.append(f"SWAP_USED_{sample.mem.swap_used_kb // (1024 * 1024)}GB")

    for gpu in sample.gpus:
        if gpu.mem_used_mb is not None and gpu.mem_total_mb:
            pct = 100.0 * gpu.mem_used_mb / gpu.mem_total_mb
            if pct >= vram_warn_pct:
                sample.flags.append(f"VRAM_HIGH_gpu{gpu.index}_{pct:.0f}pct")
            vram_history.append(gpu.mem_used_mb)
        if gpu.temp_c is not None and gpu.temp_c >= 85:
            sample.flags.append(f"GPU_HOT_{gpu.temp_c:.0f}C")

    if len(vram_history) >= 20:
        window = vram_history[-20:]
        if window[-1] - window[0] >= 512:  # +512 MB over ~20 samples
            sample.flags.append("VRAM_CREEP")

    if not sample.procs:
        sample.flags.append("TRAIN_PROC_MISSING")
        return

    for proc in sample.procs:
        prev = prev_cpu.get(proc.pid)
        if prev is None or proc.cpu_total_jiffies > prev:
            stall_counts[proc.pid] = 0
        else:
            stall_counts[proc.pid] = stall_counts.get(proc.pid, 0) + 1
            if stall_counts[proc.pid] >= stall_intervals:
                sample.flags.append(f"TRAIN_STALL_pid{proc.pid}")
        prev_cpu[proc.pid] = proc.cpu_total_jiffies


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor HC factory training for freeze/OOM/GPU issues.")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Sample period in seconds.")
    p.add_argument("--match", type=str, default=DEFAULT_MATCH, help="pgrep -f pattern for train.py.")
    p.add_argument("--pid", type=int, default=None, help="Explicit training PID (overrides --match).")
    p.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Directory for monitor logs (default: output/train_monitor).",
    )
    p.add_argument(
        "--stall-intervals",
        type=int,
        default=DEFAULT_STALL_INTERVALS,
        help="Mark TRAIN_STALL after this many samples with no CPU progress.",
    )
    p.add_argument("--mem-warn-pct", type=float, default=90.0, help="Flag when system RAM used >= this.")
    p.add_argument("--vram-warn-pct", type=float, default=95.0, help="Flag when GPU VRAM used >= this.")
    p.add_argument("--swap-warn-gb", type=float, default=1.0, help="Flag when swap used >= this many GB.")
    p.add_argument("--max-samples", type=int, default=0, help="Stop after N samples (0 = run forever).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = MonitorWriter(Path(args.output_dir), run_tag)
    kernel = KernelTail()
    prev_cpu: dict[int, int] = {}
    stall_counts: dict[int, int] = {}
    vram_history: list[float] = []

    print(
        f"[monitor_training] started tag={run_tag} interval={args.interval}s "
        f"match={args.match!r} pid={args.pid} -> {writer.text_path}",
        flush=True,
    )

    n = 0
    try:
        while True:
            ts = _now_iso()
            pids = find_pids(args.match, args.pid)
            procs = snapshot_procs(pids)
            sample = Sample(
                ts=ts,
                uptime_sec=_read_uptime(),
                mem=_read_mem(),
                loadavg_1=_read_loadavg(),
                gpus=query_gpus(),
                procs=procs,
                kernel_hits=kernel.poll(),
            )
            evaluate_flags(
                sample,
                prev_cpu,
                args.stall_intervals,
                stall_counts,
                vram_history,
                args.mem_warn_pct,
                args.vram_warn_pct,
                args.swap_warn_gb,
            )
            writer.write(sample)

            n += 1
            if args.max_samples and n >= args.max_samples:
                break
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("[monitor_training] stopped by user", flush=True)
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
