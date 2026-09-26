#!/usr/bin/env python3
"""Sidecar monitor for long Isaac Lab / HC factory training runs.

Designed to help localize freezes (desk GNOME/Xorg + NVIDIA display path, as well
as true train/GPU hangs). Logs GPU / RAM / swap / CPU, train + display processes,
kernel / Xorg hints, and optional train-log growth. Each sample is fsync'd so the
last line may survive a hard freeze.

Usage:
  python tools/monitor_training.py
  python tools/monitor_training.py --match "HcFactory-" --interval 15
  python tools/monitor_training.py --pid 12345 --interval 10 --watch-display \\
      --train-log logs/rl_games/HcFactory/hier_G0/metrics.jsonl
  python tools/monitor_training.py --freeze-hunt   # shorter interval + display watch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MATCH = "HcFactory-"
DEFAULT_INTERVAL = 30.0
DEFAULT_OUTPUT = "outputs/train_monitor"
DEFAULT_STALL_INTERVALS = 10  # no CPU progress for N samples → flag stall
DEFAULT_LOG_STALL_INTERVALS = 6
DEFAULT_DISPLAY_STALL_INTERVALS = 6

# Train process name fragments (setproctitle) + common launchers.
DEFAULT_MATCH_HINTS = ("HcFactory-", "train.py", "run_2026_journal")

# Desktop / display path — historically the freeze subject on the 4090 desk.
DISPLAY_MATCHES = (
    "gnome-shell",
    "Xorg",
    "Xwayland",
    "gdm-x-session",
    "gdm3",
    "mutter",
    "cinnamon",
    "kwin_x11",
    "kwin_wayland",
    "plasmashell",
)

KERNEL_PATTERNS = re.compile(
    r"Xid|NVRM|soft lockup|hard LOCKUP|Out of memory|Killed process|"
    r"GPU has fallen|Resetting GPU|watchdog: BUG|hung_task|blocked for more than|"
    r"modeset|Failed to grab modeset|gnome-shell|Xorg.*segfault|"
    r"NVRM: Xid|drm:.*failure|amdgpu.*ring",
    re.IGNORECASE,
)

XORG_PATTERNS = re.compile(
    r"EE\)|Fatal|segfault|GPU hang|lockup|Failed to grab modeset|NVIDIA",
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
    state: str = "?"
    wchan: str = ""
    threads: int = 0
    role: str = "train"  # train | display | other


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
    persistence: str | None = None
    pstate: str | None = None


@dataclass
class SystemCpu:
    user: int
    nice: int
    system: int
    idle: int
    iowait: int
    irq: int
    softirq: int

    @property
    def total(self) -> int:
        return self.user + self.nice + self.system + self.idle + self.iowait + self.irq + self.softirq


@dataclass
class Sample:
    ts: str
    uptime_sec: float
    mem: MemSnapshot
    loadavg_1: float
    loadavg_5: float = 0.0
    loadavg_15: float = 0.0
    gpus: list[GpuSnapshot] = field(default_factory=list)
    procs: list[ProcSnapshot] = field(default_factory=list)
    display_procs: list[ProcSnapshot] = field(default_factory=list)
    kernel_hits: list[str] = field(default_factory=list)
    xorg_hits: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hypothesis: str = ""
    d_state_count: int = 0
    d_state_top: list[str] = field(default_factory=list)
    iowait_pct: float | None = None
    train_log_bytes: int | None = None
    train_log_mtime: float | None = None
    session_type: str = ""
    graphical_active: bool | None = None


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


def _read_loadavg() -> tuple[float, float, float]:
    with open("/proc/loadavg", encoding="utf-8") as f:
        a, b, c = f.read().split()[:3]
        return float(a), float(b), float(c)


def _read_uptime() -> float:
    with open("/proc/uptime", encoding="utf-8") as f:
        return float(f.read().split()[0])


def _read_system_cpu() -> SystemCpu | None:
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            parts = f.readline().split()
    except OSError:
        return None
    if parts[0] != "cpu" or len(parts) < 8:
        return None
    vals = [int(x) for x in parts[1:8]]
    return SystemCpu(*vals)


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
            "memory.used,memory.total,power.draw,persistence_mode,pstate",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        # Older drivers may lack persistence_mode/pstate — fall back.
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
                persistence=parts[8] if len(parts) > 8 else None,
                pstate=parts[9] if len(parts) > 9 else None,
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
    after = stat.rsplit(")", 1)[-1].split()
    if len(after) < 14:
        return None
    utime = int(after[11])
    stime = int(after[12])
    return utime + stime


def _proc_state(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return "?"
    after = stat.rsplit(")", 1)[-1].split()
    return after[0] if after else "?"


def _proc_wchan(pid: int) -> str:
    try:
        text = Path(f"/proc/{pid}/wchan").read_text(encoding="utf-8").strip()
        return text[:64] if text and text != "0" else ""
    except OSError:
        return ""


def _proc_threads(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _proc_rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _is_monitor_proc(cmd: str) -> bool:
    return "monitor_training" in cmd


def find_pids(match: str, explicit_pid: int | None) -> list[int]:
    if explicit_pid is not None:
        return [explicit_pid] if Path(f"/proc/{explicit_pid}").exists() else []
    proc = _run(["pgrep", "-f", match])
    if proc.returncode != 0:
        return []
    pids = []
    for token in proc.stdout.split():
        if not token.isdigit():
            continue
        pid = int(token)
        if pid == os.getpid():
            continue
        cmd = _proc_cmdline(pid)
        if _is_monitor_proc(cmd):
            continue
        pids.append(pid)
    return sorted(set(pids))


def find_display_pids() -> list[int]:
    found: list[int] = []
    for name in DISPLAY_MATCHES:
        proc = _run(["pgrep", "-x", name])
        if proc.returncode != 0:
            # some are matched better with -f (gdm-x-session)
            proc = _run(["pgrep", "-f", name])
        if proc.returncode != 0:
            continue
        for token in proc.stdout.split():
            if token.isdigit():
                found.append(int(token))
    return sorted(set(found))


def snapshot_procs(pids: list[int], role: str) -> list[ProcSnapshot]:
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
                state=_proc_state(pid),
                wchan=_proc_wchan(pid),
                threads=_proc_threads(pid),
                role=role,
            )
        )
    return out


def count_d_state(limit: int = 8) -> tuple[int, list[str]]:
    """Count tasks in uninterruptible sleep (D) — common during I/O / driver hangs."""
    count = 0
    top: list[str] = []
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            state = _proc_state(pid)
            if state != "D":
                continue
            count += 1
            if len(top) < limit:
                cmd = _proc_cmdline(pid)[:80] or f"pid={pid}"
                wchan = _proc_wchan(pid)
                top.append(f"{pid}:{state}:{wchan}:{cmd}")
    except OSError:
        pass
    return count, top


def session_info() -> tuple[str, bool | None]:
    session = os.environ.get("XDG_SESSION_TYPE", "")
    graphical: bool | None = None
    proc = _run(["systemctl", "is-active", "graphical.target"], timeout=5)
    if proc.returncode == 0:
        graphical = proc.stdout.strip() == "active"
    elif proc.stdout.strip() in {"inactive", "failed"}:
        graphical = False
    return session, graphical


def train_log_stat(path: Path | None) -> tuple[int | None, float | None]:
    if path is None or not path.exists():
        return None, None
    try:
        st = path.stat()
        return st.st_size, st.st_mtime
    except OSError:
        return None, None


def discover_train_log(repo: Path) -> Path | None:
    """Best-effort: newest metrics.jsonl under logs/rl_games/HcFactory."""
    root = repo / "logs" / "rl_games" / "HcFactory"
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    try:
        for d in root.iterdir():
            if not d.is_dir():
                continue
            m = d / "metrics.jsonl"
            if m.is_file():
                candidates.append(m)
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


class KernelTail:
    """Tail /var/log/kern.log (or dmesg / journalctl fallback) for GPU/OOM/lockup lines."""

    def __init__(self) -> None:
        self._path = Path("/var/log/kern.log")
        self._offset = 0
        self._inode: int | None = None
        self._seen_dmesg: set[str] = set()
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
                    hits.append(line.strip()[:300])
            return hits

        # Prefer journalctl -k for recent boot (no root needed on many setups).
        proc = _run(
            ["journalctl", "-k", "-n", "40", "--no-pager", "-o", "short-iso"],
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in proc.stdout.splitlines():
                if KERNEL_PATTERNS.search(line) and line not in self._seen_dmesg:
                    self._seen_dmesg.add(line)
                    hits.append(line.strip()[:300])
            # keep set bounded
            if len(self._seen_dmesg) > 500:
                self._seen_dmesg = set(list(self._seen_dmesg)[-200:])
            return hits

        proc = _run(["dmesg", "--ctime", "--level=err,warn"], timeout=10)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if KERNEL_PATTERNS.search(line) and line not in self._seen_dmesg:
                    self._seen_dmesg.add(line)
                    hits.append(line.strip()[:300])
        return hits


class XorgTail:
    """Tail newest /var/log/Xorg.*.log for EE / modeset / NVIDIA lines."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._offset = 0
        self._pick()

    def _pick(self) -> None:
        logs = sorted(Path("/var/log").glob("Xorg.*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            self._path = None
            return
        self._path = logs[0]
        try:
            self._offset = self._path.stat().st_size
        except OSError:
            self._path = None

    def poll(self) -> list[str]:
        if self._path is None or not self._path.exists():
            self._pick()
            return []
        hits: list[str] = []
        try:
            with self._path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except OSError:
            return []
        for line in chunk.decode("utf-8", errors="replace").splitlines():
            if XORG_PATTERNS.search(line):
                hits.append(line.strip()[:300])
        return hits


def sample_to_dict(sample: Sample) -> dict[str, Any]:
    return {
        "ts": sample.ts,
        "uptime_sec": sample.uptime_sec,
        "mem_total_gb": round(sample.mem.mem_total_kb / 1024 / 1024, 2),
        "mem_used_pct": round(sample.mem.mem_used_pct, 1),
        "swap_used_gb": round(sample.mem.swap_used_kb / 1024 / 1024, 2),
        "loadavg_1": sample.loadavg_1,
        "loadavg_5": sample.loadavg_5,
        "loadavg_15": sample.loadavg_15,
        "iowait_pct": sample.iowait_pct,
        "d_state_count": sample.d_state_count,
        "d_state_top": sample.d_state_top,
        "session_type": sample.session_type,
        "graphical_active": sample.graphical_active,
        "gpus": [asdict(g) for g in sample.gpus],
        "procs": [asdict(p) for p in sample.procs],
        "display_procs": [asdict(p) for p in sample.display_procs],
        "train_log_bytes": sample.train_log_bytes,
        "train_log_mtime": sample.train_log_mtime,
        "kernel_hits": sample.kernel_hits,
        "xorg_hits": sample.xorg_hits,
        "flags": sample.flags,
        "notes": sample.notes,
        "hypothesis": sample.hypothesis,
    }


def format_line(sample: Sample) -> str:
    gpu_parts = []
    for g in sample.gpus:
        persist = f" pm={g.persistence}" if g.persistence else ""
        gpu_parts.append(
            f"GPU{g.index} {g.mem_used_mb}/{g.mem_total_mb}MB util={g.gpu_util_pct}% "
            f"T={g.temp_c}C P={g.power_w}W{persist}"
        )
    proc_parts = [
        f"pid={p.pid}[{p.state}] rss={p.rss_mb}MB cpuJ={p.cpu_total_jiffies}"
        + (f" w={p.wchan}" if p.wchan else "")
        for p in sample.procs
    ]
    disp_parts = [
        f"{Path(p.cmd.split()[0]).name if p.cmd else '?'}:{p.pid}[{p.state}]"
        for p in sample.display_procs[:4]
    ]
    flags = ",".join(sample.flags) if sample.flags else "ok"
    bits = [
        f"[{sample.ts}] flags={flags}",
        f"hyp={sample.hypothesis or '-'}",
        f"mem={sample.mem.mem_used_pct:.1f}%",
        f"swap={sample.mem.swap_used_kb // 1024}MB",
        f"load={sample.loadavg_1:.2f}",
        f"D={sample.d_state_count}",
    ]
    if sample.iowait_pct is not None:
        bits.append(f"iowait={sample.iowait_pct:.1f}%")
    if sample.graphical_active is not None:
        bits.append(f"gui={'on' if sample.graphical_active else 'off'}")
    if sample.train_log_bytes is not None:
        bits.append(f"log={sample.train_log_bytes}B")
    line = (
        " ".join(bits)
        + " | "
        + " | ".join(gpu_parts or ["no-gpu"])
        + " | train: "
        + (" ; ".join(proc_parts) if proc_parts else "none")
    )
    if disp_parts:
        line += " | disp: " + ",".join(disp_parts)
    if sample.kernel_hits:
        line += " | kern: " + " ; ".join(sample.kernel_hits[:2])
    if sample.xorg_hits:
        line += " | xorg: " + " ; ".join(sample.xorg_hits[:2])
    return line


class MonitorWriter:
    def __init__(self, output_dir: Path, run_tag: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = output_dir / f"monitor_{run_tag}.log"
        self.jsonl_path = output_dir / f"monitor_{run_tag}.jsonl"
        self.latest_path = output_dir / "latest.json"
        self.heartbeat_path = output_dir / "heartbeat.txt"
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

        # Tiny heartbeat for post-mortem: last alive stamp + hypothesis.
        hb = f"{sample.ts}\t{sample.hypothesis or 'ok'}\tflags={','.join(sample.flags) or 'ok'}\n"
        with self.heartbeat_path.open("w", encoding="utf-8") as hf:
            hf.write(hb)
            hf.flush()
            os.fsync(hf.fileno())

    def close(self) -> None:
        self._text.close()
        self._jsonl.close()


def _iowait_pct(prev: SystemCpu | None, cur: SystemCpu | None) -> float | None:
    if prev is None or cur is None:
        return None
    dt = cur.total - prev.total
    if dt <= 0:
        return None
    return 100.0 * (cur.iowait - prev.iowait) / dt


def evaluate_flags(
    sample: Sample,
    prev_cpu: dict[int, int],
    stall_counts: dict[int, int],
    display_stall_counts: dict[int, int],
    stall_intervals: int,
    display_stall_intervals: int,
    log_stall_intervals: int,
    log_stall_count: list[int],
    prev_log_bytes: list[int | None],
    vram_history: deque[float],
    mem_warn_pct: float,
    vram_warn_pct: float,
    swap_warn_gb: float,
    d_warn: int,
) -> None:
    if sample.kernel_hits:
        sample.flags.append("KERNEL_ALERT")
        joined = " ".join(sample.kernel_hits).lower()
        if "xid" in joined or "nvrm" in joined:
            sample.flags.append("NVIDIA_XID")
        if "soft lockup" in joined or "hard lockup" in joined:
            sample.flags.append("SOFT_LOCKUP")
        if "hung_task" in joined or "blocked for more than" in joined:
            sample.flags.append("HUNG_TASK")
        if "modeset" in joined:
            sample.flags.append("MODESET")
        if "gnome-shell" in joined or "xorg" in joined:
            sample.flags.append("DISPLAY_KERNEL")

    if sample.xorg_hits:
        sample.flags.append("XORG_ALERT")

    if sample.mem.mem_used_pct >= mem_warn_pct:
        sample.flags.append(f"MEM_HIGH_{sample.mem.mem_used_pct:.0f}pct")
    if sample.mem.swap_used_kb >= swap_warn_gb * 1024 * 1024:
        sample.flags.append(f"SWAP_USED_{sample.mem.swap_used_kb // (1024 * 1024)}GB")

    if sample.d_state_count >= d_warn:
        sample.flags.append(f"D_STATE_{sample.d_state_count}")

    if sample.iowait_pct is not None and sample.iowait_pct >= 40.0:
        sample.flags.append(f"IOWAIT_{sample.iowait_pct:.0f}pct")

    if sample.graphical_active:
        sample.notes.append("graphical.target active (display path shares GPU)")

    for gpu in sample.gpus:
        if gpu.mem_used_mb is not None and gpu.mem_total_mb:
            pct = 100.0 * gpu.mem_used_mb / gpu.mem_total_mb
            if pct >= vram_warn_pct:
                sample.flags.append(f"VRAM_HIGH_gpu{gpu.index}_{pct:.0f}pct")
            vram_history.append(gpu.mem_used_mb)
        if gpu.temp_c is not None and gpu.temp_c >= 85:
            sample.flags.append(f"GPU_HOT_{gpu.temp_c:.0f}C")
        if gpu.persistence and gpu.persistence.lower() in {"disabled", "off"}:
            sample.notes.append(f"gpu{gpu.index} persistence={gpu.persistence}")

    if len(vram_history) >= 20:
        window = list(vram_history)[-20:]
        if window[-1] - window[0] >= 512:
            sample.flags.append("VRAM_CREEP")

    # Train process stall (CPU jiffies flat). Logic backend may idle — pair with log stall.
    if not sample.procs:
        sample.flags.append("TRAIN_PROC_MISSING")
    else:
        for proc in sample.procs:
            if proc.state == "D":
                sample.flags.append(f"TRAIN_D_STATE_pid{proc.pid}")
            prev = prev_cpu.get(proc.pid)
            if prev is None or proc.cpu_total_jiffies > prev:
                stall_counts[proc.pid] = 0
            else:
                stall_counts[proc.pid] = stall_counts.get(proc.pid, 0) + 1
                if stall_counts[proc.pid] >= stall_intervals:
                    # Distinguish sleeping wait vs hard stall.
                    if proc.state in {"S", "I"}:
                        sample.flags.append(f"TRAIN_CPU_IDLE_pid{proc.pid}")
                    else:
                        sample.flags.append(f"TRAIN_STALL_pid{proc.pid}")
            prev_cpu[proc.pid] = proc.cpu_total_jiffies

    # Display compositor stall — strongest historical freeze signal on this desk.
    for proc in sample.display_procs:
        key = proc.pid
        if proc.state == "D":
            sample.flags.append(f"DISPLAY_D_STATE_pid{proc.pid}")
        prev = prev_cpu.get(key)
        if prev is None or proc.cpu_total_jiffies > prev:
            display_stall_counts[key] = 0
        else:
            display_stall_counts[key] = display_stall_counts.get(key, 0) + 1
            if display_stall_counts[key] >= display_stall_intervals:
                sample.flags.append(f"DISPLAY_STALL_pid{proc.pid}")
        prev_cpu[key] = proc.cpu_total_jiffies

    # Train log growth (metrics.jsonl / stdout log).
    if sample.train_log_bytes is not None:
        prev_b = prev_log_bytes[0]
        if prev_b is not None and sample.train_log_bytes <= prev_b:
            log_stall_count[0] += 1
            if log_stall_count[0] >= log_stall_intervals:
                sample.flags.append("TRAIN_LOG_STALL")
        else:
            log_stall_count[0] = 0
        prev_log_bytes[0] = sample.train_log_bytes

    sample.hypothesis = classify_hypothesis(sample)


def classify_hypothesis(sample: Sample) -> str:
    flags = set(sample.flags)
    if "NVIDIA_XID" in flags or "MODESET" in flags or "DISPLAY_KERNEL" in flags:
        return "display_nvidia_path"
    if any(f.startswith("DISPLAY_STALL") for f in flags) or any(
        f.startswith("DISPLAY_D_STATE") for f in flags
    ):
        return "display_compositor_stall"
    if "SOFT_LOCKUP" in flags or "HUNG_TASK" in flags:
        return "kernel_lockup"
    if any(f.startswith("MEM_HIGH") for f in flags) or any(f.startswith("SWAP_USED") for f in flags):
        return "memory_pressure"
    if any(f.startswith("D_STATE") for f in flags) and (
        sample.iowait_pct is not None and sample.iowait_pct >= 30
    ):
        return "io_or_driver_block"
    if "TRAIN_LOG_STALL" in flags and any(f.startswith("TRAIN_STALL") for f in flags):
        return "train_likely_stuck"
    if "TRAIN_LOG_STALL" in flags and any(f.startswith("TRAIN_CPU_IDLE") for f in flags):
        return "train_idle_or_slow"  # common for logic backend; not necessarily freeze
    if "TRAIN_PROC_MISSING" in flags:
        return "train_exited"
    if sample.graphical_active and sample.procs:
        return "gui_plus_train_risk"  # historical desk freeze combo
    return "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor HC factory training and localize freezes (train vs display/NVIDIA)."
    )
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Sample period in seconds.")
    p.add_argument("--match", type=str, default=DEFAULT_MATCH, help="pgrep -f pattern for train process.")
    p.add_argument("--pid", type=int, default=None, help="Explicit training PID (overrides --match).")
    p.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Directory for monitor logs (default: outputs/train_monitor).",
    )
    p.add_argument(
        "--stall-intervals",
        type=int,
        default=DEFAULT_STALL_INTERVALS,
        help="Mark train CPU stall after this many samples with no jiffy progress.",
    )
    p.add_argument(
        "--display-stall-intervals",
        type=int,
        default=DEFAULT_DISPLAY_STALL_INTERVALS,
        help="Mark DISPLAY_STALL after this many samples with no gnome/Xorg CPU progress.",
    )
    p.add_argument(
        "--log-stall-intervals",
        type=int,
        default=DEFAULT_LOG_STALL_INTERVALS,
        help="Mark TRAIN_LOG_STALL after this many samples with no log growth.",
    )
    p.add_argument("--mem-warn-pct", type=float, default=90.0, help="Flag when system RAM used >= this.")
    p.add_argument("--vram-warn-pct", type=float, default=95.0, help="Flag when GPU VRAM used >= this.")
    p.add_argument("--swap-warn-gb", type=float, default=1.0, help="Flag when swap used >= this many GB.")
    p.add_argument("--d-warn", type=int, default=8, help="Flag when >= N tasks are in D state.")
    p.add_argument("--max-samples", type=int, default=0, help="Stop after N samples (0 = run forever).")
    p.add_argument(
        "--watch-display",
        action="store_true",
        default=True,
        help="Watch gnome-shell/Xorg (default on).",
    )
    p.add_argument("--no-watch-display", action="store_false", dest="watch_display")
    p.add_argument(
        "--train-log",
        type=str,
        default="",
        help="Path to metrics.jsonl or train log to watch for growth (optional).",
    )
    p.add_argument(
        "--auto-train-log",
        action="store_true",
        default=True,
        help="Auto-pick newest logs/rl_games/HcFactory/*/metrics.jsonl (default on).",
    )
    p.add_argument("--no-auto-train-log", action="store_false", dest="auto_train_log")
    p.add_argument(
        "--freeze-hunt",
        action="store_true",
        help="Preset for desk freeze: interval=10s, tighter stalls, watch display.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.freeze_hunt:
        if args.interval == DEFAULT_INTERVAL:
            args.interval = 10.0
        if args.stall_intervals == DEFAULT_STALL_INTERVALS:
            args.stall_intervals = 6
        if args.display_stall_intervals == DEFAULT_DISPLAY_STALL_INTERVALS:
            args.display_stall_intervals = 4
        if args.log_stall_intervals == DEFAULT_LOG_STALL_INTERVALS:
            args.log_stall_intervals = 4
        args.watch_display = True

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = MonitorWriter(Path(args.output_dir), run_tag)
    kernel = KernelTail()
    xorg = XorgTail()
    prev_cpu: dict[int, int] = {}
    stall_counts: dict[int, int] = {}
    display_stall_counts: dict[int, int] = {}
    vram_history: deque[float] = deque(maxlen=40)
    prev_sys_cpu: SystemCpu | None = None
    log_stall_count = [0]
    prev_log_bytes: list[int | None] = [None]

    repo = Path(__file__).resolve().parents[1]
    train_log: Path | None = Path(args.train_log) if args.train_log else None
    if train_log is None and args.auto_train_log:
        train_log = discover_train_log(repo)

    print(
        f"[monitor_training] started tag={run_tag} interval={args.interval}s "
        f"match={args.match!r} pid={args.pid} display={args.watch_display} "
        f"train_log={train_log} -> {writer.text_path}",
        flush=True,
    )
    print(
        "[monitor_training] freeze tips: look for hypothesis=display_* / NVIDIA_XID / "
        "DISPLAY_STALL; TRAIN_CPU_IDLE alone is often normal on logic backend.",
        flush=True,
    )

    n = 0
    try:
        while True:
            ts = _now_iso()
            pids = find_pids(args.match, args.pid)
            procs = snapshot_procs(pids, role="train")
            display_procs: list[ProcSnapshot] = []
            if args.watch_display:
                display_procs = snapshot_procs(find_display_pids(), role="display")

            d_count, d_top = count_d_state()
            session_type, graphical = session_info()
            cur_sys = _read_system_cpu()
            iowait = _iowait_pct(prev_sys_cpu, cur_sys)
            prev_sys_cpu = cur_sys

            log_bytes, log_mtime = train_log_stat(train_log)
            # Re-discover if auto and missing.
            if train_log is None and args.auto_train_log and n % 10 == 0:
                train_log = discover_train_log(repo)

            load1, load5, load15 = _read_loadavg()
            sample = Sample(
                ts=ts,
                uptime_sec=_read_uptime(),
                mem=_read_mem(),
                loadavg_1=load1,
                loadavg_5=load5,
                loadavg_15=load15,
                gpus=query_gpus(),
                procs=procs,
                display_procs=display_procs,
                kernel_hits=kernel.poll(),
                xorg_hits=xorg.poll() if args.watch_display else [],
                d_state_count=d_count,
                d_state_top=d_top,
                iowait_pct=round(iowait, 1) if iowait is not None else None,
                train_log_bytes=log_bytes,
                train_log_mtime=log_mtime,
                session_type=session_type,
                graphical_active=graphical,
            )
            evaluate_flags(
                sample,
                prev_cpu,
                stall_counts,
                display_stall_counts,
                args.stall_intervals,
                args.display_stall_intervals,
                args.log_stall_intervals,
                log_stall_count,
                prev_log_bytes,
                vram_history,
                args.mem_warn_pct,
                args.vram_warn_pct,
                args.swap_warn_gb,
                args.d_warn,
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
