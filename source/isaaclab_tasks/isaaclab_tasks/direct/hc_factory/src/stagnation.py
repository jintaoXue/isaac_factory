"""Progress stall detector. Fingerprint from ``debug_env_dump.ongoing_fingerprint``."""
from __future__ import annotations

from .debug_env_dump import ongoing_fingerprint


class StagnationDetector:
    def __init__(self, l1: int = 400, l2: int = 600, l3: int = 800) -> None:
        self.l1 = int(l1)
        self.l2 = int(l2)
        self.l3 = int(l3)
        self.fp: str | None = None
        self.n = 0
        self._fired: set[str] = set()
        self.tried_keys: set[str] = set()

    def reset(self) -> None:
        self.fp = None
        self.n = 0
        self._fired.clear()
        self.tried_keys.clear()

    def update(self, env: dict) -> str | None:
        """Return 'L1' / 'L2' / 'L3' once per stall streak, else None."""
        fp = ongoing_fingerprint(env)
        if fp == self.fp:
            self.n += 1
        else:
            self.fp = fp
            self.n = 0
            self._fired.clear()
            self.tried_keys.clear()

        for level, thresh in (("L3", self.l3), ("L2", self.l2), ("L1", self.l1)):
            if self.n >= thresh and level not in self._fired:
                self._fired.add(level)
                return level
        return None
