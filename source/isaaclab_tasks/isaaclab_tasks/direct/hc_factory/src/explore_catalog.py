"""On-disk catalog of unique decision checkpoints (masked-random explore)."""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


def default_root(n_products: int = 16, t_max: int = 25000) -> Path:
    return Path("env_checkpoints") / "random_explore" / f"N{n_products}_T{t_max}"


class ExploreCatalog:
    def __init__(
        self,
        root: str | Path | None = None,
        n_products: int = 16,
        t_max: int = 25000,
        *,
        create_round: bool = True,
    ) -> None:
        self.root = Path(root) if root else default_root(n_products, t_max)
        self.rounds_dir = self.root / "rounds"
        self.by_nfin = self.root / "by_nfin"
        self.catalog_path = self.root / "catalog.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self._by_key: dict[str, dict] = {}
        self._rows: list[dict] = []
        if self.catalog_path.exists():
            with self.catalog_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._rows.append(row)
                    key = row.get("key")
                    if key:
                        prev = self._by_key.get(key)
                        if prev is None or int(row.get("n_finished", 0)) >= int(prev.get("n_finished", 0)):
                            self._by_key[key] = row
        self.round_id = "readonly"
        self.round_dir = self.root
        if create_round:
            self.round_id = self._next_round_id()
            self.round_dir = self.rounds_dir / self.round_id
            (self.round_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    def _next_round_id(self) -> str:
        existing = sorted(p.name for p in self.rounds_dir.glob("r*") if p.is_dir())
        n = 1
        if existing:
            try:
                n = int(existing[-1].lstrip("r")) + 1
            except ValueError:
                n = len(existing) + 1
        return f"r{n:03d}"

    def write_round_meta(self, **meta: Any) -> None:
        payload = {"round": self.round_id, "ts": datetime.now().isoformat(timespec="seconds"), **meta}
        (self.round_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def has_key(self, key: str) -> bool:
        return key in self._by_key

    def save_if_new(self, ckpt: dict, *, key: str, n_finished: int, time_step: int, n_ongoing: int) -> Path | None:
        """Append only if key is new, or this snapshot finished more products."""
        prev = self._by_key.get(key)
        if prev is not None and int(prev.get("n_finished", 0)) >= n_finished:
            return None
        name = f"nfin{n_finished:02d}_ong{n_ongoing:02d}_t{time_step:06d}_{key}.pkl"
        path = self.round_dir / "ckpts" / name
        with path.open("wb") as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)

        nfin_dir = self.by_nfin / f"{n_finished:02d}"
        nfin_dir.mkdir(parents=True, exist_ok=True)
        link = nfin_dir / name
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            os.link(path, link)
        except OSError:
            if not link.exists():
                link.symlink_to(os.path.relpath(path, nfin_dir))

        row = {
            "key": key,
            "path": str(path),
            "n_finished": n_finished,
            "n_ongoing": n_ongoing,
            "t": time_step,
            "round": self.round_id,
        }
        with self.catalog_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rows.append(row)
        self._by_key[key] = row
        return path

    def pick_by_nfin(self, n_finished: int) -> Path | None:
        """Latest catalog row with ``n_finished == n`` (fallback: closest below)."""
        exact = [r for r in self._rows if int(r.get("n_finished", -1)) == n_finished]
        pool = exact or [r for r in self._rows if int(r.get("n_finished", -1)) <= n_finished]
        if not pool:
            return None
        row = pool[-1]
        path = Path(row["path"])
        return path if path.is_file() else None

    @staticmethod
    def load_pkl(path: str | Path) -> dict:
        with Path(path).open("rb") as f:
            return pickle.load(f)
