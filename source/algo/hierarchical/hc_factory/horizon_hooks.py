"""Horizon helpers: catalog explore, stall restore, curriculum. Used by HierarchicalTPA.train()."""
from __future__ import annotations

from collections import deque
from pathlib import Path

from .hc_factory_imports import import_hc_module

_env_ckpt = import_hc_module("src.env_checkpoint")
_stag = import_hc_module("src.stagnation")
_catalog = import_hc_module("src.explore_catalog")
_curr = import_hc_module("src.curriculum")


def hc_env_list(vec_env):
    env = vec_env
    for _ in range(8):
        if env is None:
            break
        if hasattr(env, "env_list"):
            return env.env_list
        env = getattr(env, "env", None) or getattr(env, "unwrapped", None)
    raise RuntimeError("cannot find hc env_list on vec_env")


class HorizonHooks:
    def __init__(self, config: dict) -> None:
        self.explore = bool(config.get("explore") or config.get("explore_catalog"))
        self.warmstart_path = str(config.get("warmstart") or "").strip() or None
        self.ring_k = int(config.get("decision_ring_k", 20))
        self.cosine_th = float(config.get("soft_cosine_th", 0.95))
        anchor = int(config.get("t_max_anchor", _curr.T_MAX_ANCHOR))
        self.curriculum = _curr.CurriculumScheduler(
            enabled=bool(config.get("curriculum")),
            start_stage=int(config.get("curriculum_start_stage", 0)),
            t_max_anchor=anchor,
        )
        n_products = 16 if self.explore else self.curriculum.spec.n_products
        t_max = _curr.t_max_for(n_products, anchor) if self.explore else self.curriculum.spec.t_max
        catalog_root = config.get("explore_catalog_dir") or None
        self.catalog = _catalog.ExploreCatalog(
            catalog_root, n_products=n_products, t_max=t_max, create_round=self.explore
        )
        self.l1 = int(config.get("stagnation_l1", 400))
        self.l2 = int(config.get("stagnation_l2", 600))
        self.l3 = int(config.get("stagnation_l3", 800))
        self.rings: list[deque] = []
        self.detectors: list = []
        self.ep_stalled: list[bool] = []
        self.env_list = None
        if self.explore:
            self.catalog.write_round_meta(epsilon=1.0, t_max=t_max, n_products=n_products)

    def bind(self, vec_env, n_envs: int) -> None:
        self.env_list = hc_env_list(vec_env)
        self.rings = [deque(maxlen=self.ring_k) for _ in range(n_envs)]
        self.detectors = [_stag.StagnationDetector(self.l1, self.l2, self.l3) for _ in range(n_envs)]
        self.ep_stalled = [False] * n_envs
        for env in self.env_list:
            if self.explore:
                env.task_manager.max_episodic_steps = _curr.t_max_for(16, self.curriculum.anchor)
                env.env_state_action_dict.setdefault("progress", {})["stage_wip_cap"] = 10
            else:
                self.curriculum.apply(env, overlay_existing=False)
        if self.warmstart_path:
            self._restore_path(0, self.warmstart_path, overlay=bool(self.curriculum.enabled))

    def maybe_warmstart_new_episode(self, env_id: int) -> None:
        env = self.env_list[env_id]
        if self.explore:
            env.task_manager.max_episodic_steps = _curr.t_max_for(16, self.curriculum.anchor)
            env.env_state_action_dict.setdefault("progress", {})["stage_wip_cap"] = 10
            return
        self.curriculum.apply(env, overlay_existing=False)
        if not self.curriculum.enabled:
            return
        spec = self.curriculum.spec
        if spec.stage == 0:
            return
        start_nfin = max(0, spec.n_products // 2)
        path = self.catalog.pick_by_nfin(start_nfin)
        if path is not None:
            self._restore_path(env_id, path, overlay=True)
            self.curriculum.apply(env, overlay_existing=True)

    def on_decision(self, env_id: int, action: dict, env: dict) -> None:
        if not action.get("dispatch_list"):
            return
        key = _env_ckpt.progress_key(env)
        ckpt = _env_ckpt.capture(env)
        nfin = _env_ckpt.n_finished(env)
        n_ong = len((env.get("progress") or {}).get("ongoing_task_records") or {})
        t = int(env.get("time_step", 0) or 0)
        self.rings[env_id].append({"key": key, "ckpt": ckpt, "n_finished": nfin, "t": t})
        if self.explore:
            self.catalog.save_if_new(ckpt, key=key, n_finished=nfin, time_step=t, n_ongoing=n_ong)

    def after_step(self, env_id: int, env: dict) -> str | None:
        level = self.detectors[env_id].update(env)
        if level == "L1":
            print(f"[Hier] stagnation L1 env={env_id} n={self.detectors[env_id].n}")
            return "L1"
        if level == "L2":
            self.ep_stalled[env_id] = True
            if self._restore_ring(env_id):
                print(f"[Hier] stagnation L2 restore env={env_id}")
            return "L2"
        if level == "L3":
            self.ep_stalled[env_id] = True
            nfin = _env_ckpt.n_finished(env)
            path = self.catalog.pick_by_nfin(nfin)
            if path is not None:
                self._restore_path(env_id, path, overlay=True)
                print(f"[Hier] stagnation L3 catalog restore env={env_id} {path}")
            elif self._restore_ring(env_id, oldest=True):
                print(f"[Hier] stagnation L3 ring restore env={env_id}")
            else:
                self.env_list[env_id].reset_env()
                self.maybe_warmstart_new_episode(env_id)
                print(f"[Hier] stagnation L3 full reset env={env_id}")
            self.detectors[env_id].reset()
            return "L3"
        return None

    def on_episode_end(self, env_id: int, *, success: bool, ep_len: int) -> None:
        advanced = self.curriculum.observe_episode(
            success=success, stagnation=self.ep_stalled[env_id], ep_len=ep_len
        )
        self.detectors[env_id].reset()
        self.rings[env_id].clear()
        self.ep_stalled[env_id] = False
        self.maybe_warmstart_new_episode(env_id)
        if advanced:
            spec = self.curriculum.spec
            print(f"[Hier] curriculum -> stage {spec.stage} N={spec.n_products} T_max={spec.t_max}")

    def _restore_ring(self, env_id: int, oldest: bool = False) -> bool:
        det = self.detectors[env_id]
        items = list(self.rings[env_id])
        if not items:
            return False
        cur = _env_ckpt.progress_key(self.env_list[env_id].env_state_action_dict)
        order = items if oldest else list(reversed(items))
        chosen = None
        for item in order:
            if item["key"] == cur or item["key"] in det.tried_keys:
                continue
            chosen = item
            break
        if chosen is None:
            return False
        det.tried_keys.add(chosen["key"])
        self.env_list[env_id].restore_checkpoint(chosen["ckpt"])
        det.reset()
        det.tried_keys.add(chosen["key"])
        return True

    def _restore_path(self, env_id: int, path: str | Path, overlay: bool) -> None:
        ckpt = _catalog.ExploreCatalog.load_pkl(path)
        self.env_list[env_id].restore_checkpoint(ckpt)
        if overlay:
            self.curriculum.apply(self.env_list[env_id], overlay_existing=True)
