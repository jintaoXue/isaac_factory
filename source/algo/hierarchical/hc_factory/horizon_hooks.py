"""Horizon helpers: catalog explore, stall restore, curriculum. Used by HierarchicalTPA.train()."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import pickle

from .hc_factory_imports import import_hc_module

_env_ckpt = import_hc_module("src.env_checkpoint")
_stag = import_hc_module("src.stagnation")
_catalog = import_hc_module("src.explore_catalog")
_curr = import_hc_module("src.curriculum")
_dbg = import_hc_module("src.debug_env_dump")


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
        explore_n = int(config.get("explore_n_products") or 0)
        self.explore_n_products = explore_n if explore_n > 0 else _curr.N_FULL_ORDER
        self.explore_save_catalog = bool(config.get("explore_save_catalog", True))
        print(
            f"[Horizon] explore={self.explore} N={self.explore_n_products} "
            f"T_max={_curr.t_max_for(self.explore_n_products, anchor)} "
            f"save_catalog={self.explore_save_catalog}"
        )
        self.curriculum = _curr.CurriculumScheduler(
            enabled=bool(config.get("curriculum")),
            start_stage=int(config.get("curriculum_start_stage", 0)),
            t_max_anchor=anchor,
        )
        # Catalog root: explore uses explore_n_products; curriculum warmstart uses N_TRAIN_TARGET
        # (e.g. N10_T40000), NOT N_FULL_ORDER — otherwise pick_by_nfin silently misses the collect library.
        catalog_root = config.get("explore_catalog_dir") or None
        if catalog_root:
            catalog_n = self.explore_n_products
        elif self.explore:
            catalog_n = self.explore_n_products
        else:
            catalog_n = _curr.N_TRAIN_TARGET
        catalog_t = _curr.t_max_for(catalog_n, anchor)
        self.catalog = _catalog.ExploreCatalog(
            catalog_root,
            n_products=catalog_n,
            t_max=catalog_t,
            create_round=self.explore and self.explore_save_catalog,
        )
        print(
            f"[Horizon] catalog={self.catalog.root} "
            f"entries={len(self.catalog._rows)} curriculum={self.curriculum.enabled}"
        )
        self.l1 = int(config.get("stagnation_l1", 400))
        self.l2 = int(config.get("stagnation_l2", 600))
        self.l3 = int(config.get("stagnation_l3", 800))
        self.rings: list[deque] = []
        self.detectors: list = []
        self.ep_stalled: list[bool] = []
        self.env_list = None
        self.stall_counts = {"L1": 0, "L2": 0, "L3": 0}
        self.last_restore_info: dict[int, dict] = {}
        mode_dir = "collect" if self.explore else "train"
        self.stall_root = Path("env_checkpoints") / "stagnation" / mode_dir
        self.stall_root.mkdir(parents=True, exist_ok=True)
        if self.explore and self.explore_save_catalog:
            self.catalog.write_round_meta(
                epsilon=1.0,
                t_max=_curr.t_max_for(self.explore_n_products, anchor),
                n_products=self.explore_n_products,
            )

    def explore_t_max(self) -> int:
        return _curr.t_max_for(self.explore_n_products, self.curriculum.anchor)

    def apply_explore_episode(self, single_env) -> int:
        """Set explore order size (N=16 catalog or N=10 baseline) and matching T_max."""
        if self.explore_n_products >= _curr.N_FULL_ORDER:
            return _curr.apply_eval_order(single_env, anchor=self.curriculum.anchor)
        progress = single_env.env_state_action_dict.setdefault("progress", {})
        progress["stage_wip_cap"] = _curr.WIP_CAP
        return _curr.apply_train_order(
            single_env,
            n_products=self.explore_n_products,
            anchor=self.curriculum.anchor,
        )

    def ensure_explore_episode(self, env_id: int) -> None:
        """Re-apply N/T if env reset left CfgProductOrder (16) instead of explore_n_products."""
        if not self.explore or self.env_list is None:
            return
        env = self.env_list[env_id]
        progress = env.env_state_action_dict.get("progress") or {}
        order = progress.get("product_order") or {}
        n_order = (
            sum(int(v or 0) for v in order.values()) if isinstance(order, dict) and order else 0
        )
        want_t = self.explore_t_max()
        if n_order != self.explore_n_products or int(env.task_manager.max_episodic_steps) != want_t:
            self.apply_explore_episode(env)

    def apply_full_order_eval(self, horizon: int | None = None) -> int:
        """Disable segment curriculum: N_FULL_ORDER products, T_max=anchor, no catalog warmstart."""
        t_max = int(
            horizon if horizon is not None else _curr.t_max_for(_curr.N_FULL_ORDER, self.curriculum.anchor)
        )
        if self.env_list is None:
            return t_max
        self.curriculum.enabled = False
        product_type = self.curriculum.product_type
        for env in self.env_list:
            env.task_manager.max_episodic_steps = t_max
            progress = env.env_state_action_dict.setdefault("progress", {})
            progress.pop("segment_target_nfin", None)
            progress.pop("segment_start_nfin", None)
            progress.pop("segment_delta_n", None)
            progress["stage_wip_cap"] = 10
            progress["product_order"] = {product_type: _curr.N_FULL_ORDER}
            progress["not_started"] = {product_type: _curr.N_FULL_ORDER}
        return t_max

    def bind(self, vec_env, n_envs: int) -> None:
        self.env_list = hc_env_list(vec_env)
        self.rings = [deque(maxlen=self.ring_k) for _ in range(n_envs)]
        self.detectors = [_stag.StagnationDetector(self.l1, self.l2, self.l3) for _ in range(n_envs)]
        self.ep_stalled = [False] * n_envs
        for i, env in enumerate(self.env_list):
            if self.explore:
                self.apply_explore_episode(env)
            else:
                self.maybe_warmstart_new_episode(i)
        if self.warmstart_path:
            overlay = bool(self.curriculum.enabled)
            self._restore_path(0, self.warmstart_path, overlay=overlay)
            env = self.env_list[0].env_state_action_dict
            print(
                f"[Hier] warmstart env=0 path={self.warmstart_path} "
                f"ep_t={int(env.get('time_step', 0) or 0)} "
                f"n_finished={_env_ckpt.n_finished(env)} "
                f"key={_env_ckpt.progress_key(env)}"
            )

    def maybe_warmstart_new_episode(self, env_id: int) -> None:
        env = self.env_list[env_id]
        if self.explore:
            self.apply_explore_episode(env)
            return
        if not self.curriculum.enabled:
            self.curriculum.apply(env, overlay_existing=False)
            return
        spec = self.curriculum.spec
        path = self.catalog.pick_by_nfin(spec.start_nfin) if spec.start_nfin > 0 else None
        if path is not None:
            self._restore_path(env_id, path, overlay=True)
        else:
            if spec.start_nfin > 0:
                print(
                    f"[Hier] curriculum warmstart miss env={env_id} "
                    f"start_nfin={spec.start_nfin} catalog={self.catalog.root} "
                    f"(falling back to empty start)"
                )
            self.curriculum.apply(env, overlay_existing=False)

    def on_decision(self, env_id: int, action: dict, env: dict) -> None:
        if not action.get("dispatch_list"):
            return
        key = _env_ckpt.progress_key(env)
        ckpt = _env_ckpt.capture(env)
        nfin = _env_ckpt.n_finished(env)
        n_ong = len((env.get("progress") or {}).get("ongoing_task_records") or {})
        t = int(env.get("time_step", 0) or 0)
        self.rings[env_id].append({"key": key, "ckpt": ckpt, "n_finished": nfin, "t": t})
        if self.explore and self.explore_save_catalog:
            self.catalog.save_if_new(ckpt, key=key, n_finished=nfin, time_step=t, n_ongoing=n_ong)

    def after_step(self, env_id: int, env: dict) -> str | None:
        level = self.detectors[env_id].update(env)
        if level == "L1":
            self.stall_counts["L1"] += 1
            self._dump_stagnation(env_id, env, "L1")
            print(
                f"[Hier] stagnation L1 env={env_id} n={self.detectors[env_id].n} "
                f"stall_counts={self.stall_counts_str()}"
            )
            return "L1"
        if level == "L2":
            self.ep_stalled[env_id] = True
            self.stall_counts["L2"] += 1
            self._dump_stagnation(env_id, env, "L2")
            cur_t = int(env.get("time_step", 0) or 0)
            if self._restore_ring(env_id):
                info = self.last_restore_info.get(env_id) or {}
                to_t = int(info.get("to_t", -1))
                print(
                    f"[Hier] stagnation L2 restore env={env_id} "
                    f"from ep_t={cur_t} to ep_t={to_t} stall_counts={self.stall_counts_str()}"
                )
                return "L2_RESTORE"
            else:
                print(f"[Hier] stagnation L2 no-restore env={env_id} stall_counts={self.stall_counts_str()}")
            return "L2"
        if level == "L3":
            self.ep_stalled[env_id] = True
            self.stall_counts["L3"] += 1
            self._dump_stagnation(env_id, env, "L3")
            nfin = _env_ckpt.n_finished(env)
            cur_t = int(env.get("time_step", 0) or 0)
            path = self.catalog.pick_by_nfin(nfin)
            if path is not None:
                self._restore_path(env_id, path, overlay=True, reset_clock=False)
                self.env_list[env_id].env_state_action_dict["time_step"] = cur_t
                print(f"[Hier] stagnation L3 catalog restore env={env_id} {path} stall_counts={self.stall_counts_str()}")
            elif self._restore_ring(env_id, oldest=True):
                print(f"[Hier] stagnation L3 ring restore env={env_id} stall_counts={self.stall_counts_str()}")
            else:
                self.env_list[env_id].reset_env()
                self.maybe_warmstart_new_episode(env_id)
                print(f"[Hier] stagnation L3 full reset env={env_id} stall_counts={self.stall_counts_str()}")
            self.detectors[env_id].reset()
            return "L3_RESTORE"
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
            print(
                f"[Hier] curriculum -> stage {spec.stage} "
                f"target={spec.target_nfin} start={spec.start_nfin} delta={spec.delta_n} "
                f"T_budget={spec.t_max}"
            )

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
        env = self.env_list[env_id]
        cur_t = int(env.env_state_action_dict.get("time_step", 0) or 0)
        env.restore_checkpoint(chosen["ckpt"])
        env.env_state_action_dict["time_step"] = cur_t
        if self.curriculum.enabled:
            self.curriculum.apply(env, overlay_existing=True)
        self.last_restore_info[env_id] = {
            "kind": "ring",
            "key": chosen["key"],
            "to_t": int(chosen.get("t", 0) or 0),
            "n_finished": int(chosen.get("n_finished", 0) or 0),
        }
        det.reset()
        det.tried_keys.add(chosen["key"])
        return True

    def _restore_path(self, env_id: int, path: str | Path, overlay: bool, reset_clock: bool = True) -> None:
        ckpt = _catalog.ExploreCatalog.load_pkl(path)
        self.env_list[env_id].restore_checkpoint(ckpt)
        if overlay:
            env = self.env_list[env_id].env_state_action_dict
            if reset_clock:
                env["time_step"] = 0
            self.curriculum.apply(self.env_list[env_id], overlay_existing=True)

    def stall_counts_str(self) -> str:
        return f"L1={self.stall_counts['L1']} L2={self.stall_counts['L2']} L3={self.stall_counts['L3']}"

    def _nearest_ring_item(self, env_id: int, env: dict):
        det = self.detectors[env_id]
        cur = _env_ckpt.progress_key(env)
        for item in reversed(list(self.rings[env_id])):
            if item["key"] != cur and item["key"] not in det.tried_keys:
                return item
        items = list(self.rings[env_id])
        return items[-1] if items else None

    def _dump_stagnation(self, env_id: int, env: dict, level: str) -> None:
        det = self.detectors[env_id]
        ep = int(env.get("episode_num", 0) or 0)
        t = int(env.get("time_step", 0) or 0)
        key = _env_ckpt.progress_key(env)
        fp = getattr(det, "fp", None)
        name = f"{level}_env{env_id:02d}_ep{ep:06d}_t{t:06d}_{key}"
        out_dir = self.stall_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        stalled_ckpt = _env_ckpt.capture(env)
        stalled_path = out_dir / "stalled_state.pkl"
        nearest = self._nearest_ring_item(env_id, env)

        meta = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": "collect" if self.explore else "train",
            "level": level,
            "env_id": env_id,
            "episode_num": ep,
            "time_step": t,
            "progress_key": key,
            "ongoing_fingerprint": fp,
            "stall_streak_steps": int(det.n),
            "stall_counts": dict(self.stall_counts),
            "catalog_round": getattr(self.catalog, "round_id", None),
            "nearest_effective_decision": None,
        }

        with stalled_path.open("wb") as f:
            pickle.dump(stalled_ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
        slim_state = _dbg.to_jsonable(stalled_ckpt)
        (out_dir / "stalled_state.json").write_text(
            json.dumps(slim_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if nearest is not None:
            nearest_meta = {
                "key": nearest["key"],
                "n_finished": int(nearest.get("n_finished", 0)),
                "time_step": int(nearest.get("t", 0)),
            }
            meta["nearest_effective_decision"] = nearest_meta
            with (out_dir / "nearest_decision.pkl").open("wb") as f:
                pickle.dump(nearest["ckpt"], f, protocol=pickle.HIGHEST_PROTOCOL)

        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
