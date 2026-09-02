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

_CURRICULUM_NFIN_BUCKETS = (8, 6, 4, 2)


class CatalogCollectStats:
    """Per-run / per-episode catalog write counters for MetricCatalog/*."""

    def __init__(self) -> None:
        self.keys_at_run_start = 0
        self.ep_new = 0
        self.ep_updated = 0
        self.ep_skipped = 0
        self.run_new = 0
        self.run_updated = 0
        self.run_skipped = 0

    def bind_catalog(self, catalog: _catalog.ExploreCatalog) -> None:
        self.keys_at_run_start = len(catalog._by_key)

    def record(self, status: str) -> None:
        if status == "new":
            self.ep_new += 1
            self.run_new += 1
        elif status == "updated":
            self.ep_updated += 1
            self.run_updated += 1
        elif status == "skipped":
            self.ep_skipped += 1
            self.run_skipped += 1

    def reset_episode(self) -> None:
        self.ep_new = 0
        self.ep_updated = 0
        self.ep_skipped = 0

    @staticmethod
    def _nfin_buckets_covered(catalog: _catalog.ExploreCatalog) -> int:
        present = {
            int(r.get("n_finished", -1))
            for r in catalog._rows
            if int(r.get("n_finished", -1)) in _CURRICULUM_NFIN_BUCKETS
        }
        return len(present)

    def step_payload(self, catalog: _catalog.ExploreCatalog) -> dict:
        unique = len(catalog._by_key)
        joined_run = self.run_new + self.run_updated
        not_joined_run = self.run_skipped
        return {
            "MetricCatalog/01_unique_keys": int(unique),
            "MetricCatalog/02_joined_cumulative": int(joined_run),
            "MetricCatalog/03_not_joined_cumulative": int(not_joined_run),
        }

    def episode_payload(self, catalog: _catalog.ExploreCatalog, *, episode: int) -> dict:
        unique = len(catalog._by_key)
        new_since_run = max(0, unique - self.keys_at_run_start)
        joined_ep = self.ep_new + self.ep_updated
        not_joined_ep = self.ep_skipped
        joined_run = self.run_new + self.run_updated
        not_joined_run = self.run_skipped
        attempts_ep = joined_ep + not_joined_ep
        attempts_run = joined_run + not_joined_run
        payload = {
            "MetricCatalog/episode": int(episode),
            "MetricCatalog/01_unique_keys": int(unique),
            "MetricCatalog/02_new_keys": int(self.ep_new),
            "MetricCatalog/03_joined": int(joined_ep),
            "MetricCatalog/04_not_joined": int(not_joined_ep),
            "MetricCatalog/05_new_keys_since_run": int(new_since_run),
            "MetricCatalog/06_nfin_buckets_covered": int(self._nfin_buckets_covered(catalog)),
            "MetricCatalog/07_joined_cumulative": int(joined_run),
            "MetricCatalog/08_not_joined_cumulative": int(not_joined_run),
        }
        if attempts_ep > 0:
            payload["MetricCatalog/09_join_fraction"] = float(joined_ep) / float(attempts_ep)
        if attempts_run > 0:
            payload["MetricCatalog/10_join_fraction_cumulative"] = float(joined_run) / float(attempts_run)
        return payload


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
        self.catalog_collect = bool(config.get("catalog_collect"))
        self.explore_save_catalog = bool(config.get("explore_save_catalog", True))
        self.save_catalog = self.explore_save_catalog and (self.explore or self.catalog_collect)
        self.catalog_stats = CatalogCollectStats()
        collect_mode = "explore" if self.explore else ("policy_train" if self.catalog_collect else "readonly")
        print(
            f"[Horizon] explore={self.explore} catalog_collect={self.catalog_collect} "
            f"N={self.explore_n_products} "
            f"T_max={_curr.t_max_for(self.explore_n_products, anchor)} "
            f"save_catalog={self.save_catalog} collect_mode={collect_mode}"
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
            create_round=self.save_catalog,
        )
        self.catalog_stats.bind_catalog(self.catalog)
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
        mode_dir = "collect" if (self.explore or self.catalog_collect) else "train"
        self.stall_root = Path("env_checkpoints") / "stagnation" / mode_dir
        self.stall_root.mkdir(parents=True, exist_ok=True)
        if self.save_catalog:
            meta = {
                "epsilon": 1.0 if self.explore else None,
                "t_max": _curr.t_max_for(
                    self.explore_n_products if self.explore else _curr.N_TRAIN_TARGET, anchor
                ),
                "n_products": self.explore_n_products if self.explore else _curr.N_TRAIN_TARGET,
                "mode": "explore" if self.explore else "catalog_collect",
            }
            self.catalog.write_round_meta(**{k: v for k, v in meta.items() if v is not None})

    @property
    def catalog_metrics_enabled(self) -> bool:
        return bool(self.save_catalog)

    def catalog_step_metrics(self) -> dict:
        return self.catalog_stats.step_payload(self.catalog)

    def catalog_episode_metrics(self, *, episode: int) -> dict:
        payload = self.catalog_stats.episode_payload(self.catalog, episode=episode)
        self.catalog_stats.reset_episode()
        return payload

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
        return self.apply_order_eval(_curr.N_FULL_ORDER, horizon=horizon)

    def apply_order_eval(self, n_products: int, horizon: int | None = None) -> int:
        """Eval order of ``n_products`` (10 = train distribution, 16 = full-order generalization)."""
        n_products = int(n_products)
        if n_products >= _curr.N_FULL_ORDER:
            t_max = int(
                horizon if horizon is not None else _curr.t_max_for(_curr.N_FULL_ORDER, self.curriculum.anchor)
            )
        else:
            t_max = int(
                horizon if horizon is not None else _curr.t_max_for(n_products, self.curriculum.anchor)
            )
        if self.env_list is None:
            return t_max
        self.curriculum.enabled = False
        product_type = self.curriculum.product_type
        for env in self.env_list:
            if n_products >= _curr.N_FULL_ORDER:
                env.task_manager.max_episodic_steps = t_max
                progress = env.env_state_action_dict.setdefault("progress", {})
                progress.pop("segment_target_nfin", None)
                progress.pop("segment_start_nfin", None)
                progress.pop("segment_delta_n", None)
                progress["stage_wip_cap"] = 10
                progress["product_order"] = {product_type: _curr.N_FULL_ORDER}
                progress["not_started"] = {product_type: _curr.N_FULL_ORDER}
            else:
                _curr.apply_train_order(
                    env,
                    n_products=n_products,
                    product_type=product_type,
                    anchor=self.curriculum.anchor,
                )
                env.task_manager.max_episodic_steps = t_max
            env.algo_hierarchical_masker.generate_agents_mask(env.env_state_action_dict)
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
        if self.save_catalog:
            _, status = self.catalog.save_if_new(
                ckpt, key=key, n_finished=nfin, time_step=t, n_ongoing=n_ong
            )
            self.catalog_stats.record(status)

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
