"""Online / replay inference for a trained BNPDFormer checkpoint.

The model never sees raw ``job_trace.csv``. Input is Stage-C
``window_feature_table.csv`` (60s). With ``remain_to_jobs_done`` (default in
FactoryBN.json) the head predicts occupancy from now until remaining jobs
finish (A.1: start / duration / station). Old checkpoints without that flag
still forecast the next 60s window.

Example (PDFormer dir, ``bn_pdformer``)::

    python -m factory_bn.infer \\
        --ckpt libcity/cache/model_cache/evt/BNPDFormer_best.pt \\
        --run_dir ../output/bottleneck_dataset/old_machine2.0 \\
        --episode 0

Live API: assemble ``features`` as ``(12, N, F)`` in checkpoint node order,
then ``BNPredictor.predict_x(...)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PDFORMER_ROOT = Path(__file__).resolve().parent.parent
if str(_PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PDFORMER_ROOT))

from factory_bn.causes import decode_root_cause
from factory_bn.dataset import FactoryBNWindowDataset, Scaler, build_infer_sample, load_factory_bn_bundle
from factory_bn.export_dataset import load_derived_episode
from factory_bn.graph import resolve_to_node_id
from factory_bn.model import BNPDFormer
from factory_bn.remain import first_done_index, occupancy_to_events


def _as_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _data_feature_from_ckpt(meta: dict[str, Any]) -> dict[str, Any]:
    cause_classes = list(meta.get("cause_classes") or [])
    n_cause = int(meta.get("n_cause_classes") or len(cause_classes) or 10)
    return {
        "num_nodes": int(meta["num_nodes"]),
        "feature_dim": int(meta["feature_dim"]),
        "output_dim": 1,
        "adj_mx": _as_numpy(meta["adj_mx"]).astype(np.float32),
        "sh_mx": _as_numpy(meta["sh_mx"]).astype(np.float32),
        "sem_mx": _as_numpy(meta["sem_mx"]).astype(np.float32),
        "pattern_keys": _as_numpy(meta["pattern_keys"]).astype(np.float32),
        "resource_ids": [str(x) for x in meta["resource_ids"]],
        "resource_types": [str(x) for x in meta["resource_types"]],
        "cause_classes": cause_classes,
        "n_cause_classes": n_cause,
        "cause_majority": int(meta.get("cause_majority", -1)),
    }


def align_episode(ep: dict[str, Any], ckpt_ids: list[str]) -> dict[str, Any]:
    """Reindex a pivoted episode onto the checkpoint node list (missing → 0)."""
    src_ids = [str(x) for x in ep["resource_ids"]]
    src_index = {rid: i for i, rid in enumerate(src_ids)}
    ckpt_index = {rid: j for j, rid in enumerate(ckpt_ids)}
    t_len, _, f_dim = ep["features"].shape
    n_dst = len(ckpt_ids)
    features = np.zeros((t_len, n_dst, f_dim), dtype=np.float32)
    scores = np.zeros((t_len, n_dst, 1), dtype=np.float32)
    for j, rid in enumerate(ckpt_ids):
        i = src_index.get(rid)
        if i is None:
            mapped = resolve_to_node_id(rid, src_ids)
            i = src_index.get(mapped) if mapped else None
        if i is not None:
            features[:, j] = ep["features"][:, i]
            scores[:, j] = ep["scores"][:, i]

    ev_node, ev_start, ev_dur, ev_ti = [], [], [], []
    src_node = np.asarray(ep.get("event_node", []), dtype=np.int64)
    for k, ni in enumerate(src_node):
        if 0 <= int(ni) < len(src_ids):
            rid = src_ids[int(ni)]
        else:
            continue
        mapped = resolve_to_node_id(rid, ckpt_ids) or (rid if rid in ckpt_index else None)
        if mapped is None:
            continue
        ev_node.append(ckpt_index[mapped])
        ev_start.append(float(ep["event_start_s"][k]))
        ev_dur.append(float(ep["event_duration_s"][k]))
        ev_ti.append(int(ep["event_start_ti"][k]))

    out = {
        "name": ep.get("name", ""),
        "resource_ids": list(ckpt_ids),
        "features": features,
        "scores": scores,
        "window_start_s": np.asarray(ep["window_start_s"], dtype=np.float32),
        "windows": np.asarray(ep.get("windows", np.arange(t_len)), dtype=np.int64),
        "will_bottleneck": np.asarray(ep.get("will_bottleneck", ep.get("will", [])), dtype=np.float32),
        "mark_node": np.asarray(ep.get("mark_node", ep.get("mark", [])), dtype=np.int64),
        "cause": np.asarray(ep.get("cause", []), dtype=np.int64),
        "event_node": np.asarray(ev_node, dtype=np.int64),
        "event_start_s": np.asarray(ev_start, dtype=np.float32),
        "event_duration_s": np.asarray(ev_dur, dtype=np.float32),
        "event_start_ti": np.asarray(ev_ti, dtype=np.int64),
        "jobs_remaining": np.asarray(ep.get("jobs_remaining", np.zeros((t_len,), dtype=np.float32)), dtype=np.float32),
        "jobs_total": float(ep.get("jobs_total") or 0.0),
    }
    if out["will_bottleneck"].size == 0:
        out["will_bottleneck"] = np.zeros((t_len,), dtype=np.float32)
    if out["mark_node"].size == 0:
        out["mark_node"] = np.full((t_len,), -1, dtype=np.int64)
    if out["cause"].size == 0:
        out["cause"] = np.full((t_len,), -1, dtype=np.int64)
    return out


def bundle_episode_to_pivot(ep: dict[str, Any], resource_ids: list[str]) -> dict[str, Any]:
    """Adapt ``load_factory_bn_bundle`` episode dict to the pivot layout."""
    return {
        "name": ep.get("name", ""),
        "resource_ids": list(resource_ids),
        "features": ep["features"],
        "scores": ep["scores"],
        "window_start_s": ep["window_start_s"],
        "windows": ep.get("windows", np.arange(ep["features"].shape[0])),
        "will_bottleneck": ep["will"],
        "mark_node": ep["mark"],
        "cause": ep.get("cause"),
        "event_node": ep.get("event_node"),
        "event_start_s": ep.get("event_start_s"),
        "event_duration_s": ep.get("event_duration_s"),
        "event_start_ti": ep.get("event_start_ti"),
        "jobs_remaining": ep.get("jobs_remaining"),
        "jobs_total": ep.get("jobs_total"),
    }


class BNPredictor:
    """Load ``BNPDFormer_best.pt`` and run ``model.predict`` on 12-window batches."""

    def __init__(self, ckpt_path: str | Path, device: str | torch.device | None = None):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = _load_ckpt(ckpt_path, torch.device("cpu"))
        self.ckpt_path = ckpt_path
        self.epoch = int(ckpt.get("epoch") or 0)
        self.best_score_mae = float(ckpt.get("best_score_mae") or 0.0)
        self.cfg = dict(ckpt["config"])
        self.cfg["device"] = self.device
        meta = ckpt["data_meta"]
        self.resource_ids = [str(x) for x in meta["resource_ids"]]
        self.cause_classes = list(meta.get("cause_classes") or [])
        self.feature_scaler = Scaler(
            mean=_as_numpy(meta["feature_scaler_mean"]).astype(np.float32),
            std=_as_numpy(meta["feature_scaler_std"]).astype(np.float32),
        )
        self.score_scaler = Scaler(
            mean=_as_numpy(meta["score_scaler_mean"]).astype(np.float32),
            std=_as_numpy(meta["score_scaler_std"]).astype(np.float32),
        )
        self.input_window = int(self.cfg.get("input_window", 30))
        self.output_window = int(self.cfg.get("output_window", 1))
        self.horizon_s = float(self.cfg.get("horizon_s", 180))
        self.window_size_s = float(self.cfg.get("window_size_s", 60))
        self.max_hist_events = int(self.cfg.get("max_hist_events", 8))
        self.remain_to_jobs_done = bool(
            self.cfg.get("remain_to_jobs_done", meta.get("remain_to_jobs_done", False))
        )
        self.max_remain_windows = int(
            self.cfg.get("max_remain_windows") or meta.get("max_remain_windows") or 15
        )
        self.hot_eval_threshold = float(self.cfg.get("hot_eval_threshold", 0.55))
        data_feature = _data_feature_from_ckpt(meta)
        self.model = BNPDFormer(self.cfg, data_feature).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _batch_from_sample(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        item = FactoryBNWindowDataset([sample], self.feature_scaler, self.score_scaler)[0]
        return {
            k: v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v for k, v in item.items()
        }

    def _decode(self, out: dict[str, torch.Tensor], sample: dict[str, Any]) -> dict[str, Any]:
        score = self.score_scaler.inverse(_as_numpy(out["score_pred"]))  # (B,K,N,1)
        will_prob = float(_as_numpy(out["will_prob"]).reshape(-1)[0])
        mark_prob = _as_numpy(out["mark_prob"])[0]
        cause_id = int(_as_numpy(out["cause_pred"]).reshape(-1)[0])
        if 0 <= cause_id < len(self.cause_classes):
            cause_name = str(self.cause_classes[cause_id])
        else:
            cause_name = decode_root_cause(cause_id)
        lam = _as_numpy(out["Lam"])[0] if "Lam" in out else None
        dur = _as_numpy(out["dur_event"])[0] if "dur_event" in out else None

        k_use = 1
        remain_len_pred = None
        hot_grid = None
        if self.remain_to_jobs_done and "remain_len_pred" in out:
            remain_len_pred = float(_as_numpy(out["remain_len_pred"]).reshape(-1)[0])
            k_use = int(max(1, min(self.max_remain_windows, round(remain_len_pred))))
            jobs_rem = float(sample.get("jobs_remaining") or 0.0)
            if jobs_rem <= 0:
                k_use = 0
            if "hot_prob" in out:
                hot_grid = _as_numpy(out["hot_prob"])[0, : max(k_use, 1)]
        score_now = score[0, 0, :, 0]
        if k_use > 0:
            score_now = score[0, :k_use, :, 0].mean(axis=0)

        nodes = []
        for i, rid in enumerate(self.resource_ids):
            rec: dict[str, Any] = {
                "resource_id": rid,
                "score_pred": float(score_now[i]),
                "mark_prob": float(mark_prob[i]),
            }
            if lam is not None:
                rec["lam_H"] = float(np.asarray(lam).reshape(-1)[i])
            if dur is not None:
                rec["dur_min"] = float(np.asarray(dur).reshape(-1)[i])
            if hot_grid is not None:
                rec["hot_windows"] = int((hot_grid[:, i] >= self.hot_eval_threshold).sum())
            nodes.append(rec)
        nodes_sorted = sorted(nodes, key=lambda r: r["score_pred"], reverse=True)
        first_future = float(sample.get("window_start_s", 0.0)) + float(self.window_size_s)
        events = []
        if hot_grid is not None and k_use > 0:
            events = occupancy_to_events(
                hot_grid[:k_use],
                resource_ids=self.resource_ids,
                first_future_start_s=first_future,
                window_size_s=self.window_size_s,
                threshold=self.hot_eval_threshold,
            )
        y = sample.get("y_score")
        result: dict[str, Any] = {
            "t": int(sample.get("t", self.input_window)),
            "window_start_s": float(sample.get("window_start_s", 0.0)),
            "will_prob": will_prob,
            "cause_pred": cause_id,
            "cause_name": cause_name,
            "top_resource": nodes_sorted[0]["resource_id"] if nodes_sorted else "",
            "top_score": nodes_sorted[0]["score_pred"] if nodes_sorted else 0.0,
            "nodes": nodes_sorted,
            "remain_len_pred": remain_len_pred,
            "remain_windows": k_use,
            "jobs_remaining": float(sample.get("jobs_remaining") or 0.0),
            "jobs_total": float(sample.get("jobs_total") or 0.0),
            "a1_events": events,
        }
        if sample.get("has_future_score") and y is not None and np.asarray(y).shape[0] >= 1:
            y0 = np.asarray(y)
            n_cmp = min(y0.shape[0], score.shape[1])
            mask = np.asarray(sample.get("remain_mask", np.ones((n_cmp,))), dtype=np.float32)[:n_cmp]
            if mask.sum() > 0:
                err = np.abs(score[0, :n_cmp, :, 0] - y0[:n_cmp, :, 0])
                result["score_mae"] = float((err * mask[:, None]).sum() / max(mask.sum() * err.shape[1], 1.0))
            result["will_true"] = float(sample.get("will", 0.0))
            if sample.get("remain_len") is not None:
                result["remain_len_true"] = float(sample["remain_len"])
        return result

    def predict_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        batch = self._batch_from_sample(sample)
        with torch.no_grad():
            out = self.model.predict(batch)
        return self._decode(out, sample)

    def predict_x(
        self,
        features: np.ndarray,
        window_start_s: np.ndarray,
        *,
        event_node: np.ndarray | None = None,
        event_start_s: np.ndarray | None = None,
        event_duration_s: np.ndarray | None = None,
        event_start_ti: np.ndarray | None = None,
        episode_end_s: float | None = None,
        jobs_remaining: float | None = None,
        jobs_total: float | None = None,
    ) -> dict[str, Any]:
        """Live call: ``features`` is ``(Tin, N, F)`` in checkpoint node order.

        ``Tin`` must equal ``input_window`` (default 30). With remain-to-jobs-done,
        predicts occupancy until remaining jobs finish.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3 or features.shape[0] != self.input_window:
            raise ValueError(
                f"features must be ({self.input_window}, N, F), got {features.shape}"
            )
        if features.shape[1] != len(self.resource_ids):
            raise ValueError(
                f"N={features.shape[1]} != checkpoint nodes {len(self.resource_ids)}"
            )
        sample = build_infer_sample(
            features,
            window_start_s,
            t=self.input_window,
            event_node=event_node,
            event_start_s=event_start_s,
            event_duration_s=event_duration_s,
            event_start_ti=event_start_ti,
            input_window=self.input_window,
            output_window=self.output_window,
            max_hist_events=self.max_hist_events,
            window_size_s=self.window_size_s,
            horizon_s=self.horizon_s,
            episode_end_s=episode_end_s,
            remain_to_jobs_done=self.remain_to_jobs_done,
            max_remain_windows=self.max_remain_windows,
            jobs_remaining=jobs_remaining,
            jobs_total=jobs_total,
        )
        return self.predict_sample(sample)

    def predict_episode(
        self,
        ep: dict[str, Any],
        *,
        at: str | int = "last",
        open_episode: bool = True,
        remain_jobs: float | None = None,
        jobs_total: float | None = None,
    ) -> list[dict[str, Any]]:
        aligned = align_episode(ep, self.resource_ids)
        feats = aligned["features"]
        t_len = feats.shape[0]
        if t_len < self.input_window:
            raise RuntimeError(
                f"episode {aligned.get('name')} has T={t_len} < input_window={self.input_window}"
            )
        if at == "last":
            t_list = [t_len]
        elif at == "all":
            t_list = list(range(self.input_window, t_len)) + [t_len]
        else:
            t_list = [int(at)]
        ep_end = None if open_episode else float(aligned["window_start_s"][-1]) + self.window_size_s
        jobs_series = np.asarray(aligned.get("jobs_remaining"), dtype=np.float32)
        if jobs_series.size != t_len:
            jobs_series = np.zeros((t_len,), dtype=np.float32)
        jobs_tot = float(jobs_total if jobs_total is not None else (aligned.get("jobs_total") or 0.0))
        done_ti = first_done_index(jobs_series) if jobs_series.size else t_len
        rows = []
        for t in t_list:
            li = min(max(t - 1, 0), t_len - 1)
            jobs_now = float(remain_jobs) if remain_jobs is not None else float(jobs_series[li])
            sample = build_infer_sample(
                feats,
                aligned["window_start_s"],
                t=t,
                event_node=aligned["event_node"],
                event_start_s=aligned["event_start_s"],
                event_duration_s=aligned["event_duration_s"],
                event_start_ti=aligned["event_start_ti"],
                scores=aligned["scores"],
                will=aligned["will_bottleneck"],
                mark=aligned["mark_node"],
                cause=aligned["cause"],
                input_window=self.input_window,
                output_window=self.output_window,
                max_hist_events=self.max_hist_events,
                window_size_s=self.window_size_s,
                horizon_s=self.horizon_s,
                episode_end_s=ep_end,
                remain_to_jobs_done=self.remain_to_jobs_done,
                max_remain_windows=self.max_remain_windows,
                jobs_remaining=jobs_now,
                jobs_total=jobs_tot,
                done_ti=done_ti,
            )
            rec = self.predict_sample(sample)
            rec["episode"] = aligned.get("name", "")
            rec["future_window"] = t >= t_len
            rows.append(rec)
        return rows


def _resolve_episode_dir(run_dir: Path, episode: int) -> Path:
    derived = run_dir / "derived" / f"episode_{episode:02d}" / "env_00"
    if derived.is_dir():
        return derived
    raise FileNotFoundError(f"no derived episode at {derived} (run ./batch_bn_agg.sh first)")


def _print_one(rec: dict[str, Any], top_k: int, will_threshold: float) -> None:
    flag = "YES" if rec["will_prob"] >= will_threshold else "no"
    extra = ""
    if rec.get("future_window"):
        extra = (
            "  [until remaining jobs done]"
            if rec.get("remain_len_pred") is not None
            else "  [forecast beyond last observed window]"
        )
    jobs = rec.get("jobs_remaining")
    total = rec.get("jobs_total")
    rhat = rec.get("remain_len_pred")
    rw = rec.get("remain_windows")
    jobs_s = f"  jobs={jobs:.0f}/{total:.0f}" if jobs is not None else ""
    remain_s = ""
    if rhat is not None:
        remain_s = f"  R_hat={rhat:.0f}w (~{float(rw or 0) * 60:.0f}s)"
    print(
        f"t={rec['t']}  t_s={rec['window_start_s']:.0f}  "
        f"will_prob={rec['will_prob']:.3f} ({flag} @ {will_threshold})  "
        f"cause={rec['cause_name'] or rec['cause_pred']}{jobs_s}{remain_s}{extra}"
    )
    if "score_mae" in rec:
        print(f"  score_mae vs remaining occupancy (replay)={rec['score_mae']:.4f}  will_true={rec.get('will_true')}")
        if rec.get("remain_len_true") is not None:
            print(f"  remain_len_true={rec['remain_len_true']:.0f}")
    events = rec.get("a1_events") or []
    if events:
        print(f"  A.1 events until jobs done ({len(events)}):")
        for ev in events[:12]:
            print(
                f"    {ev['start_s']:.0f}-{ev['end_s']:.0f}s  dur={ev['duration_s']:.0f}s  "
                f"{ev['resource_id']}"
            )
        if len(events) > 12:
            print(f"    ... {len(events) - 12} more")
    print(f"  top-{top_k} mean score_pred over remaining windows:")
    for row in rec["nodes"][:top_k]:
        lam = f"  lam_H={row['lam_H']:.3f}" if "lam_H" in row else ""
        hot = f"  hot_win={row['hot_windows']}" if "hot_windows" in row else ""
        print(
            f"    {row['score_pred']:7.4f}  mark={row['mark_prob']:.3f}  "
            f"{row['resource_id']}{hot}{lam}"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "t",
        "window_start_s",
        "future_window",
        "will_prob",
        "cause_name",
        "top_resource",
        "top_score",
        "score_mae",
        "will_true",
    ]
    for i in range(top_k):
        fieldnames += [f"top{i+1}_id", f"top{i+1}_score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rec in rows:
            flat = {
                "episode": rec.get("episode", ""),
                "t": rec["t"],
                "window_start_s": rec["window_start_s"],
                "future_window": int(bool(rec.get("future_window"))),
                "will_prob": rec["will_prob"],
                "cause_name": rec["cause_name"],
                "top_resource": rec["top_resource"],
                "top_score": rec["top_score"],
                "score_mae": rec.get("score_mae", ""),
                "will_true": rec.get("will_true", ""),
            }
            for i, node in enumerate(rec["nodes"][:top_k]):
                flat[f"top{i+1}_id"] = node["resource_id"]
                flat[f"top{i+1}_score"] = node["score_pred"]
            w.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="BNPDFormer online / replay inference")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(_PDFORMER_ROOT / "libcity/cache/model_cache/evt/BNPDFormer_best.pt"),
        help="BNPDFormer_best.pt (must include data_meta / scalers)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: cuda if available)")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--episode_dir", type=str, default=None, help="derived/.../env_00 folder")
    src.add_argument("--run_dir", type=str, default=None, help="bottleneck_dataset/<run> (needs derived/)")
    src.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="exported raw_data/<tag> (episodes.npz); same node set as the ckpt",
    )
    parser.add_argument("--episode", type=int, default=0, help="episode index with --run_dir")
    parser.add_argument("--episode_name", type=str, default=None, help="npz episode key with --data_dir")
    parser.add_argument(
        "--at",
        type=str,
        default="last",
        help="last = forecast after the final observed window; all = replay; or an integer t",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--will_threshold", type=float, default=0.5)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument(
        "--closed",
        action="store_true",
        help="treat episode as finished (phase uses last window end); default is open/live",
    )
    args = parser.parse_args()

    at: str | int = args.at
    if at not in ("last", "all"):
        at = int(at)

    pred = BNPredictor(args.ckpt, device=args.device)
    print(
        f"[infer] ckpt={pred.ckpt_path} epoch={pred.epoch} val_mae={pred.best_score_mae:.4f} "
        f"N={len(pred.resource_ids)} Tin={pred.input_window} H={pred.horizon_s:.0f}s "
        f"remain_jobs={pred.remain_to_jobs_done} Kmax={pred.max_remain_windows} "
        f"device={pred.device}"
    )

    if args.episode_dir:
        ep = load_derived_episode(Path(args.episode_dir), window_size=pred.window_size_s)
    elif args.run_dir:
        ep = load_derived_episode(
            _resolve_episode_dir(Path(args.run_dir), args.episode),
            window_size=pred.window_size_s,
        )
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = (_PDFORMER_ROOT / data_dir).resolve()
        bundle = load_factory_bn_bundle(data_dir)
        if args.episode_name:
            match = [e for e in bundle["episodes"] if e["name"] == args.episode_name]
            if not match:
                names = ", ".join(e["name"] for e in bundle["episodes"][:8])
                raise SystemExit(f"episode {args.episode_name!r} not in {data_dir}; e.g. {names}")
            raw = match[0]
        else:
            raw = bundle["episodes"][0]
            print(f"[infer] using first npz episode {raw['name']}")
        ep = bundle_episode_to_pivot(raw, bundle["resource_ids"])
    else:
        parser.error("pass --run_dir, --episode_dir, or --data_dir")

    rows = pred.predict_episode(ep, at=at, open_episode=not args.closed)
    show = rows if at == "all" else rows[-1:]
    if at == "all":
        print(f"[infer] {len(rows)} windows")
    for rec in show:
        _print_one(rec, top_k=args.top_k, will_threshold=args.will_threshold)

    payload = {
        "ckpt": str(pred.ckpt_path),
        "epoch": pred.epoch,
        "resource_ids": pred.resource_ids,
        "predictions": [
            {k: v for k, v in rec.items() if k != "nodes"}
            | {"top_nodes": rec["nodes"][: args.top_k]}
            for rec in rows
        ],
    }
    if args.out_json:
        _write_json(Path(args.out_json), payload)
        print(f"[infer] wrote {args.out_json}")
    if args.out_csv:
        _write_csv(Path(args.out_csv), rows, top_k=args.top_k)
        print(f"[infer] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
