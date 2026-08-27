"""Sliding-window dataset for multi-episode FactoryBN (PDFormer + STGNPP)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from factory_bn.causes import ROOT_CAUSE_CLASSES
from factory_bn.remain import (
    ensure_labor_saturated_feature,
    first_done_index,
    node_hot_mask,
    occupancy_node_mask,
    pack_remain_target,
)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def _fit_scaler(x: np.ndarray) -> Scaler:
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return Scaler(mean=mean, std=std)


def _pad_events(
    nodes: list[int],
    idxs: list[int],
    durs: list[float],
    taus: list[float],
    n_nodes: int,
    max_events: int,
) -> dict[str, np.ndarray]:
    """Pack variable-length per-node histories into (N, L) tensors."""
    event_idx = np.full((n_nodes, max_events), -1, dtype=np.int64)
    event_dur = np.zeros((n_nodes, max_events), dtype=np.float32)
    event_mask = np.zeros((n_nodes, max_events), dtype=np.float32)
    inter_tau = np.zeros((n_nodes, max_events), dtype=np.float32)
    # group by node
    by_node: dict[int, list[tuple[int, float, float]]] = {}
    for ni, idx, dur, tau in zip(nodes, idxs, durs, taus):
        by_node.setdefault(ni, []).append((idx, dur, tau))
    for ni, seq in by_node.items():
        for j, (idx, dur, tau) in enumerate(seq[:max_events]):
            event_idx[ni, j] = idx
            event_dur[ni, j] = dur
            event_mask[ni, j] = 1.0
            inter_tau[ni, j] = tau
    return {
        "event_idx": event_idx,
        "event_dur": event_dur,
        "event_mask": event_mask,
        "inter_tau": inter_tau,
    }


def _collect_hist_events(
    ev_node: np.ndarray,
    ev_start_s: np.ndarray,
    ev_dur: np.ndarray,
    ev_ti: np.ndarray,
    hist_start: int,
    t: int,
    wstart: np.ndarray,
    n_nodes: int,
    max_hist_events: int,
) -> dict[str, np.ndarray]:
    """Events whose start window lies in ``[hist_start, t)``; tau in minutes."""
    nodes: list[int] = []
    idxs: list[int] = []
    durs: list[float] = []
    taus: list[float] = []
    last_s_by_node: dict[int, float] = {}
    for k in range(len(ev_node)):
        ti_abs = int(ev_ti[k])
        if hist_start <= ti_abs < t:
            ni = int(ev_node[k])
            rel = ti_abs - hist_start
            prev = last_s_by_node.get(ni, float(wstart[hist_start]))
            tau = max(float(ev_start_s[k]) - prev, 0.0) / 60.0
            nodes.append(ni)
            idxs.append(rel)
            durs.append(float(ev_dur[k]) / 60.0)
            taus.append(tau)
            last_s_by_node[ni] = float(ev_start_s[k])
    return _pad_events(nodes, idxs, durs, taus, n_nodes, max_hist_events)


class FactoryBNWindowDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        feature_scaler: Scaler,
        score_scaler: Scaler,
    ):
        self.samples = samples
        self.feature_scaler = feature_scaler
        self.score_scaler = score_scaler

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        x = self.feature_scaler.transform(s["x"])
        y_score = self.score_scaler.transform(s["y_score"])
        item = {
            "X": torch.from_numpy(x),
            "y_score": torch.from_numpy(y_score),
            "will": torch.tensor(s["will"], dtype=torch.float32),
            "mark": torch.tensor(s["mark"], dtype=torch.long),
            "cause": torch.tensor(s["cause"], dtype=torch.long),
            "tts": torch.tensor(s["tts"], dtype=torch.float32),
            "duration": torch.tensor(s["duration"], dtype=torch.float32),
            "episode_id": torch.tensor(s["episode_id"], dtype=torch.long),
            "event_idx": torch.from_numpy(s["event_idx"]),
            "event_dur": torch.from_numpy(s["event_dur"]),
            "event_mask": torch.from_numpy(s["event_mask"]),
            "inter_tau": torch.from_numpy(s["inter_tau"]),
            "next_tau": torch.from_numpy(s["next_tau"]),
            "next_dur": torch.from_numpy(s["next_dur"]),
            "next_mask": torch.from_numpy(s["next_mask"]),
            "surv_mask": torch.from_numpy(s["surv_mask"]),
            "phase": torch.from_numpy(s["phase"]),
        }
        if "remain_mask" in s:
            item["remain_mask"] = torch.from_numpy(np.asarray(s["remain_mask"], dtype=np.float32))
            item["y_hot"] = torch.from_numpy(np.asarray(s["y_hot"], dtype=np.float32))
            item["remain_len"] = torch.tensor(float(s["remain_len"]), dtype=torch.float32)
            item["jobs_remaining"] = torch.tensor(float(s["jobs_remaining"]), dtype=torch.float32)
            item["jobs_total"] = torch.tensor(float(s["jobs_total"]), dtype=torch.float32)
        if "occ_node_mask" in s:
            item["occ_node_mask"] = torch.from_numpy(
                np.asarray(s["occ_node_mask"], dtype=np.float32)
            )
        else:
            item["occ_node_mask"] = torch.from_numpy(occupancy_node_mask(s["x"]))
        item["window_hot"] = torch.tensor(float(s.get("window_hot", 0.0)), dtype=torch.float32)
        item["run_dim_id"] = torch.tensor(int(s.get("run_dim_id", -1)), dtype=torch.long)
        return item


def load_factory_bn_bundle(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    npz = np.load(data_dir / "episodes.npz", allow_pickle=True)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    episode_names = [str(x) for x in npz["episode_names"].tolist()]
    episodes = []
    for i, name in enumerate(episode_names):
        ep = {
            "name": name,
            "episode_id": i,
            "features": ensure_labor_saturated_feature(npz[f"{name}_features"]),
            "scores": npz[f"{name}_scores"],
            "will": npz[f"{name}_will"],
            "mark": npz[f"{name}_mark"],
            "tts": npz[f"{name}_tts"],
            "duration": npz[f"{name}_duration"],
            "windows": npz[f"{name}_windows"],
            "window_start_s": npz[f"{name}_window_start_s"],
        }
        if f"{name}_cause" in npz:
            ep["cause"] = np.asarray(npz[f"{name}_cause"], dtype=np.int64)
        else:
            ep["cause"] = np.full((ep["will"].shape[0],), -1, dtype=np.int64)
        # events optional for backward compat
        if f"{name}_event_node" in npz:
            ep["event_node"] = npz[f"{name}_event_node"]
            ep["event_start_s"] = npz[f"{name}_event_start_s"]
            ep["event_duration_s"] = npz[f"{name}_event_duration_s"]
            ep["event_start_ti"] = npz[f"{name}_event_start_ti"]
        else:
            ep["event_node"] = np.zeros((0,), dtype=np.int64)
            ep["event_start_s"] = np.zeros((0,), dtype=np.float32)
            ep["event_duration_s"] = np.zeros((0,), dtype=np.float32)
            ep["event_start_ti"] = np.zeros((0,), dtype=np.int64)
        if f"{name}_jobs_remaining" in npz:
            ep["jobs_remaining"] = np.asarray(npz[f"{name}_jobs_remaining"], dtype=np.float32)
            tot = np.asarray(npz[f"{name}_jobs_total"]).reshape(-1)
            ep["jobs_total"] = float(tot[0]) if tot.size else 0.0
        else:
            t_len = int(ep["features"].shape[0])
            ep["jobs_remaining"] = np.linspace(float(t_len), 1.0, t_len, dtype=np.float32)
            ep["jobs_total"] = float(t_len)
        episodes.append(ep)
    if "cause_classes" in npz:
        cause_classes = [str(x) for x in npz["cause_classes"].tolist()]
    else:
        cause_classes = list(ROOT_CAUSE_CLASSES)
    return {
        "meta": meta,
        "resource_ids": [str(x) for x in npz["resource_ids"].tolist()],
        "resource_types": [str(x) for x in npz["resource_types"].tolist()],
        "cause_classes": cause_classes,
        "adj_mx": npz["adj_mx"].astype(np.float32),
        "sh_mx": npz["sh_mx"].astype(np.float32),
        "sem_mx": npz["sem_mx"].astype(np.float32),
        "episodes": episodes,
        "window_size_s": float(npz["window_size_s"][0]),
    }


def _build_samples(
    episodes: list[dict[str, Any]],
    input_window: int,
    output_window: int,
    horizon_windows: int,
    max_hist_events: int = 8,
    window_size_s: float = 60.0,
    horizon_s: float = 180.0,
    remain_to_jobs_done: bool = False,
    max_remain_windows: int = 512,
    hot_score_threshold: float = 0.55,
    occupancy_horizon_windows: int | None = None,
    hot_min_windows: int = 2,
    hot_gap_windows: int = 1,
) -> list[dict[str, Any]]:
    """Create causal windows with STGNPP event histories.

    Absolute episode time index ``t`` is the first future step.
    History covers ``[t-input_window, t)``.
    Historical events: start_ti in that range (mapped to relative idx 0..Tin-1).
    Next event: first event with start_ti >= t; tau in minutes from
    ``window_start[t-1]``. Arrival NLL only if that tau is within
    the event horizon; otherwise the node is right-censored at H (survival Λ(H)).

    When ``remain_to_jobs_done``, occupancy y is the next
    ``occupancy_horizon_windows`` steps (A.1 fixed H, padded to
    ``max_remain_windows``). ``remain_len`` is still windows until jobs hit
    zero. Auxiliary ``will`` stays the 180s near-term label.
    """
    samples: list[dict[str, Any]] = []
    for ep in episodes:
        feats = ep["features"]
        scores = ep["scores"]
        will = ep["will"]
        mark = ep["mark"]
        cause = ep.get("cause")
        if cause is None:
            cause = np.full((will.shape[0],), -1, dtype=np.int64)
        tts = ep["tts"]
        duration = ep["duration"]
        wstart = ep["window_start_s"]
        t_len = feats.shape[0]
        n_nodes = feats.shape[1]
        ep_end_s = float(wstart[-1]) + window_size_s if t_len else 1.0
        jobs_rem = np.asarray(
            ep.get("jobs_remaining", np.linspace(t_len, 1, t_len, dtype=np.float32)),
            dtype=np.float32,
        )
        jobs_total = float(ep.get("jobs_total") or (jobs_rem[0] if jobs_rem.size else 0.0))
        done_ti = first_done_index(jobs_rem) if remain_to_jobs_done else t_len
        hot = node_hot_mask(
            feats,
            scores,
            score_threshold=hot_score_threshold,
            window_size_s=window_size_s,
            min_hot_windows=hot_min_windows,
            gap_windows=hot_gap_windows,
        )
        occ_mask = occupancy_node_mask(feats)

        ev_node = ep["event_node"]
        ev_start_s = ep["event_start_s"]
        ev_dur = ep["event_duration_s"]
        ev_ti = ep["event_start_ti"]

        if remain_to_jobs_done:
            t_hi = min(t_len, done_ti)
        else:
            t_hi = t_len - output_window + 1
        for t in range(input_window, t_hi):
            label_idx = t - 1
            hist_start = t - input_window
            obs_jobs = float(jobs_rem[label_idx]) if label_idx < len(jobs_rem) else 0.0
            if remain_to_jobs_done and obs_jobs <= 0:
                continue

            padded = _collect_hist_events(
                ev_node,
                ev_start_s,
                ev_dur,
                ev_ti,
                hist_start,
                t,
                wstart,
                n_nodes,
                max_hist_events,
            )

            if remain_to_jobs_done:
                k_occ = int(occupancy_horizon_windows) if occupancy_horizon_windows else int(max_remain_windows)
                k_occ = max(1, min(int(max_remain_windows), k_occ))
                y_score, y_hot, remain_mask, remain_len = pack_remain_target(
                    scores,
                    hot,
                    t=t,
                    done_ti=done_ti,
                    max_remain_windows=max_remain_windows,
                    occupancy_horizon_windows=k_occ,
                )
                if remain_len <= 0:
                    continue
                event_h_s = max(k_occ * float(window_size_s), float(window_size_s))
            else:
                y_score = scores[t : t + output_window].astype(np.float32)
                y_hot = None
                remain_mask = None
                remain_len = int(output_window)
                event_h_s = float(horizon_s)

            horizon_min = max(event_h_s, 1.0) / 60.0
            next_tau = np.full((n_nodes,), horizon_min, dtype=np.float32)
            next_dur = np.zeros((n_nodes,), dtype=np.float32)
            next_mask = np.zeros((n_nodes,), dtype=np.float32)
            surv_mask = np.ones((n_nodes,), dtype=np.float32)
            ref_s = float(wstart[label_idx])
            seen_next: set[int] = set()
            for k in range(len(ev_node)):
                ti_abs = int(ev_ti[k])
                if ti_abs < t:
                    continue
                ni = int(ev_node[k])
                if ni in seen_next:
                    continue
                seen_next.add(ni)
                tau_min = max(float(ev_start_s[k]) - ref_s, 1e-3) / 60.0
                if tau_min <= horizon_min + 1e-6:
                    next_tau[ni] = tau_min
                    next_dur[ni] = float(ev_dur[k]) / 60.0
                    next_mask[ni] = 1.0
                    surv_mask[ni] = 0.0

            phase = np.array(
                [ref_s / max(ep_end_s, 1.0), (ref_s / 86400.0) % 1.0],
                dtype=np.float32,
            )

            sample = {
                "x": feats[hist_start:t].astype(np.float32),
                "y_score": y_score,
                "will": float(will[label_idx]),
                "mark": int(mark[label_idx]),
                "cause": int(cause[label_idx]),
                "tts": float(tts[label_idx]),
                "duration": float(duration[label_idx]),
                "episode_id": int(ep["episode_id"]),
                "episode_name": str(ep.get("name") or ep["episode_id"]),
                **padded,
                "next_tau": next_tau,
                "next_dur": next_dur,
                "next_mask": next_mask,
                "surv_mask": surv_mask,
                "phase": phase,
            }
            if remain_mask is not None and y_hot is not None:
                sample["remain_mask"] = remain_mask
                sample["y_hot"] = y_hot
                sample["remain_len"] = float(remain_len)
                sample["jobs_remaining"] = obs_jobs
                sample["jobs_total"] = jobs_total
            sample["occ_node_mask"] = occ_mask
            ep_name = str(ep.get("name") or ep["episode_id"])
            sample["run_dim_id"] = run_dim_id(ep_name)
            if label_idx < hot.shape[0]:
                sample["window_hot"] = float((hot[label_idx] * occ_mask).sum() > 0.5)
            else:
                sample["window_hot"] = 0.0
            samples.append(sample)
    return samples


def run_dim_id(episode_name: str) -> int:
    """Map episode / run prefix to a disturbance-dimension id.

    ``machine`` / ``human`` / ``logistics`` / ``material`` / ``none|norm``.
    Unknown names get -1 so they do not share a contrastive class by accident.
    """
    prefix = _run_prefix(episode_name).lower()
    for key, idx in (
        ("logistics", 2),
        ("material", 3),
        ("machine", 0),
        ("human", 1),
        ("none", 4),
        ("norm", 4),
    ):
        if key in prefix:
            return idx
    return -1


def _run_prefix(episode_name: str) -> str:
    """``old_machine2.0__episode_00`` → ``old_machine2.0``; bare names stay as-is."""
    text = str(episode_name)
    return text.split("__", 1)[0] if text else ""


def split_episodes_by_name(
    episode_names: list[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    train_only_contains: list[str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Hold out whole episodes. Stratify by run prefix so each dim stays in all splits.

    Ratios apply **per run** (``name.split('__')[0]``). Train is always disjoint
    from val/test when a run has ≥2 episodes. Val and test are disjoint when a
    run has ≥3 episodes; smaller runs reuse val as test.
    """
    uniq: list[str] = []
    seen: set[str] = set()
    for name in episode_names:
        key = str(name)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for name in uniq:
        groups.setdefault(_run_prefix(name), []).append(name)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for run in sorted(groups):
        group = list(groups[run])
        rng.shuffle(group)
        n = len(group)
        if n == 0:
            continue
        if n == 1:
            train.extend(group)
            continue
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 1
        if n_train + n_val >= n and n > 2:
            n_val = max(1, n - n_train - 1)
        if n == 2:
            n_train, n_val = 1, 1
        n_train = min(n_train, n - 1)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    if not test and val:
        test = list(val)
    elif not test and train:
        test = list(train)
    if not val and test:
        val = list(test)
    elif not val and train:
        val = list(train)
    train_s, val_s, test_s = set(train), set(val), set(test)
    needles = [str(x) for x in (train_only_contains or []) if str(x)]
    if needles:
        def _forced(name: str) -> bool:
            text = str(name)
            return any(n in text for n in needles)

        extra = {x for x in val_s if _forced(x)} | {x for x in test_s if _forced(x)}
        train_s |= extra
        val_s -= extra
        test_s -= extra
        if not val_s and test_s:
            val_s = set(test_s)
        elif not val_s and train_s:
            val_s = set(train_s)
        if not test_s and val_s:
            test_s = set(val_s)
        elif not test_s and train_s:
            test_s = set(train_s)
    return train_s, val_s, test_s


def _count_by_run(names: list[str] | set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        key = _run_prefix(str(name))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_infer_sample(
    features: np.ndarray,
    window_start_s: np.ndarray,
    *,
    t: int,
    event_node: np.ndarray | None = None,
    event_start_s: np.ndarray | None = None,
    event_duration_s: np.ndarray | None = None,
    event_start_ti: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    will: np.ndarray | None = None,
    mark: np.ndarray | None = None,
    cause: np.ndarray | None = None,
    input_window: int = 30,
    output_window: int = 1,
    max_hist_events: int = 8,
    window_size_s: float = 60.0,
    horizon_s: float = 180.0,
    episode_end_s: float | None = None,
    episode_id: int = 0,
    remain_to_jobs_done: bool = False,
    max_remain_windows: int = 512,
    jobs_remaining: float | None = None,
    jobs_total: float | None = None,
    done_ti: int | None = None,
    hot: np.ndarray | None = None,
    hot_min_windows: int = 2,
    hot_gap_windows: int = 1,
) -> dict[str, Any]:
    """Pack one causal window for ``model.predict`` (no future labels required).

    ``t`` is the first future step: ``X = features[t-input_window:t]``.
    Online: observed windows ``0..T-1``, pass ``t=T`` to forecast remaining occupancy.
    Future event labels are unknown, so every node is right-censored at H.
    """
    t_len, n_nodes, _ = features.shape
    if t < input_window:
        raise ValueError(f"t={t} needs at least {input_window} history windows")
    if t > t_len:
        raise ValueError(f"t={t} exceeds available windows T={t_len}")
    hist_start = t - input_window
    label_idx = t - 1
    wstart = np.asarray(window_start_s, dtype=np.float32)
    ev_node = np.asarray(event_node if event_node is not None else [], dtype=np.int64)
    ev_start_s = np.asarray(event_start_s if event_start_s is not None else [], dtype=np.float32)
    ev_dur = np.asarray(
        event_duration_s if event_duration_s is not None else [], dtype=np.float32
    )
    ev_ti = np.asarray(event_start_ti if event_start_ti is not None else [], dtype=np.int64)
    padded = _collect_hist_events(
        ev_node, ev_start_s, ev_dur, ev_ti, hist_start, t, wstart, n_nodes, max_hist_events
    )
    k_out = int(max_remain_windows) if remain_to_jobs_done else int(output_window)
    event_h_s = float(k_out) * float(window_size_s) if remain_to_jobs_done else float(horizon_s)
    horizon_min = max(event_h_s, 1.0) / 60.0
    ref_s = float(wstart[label_idx])
    last_obs_end = float(wstart[-1]) + float(window_size_s)
    ep_end = float(episode_end_s) if episode_end_s is not None else max(last_obs_end, ref_s + event_h_s)
    y_score = np.zeros((k_out, n_nodes, 1), dtype=np.float32)
    y_hot = np.zeros((k_out, n_nodes), dtype=np.float32)
    remain_mask = np.zeros((k_out,), dtype=np.float32)
    remain_len = 0
    has_future = False
    if remain_to_jobs_done and scores is not None and t < t_len:
        hot_arr = (
            hot
            if hot is not None
            else node_hot_mask(
                features,
                scores,
                window_size_s=window_size_s,
                min_hot_windows=hot_min_windows,
                gap_windows=hot_gap_windows,
            )
        )
        end_i = int(done_ti) if done_ti is not None else t_len
        y_score, y_hot, remain_mask, remain_len = pack_remain_target(
            scores, hot_arr, t=t, done_ti=end_i, max_remain_windows=k_out,
            occupancy_horizon_windows=k_out,
        )
        has_future = remain_len > 0
    elif scores is not None and t + output_window <= t_len:
        y_score = np.asarray(scores[t : t + output_window], dtype=np.float32)
        has_future = True
    will_v = 0.0
    mark_v = -1
    cause_v = -1
    if will is not None and label_idx < len(will):
        will_v = float(will[label_idx])
    if mark is not None and label_idx < len(mark):
        mark_v = int(mark[label_idx])
    if cause is not None and label_idx < len(cause):
        cause_v = int(cause[label_idx])
    jobs_rem = 0.0 if jobs_remaining is None else float(jobs_remaining)
    jobs_tot = 1.0 if not jobs_total else float(jobs_total)
    sample = {
        "x": np.asarray(features[hist_start:t], dtype=np.float32),
        "y_score": y_score,
        "will": will_v,
        "mark": mark_v,
        "cause": cause_v,
        "tts": -1.0,
        "duration": -1.0,
        "episode_id": int(episode_id),
        **padded,
        "next_tau": np.full((n_nodes,), horizon_min, dtype=np.float32),
        "next_dur": np.zeros((n_nodes,), dtype=np.float32),
        "next_mask": np.zeros((n_nodes,), dtype=np.float32),
        "surv_mask": np.ones((n_nodes,), dtype=np.float32),
        "phase": np.array(
            [ref_s / max(ep_end, 1.0), (ref_s / 86400.0) % 1.0], dtype=np.float32
        ),
        "t": int(t),
        "window_start_s": ref_s,
        "has_future_score": has_future,
        "occ_node_mask": occupancy_node_mask(features),
        "run_dim_id": -1,
        "window_hot": 0.0,
    }
    if remain_to_jobs_done:
        sample["remain_mask"] = remain_mask
        sample["y_hot"] = y_hot
        sample["remain_len"] = float(remain_len)
        sample["jobs_remaining"] = jobs_rem
        sample["jobs_total"] = jobs_tot
    return sample


def _cause_stats(samples: list[dict[str, Any]], n_classes: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Inverse-frequency class weights, counts, majority class id (-1 if none)."""
    counts = np.zeros(n_classes, dtype=np.float32)
    for s in samples:
        if float(s.get("window_hot", 1.0)) < 0.5:
            continue
        cid = int(s.get("cause", -1))
        if 0 <= cid < n_classes:
            counts[cid] += 1
    weights = np.ones(n_classes, dtype=np.float32)
    pos = counts > 0
    n_pos = int(pos.sum())
    if n_pos > 0:
        weights[pos] = counts.sum() / (n_pos * counts[pos])
        weights[~pos] = 0.0
        majority = int(np.argmax(counts))
    else:
        majority = -1
    return weights, counts, majority


def build_dataloaders(
    data_dir: Path,
    input_window: int = 30,
    output_window: int = 1,
    horizon_s: float = 180.0,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    max_hist_events: int = 8,
    remain_to_jobs_done: bool = True,
    max_remain_windows: int = 512,
    hot_score_threshold: float = 0.55,
    occupancy_horizon_windows: int | None = None,
    hot_min_windows: int = 2,
    hot_gap_windows: int = 1,
    train_only_contains: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    bundle = load_factory_bn_bundle(data_dir)
    window_size = bundle["window_size_s"]
    horizon_windows = max(1, int(round(horizon_s / window_size)))

    samples = _build_samples(
        bundle["episodes"],
        input_window=input_window,
        output_window=output_window,
        horizon_windows=horizon_windows,
        max_hist_events=max_hist_events,
        window_size_s=window_size,
        horizon_s=horizon_s,
        remain_to_jobs_done=remain_to_jobs_done,
        max_remain_windows=max_remain_windows,
        hot_score_threshold=hot_score_threshold,
        occupancy_horizon_windows=occupancy_horizon_windows,
        hot_min_windows=hot_min_windows,
        hot_gap_windows=hot_gap_windows,
    )
    if not samples:
        raise RuntimeError("No training samples; check episode length vs input_window")

    episode_names = [str(ep.get("name") or ep["episode_id"]) for ep in bundle["episodes"]]
    train_eps, val_eps, test_eps = split_episodes_by_name(
        episode_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        train_only_contains=train_only_contains,
    )
    train_samples = [s for s in samples if str(s.get("episode_name")) in train_eps]
    val_samples = [s for s in samples if str(s.get("episode_name")) in val_eps]
    test_samples = [s for s in samples if str(s.get("episode_name")) in test_eps]
    if not train_samples:
        raise RuntimeError("Episode split left train empty; check episode_name on samples")
    if not val_samples:
        val_samples = list(test_samples or train_samples)
    if not test_samples:
        test_samples = list(val_samples)

    x_cat = np.stack([s["x"] for s in train_samples], axis=0)
    if remain_to_jobs_done:
        chunks = [
            s["y_score"][np.asarray(s["remain_mask"]) > 0.5]
            for s in train_samples
            if float(np.asarray(s.get("remain_mask", [0])).sum()) > 0
        ]
        y_cat = np.concatenate(chunks, axis=0) if chunks else np.stack([s["y_score"] for s in train_samples], axis=0)
    else:
        y_cat = np.stack([s["y_score"] for s in train_samples], axis=0)
    feature_scaler = _fit_scaler(x_cat)
    score_scaler = _fit_scaler(y_cat)
    cause_w, cause_counts, cause_majority = _cause_stats(
        train_samples, len(bundle["cause_classes"])
    )

    train_ds = FactoryBNWindowDataset(train_samples, feature_scaler, score_scaler)
    val_ds = FactoryBNWindowDataset(val_samples, feature_scaler, score_scaler)
    test_ds = FactoryBNWindowDataset(test_samples, feature_scaler, score_scaler)

    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    data_feature = {
        "num_nodes": bundle["adj_mx"].shape[0],
        "feature_dim": train_samples[0]["x"].shape[-1],
        "output_dim": 1,
        "adj_mx": bundle["adj_mx"],
        "sh_mx": bundle["sh_mx"],
        "sem_mx": bundle["sem_mx"],
        "resource_ids": bundle["resource_ids"],
        "resource_types": bundle["resource_types"],
        "feature_scaler": feature_scaler,
        "score_scaler": score_scaler,
        "meta": bundle["meta"],
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "split_by": "episode",
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_test_episodes": len(test_eps),
        "train_episodes": sorted(train_eps),
        "val_episodes": sorted(val_eps),
        "test_episodes": sorted(test_eps),
        "train_episodes_by_run": _count_by_run(train_eps),
        "val_episodes_by_run": _count_by_run(val_eps),
        "test_episodes_by_run": _count_by_run(test_eps),
        "window_size_s": window_size,
        "horizon_windows": horizon_windows,
        "horizon_s": float(horizon_s),
        "train_feature_windows": x_cat,
        "max_hist_events": max_hist_events,
        "n_event_positive_train": int(sum(s["next_mask"].sum() for s in train_samples)),
        "n_event_surv_train": int(sum(s["surv_mask"].sum() for s in train_samples)),
        "cause_classes": bundle["cause_classes"],
        "n_cause_classes": len(bundle["cause_classes"]),
        "cause_class_weight": cause_w,
        "cause_train_counts": cause_counts,
        "cause_majority": cause_majority,
        "n_cause_labeled_train": int(sum(1 for s in train_samples if int(s.get("cause", -1)) >= 0)),
        "remain_to_jobs_done": bool(remain_to_jobs_done),
        "max_remain_windows": int(max_remain_windows),
        "occupancy_horizon_windows": int(
            occupancy_horizon_windows if occupancy_horizon_windows else max_remain_windows
        ),
        "hot_score_threshold": float(hot_score_threshold),
    }
    return (*loaders, data_feature)


def make_pattern_keys(
    train_feature_windows: np.ndarray,
    s_attn_size: int,
    n_cluster: int = 16,
    output_channel: int = 4,
) -> np.ndarray:
    b, t, n, f = train_feature_windows.shape
    ch = min(output_channel, f - 1)
    if t < s_attn_size:
        s_attn_size = t
    patches = []
    for i in range(b):
        for node in range(n):
            for start in range(0, t - s_attn_size + 1):
                patches.append(train_feature_windows[i, start : start + s_attn_size, node, ch])
    patches = np.asarray(patches, dtype=np.float32)
    if patches.shape[0] == 0:
        return np.zeros((n_cluster, s_attn_size, 1), dtype=np.float32)

    rng = np.random.default_rng(0)
    n_cluster = min(n_cluster, patches.shape[0])
    centers = patches[rng.choice(patches.shape[0], size=n_cluster, replace=False)].copy()
    for _ in range(10):
        d = ((patches[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(axis=1)
        for k in range(n_cluster):
            mask = assign == k
            if mask.any():
                centers[k] = patches[mask].mean(axis=0)
    return centers[:, :, None].astype(np.float32)
