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
        return {
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
            "features": npz[f"{name}_features"],
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
) -> list[dict[str, Any]]:
    """Create causal windows with STGNPP event histories.

    Absolute episode time index ``t`` is the first future step.
    History covers ``[t-input_window, t)``.
    Historical events: start_ti in that range (mapped to relative idx 0..Tin-1).
    Next event: first event with start_ti >= t; tau in minutes from
    ``window_start[t-1]``. Arrival NLL only if that tau is within
    ``horizon_s``; otherwise the node is right-censored at H (survival Λ(H)).
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

        ev_node = ep["event_node"]
        ev_start_s = ep["event_start_s"]
        ev_dur = ep["event_duration_s"]
        ev_ti = ep["event_start_ti"]

        for t in range(input_window, t_len - output_window + 1):
            label_idx = t - 1
            hist_start = t - input_window

            # collect historical events inside the input window
            nodes, idxs, durs, taus = [], [], [], []
            last_s_by_node: dict[int, float] = {}
            for k in range(len(ev_node)):
                ti_abs = int(ev_ti[k])
                if hist_start <= ti_abs < t:
                    ni = int(ev_node[k])
                    rel = ti_abs - hist_start
                    prev = last_s_by_node.get(ni, float(wstart[hist_start]))
                    tau = max(float(ev_start_s[k]) - prev, 0.0) / 60.0  # minutes
                    nodes.append(ni)
                    idxs.append(rel)
                    durs.append(float(ev_dur[k]) / 60.0)
                    taus.append(tau)
                    last_s_by_node[ni] = float(ev_start_s[k])

            padded = _pad_events(nodes, idxs, durs, taus, n_nodes, max_hist_events)

            # Next event per node after label time, censored at the will horizon.
            # tau / duration are in minutes to match STGNPP intensity units.
            horizon_min = max(float(horizon_s), 1.0) / 60.0
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

            # episode phase proxies for periodic gate: [time_frac, day_frac≈0]
            phase = np.array(
                [ref_s / max(ep_end_s, 1.0), (ref_s / 86400.0) % 1.0],
                dtype=np.float32,
            )

            samples.append(
                {
                    "x": feats[hist_start:t].astype(np.float32),
                    "y_score": scores[t : t + output_window].astype(np.float32),
                    "will": float(will[label_idx]),
                    "mark": int(mark[label_idx]),
                    "cause": int(cause[label_idx]),
                    "tts": float(tts[label_idx]),
                    "duration": float(duration[label_idx]),
                    "episode_id": int(ep["episode_id"]),
                    **padded,
                    "next_tau": next_tau,
                    "next_dur": next_dur,
                    "next_mask": next_mask,
                    "surv_mask": surv_mask,
                    "phase": phase,
                }
            )
    return samples


def _cause_stats(samples: list[dict[str, Any]], n_classes: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Inverse-frequency class weights, counts, majority class id (-1 if none)."""
    counts = np.zeros(n_classes, dtype=np.float32)
    for s in samples:
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
    input_window: int = 12,
    output_window: int = 1,
    horizon_s: float = 180.0,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    max_hist_events: int = 8,
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
    )
    if not samples:
        raise RuntimeError("No training samples; check episode length vs input_window")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n = len(indices)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1) if n > 2 else 0
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    if len(test_idx) == 0:
        test_idx = val_idx

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    x_cat = np.stack([s["x"] for s in train_samples], axis=0)
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
