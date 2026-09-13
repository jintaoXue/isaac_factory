"""Fit preprocessing on diagnostic fitting episodes only, without changing frozen files."""

import numpy as np
import torch
from factory_baselines.precursor import pack_history_precursor


def validate_views(payload, views):
    if set(views) != {'fit', 'heldout', 'original_validation'}:
        raise ValueError('Expected exactly three diagnostic views')
    sets = {k: set(map(int, v)) for k, v in views.items()}
    if any(not v or len(v) != len(views[k]) for k, v in sets.items()):
        raise ValueError('Empty or repeated sample indices')
    original = {k: set(v.tolist()) for k, v in payload['split_indices'].items()}
    if sets['fit'] & sets['heldout'] or sets['fit'] | sets['heldout'] != original['train']:
        raise ValueError('Fitting and heldout must partition original training samples')
    if sets['original_validation'] != original['validation']:
        raise ValueError('Original validation must be unchanged')
    if any(v & original['test'] for v in sets.values()):
        raise ValueError('Test samples are not allowed')
    groups = {k: set(payload['sample_group_id'][sorted(v)].tolist()) for k, v in sets.items()}
    names = list(groups)
    if any(groups[names[i]] & groups[names[j]] for i in range(3) for j in range(i)):
        raise ValueError('An episode crosses fitting/evaluation views')


def _check_shapes(payload, node_applicability, old_normalization):
    x = payload['x']
    if x.ndim != 4 or x.shape[1] != 30 or x.shape[-1] != 27:
        raise ValueError('Expected frozen thirty-window, 27-channel histories')
    if payload['global_features'].shape[-1] != 0:
        raise ValueError('This diagnostic expects the fixed zero-global-channel dataset')
    if node_applicability.shape != (x.shape[2], 21) or node_applicability.dtype != torch.bool:
        raise ValueError('Invalid fixed node applicability')
    if payload['observation_mask'].shape != x.shape[:3] or payload['node_mask'].shape != (len(x), x.shape[2]):
        raise ValueError('Invalid observation or node masks')
    mean, std = (torch.as_tensor(old_normalization[k], dtype=torch.float64) for k in ('feature_mean', 'feature_std'))
    if mean.shape != (21,) or std.shape != (21,) or not torch.isfinite(mean).all() or not (torch.isfinite(std) & (std > 0)).all():
        raise ValueError('Invalid frozen inverse normalization')
    return mean, std


def _raw_and_mask(payload, ids, node_applicability, old_mean, old_std):
    raw = payload['x'][ids, ..., :21].double() * old_std + old_mean
    valid = (payload['observation_mask'][ids].bool()[..., None]
             & payload['node_mask'][ids].bool()[:, None, :, None]
             & node_applicability[None, None])
    if not torch.isfinite(raw[valid]).all(): raise ValueError('Non-finite observed raw input')
    return raw, valid


def fit_normalization(payload, fit_indices, node_applicability, old_normalization, block_size=32):
    """The old affine transform is inverted, then only fitting observations are used.

    No label, heldout input, validation input, or test input contributes to these
    statistics. Frozen float32 inversion can introduce ordinary rounding error.
    """
    old_mean, old_std = _check_shapes(payload, node_applicability, old_normalization)
    if not fit_indices or len(set(fit_indices)) != len(fit_indices): raise ValueError('Invalid fitting indices')
    if not set(fit_indices) <= set(payload['split_indices']['train'].tolist()):
        raise ValueError('Only original training samples may fit normalization')
    count = torch.zeros(21, dtype=torch.float64)
    mean = torch.zeros_like(count); m2 = torch.zeros_like(count)
    # Parallel variance merging avoids cancellation for large constant channels.
    for offset in range(0, len(fit_indices), block_size):
        ids = fit_indices[offset:offset + block_size]
        raw, valid = _raw_and_mask(payload, ids, node_applicability, old_mean, old_std)
        n = valid.sum((0, 1, 2)).double()
        batch_mean = raw.masked_fill(~valid, 0).sum((0, 1, 2)) / n.clamp_min(1)
        batch_m2 = ((raw - batch_mean).masked_fill(~valid, 0).square()).sum((0, 1, 2))
        combined = count + n; delta = batch_mean - mean
        mean += delta * n / combined.clamp_min(1)
        m2 += batch_m2 + delta.square() * count * n / combined.clamp_min(1)
        count = combined
    std = (m2 / count.clamp_min(1)).clamp_min(0).sqrt()
    mean = torch.where(count > 0, mean, torch.zeros_like(mean))
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    return dict(feature_mean=mean.tolist(), feature_std=std.tolist(), observed_count=count.long().tolist(),
                fit_sample_indices=list(map(int, fit_indices)), fit_scope='retained_fitting_episodes_only',
                original_affine_inverted=True, labels_used=False)


def prepare_inputs_in_memory(payload, views, node_applicability, old_normalization, block_size=32):
    """Mutate only fresh, in-memory train/validation x; never save dataset tensors.

    Near summaries are rebuilt from the same re-normalized encoder input with
    fitting-only imputation means. This removes the old full-train mean from
    missing observations too. Original test x is left untouched and has no near
    features attached. All original split indices and targets stay unchanged.
    """
    validate_views(payload, views)
    old_mean, old_std = _check_shapes(payload, node_applicability, old_normalization)
    norm = fit_normalization(payload, views['fit'], node_applicability, old_normalization, block_size)
    new_mean, new_std = (torch.tensor(norm[k], dtype=torch.float64) for k in ('feature_mean', 'feature_std'))
    ids_all = sorted(i for values in views.values() for i in values)
    precursor = torch.zeros(len(payload['x']), payload['x'].shape[2], 23, dtype=torch.float32)
    precursor_valid = torch.zeros(len(payload['x']), dtype=torch.bool)
    for offset in range(0, len(ids_all), block_size):
        ids = ids_all[offset:offset + block_size]
        raw, valid = _raw_and_mask(payload, ids, node_applicability, old_mean, old_std)
        normalized = ((raw - new_mean) / new_std).masked_fill(~valid, 0).float()
        payload['x'][ids, ..., :21] = normalized
        # Follow the existing near adapter's float32 inverse and masking policy.
        for i in ids:
            near = payload['x'][i].numpy().copy()
            near[..., :21] = near[..., :21] * np.asarray(norm['feature_std'], dtype=np.float32) + np.asarray(norm['feature_mean'], dtype=np.float32)
            precursor[i] = torch.from_numpy(pack_history_precursor(near, 30, 'near')) * payload['node_mask'][i, :, None]
            precursor_valid[i] = True
    if set(torch.where(precursor_valid)[0].tolist()) != set(ids_all):
        raise ValueError('Derived feature coverage differs from diagnostic views')
    return {**payload, 'event_precursor': precursor, 'event_precursor_valid': precursor_valid}, norm
