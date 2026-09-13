"""Exclude heldout labels and input statistics, preserving the frozen test data."""
from copy import deepcopy
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'))
from baseline_episode_holdout_inputs import validate_views, fit_normalization, prepare_inputs_in_memory


def fixture(mean=2., std=3.):
    raw = torch.arange(5 * 30 * 2 * 21, dtype=torch.float32).reshape(5, 30, 2, 21) / 100
    raw[..., 1] = 7.
    x = torch.cat(((raw - mean) / std, torch.ones(5, 30, 2, 6)), -1)
    observed = torch.ones(5, 30, 2, dtype=torch.bool); observed[0, :3, 0] = False
    node = torch.ones(5, 2, dtype=torch.bool); node[1, 1] = False
    applicable = torch.ones(2, 21, dtype=torch.bool); applicable[1, 2] = False
    valid = observed[..., None] & node[:, None, :, None] & applicable[None, None]
    x[..., :21].masked_fill_(~valid, 0)
    p = dict(x=x, observation_mask=observed, node_mask=node, global_features=torch.empty(5, 30, 0),
             split_indices=dict(train=torch.tensor([0, 1, 2]), validation=torch.tensor([3]), test=torch.tensor([4])),
             sample_group_id=torch.tensor([10, 11, 12, 13, 14]), y_cause=torch.arange(5))
    views = dict(fit=[0, 1], heldout=[2], original_validation=[3])
    old = dict(feature_mean=[mean] * 21, feature_std=[std] * 21)
    return p, views, applicable, old


def rejects(fn, *args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError('Invalid diagnostic input accepted')


def test_excluded_inputs_and_all_labels_do_not_change_fitted_statistics_or_fit_features():
    p, v, a, n = fixture(); q = deepcopy(p)
    q['x'][2:, ..., :21] += 1000
    q['y_cause'] += 900
    one, norm = prepare_inputs_in_memory(p, v, a, n)
    two, changed = prepare_inputs_in_memory(q, v, a, n)
    assert norm == changed
    assert torch.equal(one['x'][v['fit']], two['x'][v['fit']])
    assert torch.equal(one['event_precursor'][v['fit']], two['event_precursor'][v['fit']])


def test_affine_inverse_does_not_retain_full_train_normalization_and_missing_values_stay_masked():
    p, v, a, n = fixture(); q, _, _, other = fixture(-20., 5.)
    one, norm = prepare_inputs_in_memory(p, v, a, n)
    two, norm2 = prepare_inputs_in_memory(q, v, a, other)
    torch.testing.assert_close(torch.tensor(norm['feature_mean']), torch.tensor(norm2['feature_mean']), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(one['x'][:4], two['x'][:4], rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(one['event_precursor'][:4], two['event_precursor'][:4], rtol=1e-5, atol=2e-6)
    assert torch.count_nonzero(one['x'][0, :3, 0, :21]) == 0
    assert torch.count_nonzero(one['x'][1, :, 1, :21]) == 0
    assert norm['feature_std'][1] == 1.


def test_test_inputs_splits_and_targets_remain_unchanged_and_only_allowed_views_get_features():
    p, v, a, n = fixture(); old = deepcopy(p)
    out, _ = prepare_inputs_in_memory(p, v, a, n)
    assert torch.equal(out['x'][4], old['x'][4])
    assert torch.equal(out['x'][..., 21:], old['x'][..., 21:])
    for name in ['node_mask', 'observation_mask', 'sample_group_id', 'y_cause']:
        assert torch.equal(out[name], old[name])
    for k in old['split_indices']: assert torch.equal(out['split_indices'][k], old['split_indices'][k])
    assert out['event_precursor_valid'].tolist() == [True, True, True, True, False]
    assert torch.count_nonzero(out['event_precursor'][4]) == 0


def test_views_reject_episode_overlap_test_access_and_validation_changes():
    p, v, a, n = fixture()
    bad = deepcopy(p); bad['sample_group_id'][2] = 10; rejects(validate_views, bad, v)
    rejects(validate_views, p, dict(v, fit=[0, 1, 4]))
    rejects(validate_views, p, dict(v, original_validation=[2, 3]))
    rejects(fit_normalization, p, [0, 3], a, n)
