"""Pure in-memory safeguards for whole-episode diagnostic exclusion."""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from prepare_baseline_episode_holdout import plan_episode_holdout, summarize_cached_labels


def reject(function, *args):
    try: function(*args)
    except ValueError: return
    raise AssertionError("Invalid input was accepted")


def rows():
    return [dict(group_id=f'{run}:env_00:episode_{ep:02d}', sample_index=str(i * 2 + j),
                 split='train' if ep < 6 else 'validation', ignored_label=0)
            for i, (run, ep) in enumerate((r, e) for r in ['runA', 'runB'] for e in range(7))
            for j in range(2)]


def test_partition_ignores_labels_and_row_order_and_keeps_episodes_whole():
    data = rows(); plan = plan_episode_holdout(data)
    altered = [dict(r, ignored_label=1000 - i) for i, r in enumerate(reversed(data))]
    assert plan == plan_episode_holdout(altered)
    for group in {r['group_id'] for r in data}:
        ids = {int(r['sample_index']) for r in data if r['group_id'] == group}
        assert sum(ids <= set(v) for v in plan['sample_indices'].values()) == 1
    assert plan['sample_indices']['original_validation'] == [12, 13, 26, 27]
    assert [r['held_out_episodes'] for r in plan['raw_run_strata']] == [1, 1]


def test_partition_rejects_test_cross_split_duplicate_and_singleton():
    data = rows()
    for malformed in [data + [data[0]], [dict(data[0], split='test')] + data[1:],
                      data[:-1] + [dict(data[-1], group_id=data[0]['group_id'])],
                      [data[0], data[-1]]]:
        reject(plan_episode_holdout, malformed)


def test_label_counts_only_describe_the_frozen_partition():
    plan = plan_episode_holdout(rows()); samples = np.arange(28); nodes = np.zeros(28, dtype=int)
    labels = samples % 3
    before = repr(plan); result = summarize_cached_labels(samples, nodes, labels, plan)
    assert sum(r['samples'] for r in result.values()) == 28
    assert [sum(r[k] for r in result.values()) for k in ('negative', 'ongoing', 'upcoming')] == [10, 9, 9]
    changed = summarize_cached_labels(samples, nodes, 2 - labels, plan)
    assert repr(plan) == before and changed != result


def test_cache_coverage_and_node_identity_are_required():
    plan = plan_episode_holdout(rows()); samples = np.arange(28); nodes = np.zeros(28, dtype=int); labels = samples % 3
    for s, n, y in [(samples[:-1], nodes[:-1], labels[:-1]),
                     (np.r_[samples, 0], np.r_[nodes, 0], np.r_[labels, 0]),
                     (samples, nodes, np.full(28, 3))]:
        reject(summarize_cached_labels, s, n, y, plan)
