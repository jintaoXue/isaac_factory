"""Check tied AP, whole-episode sampling and missing-positive uncertainty."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'))
from analyze_baseline_episode_holdout import EpisodeRanking, stratified_episode_weights, interval, difference, summarize_cache
from factory_baselines.metrics import _binary_metrics


def test_weighted_tied_ap_matches_materialized_episode_resampling():
    labels = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([.9, .9, .8, .5, .8, .5])
    episode = np.array([0, 1, 1, 2, 2, 3])
    rank = EpisodeRanking(labels, scores, episode, 4)
    for weights in (np.ones(4, dtype=int), np.array([0, 2, 1, 3]), np.array([0, 0, 4, 0]), np.zeros(4, dtype=int)):
        indices = np.repeat(np.arange(len(labels)), weights[episode])
        actual = rank.ap(weights)
        expected = _binary_metrics(labels[indices], scores[indices])['pr_auc']
        if expected is None: assert actual is None
        else: assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_stratified_draws_preserve_every_raw_run_episode_count_and_repeat_exactly():
    groups = ['a:env_0:episode_0', 'a:env_1:episode_0', 'b:env_0:episode_0']
    weights = stratified_episode_weights(groups)
    assert np.array_equal(weights, stratified_episode_weights(groups))
    assert np.all(weights[:, :2].sum(1) == 2) and np.all(weights[:, 2] == 1)
    assert np.any(weights[:, 0] == 0) and np.any(weights[:, 0] == 2)


def test_undefined_positive_replicates_are_counted_and_never_treated_as_zero():
    draws = difference([None, .4, .8], [.2, .1, None])
    result = interval(draws)
    assert result['valid_replicates'] == 1 and result['total_replicates'] == 3
    assert np.allclose(result['percentile_95'], [.3, .3])
    assert interval([None])['percentile_95'] is None


def test_cache_reading_excludes_ongoing_and_inactive_nodes_and_counts_timing_miss():
    groups = ['a:env_0:episode_0', 'a:env_1:episode_0']
    arrays = dict(sample_index=np.array([9, 3]), occ_node_mask=np.array([[1, 1, 0], [1, 1, 1]]),
        event_will=np.array([[1, 1, 1], [1, 0, 0]]), event_start=np.array([[1, 0, 1], [2, 0, 0]]),
        will_probability=np.array([[.8, .99, 1.], [.9, .7, .1]]),
        hist_last_hot=np.zeros((2, 3)), predicted_start=np.array([[1, 0, 1], [7, 0, 0]]))
    report = dict(ranking={'upcoming_vs_negative': {'tie_aware_average_precision': 1.}},
                  thresholds=[{'n_matched_who_upcoming': 1}])
    result, draws = summarize_cache(arrays, groups, {9: groups[0], 3: groups[1]}, np.array([[1, 1], [2, 0]]), report)
    assert result['upcoming'] == 2 and result['hits'] == 1 and result['recall'] == .5
    assert draws == {'ap': [1., 1.], 'recall': [.5, 1.]}
    assert [r['hits'] for r in result['episodes']] == [1, 0]
