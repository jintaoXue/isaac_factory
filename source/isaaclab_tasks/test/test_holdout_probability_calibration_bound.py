from itertools import combinations, product
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'))
from bound_holdout_probability_calibration import calibration_bound, minimum_ap_for_hits
from factory_baselines.metrics import _binary_metrics


def test_exhaustive_rank_orders_thresholds_and_timing_errors_cannot_exceed_bound():
    # All 210 orders of three upcoming, two ongoing and two negative targets,
    # all 32 patterns of positive-target timing success, all seven thresholds.
    for up_indices in combinations(range(7), 3):
        rest = set(range(7)) - set(up_indices)
        for ongoing_indices in combinations(sorted(rest), 2):
            up = np.isin(np.arange(7), up_indices); ongoing = np.isin(np.arange(7), ongoing_indices)
            eligible = ~ongoing; scores = np.arange(7, 0, -1)
            ap = _binary_metrics(up[eligible].astype(int), scores[eligible])['pr_auc']; bound = calibration_bound(ap, 3, 2)['maximum_possible_upcoming_hits']
            targets = np.flatnonzero(up | ongoing)
            for success in product((0, 1), repeat=5):
                hit = np.zeros(7, dtype=int); hit[targets] = success
                for n in range(1, 8):
                    tp = int(hit[:n].sum())
                    if tp >= 4 * (n - tp): assert int((hit[:n] * up[:n]).sum()) <= bound


def test_score_ties_and_ongoing_precision_cushion_are_valid():
    labels = np.array([1, 0, 1, 0, 1]); scores = np.ones(5)
    ap = _binary_metrics(labels, scores)['pr_auc']; r = calibration_bound(ap, 3, 10)
    assert r['maximum_possible_upcoming_hits'] == 3
    assert minimum_ap_for_hits(3, 10, 3) <= ap


def test_low_ap_rules_out_half_recall_even_granting_perfect_ongoing():
    r = calibration_bound(.0078973105, 145, 950)
    assert r['half_recall_ruled_out'] and r['maximum_possible_upcoming_hits'] < 73
    assert r['half_recall_ap_necessary_lower_bound'] > .05
    assert calibration_bound(1., 145, 950)['maximum_possible_upcoming_hits'] == 145


def test_invalid_counts_and_ap_are_rejected():
    for args in [(float('nan'), 145, 950), (-.1, 145, 950), (.1, 0, 10), (.1, 2, -1), (.1, 2.5, 3)]:
        try: calibration_bound(*args)
        except ValueError: continue
        raise AssertionError('Invalid bound inputs accepted')
