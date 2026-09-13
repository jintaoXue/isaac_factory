"""Keep station matches distinct from timing-qualified upcoming report hits."""
from pathlib import Path
import sys
import pytest
TOOLS = Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'
sys.path.insert(0, str(TOOLS))
from compare_baseline_gru_capacity import view_stats, parent_filename


def example():
    row = dict(threshold=.7, n_true_upcoming=10, report_recall_upcoming=.2,
        n_matched_who_upcoming=5, upcoming_timing_misses=3,
        report_precision=.8, report_recall=.6, report_f1=.68)
    return dict(thresholds=[dict(row, threshold=.5, report_recall_upcoming=.5, upcoming_timing_misses=0), row],
        saved_report_threshold=.7, groups={'upcoming': {'count': 10, 'will_q10_q50_q90': [.01, .1, .9]}},
        ranking={'upcoming_vs_negative': {'tie_aware_average_precision': .12}})


def test_saved_threshold_and_timing_qualified_hits():
    result = view_stats(example())
    assert result['upcoming_report_hits'] == 2 and result['upcoming_station_hits'] == 5
    assert result['upcoming_timing_misses'] == 3 and result['saved_threshold'] == .7


def test_inconsistent_or_ambiguous_cached_counts_are_rejected():
    data = example(); data['thresholds'][1]['report_recall_upcoming'] = .5
    with pytest.raises(AssertionError): view_stats(data)
    data = example(); data['thresholds'].append(dict(data['thresholds'][1]))
    with pytest.raises(ValueError): view_stats(data)


def test_each_parent_view_maps_to_distinct_preexisting_filename():
    names = {parent_filename(m, s, c, v) for m in ('b4', 'b5') for s in (42, 43)
        for c in ('best', 'last') for v in ('train', 'validation')}
    assert len(names) == 16
    assert 'b4_seed42_train_diagnostics_nearpreclast20260912.json' in names
    assert 'baseline_near_parent_b5s43_last_validation20260913.json' in names
