"""Global temporal associations preserve cutoff and local-audit semantics."""
from pathlib import Path
import sys
import pytest
TOOLS = Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'
sys.path.insert(0, str(TOOLS))
from audit_factorywide_upcoming_starts import classify, summarize
from diagnose_upcoming_schedule_support import match_future_runtime


def target(start=2):
    return dict(group_id='episode_a', resource_id='station_a', onset_window_index=10,
        start_index=start, first_future_time_s=(10-start)*60, onset_window_end_s=660)


def event(name, start, node='station_a', end=None):
    return dict(event_id=name, start=start, end=start+180 if end is None else end,
        target=node, type='machine_failure', right_censored=False)


def test_half_open_boundaries_and_active_history_are_distinct():
    events = [event('active',479,end=481),event('begin',480),event('remote',599,'station_b'),
        event('within',600,'station_b'),event('excluded_end',660),event('ended',200,end=480)]
    result = classify(target(),events)
    assert result['factory_start_class']=='local_and_other_resources'
    assert result['local_future_event_ids']==['begin']
    assert result['other_resource_future_event_ids']==['remote','within']
    assert result['before_onset_window_event_ids']==['begin','remote']
    assert result['within_onset_window_event_ids']==['within']
    assert result['already_active_before_cutoff_event_ids']==['active']


def test_extension_preserves_old_local_join():
    for start in (1,2):
        for events in ([],[event('other',610,'station_b')],[event('local',500)],
                       [event('local',540),event('other',600,'station_b')]):
            t=target(start); result=classify(t,events)
            old=match_future_runtime(t,{'start_index':start},events)
            assert result['local_future_event_ids']==sorted(old['matched_runtime_event_ids'])
            assert bool(result['local_future_event_ids'])==old['matched_future_local_runtime_start']


def test_two_anchors_can_have_different_classes_for_one_onset():
    events=[event('early_remote',500,'station_b')]
    rows=[{**target(s),**classify(target(s),events)} for s in (1,2)]
    summary=summarize(rows)
    assert summary['total']['window_targets']==2 and summary['total']['unique_onsets']==1
    assert summary['by_start_class']['other_resource_only']['unique_onsets']==1
    assert summary['by_start_class']['no_recorded_factory_start']['unique_onsets']==1
    assert sum(v['window_targets'] for v in summary['by_start_class'].values())==2


def test_invalid_boundaries_duplicate_ids_and_unknown_types_rejected():
    with pytest.raises(ValueError): classify(dict(target(),first_future_time_s=479),[])
    with pytest.raises(ValueError): classify(target(),[event('same',500),event('same',600)])
    with pytest.raises(ValueError): classify(target(),[dict(event('x',500),type='unrecorded_quality_hold')])
    with pytest.raises(ValueError): classify(target(),[event('zero',500,end=500)])
