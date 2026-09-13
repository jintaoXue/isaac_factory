from pathlib import Path
import sys
import numpy as np
import pytest
TOOLS=Path(__file__).resolve().parents[1]/'isaaclab_tasks/direct/hc_factory/tools'
sys.path.insert(0,str(TOOLS))
from analyze_holdout_factory_starts import join_predictions,summary


def example():
    arrays=dict(sample_index=np.array([10,11]),occ_node_mask=np.ones((2,2)),
        event_will=np.array([[1,1],[1,0]]),event_start=np.array([[2,1],[1,0]]),
        will_probability=np.array([[.8,.1],[.7,.9]],dtype=np.float32),
        hist_last_hot=np.zeros((2,2)),predicted_start=np.array([[7.9,1],[4.9,0]]))
    samples={10:dict(group_id='ep',first_future_start_s=480),11:dict(group_id='ep',first_future_start_s=540)}
    targets=[dict(group_id='ep',resource_id=node,onset_window_index=onset,start_index=start,first_future_time_s=first,
        factory_start_class=kind,already_active_before_cutoff_event_ids=[]) for node,onset,start,first,kind in
        [('A',10,2,480,'no_recorded_factory_start'),('B',9,1,480,'other_resource_only'),('A',10,1,540,'no_recorded_factory_start')]]
    return arrays,samples,['A','B'],targets


def test_float32_threshold_and_integer_start_follow_canonical_report():
    rows=join_predictions(*example());s=summary(rows)
    assert s['report_hits']==1 and s['probability_misses']==1 and s['timing_misses']==1
    assert rows[2]['report_hit'] and rows[2]['decoded_start']==4
    assert s['unique_onsets']==2 and s['window_targets']==3


def test_hot_history_override_and_zero_group():
    args=example();args[0]['hist_last_hot'][0,0]=1
    assert summary(join_predictions(*args))['report_hits']==2
    assert summary([])['recall'] is None and summary([])['probability_q10_q50_q90'] is None


def test_incomplete_or_duplicate_identity_is_rejected():
    args=example()
    with pytest.raises(ValueError):join_predictions(*args[:3],args[3][:-1])
    args=example();args[0]['sample_index'][1]=10
    with pytest.raises(ValueError):join_predictions(*args)
    args=example();args[3].append(dict(args[3][0]))
    with pytest.raises(ValueError):join_predictions(*args)


def test_audit_cutoff_and_sample_cutoff_must_agree():
    args=example();args[3][0]['first_future_time_s']=540
    with pytest.raises(ValueError):join_predictions(*args)
    args=example();args[1][10]['first_future_start_s']=481
    with pytest.raises(ValueError):join_predictions(*args)
