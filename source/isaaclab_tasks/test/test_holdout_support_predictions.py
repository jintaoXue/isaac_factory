from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'))
from analyze_holdout_support_predictions import summarize_support_predictions, all_fitting_cells


def fixture():
    a = dict(sample_index=np.array([20, 10]), occ_node_mask=np.array([[1, 1, 0], [1, 1, 1]]),
        event_will=np.array([[1, 0, 1], [1, 1, 1]]), event_start=np.array([[1, 0, 1], [1, 2, 0]]),
        will_probability=np.array([[.8, .9, 1.], [.2, .9, .99]]), hist_last_hot=np.zeros((2, 3)),
        predicted_start=np.array([[1, 0, 1], [1, 7, 0]]))
    return a, {20: 's0', 10: 's1'}, ['n0', 'n1', 'n2'], [dict(cell=['n0', 's0'], train_unique_onsets=1), dict(cell=['n1', 's1'], train_unique_onsets=2)]


def test_sample_and_node_identity_determine_support_and_exclude_ongoing_inactive():
    r = summarize_support_predictions(*fixture())
    assert r['zero_joint_support']['upcoming'] == 1 and r['zero_joint_support']['negative'] == 1
    assert r['positive_joint_support']['upcoming'] == 2 and r['positive_joint_support']['negative'] == 0
    assert r['zero_joint_support']['ap'] == .5
    assert r['positive_joint_support']['ap'] == 1.


def test_probability_and_timing_misses_partition_fixed_recall():
    r = summarize_support_predictions(*fixture())
    zero, pos = r['zero_joint_support'], r['positive_joint_support']
    assert (zero['hits'], zero['probability_misses'], zero['timing_misses']) == (0, 1, 0)
    assert (pos['hits'], pos['probability_misses'], pos['timing_misses']) == (1, 0, 1)


def test_query_labels_do_not_redefine_fitting_support():
    a, scenarios, nodes, cells = fixture(); a['event_start'][1, 2] = 1
    r = summarize_support_predictions(a, scenarios, nodes, cells)
    assert r['zero_joint_support']['upcoming'] == 2 and r['positive_joint_support']['upcoming'] == 2


def test_empty_positive_subgroup_is_undefined_not_zero():
    a, scenarios, nodes, cells = fixture()
    r = summarize_support_predictions(a, scenarios, nodes, [])['positive_joint_support']
    assert r['upcoming'] == 0 and r['recall'] is None and r['ap'] is None


def test_fit_cells_include_negative_query_categories_and_exclude_heldout_episodes():
    episodes = [dict(group_id='f', split='train', scenario_id='s0', upcoming_onsets=[dict(resource_id='n1', anchors=[1, 2])]),
                dict(group_id='h', split='train', scenario_id='s1', upcoming_onsets=[dict(resource_id='n0', anchors=[1, 2])])]
    cells = all_fitting_cells(episodes, ['f'])
    assert cells == [dict(cell=['n1', 's0'], train_unique_onsets=1)]
    a, scenarios, nodes, _ = fixture()
    r = summarize_support_predictions(a, scenarios, nodes, cells)
    assert r['positive_joint_support']['negative'] == 1 and r['positive_joint_support']['upcoming'] == 0
    assert r['zero_joint_support']['upcoming'] == 3
