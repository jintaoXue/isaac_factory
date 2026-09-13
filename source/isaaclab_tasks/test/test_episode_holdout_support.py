"""Check fitting-only support and distinct-onset versus overlapping windows."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'))
from audit_episode_holdout_support import fitting_support


def data():
    def ep(g, split, node, scenario):
        return dict(group_id=g, split=split, scenario_id=scenario,
            upcoming_onsets=[dict(resource_id=node, anchors=[{'start_index': 1}, {'start_index': 2}])])
    return [ep('f', 'train', 'n1', 's1'), ep('h', 'train', 'n2', 's1'), ep('v', 'validation', 'n2', 's2')], dict(fit=['f'], heldout=['h'], original_validation=['v'])


def test_unseen_episode_never_supplies_training_support():
    episodes, groups = data(); result = fitting_support(episodes, groups)
    for view in ('heldout', 'original_validation'):
        cell = result['fitting_only_support'][view]['node']['cells'][0]
        assert cell['train_unique_onsets'] == 0
        assert cell['validation_unique_onsets'] == 1 and cell['validation_window_targets'] == 2
    assert episodes == data()[0]


def test_node_and_joint_scenario_support_are_distinct_and_not_double_counted():
    episodes, groups = data(); episodes[1]['upcoming_onsets'][0]['resource_id'] = 'n1'; episodes[2]['upcoming_onsets'][0]['resource_id'] = 'n1'
    result = fitting_support(episodes, groups)['fitting_only_support']
    assert result['heldout']['node_scenario']['cells'][0]['train_unique_onsets'] == 1
    assert result['original_validation']['node']['cells'][0]['train_unique_onsets'] == 1
    assert result['original_validation']['node_scenario']['cells'][0]['train_unique_onsets'] == 0


def test_other_unseen_labels_cannot_change_a_views_support():
    episodes, groups = data(); before = fitting_support(episodes, groups)
    episodes[1]['upcoming_onsets'] *= 3
    after = fitting_support(episodes, groups)
    assert before['fitting_only_support']['original_validation'] == after['fitting_only_support']['original_validation']


def test_reject_overlapping_episodes_missing_coverage_and_test():
    episodes, groups = data()
    bad = deepcopy(episodes); bad[1]['split'] = 'test'
    for e, g in [(episodes, dict(groups, heldout=['f'])), (episodes[:-1], groups), (bad, groups)]:
        try: fitting_support(e, g)
        except ValueError: continue
        raise AssertionError('Invalid partition accepted')
