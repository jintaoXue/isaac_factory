"""Regression for saved JSON tuple/list configuration and frozen preflight reuse."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'))
from run_baseline_episode_holdout import (configuration, assert_saved_configuration,
    validate_completed_preflight, PREFLIGHT_SOURCE, PREFLIGHT_DRIVER_SHA)


def reject(function, *args):
    try: function(*args)
    except AssertionError: return
    raise AssertionError('Changed scientific configuration was accepted')


def test_saved_threshold_tuple_roundtrip_is_accepted_but_changed_values_are_not():
    for name in ('b4', 'b5'):
        train, overrides, loss = configuration(name)
        saved = json.loads(json.dumps(dict(model_config=overrides, training_config=asdict(train), loss_config=asdict(loss))))
        assert isinstance(train.report_threshold_sweep, tuple)
        assert isinstance(saved['training_config']['report_threshold_sweep'], list)
        assert_saved_configuration(overrides, train, loss, saved)
        for section, key, value in [('training_config', 'learning_rate', 100.),
                                    ('training_config', 'report_threshold_sweep', [.5]),
                                    ('model_config', 'event_precursor', 'near_far'),
                                    ('loss_config', 'event_will_upcoming_pos_weight', 900.)]:
            bad = deepcopy(saved); bad[section][key] = value
            reject(assert_saved_configuration, overrides, train, loss, bad)


def test_completed_preflight_reuse_requires_identical_inputs_and_scientific_policy():
    old = dict(source_commit=PREFLIGHT_SOURCE, source_sha256={'run_baseline_episode_holdout.py': PREFLIGHT_DRIVER_SHA,
               'baseline_episode_holdout_inputs.py': 'unchanged'}, runtime_commit='runtime', plan_sha256='plan',
               checkpoint_epochs=[10, 30, 60], fitting_normalization_sha256='norm', seed=42)
    new = deepcopy(old); new['source_commit'] = 'startup_fix'; new['source_sha256']['run_baseline_episode_holdout.py'] = 'fixed'
    norm = {'feature_mean': [1.]}
    preflight = dict(identity=old, normalization=norm, status='two_models_real_fitting_batch_preflight_passed_no_optimizer_steps', test_evaluated=False)
    validate_completed_preflight(preflight, new, norm)
    for key, value in [('plan_sha256', 'changed'), ('seed', 43), ('checkpoint_epochs', [5, 60]), ('runtime_commit', 'different')]:
        bad = deepcopy(new); bad[key] = value; reject(validate_completed_preflight, preflight, bad, norm)
    bad = deepcopy(new); bad['source_sha256']['baseline_episode_holdout_inputs.py'] = 'changed'
    reject(validate_completed_preflight, preflight, bad, norm)
    reject(validate_completed_preflight, preflight, new, {'feature_mean': [2.]})
