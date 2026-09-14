"""In-memory regression checks for recovering saved B2 results without refitting."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'))
import run_remaining_baselines_matched as runner
import recover_remaining_baselines_matched as recovery
from recover_remaining_baselines_matched import RUNTIME, TASKS, validate_failure


def failure():
    plan = {'status':'failed', 'source_commit':RUNTIME, 'failed_stage':['B2',5],
            'test_evaluated':False, 'tasks':[{'model':m,'max_start':s} for m,s in TASKS]}
    record = {'status':'failed', 'source_commit':RUNTIME, 'error':"'target_cause'",
              'model':'B2', 'max_start':5, 'test_evaluated':False}
    return plan, record


@pytest.mark.parametrize('terminal', ['99 0', '99 1 0', '99 1 2'])
def test_recovery_launcher_refuses_live_successful_or_unexpected_terminal(monkeypatch, terminal):
    plan, record = failure()
    monkeypatch.setattr(recovery, 'failed_stage', lambda: (plan,record,{}))
    monkeypatch.setattr(recovery, 'out', lambda *args: terminal)
    def no_write(*args, **kwargs):
        pytest.fail('A refused recovery must not create state or respawn a pane')
    monkeypatch.setattr(recovery, 'write', no_write)
    monkeypatch.setattr(recovery.subprocess, 'run', no_write)
    with pytest.raises(ValueError, match='ended with exit 1'):
        recovery.launch('repair-source')


@pytest.mark.parametrize('damage', ['none','runtime','stage','error','complete','test','order'])
def test_recovery_only_accepts_the_known_first_stage_export_failure(damage):
    plan, record = failure()
    if damage == 'runtime': plan['source_commit'] = 'another-run'
    if damage == 'stage': plan['failed_stage'] = ['B3',5]
    if damage == 'error': record['error'] = 'CUDA out of memory'
    if damage == 'complete': record['status'] = 'validation_completed'
    if damage == 'test': record['test_evaluated'] = True
    if damage == 'order': plan['tasks'].reverse()
    if damage == 'none':
        validate_failure(plan, record)
    else:
        with pytest.raises(ValueError): validate_failure(plan, record)


@pytest.mark.parametrize('damage', ['none','foreign_index','duplicate','missing','wrong_split','cause','remain','test'])
def test_target_adapter_preserves_predictions_and_rejects_cross_split_or_changed_truth(damage):
    targets = {2:('transport_delay',11)}
    rows = [{'sample_index':'2','split':'validation','predicted_cause':'queue_buildup',
             'predicted_remain_len_windows':'12.4'}]
    split = 'validation'
    if damage == 'foreign_index': rows[0]['sample_index'] = '3'
    if damage == 'duplicate': rows *= 2
    if damage == 'missing': rows = []
    if damage == 'wrong_split': rows[0]['split'] = 'train'
    if damage == 'cause': rows[0]['target_cause'] = 'material_shortage'
    if damage == 'remain': rows[0]['target_remain_len_windows'] = '10'
    if damage == 'test': split = 'test'
    original = copy.deepcopy(rows)
    if damage == 'none':
        result = runner.b2_target_rows(rows, split, targets)
        assert result[0] == {**rows[0], 'target_cause':'transport_delay','target_remain_len_windows':11}
    else:
        with pytest.raises(ValueError): runner.b2_target_rows(rows, split, targets)
    assert rows == original


def test_canonical_truth_reads_only_registered_train_validation_indices(monkeypatch):
    from test_remaining_baselines_matched import payload, manifest
    data, meta = payload(), manifest()
    class Splits(dict):
        def __getitem__(self, key):
            assert key != 'test', 'test split must never be indexed'
            return super().__getitem__(key)
    data['split_indices'] = Splits(data['split_indices'])
    monkeypatch.setattr(runner, 'sha', lambda _: runner.SOURCE_MANIFEST)
    monkeypatch.setattr(runner, 'load_shared_dataset', lambda _: (data,meta))
    splits = {s:{'sample_indices':data['split_indices'][s].tolist()} for s in ('train','validation')}
    result = runner.b2_source_targets(splits, meta['cause_classes'])
    assert set(result) == {'train','validation'}
    assert set(result['train']) == {0,1} and set(result['validation']) == {2}
    splits['validation']['sample_indices'] = [1]
    with pytest.raises(ValueError, match='source split'):
        runner.b2_source_targets(splits, meta['cause_classes'])
