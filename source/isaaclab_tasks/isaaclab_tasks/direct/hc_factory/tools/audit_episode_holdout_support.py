#!/usr/bin/env python3
"""Count fitting-only onset support for the two frozen unseen episode views."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from diagnose_upcoming_schedule_support import summarize_support_cells

PLAN_SHA = '459945152e82ae529dd50bcb7b6427951e3b9204f4ea4ef0fdb71b9ac0ab4e2a'
SUPPORT_SHA = '56d2dc07169d12e420ed1d2877bcf3290d2cd53d0abdd317ea6581845b32cd93'
RUNTIME = 'ee838f59d2bdf2893a35ec00acae1308f9dd0e07'


def fitting_support(episodes, groups):
    expected = {'fit', 'heldout', 'original_validation'}
    if set(groups) != expected or not all(groups.values()): raise ValueError('Missing frozen view')
    membership = [g for values in groups.values() for g in values]
    by_id = {e['group_id']: e for e in episodes}
    if len(by_id) != len(episodes) or len(set(membership)) != len(membership): raise ValueError('Duplicate episode')
    if set(by_id) != set(membership): raise ValueError('Episode coverage differs')
    for view in expected:
        original = 'validation' if view == 'original_validation' else 'train'
        if any(by_id[g]['split'] != original for g in groups[view]): raise ValueError('Original split differs')
    counts = {}
    for view in sorted(expected):
        selected = [by_id[g] for g in groups[view]]
        counts[view] = dict(episodes=len(selected), unique_upcoming_onsets=sum(len(e['upcoming_onsets']) for e in selected),
            upcoming_window_targets=sum(len(o['anchors']) for e in selected for o in e['upcoming_onsets']),
            scenarios=dict(sorted(Counter(e['scenario_id'] for e in selected).items())))
    results = {}
    for view in ('heldout', 'original_validation'):
        # Only the original fitting episodes supply support counts. Unseen
        # labels describe their own targets; they never enter the fitting cells.
        selected = [dict(by_id[g], split='train') for g in groups['fit']]
        selected += [dict(by_id[g], split='validation') for g in groups[view]]
        results[view] = summarize_support_cells(selected)
    return dict(counts=counts, fitting_only_support=results)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source_commit', required=True); args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path('/home/sci/work/BSTAN_isaac_factory')
    tool_path = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'
    d = (repo / tool_path).parent / 'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
    out = d / 'baseline_episodeholdout20260913_fitting_support.json'; assert not out.exists(), 'Reuse completed support counts'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'dev_xwt'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    planned_file = d / 'baseline_episode_holdout_plan20260913.json'; assert sha(planned_file) == PLAN_SHA
    support_file = d / 'dense_event_support20260912.json'; assert sha(support_file) == SUPPORT_SHA
    planned = json.loads(planned_file.read_text()); support = json.loads(support_file.read_text())
    result = fitting_support(support['episodes'], planned['plan']['groups'])
    for view, c in result['counts'].items():
        assert c['episodes'] == planned['counts'][view]['episodes'] and c['upcoming_window_targets'] == planned['counts'][view]['upcoming']
    own = subprocess.check_output(['git', 'show', args.source_commit + ':' + tool_path + Path(__file__).name])
    import diagnose_upcoming_schedule_support as dependency
    dependency_path = Path(dependency.__file__)
    dep = subprocess.check_output(['git', 'show', RUNTIME + ':' + tool_path + dependency_path.name])
    assert sha(Path(__file__)) == hashlib.sha256(own).hexdigest() and sha(dependency_path) == hashlib.sha256(dep).hexdigest()
    record = dict(status='frozen_holdout_fitting_only_onset_support_audit_complete', source_commit=args.source_commit,
        source_sha256=sha(Path(__file__)), dependency_sha256=sha(dependency_path), runtime_commit=RUNTIME,
        plan_sha256=PLAN_SHA, original_support_sha256=SUPPORT_SHA, **result,
        scope='Counts of distinct fitting onsets by node, scenario and node-by-scenario. Unseen targets are only counted retrospectively. Positive support does not establish matched historical features or predictability. No partition, model, checkpoint or threshold changes.',
        model_training=False, model_forward=False, test_evaluated=False, goal_met=False)
    assert sha(planned_file) == PLAN_SHA and sha(support_file) == SUPPORT_SHA
    with out.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    for view, stats in result['fitting_only_support'].items():
        print('FITTING_SUPPORT', view, json.dumps(result['counts'][view]), flush=True)
        for level, level_stats in stats.items(): print(level, json.dumps(level_stats['coverage']), flush=True)
    print('FITTING_SUPPORT_COMPLETE', out.stat().st_size, sha(out), flush=True)


if __name__ == '__main__': main()
