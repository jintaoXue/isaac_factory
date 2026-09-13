#!/usr/bin/env python3
"""Extend saved local-start associations to all recorded runtime resources."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

RUNTIME = '90a41a5b9c44e625f8c80083a8a950cd50f5eb13'
OLD_SHA = 'ebeead2dff053e3b8041ad563d3e95519c5a58b3b91c4006b0586cc6e3d86d02'
PLAN_SHA = '459945152e82ae529dd50bcb7b6427951e3b9204f4ea4ef0fdb71b9ac0ab4e2a'
KINDS = ('local_only', 'other_resource_only', 'local_and_other_resources', 'no_recorded_factory_start')
TYPES = {'machine_failure', 'human_unavailable', 'transport_delay', 'material_shortage'}


def classify(target, events):
    start = target['start_index']; onset = target['onset_window_index']
    if start not in (1, 2): raise ValueError('Upcoming start must be one or two windows')
    begin, onset_begin, end = (onset - start) * 60, onset * 60, (onset + 1) * 60
    if begin < 0 or target['first_future_time_s'] != begin or target['onset_window_end_s'] != end:
        raise ValueError('Frozen forecast boundary mismatch')
    if len({e['event_id'] for e in events}) != len(events): raise ValueError('Duplicate runtime event ID')
    if any(e['type'] not in TYPES or not 0 <= e['start'] < e['end'] for e in events):
        raise ValueError('Invalid audited runtime interval or unknown event type')
    future = [e for e in events if begin <= e['start'] < end]
    local = [e for e in future if e['target'] == target['resource_id']]
    other = [e for e in future if e['target'] != target['resource_id']]
    kind = KINDS[2] if local and other else KINDS[0] if local else KINDS[1] if other else KINDS[3]
    before = [e for e in future if e['start'] < onset_begin]
    within = [e for e in future if e['start'] >= onset_begin]
    active = [e for e in events if e['start'] < begin < e['end']]
    return dict(factory_start_class=kind,
        local_future_event_ids=sorted(e['event_id'] for e in local),
        other_resource_future_event_ids=sorted(e['event_id'] for e in other),
        other_future_resource_ids=sorted({e['target'] for e in other}),
        future_event_types=sorted({e['type'] for e in future}),
        before_onset_window_event_ids=sorted(e['event_id'] for e in before),
        within_onset_window_event_ids=sorted(e['event_id'] for e in within),
        already_active_before_cutoff_event_ids=sorted(e['event_id'] for e in active))


def summarize(rows):
    def count(selected):
        return dict(window_targets=len(selected),
            unique_onsets=len({(x['group_id'], x['resource_id'], x['onset_window_index']) for x in selected}),
            episodes=len({x['group_id'] for x in selected}))
    return dict(total=count(rows), by_start_class={k: count([x for x in rows if x['factory_start_class'] == k]) for k in KINDS},
        any_factory_future_start=count([x for x in rows if x['factory_start_class'] != KINDS[3]]),
        any_start_before_onset_window=count([x for x in rows if x['before_onset_window_event_ids']]),
        starts_only_within_onset_window=count([x for x in rows if x['within_onset_window_event_ids'] and not x['before_onset_window_event_ids']]),
        already_active_before_cutoff=count([x for x in rows if x['already_active_before_cutoff_event_ids']]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source_commit', required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path('/home/sci/work/BSTAN_isaac_factory')
    tools = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'
    d = (repo / tools).parent / 'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
    output = d / 'baseline_factorywide_upcoming_starts20260913.json'; assert not output.exists()
    old_path, plan_path = d / 'baseline_upcoming_schedule_support20260913.json', d / 'baseline_episode_holdout_plan20260913.json'
    assert sha(old_path) == OLD_SHA and sha(plan_path) == PLAN_SHA
    old, plan = json.loads(old_path.read_text()), json.loads(plan_path.read_text())
    manifest_path = d / 'dataset_manifest.json'; assert sha(manifest_path) == old['manifest_sha256'] == plan['dataset_manifest_sha256']
    manifest = json.loads(manifest_path.read_text()); cohort_path = Path(manifest['cohort_audit']['path'])
    assert sha(cohort_path) == old['cohort_audit_sha256'] == manifest['cohort_audit']['sha256']
    cohort = json.loads(cohort_path.read_text())
    raw = {f"{x['run_id']}:env_{x['env_id']:02d}:episode_{x['episode_id']:02d}": x for x in cohort['episodes']}
    source_episodes = {x['group_id']: x for x in manifest['source_episodes']}
    group_view = {}
    for view, groups in plan['plan']['groups'].items():
        assert view in ('fit', 'heldout', 'original_validation')
        for group in groups:
            assert group not in group_view; group_view[group] = view
    source = subprocess.check_output(['git', 'show', args.source_commit + ':' + tools + 'audit_factorywide_upcoming_starts.py'])
    pf = json.loads((d / 'baseline_gru_capacity_preflight20260913.json').read_text())

    def guard():
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
        assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'dev_xwt'
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        assert sha(old_path) == OLD_SHA and sha(plan_path) == PLAN_SHA and sha(cohort_path) == old['cohort_audit_sha256']
        assert sha(manifest_path) == old['manifest_sha256']
        for name, h in pf['runtime_source_sha256'].items(): assert sha(repo / name) == h
        for name, value in pf['dataset_files_stat'].items():
            stat = (d / name).stat(); assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value

    guard(); rows = []; seen = set()
    for target in old['upcoming_targets']:
        group = target['group_id']; entry = raw[group]
        assert target['split'] in ('train', 'validation') and group in group_view and entry['accepted']
        assert entry['raw_episode_sha256'] == source_episodes[group]['raw_episode_sha256']
        assert (target['split'] == 'validation') == (group_view[group] == 'original_validation')
        key = (group, target['resource_id'], target['onset_window_index'], target['start_index'])
        assert key not in seen; seen.add(key)
        classified = classify(target, entry['runtime_events'])
        assert classified['local_future_event_ids'] == sorted(target['matched_runtime_event_ids'])
        assert bool(classified['local_future_event_ids']) == target['matched_future_local_runtime_start']
        rows.append({**target, **classified, 'holdout_diagnostic_view': group_view[group]})
    original = {split: summarize([x for x in rows if x['split'] == split]) for split in ('train', 'validation')}
    views = {view: summarize([x for x in rows if x['holdout_diagnostic_view'] == view]) for view in ('fit', 'heldout', 'original_validation')}
    assert [(original[s]['total']['window_targets'], original[s]['total']['unique_onsets']) for s in ('train', 'validation')] == [(595,299),(145,73)]
    assert [views[v]['total']['window_targets'] for v in ('fit', 'heldout', 'original_validation')] == [451,144,145]
    for split in original:
        local = sum(original[split]['by_start_class'][k]['window_targets'] for k in (KINDS[0], KINDS[2]))
        assert local == old['summaries'][split]['matched_future_local_runtime_start_targets']
    guard()
    record = dict(status='all_frozen_upcoming_anchors_joined_to_all_recorded_runtime_resources', source_commit=args.source_commit,
        source_sha256=hashlib.sha256(source).hexdigest(), runtime_commit=RUNTIME, old_local_audit_sha256=OLD_SHA,
        holdout_plan_sha256=PLAN_SHA, cohort_audit_file=str(cohort_path), cohort_audit_sha256=sha(cohort_path),
        dataset_manifest_sha256=sha(manifest_path), original_splits=original, holdout_diagnostic_views=views,
        upcoming_targets=rows, runtime_event_types=sorted(TYPES), raw_csv_reread=False,
        scope='Recorded runtime STARTs on any resource from first forecast time inclusive to onset-window end exclusive; before-onset and within-onset timing separate. Existing audited intervals reused. Not causal attribution.',
        limitations=['Only the four recorded runtime types are included; quality holds and all other exogenous mechanisms are not ruled out.',
            'A temporally associated start may not cause the bottleneck and is not proof of unpredictability.',
            'No-start groups can still have previously active disturbances and predictive history.',
            'One onset may span different classes at its two forecast cutoffs; class window counts partition but unique-onset counts need not.',
            'No future plan or retrospective label enters a model, filter or threshold selection.'],
        runtime_source_files_unchanged=len(pf['runtime_source_sha256']), dataset_files_stat_unchanged=6,
        model_training=False, model_forward=False, test_evaluated=False, goal_met=False)
    with output.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    for name, summary in {**original, **views}.items(): print('FACTORY_START_GROUPS',name,json.dumps(summary),flush=True)
    print('FACTORY_START_AUDIT_COMPLETE',output.stat().st_size,sha(output),flush=True)


if __name__ == '__main__': main()
