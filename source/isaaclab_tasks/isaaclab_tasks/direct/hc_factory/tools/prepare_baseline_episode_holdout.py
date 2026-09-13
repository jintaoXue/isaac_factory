#!/usr/bin/env python3
"""Plan a label-blind episode holdout inside the frozen training split; no training."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


RUNTIME = 'ee838f59d2bdf2893a35ec00acae1308f9dd0e07'
SALT = 'baseline_episode_holdout20260913_v1'
TOOL_PATH = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'
EXPORT_SHA = 'd9971652a72fc24eb6388482e9c28382fcab226b2d0e76f8f7bd96f269e35f3d'
PARENT_VERIFY_SHA = 'b44d551ad06ac86bdc228173a9a7d47f35baf670ff4e5fc251e43edb2d5c3e8f'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def plan_episode_holdout(rows):
    """Use identifiers only, keeping every window of an episode in one view.

    One fifth of each raw run's original training episodes, rounded to nearest
    integer (half upward), with at least one retained episode and one held out.
    Original validation is untouched. There is no label-dependent balancing.
    """
    groups, sample_ids = {}, set()
    for row in rows:
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Only original train/validation rows are permitted')
        group, sample = row['group_id'], int(row['sample_index'])
        if sample in sample_ids: raise ValueError('Duplicate sample')
        sample_ids.add(sample)
        if ':env_' not in group: raise ValueError('Unknown episode identifier format')
        identity = (row['split'], group.split(':env_')[0])
        if group in groups and groups[group] != identity:
            raise ValueError('Episode crosses splits')
        groups[group] = identity
    by_run = defaultdict(list)
    for group, (split, run) in groups.items():
        if split == 'train': by_run[run].append(group)
    holdout, strata = set(), []
    for run, values in sorted(by_run.items()):
        if len(values) < 2: raise ValueError('Cannot hold out a singleton raw run')
        values = sorted(values, key=lambda g: (hashlib.sha256((SALT + ':' + g).encode()).hexdigest(), g))
        count = min(len(values) - 1, max(1, (len(values) + 2) // 5))
        holdout.update(values[:count])
        strata.append(dict(raw_run=run, original_train_episodes=len(values), held_out_episodes=count))
    view_groups = dict(fit=[], heldout=[], original_validation=[])
    for group, (split, _run) in sorted(groups.items()):
        view = 'original_validation' if split == 'validation' else 'heldout' if group in holdout else 'fit'
        view_groups[view].append(group)
    membership = {g: view for view, values in view_groups.items() for g in values}
    indices = {view: sorted(int(r['sample_index']) for r in rows if membership[r['group_id']] == view)
               for view in view_groups}
    if not all(indices.values()): raise ValueError('An evaluation or fitting view is empty')
    return dict(salt=SALT, selection='identifier_hash_within_original_raw_run_no_labels',
                groups=view_groups, sample_indices=indices, raw_run_strata=strata)


def summarize_cached_labels(sample_index, node_index, label_kind, plan):
    """Labels only describe a plan already frozen from identifiers."""
    if not (len(sample_index) == len(node_index) == len(label_kind)):
        raise ValueError('Cached row lengths differ')
    pairs = list(zip(sample_index.tolist(), node_index.tolist()))
    if len(set(pairs)) != len(pairs): raise ValueError('Duplicate sample/node row')
    if not np.isin(label_kind, [0, 1, 2]).all(): raise ValueError('Invalid cached label')
    expected = {int(i) for values in plan['sample_indices'].values() for i in values}
    if set(sample_index.tolist()) != expected: raise ValueError('Sample coverage differs from frozen plan')
    result = {}
    for view, indices in plan['sample_indices'].items():
        mask = np.isin(sample_index, indices)
        counts = np.bincount(label_kind[mask].astype(np.int64), minlength=3).tolist()
        result[view] = dict(episodes=len(plan['groups'][view]), samples=len(indices),
                            negative=counts[0], ongoing=counts[1], upcoming=counts[2])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_commit', required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    assert repo == Path('/home/sci/work/BSTAN_isaac_factory')
    d = repo / TOOL_PATH / '../output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
    d = d.resolve()
    final = d / 'baseline_episode_holdout_plan20260913.json'
    assert not final.exists(), 'Reuse an existing verified plan; do not reselect episodes'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'dev_xwt'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    assert sha(d / 'baseline_readout_parent_final_verification20260913.json') == PARENT_VERIFY_SHA
    exports_path = d / 'baseline_repr_exports_verification20260913.json'
    assert sha(exports_path) == EXPORT_SHA
    exports = json.loads(exports_path.read_text())
    pf = json.loads((d / 'baseline_readout_dropout_preflight20260913.json').read_text())
    index_path = d / 'model_sample_index.csv'
    with index_path.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] in ('train', 'validation')]
    plan = plan_episode_holdout(rows)
    assert len(rows) == 29298
    assert len(plan['groups']['fit']) + len(plan['groups']['heldout']) == 138
    assert len(plan['groups']['original_validation']) == 30
    tables, sources = defaultdict(list), []
    # Raw labels/identities are already identical between the B4/B5 caches;
    # read one pair only. No neural representations, scores or model forward.
    for split in ('train', 'validation'):
        name = f'baseline_repr_b4s42_last_{split}20260913.npz'
        reference = next(r for r in exports['archives'] if r['model'] == 'b4' and r['split'] == split)
        assert reference['file'] == name and sha(d / name) == reference['sha256']
        with np.load(d / name, allow_pickle=False) as z:
            for key in ('sample_index', 'node_index', 'label_kind'): tables[key].append(z[key])
        sources.append(dict(file=name, sha256=reference['sha256']))
    counts = summarize_cached_labels(**{k: np.concatenate(v) for k, v in tables.items()}, plan=plan)
    assert [counts['fit'][k] + counts['heldout'][k] for k in ('ongoing', 'upcoming', 'negative')] == [4191, 595, 297152]
    assert [counts['original_validation'][k] for k in ('ongoing', 'upcoming', 'negative')] == [950, 145, 67331]
    for name, value in pf['dataset_files_stat'].items():
        st = (d / name).stat(); assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == value
    source = subprocess.check_output(['git', 'show', args.source_commit + ':' + TOOL_PATH + Path(__file__).name])
    record = dict(status='label_blind_episode_holdout_plan_only_no_neural_training',
        source_commit=args.source_commit, source_sha256=hashlib.sha256(source).hexdigest(), runtime_commit=RUNTIME,
        dataset_manifest_sha256=sha(d / 'dataset_manifest.json'), sample_index_sha256=sha(index_path),
        plan=plan, counts=counts, label_sources=sources, model_training=False, model_forward=False,
        frozen_split_modified=False, test_evaluated=False, goal_met=False,
        next_required='Refit normalization on fit episodes only and train fresh diagnostic weights. Existing frozen weights have seen heldout episodes and cannot answer this question.')
    with final.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print('EPISODE_HOLDOUT_PLAN_ONLY', final.stat().st_size, sha(final), flush=True)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__': main()
