#!/usr/bin/env python3
"""Read six completed unseen prediction caches; stratify by fitting support."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from factory_baselines.metrics import _binary_metrics

SUPPORT_SHA = '97b05d4f2a6f224a61865cc0aaa83772f6deee544274639f01bfd99e3cf52be4'
RUNTIME = 'ee838f59d2bdf2893a35ec00acae1308f9dd0e07'
TRAIN_SOURCE = 'ecc39e81c8f14483eaa50d90beea9743f9ac485c'
TAG = 'baseline_episodeholdout20260913'


def all_fitting_cells(episodes, fit_groups):
    by_id = {e['group_id']: e for e in episodes}
    if len(by_id) != len(episodes) or len(set(fit_groups)) != len(fit_groups): raise ValueError('Duplicate fitting identity')
    selected = [by_id[g] for g in fit_groups]
    if any(e['split'] != 'train' for e in selected): raise ValueError('Unseen split in fitting support')
    count = Counter((o['resource_id'], e['scenario_id']) for e in selected for o in e['upcoming_onsets'])
    # Include every fitting cell, including those with only negative query
    # targets. A query-positive-only coverage table would misclassify negatives.
    return [dict(cell=list(k), train_unique_onsets=n) for k, n in sorted(count.items())]


def summarize_support_predictions(arrays, sample_scenarios, node_ids, cells):
    lookup = {tuple(c['cell']): c['train_unique_onsets'] for c in cells}
    if len(lookup) != len(cells): raise ValueError('Duplicate support cell')
    support = np.asarray([[lookup.get((node, sample_scenarios[int(i)]), 0) for node in node_ids] for i in arrays['sample_index']])
    valid = arrays['occ_node_mask'] > .5
    if valid.shape != support.shape: raise ValueError('Node identity grid differs')
    upcoming = valid & (arrays['event_will'] > .5) & (arrays['event_start'] > 0)
    negative = valid & (arrays['event_will'] <= .5)
    predicted = arrays['will_probability'] >= .70
    start = np.where(arrays['hist_last_hot'] > .5, 0, arrays['predicted_start'])
    correct_start = np.abs(start - arrays['event_start']) <= 3
    result = {}
    for name, supported in [('zero_joint_support', support == 0), ('positive_joint_support', support > 0)]:
        up, neg = upcoming & supported, negative & supported
        selected = up | neg
        metrics = _binary_metrics(up[selected].astype(int), arrays['will_probability'][selected])
        count = int(up.sum()); hits = int((up & predicted & correct_start).sum())
        result[name] = dict(upcoming=count, negative=int(neg.sum()), hits=hits,
            recall=hits / count if count else None, ap=metrics['pr_auc'],
            probability_misses=int((up & ~predicted).sum()), timing_misses=int((up & predicted & ~correct_start).sum()),
            upcoming_probability_q10_q50_q90=np.quantile(arrays['will_probability'][up], [.1, .5, .9]).tolist() if count else None)
        assert hits + result[name]['probability_misses'] + result[name]['timing_misses'] == count
    return result


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source_commit', required=True)
    parser.add_argument('--model', choices=('b4', 'b5'), required=True); args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path('/home/sci/work/BSTAN_isaac_factory')
    tools = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'
    d = (repo / tools).parent / 'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
    out = d / f'{TAG}_{args.model}s42_support_strata.json'; assert not out.exists(), 'Reuse existing support strata'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
    own = subprocess.check_output(['git', 'show', args.source_commit + ':' + tools + Path(__file__).name])
    assert sha(Path(__file__)) == hashlib.sha256(own).hexdigest()
    support_path = d / f'{TAG}_fitting_support.json'; assert sha(support_path) == SUPPORT_SHA
    support = json.loads(support_path.read_text())
    plan_path = d / 'baseline_episode_holdout_plan20260913.json'; assert sha(plan_path) == support['plan_sha256']
    plan = json.loads(plan_path.read_text()); assert sha(d / 'model_sample_index.csv') == plan['sample_index_sha256']
    assert sha(d / 'dense_event_support20260912.json') == support['original_support_sha256']
    episodes = json.loads((d / 'dense_event_support20260912.json').read_text())['episodes']
    cells = all_fitting_cells(episodes, plan['plan']['groups']['fit'])
    scenarios = {e['group_id']: e['scenario_id'] for e in episodes}
    with (d / 'model_sample_index.csv').open(newline='') as f:
        sample_scenarios = {int(r['sample_index']): scenarios[r['group_id']] for r in csv.DictReader(f) if r['split'] in ('train', 'validation')}
    manifest_path = d / 'dataset_manifest.json'; assert sha(manifest_path) == plan['dataset_manifest_sha256']
    node_ids = json.loads(manifest_path.read_text())['node_ids']
    completed_path = d / f'{TAG}_{args.model}s42_complete.json'; completed_sha = sha(completed_path)
    completed = json.loads(completed_path.read_text()); identity = completed['identity']
    assert completed['status'] == 'fixed_sixty_epoch_training_and_nine_views_complete' and len(completed['diagnostics']) == 9
    assert identity['source_commit'] == TRAIN_SOURCE and identity['model'] == args.model and identity['seed'] == 42
    assert not completed['test_evaluated'] and identity['fixed_report_threshold'] == .70
    rows, sources = [], []
    for epoch in (10, 30, 60):
        for view in ('heldout', 'original_validation'):
            row = next(r for r in completed['diagnostics'] if r['result']['provenance']['view'] == view and r['result']['provenance']['epoch'] == epoch)
            assert sha(d / row['file']) == row['sha256']
            r = json.loads((d / row['file']).read_text()); assert r == row['result']
            assert r['provenance']['identity'] == identity and not r['provenance']['test_evaluated']
            assert sha(d / r['cache_file']) == r['cache_sha256']
            with np.load(d / r['cache_file'], allow_pickle=False) as z:
                assert json.loads(str(z['metadata_json'].item())) == {k: r[k] for k in ('provenance', 'report')}
                arrays = {k: z[k] for k in ('sample_index', 'occ_node_mask', 'event_will', 'event_start', 'will_probability', 'hist_last_hot', 'predicted_start')}
            assert arrays['sample_index'].tolist() == plan['plan']['sample_indices'][view]
            result = summarize_support_predictions(arrays, sample_scenarios, node_ids, cells)
            assert result['zero_joint_support']['upcoming'] == support['fitting_only_support'][view]['node_scenario']['coverage'][0]['validation_window_targets']
            assert sum(g['upcoming'] for g in result.values()) == r['report']['groups']['upcoming']['count']
            assert sum(g['hits'] for g in result.values()) == r['report']['thresholds'][0]['n_matched_who_upcoming']
            assert sum(g['negative'] for g in result.values()) == r['report']['groups']['negative']['count']
            rows.append(dict(epoch=epoch, view=view, groups=result)); sources.append(dict(file=r['cache_file'], sha256=r['cache_sha256']))
            print('HOLDOUT_SUPPORT_STRATA', args.model, epoch, view, json.dumps(result), flush=True)
    assert sha(completed_path) == completed_sha and sha(support_path) == SUPPORT_SHA
    record = dict(status='six_unseen_cached_views_fitting_support_strata_complete', model=args.model, source_commit=args.source_commit,
        source_sha256=sha(Path(__file__)), training_source=TRAIN_SOURCE, completed_model_file=completed_path.name,
        completed_model_sha256=completed_sha, fitting_support_sha256=SUPPORT_SHA, sources=sources, results=rows,
        scope='Fixed zero versus positive fitting onset support by node and scenario; six unseen caches from three prespecified epochs. AP excludes ongoing within each support group; recall uses threshold .70 and start tolerance3. Positive categorical support does not imply matched histories. No weight, threshold or episode selection.',
        model_training=False, model_forward=False, test_evaluated=False, goal_met=False)
    with out.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print('HOLDOUT_SUPPORT_STRATA_COMPLETE', out.stat().st_size, sha(out), flush=True)


if __name__ == '__main__': main()
