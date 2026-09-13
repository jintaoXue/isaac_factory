#!/usr/bin/env python3
"""Read completed holdout caches once; quantify episode-level uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

PLAN_SHA = '459945152e82ae529dd50bcb7b6427951e3b9204f4ea4ef0fdb71b9ac0ab4e2a'
SOURCE = 'ecc39e81c8f14483eaa50d90beea9743f9ac485c'
RUNTIME = 'ee838f59d2bdf2893a35ec00acae1308f9dd0e07'
TAG = 'baseline_episodeholdout20260913'
VIEWS = ('fit', 'heldout', 'original_validation')
REPLICATES = 512


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def stratified_episode_weights(groups, replicates=REPLICATES, seed=20260913):
    """Resample whole episodes within each original raw run, keeping run sizes."""
    if len(groups) != len(set(groups)) or not groups: raise ValueError('Invalid episode identities')
    runs = [g.split(':env_')[0] for g in groups]
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(groups)), dtype=np.int64)
    for run in sorted(set(runs)):
        indices = np.flatnonzero(np.asarray(runs) == run)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=replicates)
    return weights


class EpisodeRanking:
    """Tie-aware AP under whole-episode multiplicities, without model imports."""
    def __init__(self, labels, scores, episode, n_episodes):
        labels, scores, episode = map(np.asarray, (labels, scores, episode))
        if labels.ndim != 1 or labels.shape != scores.shape or scores.shape != episode.shape or not len(labels):
            raise ValueError('Invalid ranking shapes')
        if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all(): raise ValueError('Invalid ranking data')
        if np.any(episode < 0) or np.any(episode >= n_episodes): raise ValueError('Invalid episode index')
        order = np.argsort(-scores, kind='stable')
        self.labels = labels[order].astype(np.float64)
        self.episode = episode[order].astype(np.int64)
        self.ends = np.flatnonzero(np.r_[scores[order][1:] != scores[order][:-1], True])
        self.n_episodes = n_episodes

    def ap(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.n_episodes,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('Invalid episode weights')
        w = weights[self.episode]
        tp = np.cumsum(w * self.labels)[self.ends]
        total = np.cumsum(w)[self.ends]
        if tp[-1] == 0: return None
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        return float((np.diff(np.r_[0., tp]) * precision).sum() / tp[-1])


def interval(values):
    good = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return dict(valid_replicates=len(good), total_replicates=len(values),
                percentile_95=np.quantile(good, [.025, .975]).tolist() if len(good) else None)


def difference(a, b):
    return [x - y if x is not None and y is not None else None for x, y in zip(a, b)]


def summarize_cache(arrays, group_ids, sample_groups, weights, report):
    indices = arrays['sample_index'].tolist()
    groups = {g: i for i, g in enumerate(group_ids)}
    sample_episode = np.asarray([groups[sample_groups[i]] for i in indices])
    valid = arrays['occ_node_mask'] > .5
    up = valid & (arrays['event_will'] > .5) & (arrays['event_start'] > 0)
    neg = valid & (arrays['event_will'] <= .5)
    mask = up | neg
    episode_grid = np.broadcast_to(sample_episode[:, None], valid.shape)
    rank = EpisodeRanking(up[mask], arrays['will_probability'][mask], episode_grid[mask], len(groups))
    ap = rank.ap(np.ones(len(groups)))
    assert np.isclose(ap, report['ranking']['upcoming_vs_negative']['tie_aware_average_precision'], rtol=1e-10, atol=1e-12)
    decoded_start = np.where(arrays['hist_last_hot'] > .5, 0, arrays['predicted_start'])
    hit = up & (arrays['will_probability'] >= .70) & (np.abs(decoded_start - arrays['event_start']) <= 3)
    assert int(hit.sum()) == report['thresholds'][0]['n_matched_who_upcoming']
    positives = np.bincount(episode_grid[up], minlength=len(groups))
    hits = np.bincount(episode_grid[hit], minlength=len(groups))
    draws = dict(ap=[rank.ap(w) for w in weights], recall=[])
    for w in weights:
        denom = int(w @ positives)
        draws['recall'].append(float(w @ hits) / denom if denom else None)
    per_run = []
    for run in sorted({g.split(':env_')[0] for g in group_ids}):
        w = np.asarray([g.split(':env_')[0] == run for g in group_ids], dtype=np.int64)
        per_run.append(dict(raw_run=run, episodes=int(w.sum()), upcoming=int(w @ positives),
                            hits=int(w @ hits), ap=rank.ap(w)))
    return dict(ap=ap, upcoming=int(up.sum()), hits=int(hit.sum()), recall=float(hit.sum()/up.sum()),
                episode_bootstrap={k: interval(v) for k, v in draws.items()}, raw_runs=per_run,
                episodes=[dict(group_id=g, upcoming=int(positives[i]), hits=int(hits[i])) for i, g in enumerate(group_ids)]), draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_commit', required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path('/home/sci/work/BSTAN_isaac_factory')
    tool_path = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'
    d = (repo / tool_path).parent / 'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
    own = subprocess.check_output(['git', 'show', args.source_commit + ':' + tool_path + Path(__file__).name])
    assert hashlib.sha256(own).hexdigest() == sha(Path(__file__))
    out = d / f'{TAG}_episode_analysis.json'; assert not out.exists(), 'Reuse completed cache analysis'
    plan_path = d / 'baseline_episode_holdout_plan20260913.json'; assert sha(plan_path) == PLAN_SHA
    planned = json.loads(plan_path.read_text()); plan = planned['plan']
    assert sha(d / 'model_sample_index.csv') == planned['sample_index_sha256']
    complete_path = d / f'{TAG}_complete.json'; complete_sha = sha(complete_path)
    complete = json.loads(complete_path.read_text())
    assert complete['status'] == 'two_fixed_budget_neural_holdout_diagnostics_complete'
    assert complete['identity']['source_commit'] == SOURCE and not complete['test_evaluated']
    assert len(complete['runs']) == 2 and {r['identity']['model'] for r in complete['runs']} == {'b4', 'b5'}
    with (d / 'model_sample_index.csv').open(newline='') as f:
        sample_groups = {int(r['sample_index']): r['group_id'] for r in csv.DictReader(f) if r['split'] in ('train', 'validation')}
    weights = {v: stratified_episode_weights(plan['groups'][v], seed=20260913 + i) for i, v in enumerate(VIEWS)}
    results, sources = [], []
    for run in complete['runs']:
        assert len(run['diagnostics']) == 9
        assert sha(d / run['progress_file']) == run['progress_sha256']
        for epoch in (10, 30, 60):
            point, draws = {}, {}
            for view in VIEWS:
                row = next(r for r in run['diagnostics'] if r['result']['provenance']['epoch'] == epoch and r['result']['provenance']['view'] == view)
                assert sha(d / row['file']) == row['sha256']
                result = json.loads((d / row['file']).read_text()); assert result == row['result']
                prov = result['provenance']; assert prov['identity'] == run['identity']
                assert prov['sample_indices'] == plan['sample_indices'][view] and not prov['test_evaluated']
                if view == 'fit': assert sha(d / prov['checkpoint_file']) == prov['checkpoint_sha256']
                else: assert (prov['checkpoint_file'], prov['checkpoint_sha256']) == checkpoint_identity
                checkpoint_identity = (prov['checkpoint_file'], prov['checkpoint_sha256'])
                cache = d / result['cache_file']; assert sha(cache) == result['cache_sha256']
                with np.load(cache, allow_pickle=False) as z:
                    meta = json.loads(str(z['metadata_json'].item()))
                    assert meta == {k: result[k] for k in ('provenance', 'report')}
                    arrays = {k: z[k] for k in ('sample_index', 'occ_node_mask', 'event_will', 'event_start', 'will_probability', 'hist_last_hot', 'predicted_start')}
                assert arrays['sample_index'].tolist() == plan['sample_indices'][view]
                point[view], draws[view] = summarize_cache(arrays, plan['groups'][view], sample_groups, weights[view], result['report'])
                sources.append(dict(file=cache.name, sha256=result['cache_sha256'], report_file=row['file'], report_sha256=row['sha256']))
            contrasts = {}
            for left, right in (('fit', 'heldout'), ('heldout', 'original_validation')):
                contrasts[left + '_minus_' + right] = {metric: dict(point=point[left][metric] - point[right][metric], **interval(difference(draws[left][metric], draws[right][metric]))) for metric in ('ap', 'recall')}
            results.append(dict(model=run['identity']['model'], epoch=epoch, views=point, contrasts=contrasts))
            print('EPISODE_CACHE_ANALYSIS', run['identity']['model'], epoch, json.dumps({v: {k: point[v][k] for k in ('ap', 'hits', 'upcoming')} for v in VIEWS}), flush=True)
    assert sha(complete_path) == complete_sha
    reference_path = d / 'baseline_dense_readout_dropout_metrics_20260913.json'
    assert sha(reference_path) == 'b599500b3dd973393d3a997569692959ae747f7a8eea022d2ad5db417f696c78'
    reference = json.loads(reference_path.read_text())
    guarded = 0
    for run in reference['runs']:
        old = d / f'models/tuning/{run["model"]}_representation_v1/candidate_history/seed{run["seed"]}'
        for name, h in run['training']['files_sha256'].items(): assert sha(old / name) == h; guarded += 1
    assert guarded == 48
    for name, stat in reference['dataset_files_stat'].items():
        actual = (d / name).stat(); assert dict(size=actual.st_size, mtime_ns=actual.st_mtime_ns) == stat
    for name, h in reference['runtime_source_sha256'].items(): assert sha(repo / name) == h
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'dev_xwt'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    record = dict(status='eighteen_cached_views_episode_analysis_complete', source_commit=args.source_commit,
        source_sha256=sha(Path(__file__)), complete_file=complete_path.name, complete_sha256=complete_sha,
        plan_sha256=PLAN_SHA, replicates=REPLICATES, seeds={v: 20260913+i for i, v in enumerate(VIEWS)},
        method='Whole-episode bootstrap separately within each raw run and view; fixed run episode counts; same draws for every checkpoint and model; percentile intervals conditional on one fitted seed and this partition, not independent-run or architecture inference. No-positive replicates omitted and counted.',
        sources=sources, results=results, model_forward=False, model_training=False, threshold_search=False,
        checkpoint_selection=False, test_evaluated=False, goal_met=False,
        formal_model_files_unchanged=guarded, dataset_stats_unchanged=len(reference['dataset_files_stat']),
        runtime_sources_unchanged=len(reference['runtime_source_sha256']))
    with out.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print('EPISODE_CACHE_ANALYSIS_COMPLETE', out.stat().st_size, sha(out), flush=True)


if __name__ == '__main__': main()
