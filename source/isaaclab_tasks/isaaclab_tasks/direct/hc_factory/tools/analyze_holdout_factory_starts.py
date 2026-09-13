#!/usr/bin/env python3
"""Stratify eighteen completed holdout caches by audited factory-wide starts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

RUNTIME = '90a41a5b9c44e625f8c80083a8a950cd50f5eb13'
AUDIT_SHA = '9eeed528ccbe1534a0e29cbdda21dfec613659e53ba84acd3ca63a5b1ed57cf9'
PLAN_SHA = '459945152e82ae529dd50bcb7b6427951e3b9204f4ea4ef0fdb71b9ac0ab4e2a'
COMPLETED = {'b4':'77c45d0250c7ac44c63c224266d2143613f9e8d57d7a9a94e1e60d2ec858b0d5',
             'b5':'fa89dd945e0fa6c3b8f751a6bbd7e612b133d32bb9b876c34b1a3cc98a8c021d'}
KINDS = ('local_only','other_resource_only','local_and_other_resources','no_recorded_factory_start')
TAG = 'baseline_episodeholdout20260913'


def target_key(row):
    return row['group_id'],row['resource_id'],row['onset_window_index'],row['start_index']


def join_predictions(arrays, samples, node_ids, targets):
    lookup = {target_key(x):x for x in targets}
    if len(lookup)!=len(targets): raise ValueError('Duplicate target anchor')
    indices = arrays['sample_index'].reshape(-1)
    if len(set(indices.tolist()))!=len(indices): raise ValueError('Duplicate sample index')
    probability = np.asarray(arrays['will_probability'],dtype=np.float32)
    if probability.shape!=(len(indices),len(node_ids)) or not np.isfinite(probability).all():
        raise ValueError('Invalid cached probability grid')
    upcoming = (arrays['occ_node_mask']>.5)&(arrays['event_will']>.5)&(arrays['event_start']>0)
    predicted = probability>=np.float32(.70)
    decoded = np.where(arrays['hist_last_hot']>.5,0,np.asarray(arrays['predicted_start'],dtype=np.int64))
    rows=[];seen=set()
    for i,node in zip(*np.nonzero(upcoming)):
        sample=samples[int(indices[i])];start=int(arrays['event_start'][i,node]);first=float(sample['first_future_start_s'])
        if start not in (1,2) or first!=round(first/60)*60: raise ValueError('Changed forecast boundary')
        key=(sample['group_id'],node_ids[node],round(first/60)+start,start)
        if key not in lookup or key in seen: raise ValueError('Unmatched or duplicate cached target')
        seen.add(key);target=lookup[key]
        if target['first_future_time_s']!=first: raise ValueError('Audit/cache cutoff mismatch')
        timing=abs(int(decoded[i,node])-start)<=3;alarm=bool(predicted[i,node])
        rows.append(dict(sample_index=int(indices[i]),group_id=key[0],resource_id=key[1],onset_window_index=key[2],start_index=start,
            factory_start_class=target['factory_start_class'],already_active_before_cutoff=bool(target['already_active_before_cutoff_event_ids']),
            probability=float(probability[i,node]),decoded_start=int(decoded[i,node]),
            report_hit=alarm and timing,probability_miss=not alarm,timing_miss=alarm and not timing))
    if seen!=set(lookup): raise ValueError('Incomplete audited target coverage')
    return rows


def summary(rows):
    hits=sum(x['report_hit'] for x in rows);pm=sum(x['probability_miss'] for x in rows);tm=sum(x['timing_miss'] for x in rows)
    assert hits+pm+tm==len(rows)
    return dict(window_targets=len(rows),unique_onsets=len({(x['group_id'],x['resource_id'],x['onset_window_index']) for x in rows}),
        report_hits=hits,recall=hits/len(rows) if rows else None,probability_misses=pm,timing_misses=tm,
        unique_onsets_with_any_hit=len({(x['group_id'],x['resource_id'],x['onset_window_index']) for x in rows if x['report_hit']}),
        probability_q10_q50_q90=np.quantile([x['probability'] for x in rows],[.1,.5,.9]).tolist() if rows else None)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source_commit',required=True);args=parser.parse_args()
    repo=Path.cwd().resolve();assert repo==Path('/home/sci/work/BSTAN_isaac_factory')
    rel='source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'
    d=(repo/rel).parent/'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
    out=d/f'{TAG}_factory_start_predictions.json';assert not out.exists()
    ap=d/'baseline_factorywide_upcoming_starts20260913.json';pp=d/'baseline_episode_holdout_plan20260913.json'
    assert sha(ap)==AUDIT_SHA and sha(pp)==PLAN_SHA
    audit,plan=json.loads(ap.read_text()),json.loads(pp.read_text())
    assert sha(d/'model_sample_index.csv')==plan['sample_index_sha256']
    manifest=json.loads((d/'dataset_manifest.json').read_text());assert sha(d/'dataset_manifest.json')==audit['dataset_manifest_sha256']==plan['dataset_manifest_sha256']
    with (d/'model_sample_index.csv').open(newline='') as f:
        sample_rows=[x for x in csv.DictReader(f) if x['split'] in ('train','validation')]
    samples={int(x['sample_index']):x for x in sample_rows};assert len(samples)==len(sample_rows)
    source=subprocess.check_output(['git','show',args.source_commit+':'+rel+'analyze_holdout_factory_starts.py'])
    pf=json.loads((d/'baseline_gru_capacity_preflight20260913.json').read_text())
    sources={ap.name:AUDIT_SHA,pp.name:PLAN_SHA};results=[]

    def guard():
        assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==RUNTIME
        assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='dev_xwt'
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
        for name,h in pf['runtime_source_sha256'].items():assert sha(repo/name)==h
        for name,v in pf['dataset_files_stat'].items():
            s=(d/name).stat();assert dict(size=s.st_size,mtime_ns=s.st_mtime_ns)==v

    guard()
    for model,h in COMPLETED.items():
        cp=d/f'{TAG}_{model}s42_complete.json';assert sha(cp)==h;sources[cp.name]=h
        completed=json.loads(cp.read_text());identity=completed['identity']
        assert completed['status']=='fixed_sixty_epoch_training_and_nine_views_complete' and len(completed['diagnostics'])==9
        assert identity['model']==model and identity['seed']==42 and identity['fixed_report_threshold']==.70 and not completed['test_evaluated']
        for epoch in (10,30,60):
            for view in ('fit','heldout','original_validation'):
                item=next(x for x in completed['diagnostics'] if x['result']['provenance']['view']==view and x['result']['provenance']['epoch']==epoch)
                p=d/item['file'];assert sha(p)==item['sha256'];r=json.loads(p.read_text());assert r==item['result']
                assert r['provenance']['identity']==identity and not r['provenance']['test_evaluated'];sources[p.name]=item['sha256']
                cache=d/r['cache_file'];assert sha(cache)==r['cache_sha256'];sources[cache.name]=r['cache_sha256']
                with np.load(cache,allow_pickle=False) as z:
                    assert json.loads(str(z['metadata_json'].item()))=={k:r[k] for k in ('provenance','report')}
                    arrays={k:z[k] for k in ('sample_index','occ_node_mask','event_will','event_start','will_probability','hist_last_hot','predicted_start')}
                assert arrays['sample_index'].tolist()==plan['plan']['sample_indices'][view]
                targets=[x for x in audit['upcoming_targets'] if x['holdout_diagnostic_view']==view]
                records=join_predictions(arrays,samples,manifest['node_ids'],targets)
                overall=summary(records);canonical=next(x for x in r['report']['thresholds'] if x['threshold']==.70)
                assert overall['window_targets']==canonical['n_true_upcoming']==r['report']['groups']['upcoming']['count']
                assert abs(overall['report_hits']-canonical['report_recall_upcoming']*canonical['n_true_upcoming'])<1e-8
                assert overall['probability_misses']==canonical['upcoming_probability_misses'] and overall['timing_misses']==canonical['upcoming_timing_misses']
                groups={k:summary([x for x in records if x['factory_start_class']==k]) for k in KINDS}
                for k,g in groups.items():assert g['window_targets']==audit['holdout_diagnostic_views'][view]['by_start_class'][k]['window_targets']
                assert sum(x['report_hits'] for x in groups.values())==overall['report_hits']
                active={str(v):summary([x for x in records if x['factory_start_class']==KINDS[3] and x['already_active_before_cutoff']==v]) for v in (False,True)}
                results.append(dict(model=model,epoch=epoch,view=view,overall=overall,groups=groups,no_start_by_already_active=active,targets=records,cache_file=cache.name,cache_sha256=r['cache_sha256']))
                print('FACTORY_CACHED_STRATA',model,epoch,view,{k:(v['report_hits'],v['window_targets']) for k,v in groups.items()},flush=True)
                del arrays
    guard()
    for name,h in sources.items():assert sha(d/name)==h
    assert len(results)==18
    record=dict(status='eighteen_existing_holdout_caches_factory_start_strata_completed_and_verified',source_commit=args.source_commit,
        source_sha256=hashlib.sha256(source).hexdigest(),runtime_commit=RUNTIME,sources_sha256=sources,results=results,
        scope='All predefined 10/30/60 checkpoints, fit/heldout/original validation, both seed42 models. Recall at fixed .70 and integer start tolerance3. Retrospective positive-target groups; no subgroup AP is invented without a matching negative population.',
        model_forward=False,model_training=False,threshold_or_checkpoint_selection=False,test_evaluated=False,goal_met=False)
    with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print('FACTORY_CACHED_STRATA_COMPLETE',out.stat().st_size,sha(out),flush=True)


if __name__=='__main__':main()
