#!/usr/bin/env python3
"""Two fixed-budget neural diagnostics on genuinely unseen training episodes."""

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_episode_holdout_inputs import prepare_inputs_in_memory
from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.schema import CONTINUOUS_FEATURES, feature_is_applicable
from factory_baselines.torch_losses import compute_multitask_loss
from factory_baselines.torch_trainer import _model_spec, _model_inputs, _occupancy_type_masks, _run_train_epoch, _seed_everything
from train_dense_baseline_control import dense_configuration
import diagnose_baseline_events as diagnostic


RUNTIME = 'ee838f59d2bdf2893a35ec00acae1308f9dd0e07'
PLAN_SHA = '459945152e82ae529dd50bcb7b6427951e3b9204f4ea4ef0fdb71b9ac0ab4e2a'
PLAN_VERIFY_SHA = 'c97523220d5bf711d4b678dd95f3a9eefed4207c91a0014eed8c16e35bb44ca6'
NEAR_SHA = 'e71c84ded25a2b5fe861b45ecc12e2a0941193043a526654a9d8327a9d7a4b27'
READOUT_SHA = 'b599500b3dd973393d3a997569692959ae747f7a8eea022d2ad5db417f696c78'
PREFLIGHT_SOURCE = '9782fff8d849503f8baac5bf101536c454ac1152'
PREFLIGHT_SHA = 'f9fce5742c0cfafe68972ba443ad528795e509ae76a5c88f9f5b8dc4d24bf7da'
PREFLIGHT_DRIVER_SHA = 'e3ea0a2a0e5ab5c17cdfc93fe3a9350bb0efb265664393d2e456df029fdad987'
CHECKPOINT_EPOCHS = (10, 30, 60)
THRESHOLD = .70
TAG = 'episodeholdout20260913'
TOOL_PATH = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_new(path, record):
    with path.open('x') as f: json.dump(record, f, indent=2); f.write('\n')


def configuration(model):
    train, overrides, loss = dense_configuration(model.upper(), 'near_precursor', 42, 'cuda:0')
    assert train.max_epochs == 60 and train.event_oversample_factor == 1
    assert train.lr_schedule == 'cosine' and not train.evaluate_test
    return train, overrides, loss


def assert_saved_configuration(model_config, train, loss, tested):
    # Saved JSON converts the threshold-sweep tuple to a list. Compare the full
    # JSON representation without dropping any scientific configuration field.
    actual = dict(model_config=model_config, training_config=asdict(train), loss_config=asdict(loss))
    expected = {k: tested[k] for k in actual}
    assert json.loads(json.dumps(actual)) == json.loads(json.dumps(expected)), 'Preflight configuration differs'


def validate_completed_preflight(preflight, identity, normalization):
    old = preflight['identity']
    assert old['source_commit'] == PREFLIGHT_SOURCE
    assert old['source_sha256']['run_baseline_episode_holdout.py'] == PREFLIGHT_DRIVER_SHA
    assert old['source_sha256']['baseline_episode_holdout_inputs.py'] == identity['source_sha256']['baseline_episode_holdout_inputs.py']
    # Only the startup comparison/provenance handling changed. Every scientific
    # field, runtime source, partition and fitted preprocessing must still match.
    ignored = {'source_commit', 'source_sha256'}
    assert {k: v for k, v in old.items() if k not in ignored} == {k: v for k, v in identity.items() if k not in ignored}
    assert preflight['normalization'] == normalization
    assert preflight['status'] == 'two_models_real_fitting_batch_preflight_passed_no_optimizer_steps'
    assert not preflight['test_evaluated']


def make_model(model, payload, manifest):
    train, overrides, loss = configuration(model)
    kind = 'b4_gcn_gru' if model == 'b4' else 'b5_gat_gru'
    cls, config_cls, _, _ = _model_spec(kind)
    config = config_cls(input_dim=27, global_dim=0, num_nodes=payload['x'].shape[2],
        prediction_horizon=loss.prediction_horizon, max_remain_windows=manifest['max_remain_windows'],
        num_causes=len(manifest['cause_classes']), **overrides)
    _seed_everything(42)
    return cls(config).cuda(), config, train, loss


def snapshot(model, optimizer, scheduler, generator, epoch, history, identity):
    return dict(identity=identity, epoch=epoch, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(), loader_generator_state=generator.get_state(),
        torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all(),
        numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate(), history=history)


def save_progress(path, state):
    pending = path.with_name(path.name + '.pending')
    torch.save(state, pending)
    os.replace(pending, path)


def evaluate_view(model, payload, indices, batch_size, checkpoint_path, identity, d, view, epoch, counts):
    name = f'baseline_{TAG}_{identity["model"]}s42_epoch{epoch}_{view}'
    cache, result_path = d / (name + '.npz'), d / (name + '.json')
    checkpoint_sha = sha(checkpoint_path)
    expected = dict(identity=identity, view=view, epoch=epoch, checkpoint_file=checkpoint_path.name,
        checkpoint_sha256=checkpoint_sha, sample_indices=indices, threshold=THRESHOLD,
        weights_selected_from_evaluation=False, test_evaluated=False)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata_json'].item()))
        assert meta['provenance'] == expected
        record = dict(**meta, cache_file=cache.name, cache_sha256=sha(cache))
        if result_path.exists(): assert json.loads(result_path.read_text()) == record
        else: write_new(result_path, record)
        print('REUSED_HOLDOUT_VIEW', identity['model'], epoch, view, flush=True)
        return dict(file=result_path.name, sha256=sha(result_path), result=record)
    assert not result_path.exists()
    loader = DataLoader(FactoryBaselineTensorDataset(payload, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    collected = {}; model.eval()
    # DataLoader iterator creation consumes CPU RNG even without shuffle. Keep
    # diagnostic inference from changing the subsequent training trajectory.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        for step, cpu in enumerate(loader):
            batch = {k: v.cuda() for k, v in cpu.items()}
            result = model(**_model_inputs(batch, model))
            values = {k: cpu[k].numpy() for k in ('sample_index', 'y_hot', 'remain_mask', 'occ_node_mask', 'hist_last_hot', 'event_will', 'event_start')}
            values.update(will_probability=result['event_will_logit'].sigmoid().cpu().numpy(),
                predicted_start=result['event_start_logit'].argmax(-1).cpu().numpy(),
                predicted_duration=result['event_duration'].cpu().numpy())
            for k, v in values.items(): collected.setdefault(k, []).append(v)
            if step % 250 == 0: print('HOLDOUT_VIEW_BATCH', identity['model'], epoch, view, step, len(loader), flush=True)
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    assert arrays['sample_index'].tolist() == indices
    report = diagnostic.summarize_events(arrays, [THRESHOLD])
    for group in ('negative', 'ongoing', 'upcoming'): assert report['groups'][group]['count'] == counts[group]
    meta = dict(provenance=expected, report=report)
    with cache.open('xb') as f: np.savez_compressed(f, **arrays, metadata_json=np.asarray(json.dumps(meta)))
    record = dict(**meta, cache_file=cache.name, cache_sha256=sha(cache))
    write_new(result_path, record)
    ap = report['ranking']['upcoming_vs_negative']['tie_aware_average_precision']
    print('HOLDOUT_VIEW_COMPLETE', identity['model'], epoch, view, 'AP', ap, 'UP', report['thresholds'][0]['n_matched_who_upcoming'], flush=True)
    return dict(file=result_path.name, sha256=sha(result_path), result=record)


def main():
    import csv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_commit', required=True)
    parser.add_argument('--phase', choices=['preflight', 'train'], required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path('/home/sci/work/BSTAN_isaac_factory')
    tools = repo / TOOL_PATH; d = tools.parent / 'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
    assert d.is_dir()
    lock = (d / f'baseline_{TAG}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = d / 'baseline_episode_holdout_plan20260913.json'; assert sha(plan_path) == PLAN_SHA
    assert sha(d / 'baseline_episode_holdout_plan_verification20260913.json') == PLAN_VERIFY_SHA
    near_path = d / 'baseline_dense_near_metrics_20260912.json'; assert sha(near_path) == NEAR_SHA
    near = json.loads(near_path.read_text())
    planned = json.loads(plan_path.read_text()); views, counts = planned['plan']['sample_indices'], planned['counts']
    assert counts == dict(fit=dict(episodes=107, samples=18095, negative=226202, ongoing=3040, upcoming=451),
        heldout=dict(episodes=31, samples=5764, negative=70950, ongoing=1151, upcoming=144),
        original_validation=dict(episodes=30, samples=5439, negative=67331, ongoing=950, upcoming=145))
    reference_path = d / 'baseline_dense_readout_dropout_metrics_20260913.json'; assert sha(reference_path) == READOUT_SHA
    reference = json.loads(reference_path.read_text())
    source_hashes = {}
    for name, actual in [('baseline_episode_holdout_inputs.py', Path(sys.modules['baseline_episode_holdout_inputs'].__file__)),
                         ('run_baseline_episode_holdout.py', Path(__file__))]:
        expected = hashlib.sha256(subprocess.check_output(['git', 'show', args.source_commit + ':' + TOOL_PATH + name])).hexdigest()
        assert sha(actual) == expected
        source_hashes[name] = expected
    identity_common = dict(source_commit=args.source_commit, source_sha256=source_hashes, runtime_commit=RUNTIME,
        plan_sha256=PLAN_SHA, seed=42, checkpoint_epochs=list(CHECKPOINT_EPOCHS), fixed_report_threshold=THRESHOLD,
        selection='fixed_epochs_no_early_stopping_no_threshold_search', test_evaluated=False,
        role='diagnostic_neural_training_not_formal_baseline_candidate')

    def guard(check_models=False):
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
        assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'dev_xwt'
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        assert sha(plan_path) == PLAN_SHA and sha(d / 'dataset_manifest.json') == planned['dataset_manifest_sha256']
        for name, h in reference['runtime_source_sha256'].items(): assert sha(repo / name) == h
        for name, value in reference['dataset_files_stat'].items():
            st = (d / name).stat(); assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == value
        if check_models:
            for run in reference['runs']:
                out = d / f'models/tuning/{run["model"]}_representation_v1/candidate_history/seed{run["seed"]}'
                for name, h in run['training']['files_sha256'].items(): assert sha(out / name) == h
    guard(True)
    preflight_path = d / f'baseline_{TAG}_preflight.json'
    final_path = d / f'baseline_{TAG}_complete.json'
    assert not final_path.exists(), 'Completed diagnostic must be reused, not restarted'
    if args.phase == 'preflight': assert not preflight_path.exists(), 'Reuse completed preflight'
    torch.set_num_threads(2)
    payload, manifest = load_shared_dataset(d)
    with (d / 'node_catalog.csv').open(newline='') as f: catalog = sorted(csv.DictReader(f), key=lambda r: int(r['node_index']))
    assert [r['resource_id'] for r in catalog] == manifest['node_ids']
    applicable = torch.tensor([[feature_is_applicable(k, r['resource_id'], r['resource_type']) for k in CONTINUOUS_FEATURES] for r in catalog], dtype=torch.bool)
    old_norm = json.loads((d / 'normalization.json').read_text())
    payload, norm = prepare_inputs_in_memory(payload, views, applicable, old_norm)
    norm_sha = hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()
    print('FITTING_ONLY_NORMALIZATION_READY', len(views['fit']), norm_sha, flush=True)
    identity_common['fitting_normalization_sha256'] = norm_sha
    if args.phase == 'preflight':
        cases = []
        for name in ('b4', 'b5'):
            model, config, train, loss = make_model(name, payload, manifest)
            parent = next(r for r in near['runs'] if r['model'] == name and r['seed'] == 42)
            assert config.to_dict() == config.__class__(**parent['config']['model']).to_dict()
            assert json.loads(json.dumps(asdict(train))) == json.loads(json.dumps(asdict(train.__class__(**parent['config']['training']))))
            assert asdict(loss) == asdict(loss.__class__(**parent['config']['loss']))
            assert sum(p.numel() for p in model.parameters()) == (273054 if name == 'b4' else 285982)
            loader = DataLoader(FactoryBaselineTensorDataset(payload, views['fit'][:train.batch_size]), batch_size=train.batch_size)
            cpu = next(iter(loader)); batch = {k: v.cuda() for k, v in cpu.items()}
            masks = _occupancy_type_masks(d, payload, torch.device('cuda:0'))
            torch.cuda.reset_peak_memory_stats(); model.train()
            result = model(**_model_inputs(batch, model))
            total, components = compute_multitask_loss(result, batch, loss, torch.tensor(1., device='cuda:0'), occupancy_type_masks=masks)
            assert torch.isfinite(total) and all(torch.isfinite(v) for v in components.values())
            total.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
            cases.append(dict(model=name, model_config=config.to_dict(), training_config=asdict(train), loss_config=asdict(loss),
                parameters=sum(p.numel() for p in model.parameters()), batch_samples=len(cpu['sample_index']),
                loss=float(total.detach()), parent_near_configuration_verified=True, finite_gradients=True, optimizer_steps=0, cuda_peak_bytes=torch.cuda.max_memory_allocated()))
            print('HOLDOUT_PREFLIGHT_CASE', name, cases[-1]['parameters'], float(total.detach()), flush=True)
            del model, result, total, components, grads, batch; torch.cuda.empty_cache()
        guard(True)
        write_new(preflight_path, dict(status='two_models_real_fitting_batch_preflight_passed_no_optimizer_steps',
            identity=identity_common, normalization=norm, counts=counts, cases=cases, formal_weights_unchanged=True, test_evaluated=False))
        print('HOLDOUT_PREFLIGHT_COMPLETE', preflight_path.stat().st_size, sha(preflight_path), flush=True)
        return
    assert sha(preflight_path) == PREFLIGHT_SHA
    preflight = json.loads(preflight_path.read_text())
    validate_completed_preflight(preflight, identity_common, norm)
    all_results = []
    for name in ('b4', 'b5'):
        guard()
        model, config, train, loss = make_model(name, payload, manifest)
        identity = dict(identity_common, model=name, model_config=config.to_dict(), training_config=asdict(train), loss_config=asdict(loss), preflight_sha256=sha(preflight_path))
        tested = next(c for c in preflight['cases'] if c['model'] == name)
        assert_saved_configuration(config.to_dict(), train, loss, tested)
        last_path = d / f'baseline_{TAG}_{name}s42_progress.pt'
        summary_path = d / f'baseline_{TAG}_{name}s42_complete.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text()); assert summary['identity'] == identity
            for row in summary['diagnostics']: assert sha(d / row['file']) == row['sha256']
            all_results.append(summary); del model; torch.cuda.empty_cache(); continue
        if last_path.exists():
            state = torch.load(last_path, map_location='cpu', weights_only=False)
            assert state['identity'] == identity
            assert state['epoch'] == 60, 'Partial training requires explicit checkpoint recovery; never restart it automatically'
            history = state['history']; del state
            print('REUSING_COMPLETED_TRAINING', name, flush=True)
        else:
            for epoch in CHECKPOINT_EPOCHS:
                assert not (d / f'baseline_{TAG}_{name}s42_epoch{epoch}.pt').exists()
            generator = torch.Generator().manual_seed(42)
            loader = DataLoader(FactoryBaselineTensorDataset(payload, views['fit']), batch_size=train.batch_size,
                shuffle=True, generator=generator, num_workers=0, pin_memory=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate, weight_decay=train.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=train.lr_min)
            masks = _occupancy_type_masks(d, payload, torch.device('cuda:0')); history = []
            for epoch in range(1, 61):
                losses = _run_train_epoch(model, loader, optimizer, loss, torch.tensor(1., device='cuda:0'), torch.device('cuda:0'), train.gradient_clip_norm, masks)
                assert all(math.isfinite(v) for v in losses.values())
                history.append(dict(epoch=epoch, optimizer_steps=epoch * len(loader), learning_rate=optimizer.param_groups[0]['lr'], losses=losses))
                scheduler.step()
                state = snapshot(model, optimizer, scheduler, generator, epoch, history, identity)
                save_progress(last_path, state)
                if epoch in CHECKPOINT_EPOCHS:
                    path = d / f'baseline_{TAG}_{name}s42_epoch{epoch}.pt'
                    assert not path.exists(); torch.save(state, path)
                print('HOLDOUT_TRAIN_EPOCH', name, epoch, 'loss', losses['total'], flush=True)
            del optimizer, scheduler, state, loader, masks
        diagnostics = []
        for epoch in CHECKPOINT_EPOCHS:
            guard(); checkpoint = d / f'baseline_{TAG}_{name}s42_epoch{epoch}.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert state['identity'] == identity and state['epoch'] == epoch
            model.load_state_dict(state['model_state'], strict=True); del state
            for view in ('fit', 'heldout', 'original_validation'):
                diagnostics.append(evaluate_view(model, payload, views[view], train.batch_size, checkpoint, identity, d, view, epoch, counts[view]))
        guard(True)
        summary = dict(status='fixed_sixty_epoch_training_and_nine_views_complete', identity=identity, history=history,
            progress_file=last_path.name, progress_sha256=sha(last_path), diagnostics=diagnostics, test_evaluated=False, goal_met=False)
        write_new(summary_path, summary); all_results.append(summary)
        print('HOLDOUT_MODEL_COMPLETE', name, sha(summary_path), flush=True)
        del model; torch.cuda.empty_cache()
    guard(True)
    write_new(final_path, dict(status='two_fixed_budget_neural_holdout_diagnostics_complete', identity=identity_common,
        counts=counts, runs=all_results, original_frozen_split_modified=False, formal_weights_unchanged=True, test_evaluated=False, goal_met=False))
    print('HOLDOUT_BATCH_COMPLETE', final_path.stat().st_size, sha(final_path), flush=True)


if __name__ == '__main__': main()
