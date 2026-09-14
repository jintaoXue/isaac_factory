#!/usr/bin/env python3
"""Read and verify the twelve completed B2--B5 tasks; print a compact report source."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from verify_baseline_matched_results import artifact_snapshot, close, sha, station_counts

ROOT = Path('/home/sci/work/BSTAN_isaac_factory')
DATASET = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
GROUPS = (
    ('baseline_matched_protocol_training20260913', ('B4', 'B5'),
     'baseline_matched_protocol_training20260913_final_verification.json',
     'six_matched_tasks_independently_verified', 'model_before_matched20260913_s'),
    ('remaining_matched20260914', ('B2', 'B3'),
     'remaining_matched20260914_results.json', 'six_remaining_tasks_verified',
     'model_before_remaining_matched20260914_s'),
)


def read(path):
    return json.loads(path.read_text())


def validate_identities(rows, expected):
    if [(r['model'], r['max_start']) for r in rows] != expected:
        raise ValueError('Missing, duplicate or reordered registered task')


def compact_metrics(metrics):
    report, cause, remain = (metrics[k] for k in ('station_report', 'cause', 'remain'))
    result = {k: report[k] for k in (
        'will15_precision', 'will15_recall', 'will15_f1', 'report_threshold_used',
        'n_true_upcoming', 'n_matched_report_upcoming', 'n_matched_who_upcoming',
        'report_recall_upcoming', 'who_recall_upcoming', 'who_recall_ongoing',
        'n_true_ongoing', 'n_matched_who_ongoing')}
    result.update(sample_count=metrics['sample_count'], cause_acc=cause['cause_acc'],
                  cause_macro_recall=cause['cause_macro_recall'], cause_n=cause['cause_n'],
                  remain_len_mae_middle_weighted=remain['remain_len_mae_middle_weighted'],
                  remain_len_mae=remain['remain_len_mae'])
    return result


def collect(dataset):
    prior_path = dataset / 'baseline_matched_protocol_preflight20260913.json'
    prior = read(prior_path)
    if sha(prior_path) != '532d3b715d95106d54ba1fae01dc8bbc0e5d6995cfdda4ff2be2216b51281904':
        raise ValueError('Common protocol preflight changed')
    manifest_sha = sha(dataset / 'dataset_manifest.json')
    if manifest_sha != 'e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f':
        raise ValueError('Source dataset changed')
    sources, results = {}, []
    for tag, models, final_name, status, archive_prefix in GROUPS:
        plan_path, final_path = dataset / (tag + '_plan.json'), dataset / final_name
        plan, final = read(plan_path), read(final_path)
        if plan['status'] != 'completed' or final['status'] != status or final['test_evaluated']:
            raise ValueError('All tasks must be completed without test evaluation')
        expected = [(m, cap) for cap in (5, 10, 15) for m in models]
        validate_identities(plan['tasks'], expected)
        validate_identities(final['results'], expected)
        if models == ('B4', 'B5'):
            batch_path = dataset / (tag + '_results.json')
            batch = read(batch_path)
            if (sha(batch_path) != final['batch_result_sha256'] or batch['status'] != 'completed'
                    or batch['test_evaluated'] or sha(plan_path) != final['plan_sha256']):
                raise ValueError('B4/B5 completed batch or plan changed')
            validate_identities(batch['tasks'], expected)
            finish = read(dataset / 'baseline_matched_protocol_finish20260914_state.json')
            if finish['status'] != 'completed' or finish['final_sha256'] != sha(final_path):
                raise ValueError('B4/B5 completion state changed')
            for name, digest in final['stage_verification_sha256'].items():
                if sha(dataset / name) != digest:
                    raise ValueError('B4/B5 numerical verification proof changed')
        else:
            if plan['source_commit'] != final['source_commit'] or plan['test_evaluated']:
                raise ValueError('B2/B3 runtime or test scope changed')
            if sha(dataset / GROUPS[0][2]) != plan['b45_final_sha256'] or final['b45_final_sha256'] != plan['b45_final_sha256']:
                raise ValueError('B4/B5 results were not preserved through B2/B3')
            for name, stat in plan['protected_stats'].items():
                s = (dataset / name).stat()
                if [s.st_size, s.st_mtime_ns] != stat:
                    raise ValueError('Protected pre-existing artifact changed: ' + name)
        for position, (task, saved) in enumerate(zip(plan['tasks'], final['results'])):
            directory, record_path = Path(task['output_dir']), Path(task['record'])
            if not directory.resolve().is_relative_to((dataset / 'models').resolve()) or record_path.parent != directory:
                raise ValueError('Record outside the authorized model directories')
            record = read(record_path)
            cap, model = task['max_start'], task['model']
            if (record['status'] != 'validation_completed' or record['test_evaluated']
                    or (record['model'], record['max_start']) != (model, cap)
                    or record['source_commit'] != plan['source_commit']):
                raise ValueError('Wrong completed stage identity')
            archive = directory / f'{archive_prefix}{cap + 5}.zip' if cap < 15 else None
            files, origin = artifact_snapshot(directory, archive, record['artifact_sha256'])
            summary, metrics, config = (json.loads(files[name]) for name in ('run_summary.json', 'metrics.json', 'config.json'))
            if (summary != record['summary'] or metrics != record['metrics'] or metrics != saved['metrics']
                    or set(metrics) != {'train', 'validation'} or summary['status'] != 'validation_completed'):
                raise ValueError('Saved numerical results differ from the verified artifacts')
            training = config.get('training', config.get('config'))
            if training != task['training'] or training['evaluate_test'] or not training['evaluate_train'] or training['seed'] != 42:
                raise ValueError('Training recipe or evaluation scope changed')
            for split, values in metrics.items():
                contract = {**prior['tasks'][str(cap)]['contract'], 'window_size_s': 60.}
                if values['evaluation_contract'] != contract:
                    raise ValueError('Evaluation contract changed')
                score = station_counts(values['station_report'], prior['tasks'][str(cap)]['splits'][split])
                if score != saved['scores'][split]:
                    raise ValueError('Count-derived station scores differ')
                close(score['threshold'], summary['event_report_threshold'])
                close(values['remain']['remain_len_mae_primary'], values['remain']['remain_len_mae_middle_weighted'])
            if model in ('B4', 'B5'):
                if record != batch['tasks'][position] or summary['best_epoch'] != saved['selected_epoch'] or summary['epochs_trained'] != saved['epochs_trained']:
                    raise ValueError('B4/B5 final selection changed')
                proof_name = f'baseline_matched_{model.lower()}_start{cap}_verification20260913.json'
                if read(dataset / proof_name)['record_sha256'] != sha(record_path):
                    raise ValueError('B4/B5 record differs from its numerical proof')
            else:
                if record != saved:
                    raise ValueError('B2/B3 final record changed')
                if record['prior_archive'] and sha(Path(record['prior_archive']['path'])) != record['prior_archive']['sha256']:
                    raise ValueError('B2/B3 prior archive changed')
            result = {'model': model, 'max_start': cap, 'seed': training['seed'],
                      'source_commit': record['source_commit'], 'artifact_origin': origin,
                      'record_sha256': sha(record_path), 'checkpoint_constraint_met': summary['checkpoint_constraint_met'],
                      'initialization': summary['initialization'],
                      'selected_epoch': summary.get('best_epoch'), 'epochs_trained': summary.get('epochs_trained'),
                      'boosting_rounds_per_head': summary.get('n_estimators_per_head'),
                      'training_budget': summary.get('training_budget'),
                      **{split: compact_metrics(values) for split, values in metrics.items()}}
            results.append(result)
        sources[final_name] = {'sha256': sha(final_path), 'plan_sha256': sha(plan_path),
                               'runtime_commit': plan['source_commit'],
                               'verification_repair_commit': plan.get('recovery', {}).get('repair_source_commit')}
    results.sort(key=lambda r: (r['max_start'], r['model']))
    validate_identities(results, [(m, s) for s in (5, 10, 15) for m in ('B2', 'B3', 'B4', 'B5')])
    return {'status': 'twelve_matched_baseline_results_verified', 'test_evaluated': False,
            'dataset_manifest_sha256': manifest_sha, 'sources': sources, 'results': results,
            'verification_scope': 'Completed plans and final proofs; all current/archive artifact hashes; frozen metrics, counts and selected result identities. No model inference or new selection.'}


if __name__ == '__main__':
    if Path.cwd().resolve() != ROOT or subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'dev_xwt':
        raise ValueError('Use only BSTAN_isaac_factory on dev_xwt')
    print(json.dumps(collect(DATASET), ensure_ascii=False, separators=(',', ':'), allow_nan=False))
