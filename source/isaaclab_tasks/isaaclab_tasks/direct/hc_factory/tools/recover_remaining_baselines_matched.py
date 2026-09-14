#!/usr/bin/env python3
"""Recover only the B2 Start5 missing-CSV-target verification failure, without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import types

ROOT = Path('/home/sci/work/BSTAN_isaac_factory')
TOOLS = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools'
DATASET = TOOLS.parent / 'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'
RUNTIME = 'cd29f5011a9c3bd14ea4aaad38ea9ad3bbc216a3'
TAG = 'remaining_matched20260914'
RECOVERY = TAG + '_recovery20260915'
SELF = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/recover_remaining_baselines_matched.py'
DRIVER = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/run_remaining_baselines_matched.py'
PYTHON = '/home/sci/repos/miniconda3/envs/env_isaaclab/bin/python'
PANE = 'baseline_dense_v6:0.0'
STATE = DATASET / (RECOVERY + '_state.json')
LOG = DATASET / (RECOVERY + '.log')
TASKS = [(m, s) for s in (5, 10, 15) for m in ('B2', 'B3')]


def out(*args):
    return subprocess.check_output(args, text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_failure(plan, record):
    if (plan['source_commit'] != RUNTIME or plan['status'] != 'failed'
            or plan['failed_stage'] != ['B2', 5] or plan['test_evaluated']
            or [(t['model'], t['max_start']) for t in plan['tasks']] != TASKS):
        raise ValueError('Recovery only applies to the original six-task B2 Start5 failure')
    if (record['status'] != 'failed' or record['source_commit'] != RUNTIME
            or record['model'] != 'B2' or record['max_start'] != 5
            or record['error'] != "'target_cause'" or record['test_evaluated']):
        raise ValueError('Do not recover an unknown failure or another training stage')


def require_runtime():
    if (Path.cwd().resolve() != ROOT or out('git', 'branch', '--show-current') != 'dev_xwt'
            or out('git', 'rev-parse', 'HEAD') != RUNTIME
            or out('git', 'status', '--porcelain', '--untracked-files=no')):
        raise ValueError('Keep the clean BSTAN dev_xwt runtime at cd29f50; fetch only, do not pull')


def failed_stage():
    require_runtime()
    plan = read(DATASET / (TAG + '_plan.json'))
    first = plan['tasks'][0]
    directory, record_path = Path(first['output_dir']), Path(first['record'])
    if (not directory.resolve().is_relative_to(DATASET / 'models')
            or record_path != directory / (TAG + '_s5.json')):
        raise ValueError('Unexpected B2 output or record path')
    record = read(record_path)
    validate_failure(plan, record)
    if (DATASET / (TAG + '_results.json')).exists():
        raise FileExistsError('The completed batch must not be recovered again')
    for task in plan['tasks'][1:]:
        if Path(task['record']).exists() or Path(task['archive']).exists():
            raise ValueError('A later stage already exists; inspect instead of duplicating it')
    summary = read(directory / 'run_summary.json')
    if summary['status'] != 'validation_completed' or summary['baseline_id'] != 'B2':
        raise ValueError('B2 has no complete saved training summary; do not refit automatically')
    return plan, record, summary


def launch(source):
    plan, record, _ = failed_stage()
    terminal = out('tmux', 'display-message', '-p', '-t', PANE,
                   '#{pane_pid} #{pane_dead} #{pane_dead_status}').split()
    if len(terminal) != 3 or terminal[1:] != ['1', '1']:
        raise ValueError('Original pane must have ended with exit 1; never replace a live job')
    command = out('tmux', 'display-message', '-p', '-t', PANE, '#{pane_start_command}')
    if 'run_remaining_baselines_matched.py' not in command or RUNTIME not in command:
        raise ValueError('The failed pane is not the original B2/B3 runner')
    diag = out('tmux', 'display-message', '-p', '-t', 'baseline_dense_diag:0.0',
               '#{pane_dead} #{pane_dead_status}')
    if diag.split() != ['1', '0']:
        raise ValueError('The B4/B5 finisher pane must remain successfully completed')
    if STATE.exists() or LOG.exists():
        raise FileExistsError('Recovery already exists; inspect its state/log, do not resubmit')
    saved = out('tmux', 'show-environment', '-g', 'PYTHONPATH')
    if not saved.startswith('PYTHONPATH=') or not Path(PYTHON).is_file():
        raise ValueError('Original Python environment is unavailable')
    # Validate source availability before creating any state; never update checkout.
    for path in (SELF, DRIVER):
        subprocess.check_output(['git', 'show', source + ':' + path])
    write(STATE, {'status': 'launching', 'repair_source_commit': source,
                 'runtime_commit': RUNTIME, 'failed_pane': terminal,
                 'failed_plan_sha256': sha(DATASET / (TAG + '_plan.json')),
                 'failed_record_sha256': sha(Path(plan['tasks'][0]['record'])),
                 'test_evaluated': False, 'b2_start5_refit': False}, exclusive=True)
    with LOG.open('x', encoding='utf-8') as stream:
        stream.write('Repair source: ' + source + '; original training runtime: ' + RUNTIME + '\n')
    loader = (f'import subprocess,sys; p={SELF!r}; s={source!r}; '
              "sys.argv=[p,'--source_commit',s,'--worker']; "
              "exec(compile(subprocess.check_output(['git','show',s+':'+p]),p,'exec'),"
              "{'__name__':'__main__','__file__':p})")
    env = ['env', '-u', 'PYTHONHOME', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2',
           'OPENBLAS_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'PYTHONPATH=' + str(TOOLS) + ':' + saved.split('=', 1)[1]]
    command = 'exec ' + shlex.join(env + [PYTHON, '-B', '-u', '-c', loader])
    command += ' >> ' + shlex.quote(str(LOG)) + ' 2>&1'
    subprocess.run(['tmux', 'respawn-pane', '-t', PANE, '-c', str(ROOT), command], check=True)
    print('恢复入口已提交：先核验已保存 B2 Start5，通过后跳过训练并继续剩余五组。')
    print('训练 checkout 保持 cd29f50，原失败日志和模型文件保留。')
    print('查看：tail -n 30 ' + str(LOG))


def worker(source):
    state = read(STATE)
    if state['status'] != 'launching' or state['repair_source_commit'] != source:
        raise ValueError('Recovery identity changed or has already run')
    try:
        plan, failed, summary = failed_stage()
        plan_path = DATASET / (TAG + '_plan.json')
        task = plan['tasks'][0]
        record_path, directory = Path(task['record']), Path(task['output_dir'])
        if sha(plan_path) != state['failed_plan_sha256'] or sha(record_path) != state['failed_record_sha256']:
            raise ValueError('Failed state changed between launch and verification')
        module = types.ModuleType('remaining_matched_fixed')
        module.__file__ = str(ROOT / DRIVER)
        exec(compile(subprocess.check_output(['git', 'show', source + ':' + DRIVER]), module.__file__, 'exec'), module.__dict__)
        module.torch.set_num_threads(2)
        module.require_runtime(RUNTIME)
        module.check_stats(plan['protected_stats'])
        if module.completed_b45() != plan['b45_final_sha256']:
            raise ValueError('B4/B5 completed result changed')
        prior = failed['prior_archive']
        if prior and sha(Path(prior['path'])) != prior['sha256']:
            raise ValueError('Pre-existing B2 model archive changed')
        before = {name: sha(directory / name) for name in module.FILES if (directory / name).exists()}
        state['status'] = 'verifying_saved_b2_without_refitting'
        write(STATE, state)
        print('VERIFYING_SAVED_B2_START5_NO_REFIT', flush=True)
        verified = module.verify_output(task, summary, RUNTIME, plan['label_counts']['5'])
        if before != verified['artifact_sha256']:
            raise ValueError('Verification changed a saved training artifact')
        # Preserve the original failure evidence byte-for-byte before updating registration.
        for path, suffix in ((plan_path, '_original_failed_plan.json'), (record_path, '_original_failed_b2s5.json')):
            with (DATASET / (RECOVERY + suffix)).open('xb') as stream:
                stream.write(path.read_bytes())
        repaired = {**verified, 'prior_archive': prior,
                    'recovery': {'repair_source_commit': source, 'failed_record_sha256': state['failed_record_sha256'],
                                 'b2_start5_refit': False, 'original_artifact_sha256': before,
                                 'method': 'Canonical scalar truth supplied in verifier memory; all saved artifact bytes retained'}}
        write(record_path, repaired)
        plan.update(status='registered', recovery={'repair_source_commit': source,
                    'original_failed_plan_sha256': state['failed_plan_sha256'],
                    'reused_completed_tasks': [['B2', 5]], 'csv_target_verification': 'original train/validation tensors'})
        plan.pop('failed_stage')
        write(plan_path, plan)
        state.update(status='running_remaining_five', saved_b2_artifact_sha256=before,
                     saved_b2_scores=verified['scores'])
        write(STATE, state)
        print('B2_START5_RECOVERED_WITHOUT_REFIT', json.dumps(verified['scores']['validation']), flush=True)
        # run() reuses the verified first record and begins with B3 Start5.
        # Core trainers are imported from the original frozen cd29f50 checkout.
        module.run(RUNTIME, 'cuda:0')
        state.update(status='completed', final_sha256=sha(DATASET / (TAG + '_results.json')))
        write(STATE, state)
        print('REMAINING_RECOVERY_COMPLETED', flush=True)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        write(STATE, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_commit', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    try:
        (worker if args.worker else launch)(args.source_commit)
    except Exception as error:
        print('REMAINING_RECOVERY_STOPPED', type(error).__name__, str(error), file=sys.stderr)
        raise
