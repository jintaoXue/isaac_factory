#!/usr/bin/env python3
"""Explicit, reproducible R-series launch recipes; no changes to G/E launch recipes."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

if __package__:
    from .training_supervisor import supervise
else:
    from training_supervisor import supervise

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('R0', 'R1', 'R2', 'R2-mismatch', 'R2-fatigue', 'R2-both', 'R2-greedy')


def recipe(variant, evaluation=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    mismatch = variant in ('R2-mismatch', 'R2-both') and not evaluation
    fatigue = variant in ('R2-fatigue', 'R2-both') and not evaluation
    return dict(decision_consistent=True, decision_target_encoder=variant != 'R0',
                human_match_head=variant.startswith('R2') and variant != 'R2-greedy',
                human_policy='greedy' if variant == 'R2-greedy' else 'rl',
                human_aware_reward=mismatch or fatigue, human_reward_metrics=True,
                human_mismatch_coef=.05 if mismatch else 0.,
                human_overwork_coef=.01 if fatigue else 0., human_recovery_coef=0.,
                human_shaping_cap=.04, human_fatigue_threshold=.8)


def command(mode, device, env, dry_run=False):
    evaluation = mode.startswith('eval-')
    variant = mode.removeprefix('eval-')
    knobs = recipe(variant, evaluation)
    seed = int(env.get('HC_R_SEED', '42'))
    base = env.get('HC_R_RUN_TAG', f'{variant}-S{seed}')
    # An eval must never silently pick its checkpoint using test-set performance.
    if evaluation:
        directory, step = env.get('HC_LOAD_DIR'), env.get('HC_LOAD_STEP')
        if not directory or step is None:
            raise ValueError('eval-R requires HC_LOAD_DIR and HC_LOAD_STEP (select on development seeds first)')
        int(step)
        base = f'{variant}-S{seed}-step{step}-eval'
    else:
        directory = step = None
    if Path(base).name != base or base in ('.', '..'):
        raise ValueError('HC_R_RUN_TAG must be a directory name, not a path')
    name, version = base, 0
    prefix = 'test_hier_' if evaluation else 'hier_'
    while (ROOT / 'logs/rl_games/HcFactory' / f'{prefix}{name}').exists():
        version += 1
        name = f'{base}-v{version}'
    argv = [sys.executable, str(ROOT/'train.py'), '--task', 'HRTPaHC-v1', '--algo', 'hier',
            '--device', device, '--num_envs', '1', '--headless', '--seed', str(seed),
            '--train_n_products', '10', '--max_parallel_cd_dispatch', '10',
            '--ftg_thresh_phy', '.95', '--algo_variant', variant,
            '--wandb_name', f'{name}-N10']
    if evaluation:
        argv += ['--test', '--test_epsilon', '0', '--test_seeds',
                 env.get('HC_TEST_SEEDS', '43,44,45,46,47,48,49,50,51,52'),
                 '--test_times', env.get('HC_TEST_TIMES', '1'), '--load_dir', directory, '--load_step', step]
    else:
        argv += ['--max_sim_episodes', env.get('HC_MAX_HARD_EPISODES', '200')]
    if env.get('HC_R_WANDB', '1') != '0':
        argv += ['--wandb_activate', '--wandb_project',
                 env.get('HC_WANDB_TEST_PROJECT' if evaluation else 'HC_WANDB_TRAIN_PROJECT',
                         'HcFactory_TPA_Eval' if evaluation else 'HcFactory_TPA')]
    # N10 T=25000; N16 uses the same per-product budget (anchor 40000 → 40000).
    knobs.update(dict(t_max_anchor=40000, max_episodic_steps=25000, parallel_producing_limit=10,
        c_forbid_none_mode='always', prioritized_replay=True, dueling_dqn=True,
        gamma=.9999, decision_reward_scale=.01, learning_rate=.0001, encoder_learning_rate=.0001,
        epsilon_start=1., epsilon_end=.05, epsilon_decay_steps=1500000,
        batch_size=64, batch_size_A=16, replay_buffer_size=50000, replay_buffer_size_A=5000,
        target_tau=.005, learn_interval=8, greedy_human_eval=False,
        warmstart='', load_name='', wandb_activate=env.get('HC_R_WANDB', '1') != '0'))
    for key in ('oru', 'teacher_explore', 'autoregressive', 'curriculum', 'explore', 'explore_catalog',
                'catalog_collect', 'teacher_collect', 'human_pair_head', 'task_pair_head', 'task_match_head',
                'human_duration_aux', 'noisy_net', 'hierarchical_credit', 'b_score_rl', 'env_rule_based_exploration'):
        knobs[key] = False
    if not evaluation:
        knobs['load_dir'] = ''
    argv.append('+agent.params.config.full_experiment_name=' + name)
    for key, value in knobs.items():
        argv.append('agent.params.config.' + key + '=' + json.dumps(value, separators=(',', ':')))
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=VARIANTS + tuple('eval-' + v for v in VARIANTS))
    parser.add_argument('device', nargs='?', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    env = dict(os.environ)
    # Read identity only for a real run. Never print secrets or execute an env file as shell code.
    if not args.dry_run and (ROOT/'.wandb_local.env').exists():
        for raw in (ROOT/'.wandb_local.env').read_text().splitlines():
            line = raw.strip().removeprefix('export ')
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key.strip().startswith(('WANDB_', 'HC_WANDB_')):
                fields = shlex.split(value, comments=True)
                env[key.strip()] = fields[0] if fields else ''
    env['HC_HUMAN_SKILL_PROFILE'] = 'gap'
    env['HC_WARMSTART'] = ''
    for key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'HC_WANDB_RUN_ID', 'HC_WANDB_RESUME'):
        env.pop(key, None)
    try:
        argv = command(args.mode, args.device, env, args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))
    print('[R] gap / N10 / K10 / T25000; evaluation epsilon=0; reward ablations are separate', flush=True)
    if args.dry_run:
        print(shlex.join(argv))
        return
    raise SystemExit(supervise(argv, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
