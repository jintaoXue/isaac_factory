"""CPU checks of real reward/dispatch/logging methods without importing Isaac Sim."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import re
import types
import unittest
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory'
ALGO = ROOT / 'source/algo/hierarchical/hc_factory'
spec = importlib.util.spec_from_file_location('human_reward', ENV / 'src/human_aware_reward.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HumanAwareReward = module.HumanAwareReward
# Execute the real human-factor constants/functions, excluding simulator asset setup.
text = (ENV / 'env_asset_cfg/cfg_human.py').read_text()
physics = {'os': __import__('os')}
exec(text[text.index('HUMAN_EFFICIENCY_ETA_MIN ='):text.index('# Static privileged obs')], physics)
skill, efficiency = physics['human_effective_skill'], physics['human_efficiency']


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


ns = dict(torch=types.SimpleNamespace(device=object), copy=copy, CfgStorage={},
          CfgProcessTaskGalleryInAll={}, HumanAwareReward=HumanAwareReward,
          human_effective_skill=skill, human_efficiency=efficiency,
          CfgProcessTaskGalleryDetailedClassified={'P': {'pipe_cutting': {}}})
extract(ENV/'src/task_progress_manager.py', {'TaskManager', 'find_workstation_index_for_task'}, ns)
TaskManager = ns['TaskManager']
log_ns = dict(Any=Any, re=re, _DEFAULT_NUM_HUMAN=5, _DEFAULT_FATIGUE_EMA_ALPHA=0.02)
extract(ALGO/'wandb_metrics.py', {'_human_idx_from_entity', 'HumanFatigueTracker', 'HumanFatigueMonitor'}, log_ns)


def state():
    return {'time_step': 0, 'human': {
        f'num_{i:02d}_Human': {'fatigue': 0., 'state': 'free', 'ongoing_task_record_index': None}
        for i in range(5)}, 'progress': {'ongoing_task_records': {}, 'production_done': False}}


def reward(**cfg):
    return HumanAwareReward(dict(human_aware_reward=True, **cfg), skill, efficiency)


def assignment(i):
    return dict(human=f'num_{i:02d}_Human', task='pipe_cutting')


class RewardTests(unittest.TestCase):
    def test_disabled_exact_legacy_reward(self):
        m = TaskManager(None, step_penalty=.08, success_bonus=50)
        s = state(); m.human_reward.begin_step(s)
        m.update_rl_signals(s, {'n_task_done': 2, 'n_product_finished': 1})
        self.assertEqual(s['rl']['reward'], -.08 + 2 + .2)
        self.assertEqual(set(s['rl']['reward_parts']), {'step','finish','task','success'})
        self.assertNotIn('human_reward_stats', s['rl'])

    def test_specialist_is_not_penalized(self):
        h=reward(); s=state(); h.begin_step(s); h.on_assignment(s,assignment(0))
        p, k=h.finish_step(s)
        self.assertEqual(p['human_mismatch'], 0)
        self.assertEqual(k['assignments'], 1)

    def test_available_specialist_makes_mismatch_negative(self):
        h=reward(); s=state(); h.begin_step(s); h.on_assignment(s,assignment(1))
        p,k=h.finish_step(s)
        self.assertAlmostEqual(p['human_mismatch'], -.05*(1-.58/1.4))
        self.assertEqual(k['mismatch_count'], 1)
        self.assertEqual(k['assigned_skill_sum'], .58)

    def test_busy_specialist_is_excluded(self):
        h=reward(); s=state()
        for i,e in enumerate(s['human'].values()):
            if i != 1: e['state']='working'
        h.begin_step(s); h.on_assignment(s,assignment(1))
        self.assertEqual(h.finish_step(s)[0]['human_mismatch'], 0)

    def test_fatigue_can_reverse_preferred_worker(self):
        h=reward(); s=state(); s['human']['num_00_Human']['fatigue']=1.
        h.begin_step(s); h.on_assignment(s,assignment(0))
        self.assertLess(h.finish_step(s)[0]['human_mismatch'], 0)

    def test_overwork_only_for_active_subtask(self):
        s=state(); e=s['human']['num_00_Human']; e.update(fatigue=.9, state='working', ongoing_task_record_index=0)
        s['progress']['ongoing_task_records'][0]={'subtasks_dict':{'ongoing':['control_machine']}}
        h=reward(); h.begin_step(s)
        self.assertAlmostEqual(h.finish_step(s)[0]['human_overwork'], -.01*.25/5)
        s['progress']['ongoing_task_records'][0]['subtasks_dict']['ongoing']=['wait']
        h.begin_step(s); self.assertEqual(h.finish_step(s)[0]['human_overwork'],0)

    def test_recovery_optional_and_no_reset_bonus(self):
        s=state(); s['human']['num_00_Human']['fatigue']=.9
        h=reward(human_recovery_coef=.5); h.begin_step(s)
        s['human']['num_00_Human']['fatigue']=.899
        self.assertAlmostEqual(h.finish_step(s)[0]['human_recovery'],.5*.001/5)
        h.reset(); self.assertEqual(h.finish_step(state())[0]['human_recovery'],0)

    def test_cap_and_events_consumed(self):
        s=state(); h=reward(human_mismatch_coef=100)
        h.begin_step(s); h.on_assignment(s,assignment(1)); p,k=h.finish_step(s)
        self.assertAlmostEqual(sum(abs(x) for x in p.values()),.04)
        self.assertEqual(k['cap_hit'],1)
        self.assertEqual(h.finish_step(s)[1]['assignments'],0)

    def test_eval_metrics_without_reward_change(self):
        s=state(); h=HumanAwareReward({'human_reward_metrics':True},skill,efficiency)
        h.begin_step(s); h.on_assignment(s,assignment(1)); p,k=h.finish_step(s)
        self.assertEqual(sum(p.values()),0); self.assertEqual(k['assignments'],1)
        self.assertEqual(k['enabled'],0)

    def test_reward_parts_integrate_and_terminal_is_preserved(self):
        m=TaskManager(None,step_penalty=.08,success_bonus=50); m.configure_human_reward({'human_aware_reward':True})
        s=state(); m.human_reward.begin_step(s); m.human_reward.on_assignment(s,assignment(1))
        s['progress']['production_done']=True; m.update_rl_signals(s)
        self.assertTrue(s['rl']['success']); self.assertAlmostEqual(s['rl']['reward'],sum(s['rl']['reward_parts'].values()))
        tracker=log_ns['HumanFatigueTracker']()
        tracker.update(state(), rl=s['rl'])  # terminal reward with reset observation
        result=tracker.episode_payload(episode=1)
        self.assertEqual(result['MetricHuman/ep_dispatch_count'],1)
        self.assertLess(result['MetricReward/ep_human_mismatch_sum'],0)
        self.assertEqual(result['MetricHuman/ep_mismatch_rate'],1)
        tracker.reset_episode(); self.assertNotIn('MetricHuman/ep_dispatch_count',tracker.episode_payload(episode=2))

    def test_successful_dispatch_only_and_multiple_dispatch_availability(self):
        m=TaskManager(None); m.configure_human_reward({'human_aware_reward':True})
        s=state(); s['machine']={'M':{'state':['materialReadyFor_pipe_cutting'], 'ongoing_task_record_index':[None], 'key_variables':{'working_area_ids':{'w0':[0]}}}}
        s['material']={'num_00_P':{'ongoing_task_record_index':None}}
        record=dict(assignment(1),product='P',product_index=0,target_machine='M',task_type='processing',robot=None,logistic_machine=None)
        m.human_reward.begin_step(s); m.initialze_subtasks=lambda *args: None
        self.assertFalse(m.update_new_task_record(s,copy.deepcopy(record)))
        self.assertEqual(len(m.human_reward.assignments),0)
        m.initialze_subtasks=lambda *args: {}
        self.assertTrue(m.update_new_task_record(s,copy.deepcopy(record)))
        self.assertEqual(len(m.human_reward.assignments),1)
        self.assertNotEqual(s['human']['num_01_Human']['state'],'free')
        with self.assertRaises(ValueError): m.human_reward.on_assignment(s,assignment(1))

    def test_real_duration_grows_with_fatigue(self):
        tree=ast.parse((ENV/'src/human.py').read_text())
        method=next(m for c in tree.body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_time_counting_subtask')
        duration_ns=dict(human_effective_skill=skill, math=math, CfgSubtaskPredefinedTimeGallery={'control_machine':200}, sample_noisy_steps=lambda base,std:base, SubtaskTimeNoiseStdSteps=0)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'human.py','exec'),duration_ns)
        values=[]
        for fatigue in (0.,.9):
            actor=types.SimpleNamespace(idx=0,state={'efficiency':efficiency(fatigue)})
            duration_ns['_time_counting_subtask'](actor,{'finished':[False]},'control_machine',task_name='pipe_cutting')
            values.append(actor.state['subtask_time_target'])
        self.assertGreater(values[1],values[0])
        self.assertGreater(physics['human_step_fatigue'](0,.5,idle=False,subtask_name='control_machine'),.5)
        self.assertLess(physics['human_step_fatigue'](0,.5,idle=True,subtask_name=None),.5)

    def test_bad_hyperparameters_rejected(self):
        for cfg in ({'human_mismatch_coef':-1},{'human_shaping_cap':float('nan')},{'human_fatigue_threshold':1}):
            with self.assertRaises(ValueError): reward(**cfg)


if __name__ == '__main__': unittest.main()
