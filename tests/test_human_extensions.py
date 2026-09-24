"""CPU tests for optional C residual and completed-task auxiliary supervision."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import torch
from test_human_pair import Encoder, pre, task
from source.algo.hierarchical.hc_factory.human_pair import task_human_summary
from source.algo.hierarchical.hc_factory.hier_rl_agents import MaskedDQNAgent, RLProcessTaskPlanningAgent, RLHumanRobotAllocatorAgent
from source.algo.hierarchical.hc_factory.duration_aux import DurationAuxReplay
from source.algo.hierarchical.hc_factory import teacher_explore


class CEncoder(Encoder):
    def encode_C(self, p, a, *, pre=None):
        p = self._resolve_pre(p, pre)
        return torch.cat((p['z'], a))
    def get_obs_dim_C(self, p, a): return self.encode_C(p, a).numel()


def c_pre():
    p = pre()
    p['agent_action_mask']['agent_C_process_task_planner'] = torch.ones(3, 13)
    return p


def selection(): return torch.tensor([1., 0., 0.])


def c_agent(enabled):
    return RLProcessTaskPlanningAgent(CEncoder(), torch.device('cpu'), task_pair_head=enabled, hidden_dim=16, batch_size=1)


def d_agent(aux):
    return RLHumanRobotAllocatorAgent(CEncoder(), torch.device('cpu'), human_pair_head=True,
                                     duration_aux=aux, hidden_dim=16, batch_size=1)


def event(kind, *, product=1, human=0, start=10, duration=50):
    return dict(kind=kind, product=product, task=2, human=human, start=start, duration=duration)


class ExtensionTests(unittest.TestCase):
    def test_task_summary_masks_and_batch_reservations(self):
        p = c_pre(); x = task_human_summary(p)
        valid = torch.tensor([0, 2, 3, 4])
        eff = p['human']['efficiency'][valid, None] * p['human']['skill_task'][valid]
        torch.testing.assert_close(x[1:, 0], eff.max(0).values)
        torch.testing.assert_close(x[1:, 1], eff.mean(0))
        torch.testing.assert_close(x[1:, 2], torch.full((12,), .8))
        self.assertEqual(x[0].abs().sum(), 0)
        p['_task_pair_available'] = torch.zeros(6)
        self.assertEqual(task_human_summary(p).abs().sum(), 0)
        p['_task_pair_available'][0] = 1
        y = task_human_summary(p)
        torch.testing.assert_close(y[1:, 0], p['human']['efficiency'][0] * p['human']['skill_task'][0])
        self.assertEqual(y[1, 3].item(), p['human']['fatigue'][0].item())

    def test_c_zero_equivalence_replay_gradients_and_mask(self):
        p = c_pre(); old, new = c_agent(False), c_agent(True)
        for a in (old, new): a._ensure_dqn(p, selection())
        new.dqn.q_net.load_compatible_state_dict(old.dqn.q_net.state_dict())
        torch.testing.assert_close(old.dqn.q_net(old.encode_task_obs(p, selection())),
                                   new.dqn.q_net(new.encode_task_obs(p, selection())), rtol=0, atol=0)
        action = new.act_with_mask(p, selection(), task(3), 0.)
        self.assertEqual(action.argmax(), 3)
        p['_task_pair_available'] = torch.tensor([1., 0, 0, 0, 0, 0])
        nxt = c_pre()
        loss = new.observe_step(p, selection(), action, 2., nxt, False, 0.)
        new.dqn.optimizer.zero_grad(); loss.backward(); new.dqn.optimizer.step()
        self.assertGreater(new.dqn.q_net.pair_residual[-1].weight.grad.abs().sum(), 0)
        transitions = list(new.dqn.buffer.buffer)
        encode = lambda p,t: new.encode_task_obs(p, t.context)
        current = new.dqn._encode_batch(transitions, encode, use_next=False)
        future = new.dqn._encode_batch(transitions, encode, use_next=True)
        torch.testing.assert_close(current[0], new.encode_task_obs(p, selection()))
        torch.testing.assert_close(future[0], new.encode_task_obs(nxt, selection()))
        self.assertFalse(torch.equal(current, future))

    def test_c_checkpoint_and_teacher(self):
        p = c_pre(); c = c_agent(True); c._ensure_dqn(p, selection())
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d)/'c.pth'); c.dqn.save(path)
            other = c_agent(True); other._ensure_dqn(p, selection()); other.dqn.load(path)
            torch.testing.assert_close(c.dqn.q_net(c.encode_task_obs(p, selection())), other.dqn.q_net(other.encode_task_obs(p, selection())))
            old = c_agent(False); old._ensure_dqn(p, selection())
            with self.assertRaisesRegex(RuntimeError, 'task_pair_head=true'): old.dqn.load(path)
        d = d_agent(True); d._ensure_dqn(p, task())
        empty = SimpleNamespace(dqn=None, dqn_kwargs={})
        student = SimpleNamespace(obs_encoder=CEncoder(), cuda_device=torch.device('cpu'), agent_A=empty, agent_B=empty, agent_C=c, agent_D=d)
        with patch.object(teacher_explore, 'HierObsEncoder', CEncoder): frozen = teacher_explore.build_frozen_teacher(student)
        self.assertTrue(frozen.agent_C.task_pair_head)
        self.assertTrue(all(not v.requires_grad for v in frozen.agent_C.dqn.q_net.parameters()))
        self.assertEqual(frozen.agent_C.act_with_mask(p, selection(), task(2), 0.).argmax(), 2)
        self.assertTrue(all(not v.requires_grad for v in frozen.agent_D.human_dqn.q_net.parameters()))

    def test_aux_boundaries_and_snapshot_isolation(self):
        r = DurationAuxReplay(batch_size=1); p = pre(); original = p['z'].clone()
        r.observe(0, p, [event('start')]); p['z'].fill_(99)
        self.assertEqual(len(r.samples), 0)
        r.observe(1, p, [event('complete')])  # another env cannot complete env0 task
        self.assertEqual(r.unmatched, 1)
        r.observe(0, p, [event('complete')], done=True)
        self.assertEqual(len(r.samples), 1); self.assertEqual(len(r.pending), 0)
        torch.testing.assert_close(r.samples[0][0]['z'], original)
        r.observe(0, p, [event('start')], done=True)
        self.assertEqual(len(r.samples), 1); self.assertEqual(r.discarded, 1)
        r.observe(0, p, [event('start')])
        r.observe(0, p, [event('complete')], restored=True)
        self.assertEqual(len(r.samples), 1); self.assertEqual(r.discarded, 2)
        r.observe(0, p, [event('start'), event('complete', duration=0)])
        self.assertEqual(len(r.samples), 1)

    def test_aux_updates_shared_representation_and_checkpoint(self):
        p = pre(); old, a = d_agent(False), d_agent(True)
        for x in (old, a): x._ensure_dqn(p, task())
        a.human_dqn.q_net.load_compatible_state_dict(old.human_dqn.q_net.state_dict())
        obs = a.encode_human_obs(p, task())
        torch.testing.assert_close(old.human_dqn.q_net(obs), a.human_dqn.q_net(obs), rtol=0, atol=0)
        r = DurationAuxReplay(batch_size=1)
        r.observe(0, p, [event('start'), event('complete')])
        loss = r.compute_loss(a)
        a.human_dqn.optimizer.zero_grad(); loss.backward()
        self.assertGreater(a.human_dqn.q_net.pair_residual[0].weight.grad.abs().sum(), 0)
        self.assertGreater(a.human_dqn.q_net.duration_head.weight.grad.abs().sum(), 0)
        a.human_dqn.optimizer.step()
        self.assertEqual(r.metrics()['MetricAux/duration_updates'], 1)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d)/'aux.pth'); a.human_dqn.save(path)
            b = d_agent(True); b._ensure_dqn(p, task()); b.human_dqn.load(path)
            torch.testing.assert_close(a.human_dqn.q_net.predict_duration(obs), b.human_dqn.q_net.predict_duration(obs))
            with self.assertRaises(RuntimeError): old.human_dqn.load(path)
        state = dict(a.human_dqn.q_net.state_dict()); del state['duration_head.bias']
        with self.assertRaises(RuntimeError): a.human_dqn.q_net.load_compatible_state_dict(state)

    def test_real_t0_c_and_d_aux_migration(self):
        base = ROOT/'logs/rl_games/HcFactory/hier_2026-08-27_23-17-41/nn'
        for name, extra, kwargs in [('agent_C', 5, {'task_pair_head':True}), ('agent_D_human',8, {'human_pair_dim':8,'duration_aux':True})]:
            path = base/f'{name}_step_1290000.pth'
            if not path.exists(): self.skipTest('local T0 fixture unavailable')
            state = torch.load(path, map_location='cpu', weights_only=True)['q_net']
            hidden, dim = state['net.0.weight'].shape; actions = state['net.4.weight'].shape[0]
            old = MaskedDQNAgent(name, dim, actions, torch.device('cpu'), hidden_dim=hidden)
            new = MaskedDQNAgent(name, dim+actions*extra, actions, torch.device('cpu'), hidden_dim=hidden, **kwargs)
            old.load(str(path)); new.load(str(path))
            x = torch.randn(2,dim); pairs = torch.randn(2,actions*extra)
            torch.testing.assert_close(old.q_net(x), new.q_net(torch.cat((x,pairs),-1)),rtol=0,atol=0)

    def test_actual_environment_events_only_successful_starts(self):
        # Execute real manager methods without importing Isaac/Omniverse.
        path = ROOT/'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/task_progress_manager.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TaskManager')
        wanted = {'_duration_event','update_new_task_record','step_task_records'}
        methods = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Manager',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
        scope = dict(copy=copy,CfgProcessTaskGalleryDetailedClassified={'p':{'t':{}}},find_workstation_index_for_task=lambda *a:0,finalize_material_batch_task_done=lambda *a:None)
        exec(compile(tree,str(path),'exec'),scope)
        m = scope['Manager'](); m.duration_events_enabled=True
        m.human_reward=SimpleNamespace(on_assignment=lambda *a:None)
        m.apply_new_task_record_to_human_robot_machine_material=lambda *a:None
        m.initialze_subtasks=lambda *a:None
        state={'time_step':10,'rl':{'duration_events':[]},'progress':{'ongoing_task_records':{}},'machine':{'m':{'state':[], 'key_variables':{'working_area_ids':{'w':0}}}},'material':{'num_01_p':{}}}
        record=dict(human='h',human_index=0,product_index=1,product='p',task='t',task_index=2,task_type='processing',target_machine='m',is_final_task=False)
        self.assertFalse(m.update_new_task_record(state,copy.deepcopy(record)))
        self.assertEqual(state['rl']['duration_events'],[])
        m.initialze_subtasks=lambda *a:{}
        self.assertTrue(m.update_new_task_record(state,copy.deepcopy(record)))
        self.assertEqual(state['rl']['duration_events'][0]['kind'],'start')
        m._check_subtask_and_task_done=lambda *a:True; state['time_step']=29
        m.step_task_records(state)
        self.assertEqual(state['rl']['duration_events'][-1]['duration'],20)
        self.assertEqual(state['progress']['ongoing_task_records'],{})


if __name__ == '__main__': unittest.main()
