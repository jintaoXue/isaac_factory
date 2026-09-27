"""R-series semantic regressions; CPU only, no Isaac engine or external services."""
import os
os.environ['HC_SIM_BACKEND'] = 'logic'
import copy
from collections import deque
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
from source.algo.hierarchical.hc_factory.decision_consistent import (
    DecisionReplay, DecisionPool, build_decision_action, check_checkpoint, event, HEADS)
from source.algo.hierarchical.hc_factory.hier_obs import HierObsEncoder
from source.algo.hierarchical.hc_factory.hier_rl_agents import (
    RLProductSequencingAgent, RLProductSelectionAgent, RLProcessTaskPlanningAgent, RLHumanRobotAllocatorAgent)
from source.algo.hierarchical.hc_factory.hierarchical_tpa import HierarchicalTPA
from source.algo.hierarchical.hc_factory.hier_buffer import ReplayBuffer
from source.hc_logic_runtime import HcLogicVectorEnv, HcLogicEnvCfg
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.src.curriculum import apply_train_order
from tools.run_r_series import recipe, command, VARIANTS


def owner(match=False, target=True):
    device=torch.device('cpu');enc=HierObsEncoder(device)
    kw=dict(batch_size=4, hidden_dim=16)
    o=SimpleNamespace(cuda_device=device,obs_encoder=enc,config={'decision_target_encoder':target},
        gamma=.9999,decision_reward_scale=.01,max_parallel_cd_dispatch=10,decision_consistent=True)
    o.agent_A=RLProductSequencingAgent(enc,device,**kw)
    o.agent_B=RLProductSelectionAgent(enc,device,**kw)
    o.agent_C=RLProcessTaskPlanningAgent(enc,device,**kw)
    o.agent_D=RLHumanRobotAllocatorAgent(enc,device,human_match_head=match,**kw)
    o.encoder_optimizer=torch.optim.Adam(enc.parameters(),lr=.0001)
    o.grad_clip_norm=10.;o._loss_window={h:deque(maxlen=10) for h in HEADS}
    o._joint_learn=lambda entries:HierarchicalTPA._joint_learn(o,entries)
    return o


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.o=SimpleNamespace(config={},gamma=.9,decision_reward_scale=1.,obs_encoder=None)
        self.r=DecisionReplay(self.o)
        self.buffers={h:ReplayBuffer(100) for h in HEADS}
        self.r.dqn=lambda h:SimpleNamespace(buffer=self.buffers[h])

    def e(self,head='C',context=1,action=1,dispatch=0):
        return event(head,{'x':torch.tensor([float(context)])},torch.eye(3)[action],
                     torch.tensor([0,1,1]),torch.eye(3)[context],dispatch)

    def test_event_is_an_immutable_cpu_snapshot(self):
        pre={"x":torch.tensor([1.])}
        e=event("A",pre,torch.tensor([1,0]),torch.tensor([1,1]))
        pre["x"].fill_(9.)
        self.assertEqual(float(e["pre"]["x"]),1.)

    def test_next_context_reward_discount_and_terminal(self):
        e=self.e();a={'dispatch_list':[{}],'_decision_trace':[e]}
        ok={'dispatch_outcomes':[{'index':0,'accepted':True}]}
        self.r.observe(0,a,ok,2,False)
        self.r.observe(0,{}, {},3,False)
        self.r.observe(0,{'dispatch_list':[{}],'_decision_trace':[self.e(context=2)]},ok,4,False)
        t=self.buffers['C'].buffer[0]
        self.assertAlmostEqual(t.reward,2+.9*3);self.assertAlmostEqual(t.discount,.9**2)
        self.assertEqual(int(t.context.argmax()),1);self.assertEqual(int(t.next_context.argmax()),2)
        self.assertEqual(t.next_mask.tolist(),[0,1,1])
        self.r.observe(0,{}, {},5,True)
        last=self.buffers['C'].buffer[-1]
        self.assertTrue(last.done);self.assertEqual(int(last.next_mask.sum()),0)
        self.assertAlmostEqual(last.reward,4+.9*5)
        self.assertNotIn(0,self.r.pending)

    def test_a_only_and_rejected_dispatch(self):
        a={'_decision_trace':[self.e('A',dispatch=None)]}
        self.r.observe(0,a,{},1,False)
        self.r.observe(0,{'dispatch_list':[{}],'_decision_trace':[self.e('D_human')]},
                       {'dispatch_outcomes':[{'index':0,'accepted':False}]},2,True)
        self.assertEqual(len(self.buffers['A']),1);self.assertEqual(len(self.buffers['D_human']),0)
        self.assertEqual(self.r.counts['A_only'],1)

    def test_zero_time_microsteps_and_restore(self):
        a={'dispatch_list':[{},{}],'_decision_trace':[self.e(),{**self.e(context=2),'dispatch':1}]}
        self.r.observe(0,a,{'dispatch_outcomes':[{'index':i,'accepted':True} for i in range(2)]},7,False)
        t=self.buffers['C'].buffer[0]
        self.assertEqual(t.reward,0);self.assertEqual(t.discount,1)
        self.r.observe(0,{}, {},-1,False,restored=True)
        self.assertEqual(len(self.buffers['C']),1);self.assertNotIn(0,self.r.pending)

    def test_missing_feedback_and_actual_robot_override(self):
        with self.assertRaisesRegex(RuntimeError,'feedback'):
            self.r.observe(0,{'dispatch_list':[{}]}, {},0,False)
        self.r.observe(0,{'dispatch_list':[{}],'_decision_trace':[self.e('D_robot')]},
            {'dispatch_outcomes':[{'index':0,'accepted':True,'robot_index':None}]},0,True)
        self.assertEqual(len(self.buffers['D_robot']),0)


class IntegrationTests(unittest.TestCase):
    def test_real_dispatch_trace_replay_and_target_update(self):
        env=HcLogicVectorEnv(HcLogicEnvCfg(cuda_device_str='cpu',seed=9002,
            train_cfg={'params':{'config':{'decision_consistent':True,'human_duration_aux':True}}}))
        state=env.reset()[0];apply_train_order(env.env_list[0],n_products=10,anchor=56000)
        o=owner(match=True);replay=DecisionReplay(o)
        accepted=requested=0; multi=False
        for tick in range(6000):
            before=copy.deepcopy(state['progress'])
            action=build_decision_action(state,o,1.)
            self.assertEqual(state['progress']['next_product'],before['next_product'])
            # Returned traces are detached snapshots; C-none is never a target candidate.
            for e in action['_decision_trace']:
                if e['head']=='C':self.assertEqual(int(e['mask'][0]),0)
            if len(action['dispatch_list'])>1:
                multi=True
                humans=[e for e in action['_decision_trace'] if e['head']=='D_human']
                self.assertEqual(int(humans[1]['mask'][humans[0]['action']]),0)
                self.assertEqual(int(humans[1]['pre']['agent_action_mask']['human']['self_availability_mask'][humans[0]['action']]),0)
            state=env.step([action])[0]
            requested+=len(action['dispatch_list'])
            accepted+=sum(x['accepted'] for x in state['rl']['dispatch_outcomes'])
            replay.observe(0,action,state['rl'],state['rl']['reward'],state['rl']['done'])
            if tick >= 499 and multi:break
        self.assertTrue(multi)
        self.assertEqual(float(o.agent_D.human_dqn.q_net.prior_weight),0.)
        self.assertGreater(requested,1);self.assertEqual(accepted,requested)
        self.assertGreater(replay.counts['A_only'],0)
        self.assertTrue(any(len(replay.dqn(h).buffer)>0 for h in HEADS))
        online_before=[p.clone() for p in o.obs_encoder.parameters()]
        target_before=[p.clone() for p in replay.target_encoder.parameters()]
        self.assertTrue(replay.learn())
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(online_before,o.obs_encoder.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(target_before,replay.target_encoder.parameters())))
        self.assertTrue(all(p.grad is None for p in replay.target_encoder.parameters()))
        self.assertNotEqual(replay.metrics()['MetricDecision/updates_D_human'],0)
        print('R integration:',dict(ticks=tick+1,requested=requested,accepted=accepted,multi=multi,
                                    A_only=replay.counts['A_only']))

    def test_checkpoint_semantics_and_recipes(self):
        o=owner();r=DecisionReplay(o)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'semantics'):check_checkpoint(o,d,1)
            r.save(d,1)
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):check_checkpoint(o,d,1)
            for n in ('state_encoder','agent_A','agent_B','agent_C','agent_D_human','agent_D_robot'):
                (Path(d)/f'{n}_step_1.pth').touch()
            check_checkpoint(o,d,1)
            with self.assertRaisesRegex(RuntimeError,'semantics'):
                check_checkpoint(SimpleNamespace(decision_consistent=False),d,1)
        for v in VARIANTS:
            self.assertFalse(recipe(v,True)['human_aware_reward'])
            argv=command(v,'cpu',{'HC_R_WANDB':'0'},True)
            self.assertIn('agent.params.config.decision_consistent=true',argv)
        self.assertFalse(recipe('R2')['human_aware_reward'])
        self.assertEqual(recipe('R2-mismatch')['human_overwork_coef'],0)
        self.assertEqual(recipe('R2-fatigue')['human_mismatch_coef'],0)
        o.config['decision_target_encoder']=False
        self.assertIsNone(DecisionReplay(o).target_encoder)
        with self.assertRaises(ValueError):command('eval-R2','cpu',{},True)


if __name__=='__main__':unittest.main()
