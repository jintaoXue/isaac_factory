"""CPU regression tests for opt-in D-human residual, no simulator required."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from source.algo.hierarchical.hc_factory.human_pair import HumanPairQNetwork, human_task_pair_features, PAIR_FEATURE_DIM
from source.algo.hierarchical.hc_factory.hier_networks import QNetwork
from source.algo.hierarchical.hc_factory.hier_rl_agents import MaskedDQNAgent, RLHumanRobotAllocatorAgent
from source.algo.hierarchical.hc_factory import teacher_explore


def pre():
    return {'z':torch.randn(8), 'human':{
        'mask':torch.tensor([1.,1.,1.,1.,1.,0.]),
        'fatigue':torch.tensor([.1,.2,.3,.4,.5,0.]),
        'efficiency':torch.tensor([.9,.8,.7,.6,.5,0.]),
        'skill_task':torch.arange(72).reshape(6,12).float()/50+.35,
        'skill_effective':torch.full((6,),999.),
        'fatigue_work_rate':torch.ones(6)*.45,
        'fatigue_recover_rate':torch.ones(6)*.18},
        'agent_action_mask':{
            'human':{'self_availability_mask':torch.tensor([1.,0.,1.,1.,1.,0.])},
            'robot':{'self_availability_mask':torch.tensor([1.,1.,0.,0.])}}}


def task(i=2): return torch.nn.functional.one_hot(torch.tensor(i),13).float()


class Encoder(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__(); self.parallel_producing_limit=10; self.state_dim=8
    def _resolve_pre(self,raw,p): return raw if p is None else p
    def encode_D(self,p,a,*,pre=None):
        p=self._resolve_pre(p,pre)
        return torch.cat((p['z'],a,p['agent_action_mask']['human']['self_availability_mask'],p['agent_action_mask']['robot']['self_availability_mask']))
    def get_obs_dim_D(self,p,a): return self.encode_D(p,a).numel()


def agent(pair):
    return RLHumanRobotAllocatorAgent(Encoder(),torch.device('cpu'),human_pair_head=pair,batch_size=1,hidden_dim=16)


class PairTests(unittest.TestCase):
    def test_current_task_features_not_stale_effective_skill(self):
        p=pre(); f=human_task_pair_features(p,task(2),6)
        torch.testing.assert_close(f[:5,2],p['human']['skill_task'][:5,1])
        torch.testing.assert_close(f[:5,3],p['human']['efficiency'][:5]*p['human']['skill_task'][:5,1])
        self.assertEqual(f[5].abs().sum(),0)
        self.assertFalse(torch.equal(f,human_task_pair_features(p,task(3),6)))
        p['human']['skill_effective'].fill_(-99)
        torch.testing.assert_close(f,human_task_pair_features(p,task(2),6))

    def test_none_and_invalid_schema(self):
        p=pre(); self.assertEqual(human_task_pair_features(p,task(0),6)[:,2].sum(),0)
        p['human']['skill_task']=torch.zeros(6,11)
        with self.assertRaises(ValueError): human_task_pair_features(p,task(),6)

    def test_task_mapping_matches_source_tables(self):
        base=ROOT/'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg'
        def assignment(path,name):
            for n in ast.parse(path.read_text()).body:
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets): return ast.literal_eval(n.value)
        names=assignment(base/'cfg_human.py','_HUMAN_TASK_NAMES')
        gallery=assignment(base/'cfg_process_task_gallery.py','CfgProcessTaskGalleryInAll')
        self.assertEqual([gallery[n] for n in names],list(range(1,13)))

    def test_zero_residual_equal_to_old_q_and_masks(self):
        p=pre(); old=agent(False); new=agent(True)
        for x in (old,new): x._ensure_dqn(p,task())
        new.human_dqn.q_net.load_compatible_state_dict(old.human_dqn.q_net.state_dict())
        q0=old.human_dqn.q_net(old.encode_human_obs(p,task()))
        q1=new.human_dqn.q_net(new.encode_human_obs(p,task()))
        torch.testing.assert_close(q0,q1,rtol=0,atol=0)
        self.assertIs(type(old.human_dqn.q_net),QNetwork)
        self.assertIs(type(new.robot_dqn.q_net),QNetwork)
        for _ in range(10):
            a=new.act_with_masks(p,task(),{'human':p['agent_action_mask']['human']['self_availability_mask'],'robot':p['agent_action_mask']['robot']['self_availability_mask']},epsilon=1.)
            self.assertNotIn(a['human'].argmax().item(),(1,5))
        zero=torch.zeros(6)
        self.assertEqual(new.human_dqn.act_tensor(new.encode_human_obs(p,task()),zero,0).sum(),0)

    def test_gradients_and_target_update(self):
        p=pre(); new=agent(True); new._ensure_dqn(p,task())
        for _ in range(2):
            act={'human':torch.nn.functional.one_hot(torch.tensor(0),6),'robot':torch.zeros(4)}
            loss,_=new.observe_step(p,task(),act,3.,p,False,0.)
            self.assertTrue(torch.isfinite(loss)); new.human_dqn.optimizer.zero_grad(); loss.backward()
            grad=new.human_dqn.q_net.pair_residual[-1].weight.grad
            self.assertGreater(grad.abs().sum().item(),0.)
            new.human_dqn.optimizer.step()
        self.assertGreater(new.human_dqn.q_net.pair_residual[0].weight.grad.abs().sum().item(),0.)
        before=new.human_dqn.target_net.pair_residual[-1].weight.clone()
        new.human_dqn.register_train_step()
        self.assertFalse(torch.equal(before,new.human_dqn.target_net.pair_residual[-1].weight))

    def test_replay_next_state_and_offline_share_encoding(self):
        p=pre(); nxt=copy.deepcopy(p); nxt['human']['efficiency']*=.5
        a=agent(True); a._ensure_dqn(p,task())
        mask=p['agent_action_mask']['human']['self_availability_mask']
        a.human_dqn.store_pre(p,0,1.,nxt,mask,mask,False,context=task())
        transitions=list(a.human_dqn.buffer.buffer)
        encode=lambda p,t:a.encode_human_obs(p,t.context)
        current=a.human_dqn._encode_batch(transitions,encode,use_next=False)
        future=a.human_dqn._encode_batch(transitions,encode,use_next=True)
        self.assertFalse(torch.equal(current,future))
        torch.testing.assert_close(current[0],a.encode_human_obs(p,task()))
        loss=a.human_dqn.compute_loss(encode,offline_buffer=a.human_dqn.buffer,mix_ratio=1.)
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_roundtrip_and_wrong_switch_fails(self):
        p=pre(); a=agent(True); a._ensure_dqn(p,task())
        nn.init.constant_(a.human_dqn.q_net.pair_residual[-1].weight,.02)
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'q.pth'); a.human_dqn.save(path)
            b=agent(True); b._ensure_dqn(p,task()); b.human_dqn.load(path)
            x=a.encode_human_obs(p,task())
            torch.testing.assert_close(a.human_dqn.q_net(x),b.human_dqn.q_net(x),rtol=0,atol=0)
            torch.testing.assert_close(b.human_dqn.q_net(x),b.human_dqn.target_net(x),rtol=0,atol=0)
            c=agent(False); c._ensure_dqn(p,task())
            with self.assertRaisesRegex(RuntimeError,'human_pair_head=true'): c.human_dqn.load(path)

    def test_corrupt_base_and_partial_pair_rejected(self):
        q=HumanPairQNetwork(31+48,6,16)
        base=QNetwork(31,6,16).state_dict(); base.pop('net.0.bias')
        with self.assertRaises(RuntimeError): q.load_compatible_state_dict(base)
        broken=q.state_dict(); broken.pop('pair_residual.2.bias')
        with self.assertRaises(RuntimeError): q.load_compatible_state_dict(broken)

    def test_frozen_teacher_keeps_pair_interface_and_does_not_learn(self):
        p=pre(); a=agent(True); a._ensure_dqn(p,task())
        empty=SimpleNamespace(dqn=None,dqn_kwargs={})
        student=SimpleNamespace(obs_encoder=Encoder(),cuda_device=torch.device('cpu'),agent_A=empty,agent_B=empty,agent_C=empty,agent_D=a)
        with patch.object(teacher_explore,'HierObsEncoder',Encoder): teacher=teacher_explore.build_frozen_teacher(student)
        self.assertTrue(teacher.agent_D.human_pair_head)
        self.assertTrue(all(not x.requires_grad for x in teacher.agent_D.human_dqn.q_net.parameters()))
        x=a.encode_human_obs(p,task()); initial=teacher.agent_D.human_dqn.q_net(x).clone()
        with torch.no_grad(): a.human_dqn.q_net.pair_residual[-1].bias.add_(1)
        torch.testing.assert_close(initial,teacher.agent_D.human_dqn.q_net(x),rtol=0,atol=0)
        actions=teacher.agent_D.act_with_masks(p,task(),{'human':p['agent_action_mask']['human']['self_availability_mask'],'robot':p['agent_action_mask']['robot']['self_availability_mask']},0.)
        self.assertEqual(actions['human'].sum(),1)

    def test_real_t0_checkpoint_migration(self):
        path=ROOT/'logs/rl_games/HcFactory/hier_2026-08-27_23-17-41/nn/agent_D_human_step_1290000.pth'
        if not path.exists(): self.skipTest('T0 fixture not on this machine')
        state=torch.load(path,map_location='cpu',weights_only=True)['q_net']
        w=state.get('feature.0.weight',state.get('net.0.weight'))
        last=state.get('net.4.weight',state.get('net.0.weight'))
        h,dim=w.shape; actions=last.shape[0]
        old=MaskedDQNAgent('old',dim,actions,torch.device('cpu'),hidden_dim=h)
        new=MaskedDQNAgent('new',dim+actions*8,actions,torch.device('cpu'),hidden_dim=h,human_pair_dim=8)
        old.load(str(path)); new.load(str(path))
        x=torch.randn(4,dim); pairs=torch.zeros(4,actions,8); pairs[:,:,-1]=1
        torch.testing.assert_close(old.q_net(x),new.q_net(torch.cat((x,pairs.flatten(1)),1)),rtol=0,atol=0)

if __name__=='__main__': unittest.main()
