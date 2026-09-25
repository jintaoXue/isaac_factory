"""CPU tests for D/C human–task match scoring (no simulator)."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn

from source.algo.hierarchical.hc_factory.human_match import (
    HumanMatchQNetwork,
    MATCH_FEATURE_DIM,
    _skill_eff_clips,
    human_task_match_features,
    task_human_match_summary,
)
from source.algo.hierarchical.hc_factory.hier_networks import QNetwork
from source.algo.hierarchical.hc_factory.hier_rl_agents import RLHumanRobotAllocatorAgent, RLProcessTaskPlanningAgent


def pre():
    return {
        "z": torch.randn(8),
        "human": {
            "mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            "fatigue": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.0]),
            "efficiency": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.0]),
            "skill_task": torch.arange(72).reshape(6, 12).float() / 50 + 0.35,
            # control_machine is last column in cfg_human._HUMAN_SUBTASK_NAMES
            "skill_subtask": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.30],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.25],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.72],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.28],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.72],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "fatigue_work_rate": torch.ones(6) * 0.45,
            "fatigue_recover_rate": torch.ones(6) * 0.18,
        },
        "agent_action_mask": {
            "human": {"self_availability_mask": torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])},
            "robot": {"self_availability_mask": torch.tensor([1.0, 1.0, 0.0, 0.0])},
            "agent_C_process_task_planner": torch.ones(3, 13),
        },
    }


def task(i=2):
    return torch.nn.functional.one_hot(torch.tensor(i), 13).float()


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.parallel_producing_limit = 10
        self.state_dim = 8

    def _resolve_pre(self, raw, p):
        return raw if p is None else p

    def encode_D(self, p, a, *, pre=None):
        p = self._resolve_pre(p, pre)
        return torch.cat(
            (
                p["z"],
                a,
                p["agent_action_mask"]["human"]["self_availability_mask"],
                p["agent_action_mask"]["robot"]["self_availability_mask"],
            )
        )

    def encode_C(self, p, a, *, pre=None):
        p = self._resolve_pre(p, pre)
        return torch.cat((p["z"], a))

    def get_obs_dim_D(self, p, a):
        return self.encode_D(p, a).numel()

    def get_obs_dim_C(self, p, a):
        return self.encode_C(p, a).numel()


class MatchTests(unittest.TestCase):
    def test_features_use_speed_not_stale_effective(self):
        p = pre()
        f = human_task_match_features(p, task(2), 6)
        skill_t = p["human"]["skill_task"][:5, 1]
        skill_s = p["human"]["skill_subtask"][:5, -1]
        eta = p["human"]["efficiency"][:5]
        lo, hi = _skill_eff_clips()
        eff = (skill_t * skill_s).clamp(lo, hi)
        torch.testing.assert_close(f[:5, 2], skill_t)
        torch.testing.assert_close(f[:5, 3], skill_s)
        torch.testing.assert_close(f[:5, 4], (eta * eff).clamp(0.05, hi))
        self.assertEqual(f[5].abs().sum(), 0)
        self.assertFalse(torch.equal(f, human_task_match_features(p, task(3), 6)))

    def test_speed_clip_follows_legacy_profile(self):
        import os
        from source.algo.hierarchical.hc_factory.human_match import human_task_match_features as feats
        os.environ["HC_HUMAN_SKILL_PROFILE"] = "legacy"
        p = pre()
        # Force specialist-scale product > 1.80
        p["human"]["skill_task"][0] = torch.ones(12) * 1.40
        p["human"]["skill_subtask"][0, -1] = 1.35
        p["human"]["efficiency"][0] = 1.0
        f = feats(p, task(2), 6)
        self.assertLessEqual(f[0, 4].item(), 1.80 + 1e-5)
        os.environ["HC_HUMAN_SKILL_PROFILE"] = "strong"
        f2 = feats(p, task(2), 6)
        self.assertGreater(f2[0, 4].item(), 1.80)
        os.environ["HC_HUMAN_SKILL_PROFILE"] = "legacy"

    def test_prior_prefers_faster_worker_after_legacy_load(self):
        p = pre()
        old = RLHumanRobotAllocatorAgent(Encoder(), torch.device("cpu"), batch_size=1, hidden_dim=16)
        new = RLHumanRobotAllocatorAgent(
            Encoder(), torch.device("cpu"), human_match_head=True, batch_size=1, hidden_dim=16
        )
        for a in (old, new):
            a._ensure_dqn(p, task())
        new.human_dqn.q_net.load_compatible_state_dict(old.human_dqn.q_net.state_dict())
        obs = new.encode_human_obs(p, task())
        q = new.human_dqn.q_net(obs)
        feats = human_task_match_features(p, task(), 6)
        avail = [0, 2, 3, 4]
        order_q = sorted(avail, key=lambda i: q[i].item(), reverse=True)
        order_s = sorted(avail, key=lambda i: feats[i, 5].item(), reverse=True)
        self.assertEqual(order_q[0], order_s[0])

    def test_legacy_load_and_gradients(self):
        p = pre()
        old = RLHumanRobotAllocatorAgent(Encoder(), torch.device("cpu"), batch_size=1, hidden_dim=16)
        new = RLHumanRobotAllocatorAgent(
            Encoder(), torch.device("cpu"), human_match_head=True, batch_size=1, hidden_dim=16
        )
        for a in (old, new):
            a._ensure_dqn(p, task())
        new.human_dqn.q_net.load_compatible_state_dict(old.human_dqn.q_net.state_dict())
        self.assertIs(type(new.human_dqn.q_net), HumanMatchQNetwork)
        self.assertIs(type(new.robot_dqn.q_net), QNetwork)
        act = {"human": torch.nn.functional.one_hot(torch.tensor(0), 6), "robot": torch.zeros(4)}
        loss, _ = new.observe_step(p, task(), act, 2.0, p, False, 0.0)
        new.human_dqn.optimizer.zero_grad()
        loss.backward()
        self.assertGreater(new.human_dqn.q_net.human_tower[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(new.human_dqn.q_net.prior_weight.grad.abs().sum().item(), 0.0)

    def test_c_match_summary_and_agent(self):
        p = pre()
        summary = task_human_match_summary(p)
        self.assertEqual(summary.shape, (13, 5))
        self.assertEqual(summary[0].abs().sum(), 0)
        agent = RLProcessTaskPlanningAgent(
            Encoder(), torch.device("cpu"), task_match_head=True, hidden_dim=16, batch_size=1
        )
        sel = torch.tensor([1.0, 0.0, 0.0])
        agent._ensure_dqn(p, sel)
        obs = agent.encode_task_obs(p, sel)
        self.assertEqual(obs.numel(), agent.dqn.obs_dim)
        self.assertEqual(agent.dqn.q_net(obs).numel(), 13)

    def test_checkpoint_roundtrip(self):
        p = pre()
        agent = RLHumanRobotAllocatorAgent(
            Encoder(), torch.device("cpu"), human_match_head=True, batch_size=1, hidden_dim=16
        )
        agent._ensure_dqn(p, task())
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/d.pth"
            agent.human_dqn.save(path)
            other = RLHumanRobotAllocatorAgent(
                Encoder(), torch.device("cpu"), human_match_head=True, batch_size=1, hidden_dim=16
            )
            other._ensure_dqn(p, task())
            other.human_dqn.load(path)
            a = agent.encode_human_obs(p, task())
            torch.testing.assert_close(agent.human_dqn.q_net(a), other.human_dqn.q_net(a))


if __name__ == "__main__":
    unittest.main()
