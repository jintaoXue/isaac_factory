"""Safety and integration checks for the fixed human rule."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch
from test_human_match import pre, task, Encoder
from source.algo.hierarchical.hc_factory.greedy_human import greedy_human_action
from source.algo.hierarchical.hc_factory.hier_rl_agents import RLHumanRobotAllocatorAgent


class GreedyHumanTests(unittest.TestCase):
    def setUp(self):
        self.p = pre()
        self.p['human']['skill_task'].fill_(1)
        self.p['human']['efficiency'] = torch.tensor([.2, .9, .5, .4, .3, 1.])
        self.mask = torch.tensor([1, 1, 1, 1, 1, 0])

    def test_current_pool_mask_and_nonexistent_workers(self):
        # Observation says worker 1 unavailable, but current dispatch mask is authoritative.
        self.assertEqual(greedy_human_action(self.p, task(), self.mask).argmax(), 1)
        self.mask[1] = 0
        self.mask[5] = 1  # nonexistent slot must never win
        self.assertEqual(greedy_human_action(self.p, task(), self.mask).argmax(), 2)

    def test_fatigue_efficiency_skill_and_ties(self):
        self.p['human']['efficiency'][1] = .1
        self.assertEqual(greedy_human_action(self.p, task(), self.mask).argmax(), 2)
        self.p['human']['skill_task'][0, 1] = 1.8
        self.p['human']['efficiency'][0] = .4
        self.assertEqual(greedy_human_action(self.p, task(), self.mask).argmax(), 0)
        self.p['human']['skill_task'].fill_(1)
        self.p['human']['efficiency'].fill_(.5)
        self.assertEqual(greedy_human_action(self.p, task(), self.mask).argmax(), 0)

    def test_no_task_or_no_candidates(self):
        self.assertEqual(greedy_human_action(self.p, task(0), self.mask).sum(), 0)
        self.assertEqual(greedy_human_action(self.p, task(), self.mask * 0).sum(), 0)

    def test_allocator_replaces_only_human_and_switches_back(self):
        a = RLHumanRobotAllocatorAgent(Encoder(), torch.device('cpu'), greedy_human_eval=True)
        a._ensure_dqn(self.p, task())
        original_human = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.int32)
        robot = torch.tensor([0, 1, 0, 0], dtype=torch.int32)
        a.human_dqn.act_tensor = Mock(return_value=original_human)
        a.robot_dqn.act_tensor = Mock(return_value=robot)
        masks = {'human': self.mask, 'robot': torch.tensor([1, 1, 0, 0])}
        result = a.act_with_masks(self.p, task(), masks, 0, pre=self.p)
        self.assertEqual(result['human'].argmax(), 1)
        torch.testing.assert_close(result['robot'], robot)
        a.human_dqn.act_tensor.assert_not_called()
        a.greedy_human_eval = False
        result = a.act_with_masks(self.p, task(), masks, 0, pre=self.p)
        torch.testing.assert_close(result['human'], original_human)
        self.assertEqual(a.robot_dqn.act_tensor.call_count, 2)

    def test_training_skips_human_replay_and_updates_robot(self):
        a = RLHumanRobotAllocatorAgent(Encoder(), torch.device('cpu'), human_policy="greedy", batch_size=1)
        masks = {'human': self.mask, 'robot': torch.tensor([1, 1, 0, 0])}
        result = a.act_with_masks(self.p, task(), masks, 1.0, pre=self.p)
        self.assertEqual(result['human'].argmax(), 1)  # fixed rule even at epsilon=1
        before = [p.detach().clone() for p in a.robot_dqn.q_net.parameters()]
        lh, lr = a.observe_step(self.p, task(), result, 2.0, self.p, False, 1.0)
        self.assertIsNone(lh)
        self.assertEqual(len(a.human_dqn.buffer), 0)
        self.assertFalse(any(p.requires_grad for p in a.human_dqn.q_net.parameters()))
        self.assertIsNotNone(lr)
        a.robot_dqn.optimizer.zero_grad()
        lr.backward()
        a.robot_dqn.optimizer.step()
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, a.robot_dqn.q_net.parameters())))

    def test_checkpoint_cannot_silently_switch_fixed_rule_to_rl(self):
        import tempfile
        from pathlib import Path
        a = RLHumanRobotAllocatorAgent(Encoder(), torch.device('cpu'), human_policy="greedy")
        b = RLHumanRobotAllocatorAgent(Encoder(), torch.device('cpu'))
        for agent in (a, b):
            agent._ensure_dqn(self.p, task())
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'human.pth')
            a.human_dqn.save(path)
            a.human_dqn.load(path)
            with self.assertRaisesRegex(RuntimeError, 'human_policy=greedy'):
                b.human_dqn.load(path)


if __name__ == '__main__':
    unittest.main()
