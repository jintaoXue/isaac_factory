"""Logic-only integration tests. Run in a fresh process; no Isaac installation needed."""
import os
os.environ['HC_SIM_BACKEND'] = 'logic'
import ast
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from source.hc_backend import select_backend
from source.hc_logic_runtime import HcLogicVectorEnv, HcLogicEnvCfg
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.src import env_checkpoint
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.src.curriculum import apply_train_order
from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.env_asset_cfg.cfg_env_state_action_sample import EnvStateActionSample
from source.algo.hierarchical.hc_factory.hierarchical_dispatch import build_rule_based_action


class LogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = HcLogicVectorEnv(HcLogicEnvCfg(cuda_device_str='cpu', seed=9002,
                           train_cfg={'params': {'config': {'human_duration_aux': True}}}))

    def test_backend_selection(self):
        for argv in ([], ['--test'], ['--headless'], ['--test','--headless']):
            self.assertTrue(select_backend(argv, {})[0])
        for argv in (['--visualize'],['--video','--headless'],['--enable_cameras'],['--active_livestream'],['--sim_backend','isaac']):
            self.assertFalse(select_backend(argv, {})[0])
        self.assertFalse(select_backend([], {'HC_SIM_BACKEND':'isaac'})[0])
        with self.assertRaises(SystemExit): select_backend(['--sim_backend','logic','--visualize'], {})

    def test_storage_matches_existing_isaac_dump(self):
        obs = self.env.reset()[0]
        for name, storage in obs['storage'].items():
            a = storage['key_variables']['placement_cfg']['pose_list']
            b = EnvStateActionSample['storage'][name]['key_variables']['placement_cfg']['pose_list']
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):
                torch.testing.assert_close(x['position'],y['position'],atol=1e-5,rtol=0)
                torch.testing.assert_close(x['orientation'],y['orientation'],atol=1e-5,rtol=0)

    def test_single_tick_fatigue_reward_and_terminal_events(self):
        obs = self.env.reset()[0]; single = self.env.env_list[0]
        apply_train_order(single,n_products=1,anchor=64) # four logical steps to timeout
        obs['human']['num_00_HeterogeneousHuman']['fatigue'] = .7
        action = build_rule_based_action(obs,self.env.cuda_device,max_parallel_cd_dispatch=10)
        result = self.env.step([action])[0]
        self.assertEqual(result['time_step'],1)
        self.assertTrue(result['rl']['duration_events'])
        self.assertTrue(all(e['kind']=='start' for e in result['rl']['duration_events']))
        self.assertTrue(any(h['state']!='free' for h in result['human'].values()))
        for _ in range(3):
            result=self.env.step([build_rule_based_action(result,self.env.cuda_device,10)])[0]
        self.assertTrue(result['rl']['done']); self.assertTrue(result['rl']['truncated'])
        self.assertEqual(result['time_step'],0)  # original automatic-reset contract
        self.assertIn('reward_parts',result['rl'])

    def test_restore_no_scene_objects(self):
        self.env.reset();single=self.env.env_list[0]
        single.env_state_action_dict['time_step']=7
        saved=env_checkpoint.capture(single.env_state_action_dict)
        single.env_state_action_dict['time_step']=99
        single.restore_checkpoint(saved)
        self.assertEqual(single.env_state_action_dict['time_step'],7)
        self.assertIs(single.human_manager.human_list[0].state,single.env_state_action_dict['human']['num_00_HeterogeneousHuman'])
        single.apply_data_to_sim()  # safe no-op, including checkpoint restore path

    def test_shared_logic_tick_method_unchanged(self):
        # Structural guard: this backend calls the same manager loop as visualization.
        from source.isaaclab_tasks.isaaclab_tasks.direct.hc_factory.hc_single_env_base import HcSingleEnvBase
        self.assertIs(type(self.env.env_list[0]),HcSingleEnvBase)
        names=[type(m).__name__ for m in self.env.env_list[0].iter_managers()]
        self.assertEqual(names,['StorageManager','ProductMaterialManager','HumanManager','RobotManager',
                                'RouteManagerVectorEnv','MachineManager','TaskManager','AlgoHierarchicalMasker'])
        self.assertIsNone(self.env.env_list[0].camera_manager)

    def test_complete_order_and_no_engine(self):
        obs = self.env.reset()[0]
        apply_train_order(self.env.env_list[0], n_products=1, anchor=160000)
        completed = 0
        for step in range(10000):
            action = build_rule_based_action(obs, self.env.cuda_device, 10)
            obs = self.env.step([action])[0]
            completed += sum(e['kind']=='complete' for e in obs['rl']['duration_events'])
            if obs['rl']['done']:
                self.assertTrue(obs['rl']['success'])
                self.assertEqual(completed, 7)
                self.assertEqual(obs['time_step'], 0)
                break
        else:
            self.fail('One-product production did not finish')
        self.test_no_engine_modules_loaded()

    def test_no_engine_modules_loaded(self):
        forbidden={'isaaclab','isaacsim','omni','pxr','carb'}
        self.assertEqual([name for name in sys.modules if name.split('.')[0] in forbidden],[])


if __name__ == '__main__': unittest.main()
