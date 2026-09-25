"""Fresh-process train/eval smoke with an engine-import assertion after execution."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EntryTests(unittest.TestCase):
    def run_entry(self, args):
        code = ('import runpy,sys; sys.path.insert(0,'+repr(str(ROOT))+'); '
                'sys.argv='+repr([str(ROOT/'train.py'), *args])+'; '
                'runpy.run_path('+repr(str(ROOT/'train.py'))+',run_name="__main__"); '
                'bad=[n for n in sys.modules if n.split(".")[0] in ("isaaclab","isaacsim","omni","carb","pxr")]; '
                'assert not bad,bad; print("NO_ENGINE_IMPORTS")')
        env = {**os.environ, 'HC_SIM_BACKEND':'auto', 'LIVESTREAM':'0', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable,'-c',code],cwd=cwd,env=env,text=True,
                                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout[-7000:])
        self.assertIn('NO_ENGINE_IMPORTS',result.stdout)
        return result.stdout

    def args(self):
        return ['--task','HRTPaHC-v1','--algo','hier','--device','cpu','--headless','--num_envs','1',
                '--train_n_products','10','agent.params.config.t_max_anchor=32',
                'agent.params.config.max_episodic_steps=20','agent.params.config.wandb_activate=false',
                '+agent.params.config.full_experiment_name=logic_test']

    def test_train_and_save(self):
        out=self.run_entry(self.args()+['--seed','9003','--max_sim_episodes','1'])
        self.assertIn('physics=OFF; render=OFF',out)
        self.assertIn('checkpoint saved at step 20',out)

    def test_teacher_checkpoint_eval(self):
        teacher=ROOT/'logs/rl_games/HcFactory/hier_2026-08-27_23-17-41'
        if not (teacher/'nn/state_encoder_step_1290000.pth').is_file():
            self.skipTest('local T0 fixture missing')
        out=self.run_entry(self.args()+['--test','--test_seeds','9003','--test_times','1','--test_epsilon','0',
                                       '--load_dir',str(teacher),'--load_step','1290000'])
        self.assertIn('loaded encoder:',out)
        self.assertIn('eval saved:',out)


if __name__ == '__main__': unittest.main()
