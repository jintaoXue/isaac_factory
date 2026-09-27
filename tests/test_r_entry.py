"""End-to-end R recipes: short training, weight bundle, epsilon=0 evaluation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_r_series import command


def overrides(argv, values):
    replacements={f'agent.params.config.{k}':f'agent.params.config.{k}={v}' for k,v in values.items()}
    result=[replacements.pop(x.split('=')[0],x) if '=' in x else x for x in argv]
    return result+list(replacements.values())


class EntryTests(unittest.TestCase):
    def test_r_train_checkpoint_eval(self):
        env={**os.environ,'HC_SIM_BACKEND':'logic','HC_HUMAN_SKILL_PROFILE':'gap',
             'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HC_R_WANDB':'0','HC_MAX_HARD_EPISODES':'1'}
        with tempfile.TemporaryDirectory(prefix='r-entry-') as work:
            for variant in ('R0','R2'):
                with self.subTest(variant=variant):
                    argv=overrides(command(variant,'cpu',env),dict(max_episodic_steps=160,
                        t_max_anchor=256,batch_size=4,batch_size_A=4,hidden_dim=16,log_interval=40))
                    run=subprocess.run(argv,cwd=work,env=env,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,timeout=90)
                    self.assertEqual(run.returncode,0,run.stdout[-6000:])
                    name=next(x.split('=',1)[1] for x in argv if x.startswith('+agent.params.config.full_experiment_name='))
                    load=Path(work)/'logs/rl_games/HcFactory'/('hier_'+name)
                    self.assertTrue((load/'nn/decision_schema_step_160.json').is_file())
                    metrics=(load/'metrics.jsonl').read_text()
                    self.assertIn('MetricDecision/updates_D_human',metrics)
                    ee={**env,'HC_LOAD_DIR':str(load),'HC_LOAD_STEP':'160',
                        'HC_TEST_SEEDS':'9003','HC_TEST_TIMES':'1'}
                    argv=overrides(command('eval-'+variant,'cpu',ee),dict(
                        max_episodic_steps=160,t_max_anchor=256,hidden_dim=16))
                    run=subprocess.run(argv,cwd=work,env=env,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,timeout=90)
                    self.assertEqual(run.returncode,0,run.stdout[-6000:])
                    self.assertIn('loaded agent_D_human:',run.stdout)
                    self.assertIn('eval saved:',run.stdout)


if __name__=='__main__':unittest.main()
