"""Exercise real child exits, including a controlled fatal signal (no core dump)."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.training_supervisor import supervise
from tools.monitor_training import KERNEL_PATTERNS


class SupervisorTests(unittest.TestCase):
    def run_child(self, code):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            rc = supervise([sys.executable, '-c', code], cwd=root, env=os.environ,
                           output_root=root, resources=False, kernel=False)
            directory = next(Path(root).glob('run_*'))
            return rc, json.loads((directory/'status.json').read_text()), (directory/'console.log').read_text()

    def test_normal(self):
        rc, status, output = self.run_child("import sys; print('out'); print('err', file=sys.stderr)")
        self.assertEqual(rc, 0)
        self.assertEqual(status['state'], 'finished')
        self.assertIn('out', output)
        self.assertIn('err', output)

    def test_exception(self):
        rc, status, output = self.run_child("raise RuntimeError('test failure')")
        self.assertEqual(rc, 1)
        self.assertIsNone(status['signal'])
        self.assertIn('RuntimeError: test failure', output)

    def test_segfault(self):
        rc, status, output = self.run_child(
            'import resource, os, signal; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); '
            'os.kill(os.getpid(), signal.SIGSEGV)')
        self.assertEqual(rc, 139)
        self.assertEqual(status['returncode'], -11)
        self.assertEqual(status['signal'], 'SIGSEGV')
        self.assertIn('Fatal Python error: Segmentation fault', output)
        self.assertIn('File "<string>"', output)

    def test_sigterm(self):
        rc, status, output = self.run_child('import os, signal; os.kill(os.getpid(), signal.SIGTERM)')
        self.assertEqual(rc, 143)
        self.assertEqual(status['signal'], 'SIGTERM')

    def test_training_kernel_segfault(self):
        self.assertIsNotNone(KERNEL_PATTERNS.search('HcFactory-hier-[109879]: segfault at 7ffc52d3d4b0'))


if __name__ == '__main__':
    unittest.main()
