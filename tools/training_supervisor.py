"""Save child output, fatal Python stacks and exit status independently of W&B."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def _save(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def supervise(argv, *, cwd, env, output_root=None, resources=True, kernel=True):
    root = Path(output_root or Path(cwd) / 'outputs/train_monitor')
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('run_%Y%m%d_%H%M%S_'), dir=root)).resolve()
    print(f'[monitor] evidence: {directory}', flush=True)
    child_env = dict(env, PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1')
    started = time.time()
    status = dict(started_at=datetime.now(timezone.utc).isoformat(), state='starting',
                  python=argv[0], faulthandler=True)
    # Never serialize the environment: it contains credentials.
    status_path = directory / 'status.json'
    _save(status_path, status)
    monitor = child = None
    handlers = {}
    console_ok = True

    def display(data):
        nonlocal console_ok
        if console_ok:
            try:
                sys.stdout.write(data)
                sys.stdout.flush()
            except (OSError, ValueError):
                console_ok = False

    def forward(signum, frame):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    with (directory / 'console.log').open('wb', buffering=0) as log, \
            (directory / 'monitor.log').open('wb', buffering=0) as monitor_log:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, forward)
            child = subprocess.Popen(argv, cwd=cwd, env=child_env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            status.update(state='running', pid=child.pid)
            _save(status_path, status)
            if resources:
                try:
                    monitor = subprocess.Popen(
                        [sys.executable, str(Path(__file__).with_name('monitor_training.py')),
                         '--pid', str(child.pid), '--interval', '10', '--output-dir', str(directory),
                         '--no-auto-train-log'], cwd=cwd, stdout=monitor_log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as exc:
                    status['monitor_error'] = str(exc)
            # Direct-to-file output avoids pipes held open by surviving descendants.
            import codecs
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            with (directory / 'console.log').open('rb') as reader:
                while child.poll() is None:
                    display(decoder.decode(reader.read(65536)))
                    time.sleep(.2)
                while True:
                    data = reader.read(65536)
                    if not data:
                        break
                    display(decoder.decode(data))
                display(decoder.decode(b'', final=True))
            rc = child.wait()
            sig_name = signal.Signals(-rc).name if rc < 0 else None
            status.update(state='finished' if rc == 0 else 'failed', returncode=rc,
                          signal=sig_name, shell_exit_code=128-rc if rc < 0 else rc,
                          ended_at=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=round(time.time()-started, 3))
            _save(status_path, status)
            os.fsync(log.fileno())
        except BaseException as exc:
            status.update(state='supervisor_error', error=str(exc))
            _save(status_path, status)
            raise
        finally:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if monitor is not None:
                status['monitor_returncode_before_cleanup'] = monitor.poll()
                monitor.terminate()
                try:
                    monitor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if kernel:
        try:
            result = subprocess.run(
                ['journalctl', '-k', '--since', f'@{int(started)}', '--no-pager', '-o', 'short-iso'],
                capture_output=True, text=True, timeout=10)
            (directory / 'kernel.log').write_text(result.stdout + result.stderr)
            status['kernel_query_returncode'] = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            status['kernel_error'] = str(exc)
    _save(status_path, status)
    display(f'[monitor] returncode={rc} signal={sig_name}; evidence: {directory}\n')
    return status['shell_exit_code']
