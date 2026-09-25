"""Backend selection, resolved before importing environment modules."""
import argparse
import os


def logic_enabled():
    return os.environ.get('HC_SIM_BACKEND', 'isaac') == 'logic'


def select_backend(argv, environ=None):
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sim_backend', choices=('auto', 'logic', 'isaac'), default=env.get('HC_SIM_BACKEND', 'auto'))
    for flag in ('visualize', 'video', 'enable_cameras', 'active_livestream'):
        parser.add_argument('--'+flag, action='store_true')
    parser.add_argument('--livestream', type=int, default=0)
    args, _ = parser.parse_known_args(argv)
    if args.sim_backend not in ('auto', 'logic', 'isaac'):
        parser.error('HC_SIM_BACKEND must be auto, logic, or isaac')
    visual = (args.visualize or args.video or args.enable_cameras or args.active_livestream
              or args.livestream != 0 or env.get('LIVESTREAM', '0') not in ('', '0'))
    if args.sim_backend == 'logic' and visual:
        parser.error('logic backend cannot render; use --sim_backend isaac or --visualize with auto')
    return args.sim_backend != 'isaac' and not visual, args.sim_backend
