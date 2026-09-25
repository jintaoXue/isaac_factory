"""Offline USD reader: export initial local poses; never starts Isaac or PhysX.

Run with a Python environment containing pxr (e.g. Blender's bundled Python).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from pxr import Usd, UsdGeom, Gf

parser = argparse.ArgumentParser()
parser.add_argument('usd', type=Path)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/static_layout.json')
args = parser.parse_args()
cfg = Path(__file__).resolve().parents[1] / 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg'
paths = set()
for filename in ('cfg_storage.py', 'cfg_material_product.py'):
    for path in re.findall(r'"prim_paths_expr":\s*"([^"]+)"', (cfg / filename).read_text()):
        for idx in range(16) if '{idx}' in path else [0]:
            paths.add('/obj/' + path.format(i=0, idx=f'{idx:02d}').split('/obj/', 1)[1])
stage = Usd.Stage.Open(str(args.usd))
poses = {}
for path in sorted(paths):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f'Missing prim in source USD: {path}')
    transform = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
    q = transform.GetRotation().GetQuat()
    poses[path] = dict(position=list(transform.GetTranslation()), orientation=[q.GetReal(), *q.GetImaginary()])
payload = dict(source_asset=args.usd.name, source_sha256=hashlib.sha256(args.usd.read_bytes()).hexdigest(),
               description='Authored local transforms; wxyz quaternions; no physics stepping.', poses=poses)
args.output.write_text(json.dumps(payload, indent=2)+'\n')
print(f'Exported {len(poses)} local poses to {args.output}')
