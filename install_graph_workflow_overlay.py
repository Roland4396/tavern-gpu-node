#!/usr/bin/env python3
"""Install a resolved additive dependency overlay; never alter the base environment."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path('/home/worker/.local/share/graph-dependencies/anima-krea-20260916')
normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
base = {normalize(k): v for k, v in json.loads((root / 'base-packages.json').read_text()).items()}
plan = json.loads((root / 'pip-plan.json').read_text())['install']
for package in plan:
    name = normalize(package['metadata']['name'])
    if name in base:
        raise RuntimeError(f'Resolver proposed replacing existing package {name}; base environment protected')
overlay = root / 'python'
overlay.mkdir(exist_ok=True)
requirements = root / 'overlay-requirements.txt'
requirements.write_text('\n'.join(p['metadata']['name'] + '==' + p['metadata']['version'] for p in plan) + '\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                '--index-url', 'https://pypi.org/simple', '--timeout', '120', '--retries', '4',
                '--target', str(overlay), '-r', str(requirements)], check=True)
files = {}
for path in sorted(overlay.rglob('*')):
    if path.is_file():
        files[str(path.relative_to(overlay))] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'overlay-manifest.json').write_text(json.dumps({'packages': [p['metadata'] for p in plan], 'files': files}))
print(json.dumps({'status': 'installed', 'additional_packages': len(plan), 'base_environment_unchanged': True}))
