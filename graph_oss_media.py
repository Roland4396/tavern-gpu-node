"""Sign narrowly scoped Graph media using DSW's existing temporary credentials.

Only the short-lived object URL leaves the GPU node. No permanent keys are
stored, no bucket ACL/CORS is modified, and media bytes never use this endpoint.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

GRAPH_ROOT = Path('/mnt/data/worker/.runtime/state/graph/v1')
DATASETS_FILE = Path('/etc/dsw/instance/extra-info')
CREDENTIAL_URI = os.environ.get('ALIBABA_CLOUD_CREDENTIALS_URI', 'http://localhost:7002/api/v1/credentials/0')
OSSUTIL = '/usr/local/bin/ossutil'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.avif', '.bmp'}


def resolve_media(query: dict, root: Path = GRAPH_ROOT) -> Path:
    kind = query.get('type', 'output')
    filename = query.get('filename', '')
    subfolder = query.get('subfolder', '')
    if kind not in {'input', 'output'}:
        raise ValueError('only shared input/output media is supported')
    if not isinstance(filename, str) or not filename or '/' in filename or '\\' in filename or '\0' in filename:
        raise ValueError('invalid media filename')
    if not isinstance(subfolder, str) or '\\' in subfolder or '\0' in subfolder:
        raise ValueError('invalid media subfolder')
    if Path(subfolder).is_absolute() or '..' in Path(subfolder).parts:
        raise ValueError('media path must stay inside the Graph directory')
    if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError('unsupported image format')
    base = (root / kind).resolve()
    path = (base / subfolder / filename).resolve()
    if not path.is_relative_to(base):
        raise ValueError('media path escapes the Graph directory')
    if not path.is_file():
        raise FileNotFoundError('Graph image not found')
    return path


def oss_location(path: Path, datasets: list[dict]) -> tuple[str, str, str]:
    for item in datasets:
        if item.get('type') != 'OSS':
            continue
        mount = Path(item['mountPath']).resolve()
        if not path.is_relative_to(mount):
            continue
        source = urllib.parse.urlsplit(item['path'])
        host = urllib.parse.urlsplit(item['ossUrl']).hostname or ''
        match = re.fullmatch(r'([a-z0-9-]+)\.oss-([a-z0-9-]+?)(?:-internal)?\.aliyuncs\.com', host)
        if source.scheme != 'oss' or not match or source.netloc != match[1]:
            raise ValueError('unsupported OSS mount mapping')
        key = '/'.join(filter(None, [source.path.strip('/'), path.relative_to(mount).as_posix()]))
        return source.netloc, match[2], key
    raise ValueError('Graph media is not on an OSS dataset')


def sign_media(query: dict) -> dict:
    path = resolve_media(query)
    datasets = json.loads(DATASETS_FILE.read_text())['datasets']
    bucket, region, key = oss_location(path, datasets)
    with urllib.request.urlopen(CREDENTIAL_URI, timeout=5) as response:
        credentials = json.load(response)
    if credentials.get('Code') != 'Success':
        raise RuntimeError('DSW temporary credentials unavailable')
    env = dict(os.environ)
    # Clear stale OSS settings so the signature always targets the actual mount.
    for name in list(env):
        if name.startswith('OSS_'):
            del env[name]
    env.update(OSS_ACCESS_KEY_ID=credentials['AccessKeyId'],
               OSS_ACCESS_KEY_SECRET=credentials['AccessKeySecret'],
               OSS_SESSION_TOKEN=credentials['SecurityToken'])
    result = subprocess.run(
        [OSSUTIL, 'sign', f'oss://{bucket}/{key}', '--mode', 'StsToken',
         '--region', region, '--endpoint', f'https://oss-{region}.aliyuncs.com',
         '--expires-duration', '5m', '-q'],
        env=env, capture_output=True, text=True, timeout=15,
    )
    urls = [line.strip() for line in result.stdout.splitlines() if line.startswith('https://')]
    if result.returncode or len(urls) != 1:
        # CLI error text can include credential/query data; never log it.
        raise RuntimeError('OSS image signing failed')
    parsed = urllib.parse.urlsplit(urls[0])
    if parsed.hostname != f'{bucket}.oss-{region}.aliyuncs.com':
        raise RuntimeError('unexpected signed OSS origin')
    return {'url': urls[0], 'size': path.stat().st_size, 'expires_in': 300}
