#!/usr/bin/env python3
"""Remove original-node tensor IPC sessions only AFTER the H3 process stops."""
import argparse
import json
from pathlib import Path
import re
import shutil


def cleanup(manifest_path):
    root = Path(json.loads(Path(manifest_path).read_text())['ipc_root'])
    removed = 0
    if root.exists():
        for path in root.iterdir():
            if re.fullmatch('[0-9a-f]{32}', path.name) and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                removed += 1
    return {'removed_ipc_sessions': removed}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');args=p.parse_args()
    print(json.dumps(cleanup(args.manifest)))
