#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


DEFAULT_CONFIG = Path("/home/worker/.config/gpu-runtime/node.json")
GLM_COMMAND = [
    "/usr/bin/python3",
    "/home/worker/.local/libexec/gpu-runtime/glm_launcher.py",
]


def configure(config: dict) -> dict:
    result = dict(config)
    workloads = dict(result.get("workloads") or {})
    glm = dict(workloads["glm"])
    glm["command"] = list(GLM_COMMAND)
    workloads["glm"] = glm
    result["workloads"] = workloads
    return result


def atomic_write(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    current = json.loads(args.config.read_text(encoding="utf-8"))
    updated = configure(current)
    atomic_write(args.config, updated)
    print(json.dumps({"changed": updated != current, "command": GLM_COMMAND}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
