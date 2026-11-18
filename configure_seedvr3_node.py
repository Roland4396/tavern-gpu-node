from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("/home/worker/.config/gpu-runtime/node.json")

SEEDVR3_SPEC = {
    "gpu_count": 3,
    "allowed_gpus": list(range(8)),
    "idle_seconds": 7200,
    "startup_seconds": 300,
    "stop_seconds": 120,
    "cwd": "/home/worker",
    "command": [
        "/home/worker/.venvs/deep-learning/bin/python",
        "/home/worker/.local/libexec/gpu-runtime/seedvr3_service.py",
    ],
    "env": {
        "HOME": "/home/worker",
        "USER": "worker",
        "TMPDIR": "/home/worker/tmp",
        "PYTHONPATH": "/home/worker/.local/lib/torch-cu128-overlay",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SEEDVR3_SOCKET": "/home/worker/.local/state/gpu-runtime/seedvr3.sock",
        "SEEDVR3_ROOT": "/home/worker/.cache/gpu-runtime/seedvr3-service",
        "SEEDVR3_ALLOWED_INPUT_ROOT": "/mnt/data/worker/.runtime/state/graph/v1/input",
        "SEEDVR3_JOB_TTL_SECONDS": "3600",
    },
    "socket_path": "/home/worker/.local/state/gpu-runtime/seedvr3.sock",
    "ready": {
        "type": "unix_http",
        "socket": "/home/worker/.local/state/gpu-runtime/seedvr3.sock",
        "path": "/health",
    },
    "busy": {
        "type": "unix_http_json",
        "socket": "/home/worker/.local/state/gpu-runtime/seedvr3.sock",
        "path": "/health",
        "fields": ["active_jobs", "queued_jobs"],
    },
}


def configure(config: dict[str, Any], remove: bool = False) -> dict[str, Any]:
    result = dict(config)
    workloads = dict(result.get("workloads") or {})
    if remove:
        workloads.pop("seedvr3", None)
    else:
        workloads["seedvr3"] = SEEDVR3_SPEC
    result["workloads"] = workloads
    return result


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure the persistent three-GPU SeedVR2 service")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    current = json.loads(args.config.read_text(encoding="utf-8"))
    atomic_write(args.config, configure(current, remove=args.remove))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
