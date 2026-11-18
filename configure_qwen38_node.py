#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


DEFAULT_CONFIG = Path("/home/worker/.config/gpu-runtime/node.json")
LAUNCHER = "/home/worker/.local/libexec/gpu-runtime/qwen38_launcher.py"
MODEL = "/mnt/data/worker/.runtime/artifacts/text/qwen3.8-27b"
PRESHARDED_PATH = "/mnt/data/worker/.runtime/artifacts/text/qwen3.8-27b-tp2-presharded"
QWEN38_PYTHON = "/home/worker/.venvs/qwen38-sglang-0.5.17/bin/python"
CUDA_COMPAT = "/home/worker/.local/lib/runtime-core/cuda-compat-13-0/usr/local/cuda-13.0/compat"


def workload_spec(mode: str, port: int) -> dict:
    return {
        "gpu_count": 2,
        "allowed_gpus": list(range(8)),
        "idle_seconds": 1800,
        "startup_seconds": 1800,
        "stop_seconds": 120,
        "cwd": "/home/worker",
        "command": ["/usr/bin/python3", LAUNCHER],
        "env": {
            "HOME": "/home/worker",
            "USER": "worker",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "LD_LIBRARY_PATH": CUDA_COMPAT,
            "QWEN38_MODEL_PATH": MODEL,
            "QWEN38_PRESHARDED_PATH": PRESHARDED_PATH,
            "QWEN38_PYTHON": QWEN38_PYTHON,
            "QWEN38_CONTEXT_MODE": mode,
            "QWEN38_PORT": str(port),
        },
        "ready": {"type": "http", "url": f"http://127.0.0.1:{port}/health"},
    }


def configure(config: dict) -> dict:
    result = dict(config)
    workloads = dict(result.get("workloads") or {})
    workloads["qwen38_native"] = workload_spec("native", 30_020)
    workloads["qwen38_yarn"] = workload_spec("yarn", 30_021)
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
    print(json.dumps({"changed": updated != current, "workloads": ["qwen38_native", "qwen38_yarn"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
