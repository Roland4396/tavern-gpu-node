#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

from deepseek_launcher import (
    DEFAULT_MODEL,
    DEFAULT_SHARDED,
    MODEL_ID,
    READY_MARKER,
    TP_SIZE,
    sharded_checkpoint_ready,
)


PATTERN = "model-rank-{rank}-part-{part}.safetensors"
CUDA_COMPAT = "/home/worker/.local/lib/runtime-core/cuda-compat-13-0/usr/local/cuda-13.0/compat"
RUNTIME_BIN = "/home/worker/miniconda3/envs/runtime-core/bin"
DEFAULT_STAGING = DEFAULT_SHARDED
COPY_BUFFER_BYTES = 16 * 1024 * 1024


def acquire_gpu_leases(lock_dir: Path, gpu_count: int = TP_SIZE) -> list[int]:
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptors: list[int] = []
    try:
        for gpu in range(gpu_count):
            descriptor = os.open(lock_dir / f"gpu-{gpu}.lock", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.set_inheritable(descriptor, True)
            descriptors.append(descriptor)
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise
    return descriptors


def copy_metadata(source: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if not item.is_file() or item.suffix in {".safetensors", ".bin", ".pt"}:
            continue
        relative = item.relative_to(source)
        if relative.name == "model.safetensors.index.json":
            continue
        target = destination / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(item, target)


def copy_file_contents(source: Path, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        print(f"publish_skip name={source.relative_to(source.parents[1])} bytes={source.stat().st_size}", flush=True)
        return
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=COPY_BUFFER_BYTES)
    print(f"publish_done name={source.name} bytes={source.stat().st_size}", flush=True)


def publish_checkpoint(staging: Path, destination: Path, workers: int = TP_SIZE) -> None:
    if not sharded_checkpoint_ready(staging):
        raise RuntimeError(f"staging checkpoint is not ready: {staging}")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = sorted(
        item
        for item in staging.rglob("*")
        if item.is_file() and item.relative_to(staging) != Path(READY_MARKER)
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(copy_file_contents, item, destination / item.relative_to(staging))
            for item in files
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    copy_file_contents(staging / READY_MARKER, destination / READY_MARKER)
    if not sharded_checkpoint_ready(destination):
        raise RuntimeError(f"published checkpoint failed validation: {destination}")
    print(f"published checkpoint ready: {destination}", flush=True)


def checkpoint_manifest(destination: Path, source: Path) -> dict[str, object]:
    ranks: dict[str, list[dict[str, object]]] = {}
    for rank in range(TP_SIZE):
        parts = sorted(destination.glob(f"model-rank-{rank}-part-*.safetensors"))
        if not parts or any(part.stat().st_size <= 0 for part in parts):
            raise RuntimeError(f"incomplete sharded checkpoint for rank {rank}")
        ranks[str(rank)] = [{"name": part.name, "bytes": part.stat().st_size} for part in parts]
    return {
        "source": str(source),
        "tp_size": TP_SIZE,
        "format": "sharded_state",
        "sglang_version": importlib.metadata.version("sglang"),
        "created_at": int(time.time()),
        "ranks": ranks,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def export(source: Path, destination: Path, max_part_gib: int) -> None:
    if sharded_checkpoint_ready(destination):
        print(f"sharded checkpoint already ready: {destination}", flush=True)
        return
    copy_metadata(source, destination)

    from sglang import Engine

    engine = Engine(
        model_path=str(source),
        served_model_name=MODEL_ID,
        tp_size=TP_SIZE,
        moe_runner_backend="marlin",
        tool_call_parser="deepseekv4",
        context_length=4096,
        max_total_tokens=4096,
        max_running_requests=1,
        mem_fraction_static=0.65,
        disable_cuda_graph=True,
        skip_server_warmup=True,
        trust_remote_code=True,
    )
    try:
        engine.save_sharded_model(
            path=str(destination),
            pattern=PATTERN,
            max_size=max_part_gib * 1024**3,
        )
    finally:
        engine.shutdown()

    atomic_json(destination / "manifest.json", checkpoint_manifest(destination, source))
    atomic_json(destination / READY_MARKER, {"tp_size": TP_SIZE, "manifest": "manifest.json"})
    if not sharded_checkpoint_ready(destination):
        raise RuntimeError(f"published checkpoint failed validation: {destination}")
    print(f"sharded checkpoint ready: {destination}", flush=True)


def main() -> None:
    current_path = os.environ.get("PATH", "")
    if os.environ.get("LD_LIBRARY_PATH") != CUDA_COMPAT or current_path.split(os.pathsep)[0] != RUNTIME_BIN:
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = CUDA_COMPAT
        environment["PATH"] = os.pathsep.join((RUNTIME_BIN, "/usr/local/cuda/bin", current_path))
        environment["CUDA_HOME"] = "/usr/local/cuda"
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], environment)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--destination", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--publish-destination", type=Path, default=DEFAULT_SHARDED)
    parser.add_argument("--max-part-gib", type=int, default=8)
    args = parser.parse_args()
    if args.max_part_gib <= 0:
        parser.error("--max-part-gib must be positive")
    leases = acquire_gpu_leases(Path("/home/worker/.local/state/gpu-runtime/gpu-locks"))
    print(f"acquired_gpu_leases count={len(leases)}", flush=True)
    try:
        export(args.source, args.destination, args.max_part_gib)
    finally:
        for descriptor in leases:
            os.close(descriptor)
        print("released_gpu_leases", flush=True)
    if args.publish_destination:
        publish_checkpoint(args.destination, args.publish_destination)


if __name__ == "__main__":
    main()
