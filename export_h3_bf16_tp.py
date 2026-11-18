"""CPU-only candidate TP2/Ulysses2 DiT export. Does not change serving config.

Writes the existing SGLang rank-shard v1 format to a NEW directory. A complete
export is not a startup/quality validation; activation is a separate operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from contextlib import ExitStack
from pathlib import Path

from h3_bf16_tp_plan import plan, read_source, slice_tensor

FORMAT = "sglang-h3-rank-sharded-v1"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_fingerprint(source):
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update(name.encode("ascii"))
        digest.update((source / name).read_bytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(source: Path, destination: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from safetensors import safe_open

    def no_cuda(*args, **kwargs):
        raise RuntimeError("CPU-only checkpoint export attempted CUDA initialization")

    torch.cuda._lazy_init = no_cuda
    torch.set_num_threads(2)
    config, headers = read_source(source)
    recipe = plan(config, headers)
    if destination.exists():
        raise FileExistsError(f"Export needs a new candidate directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = recipe["v1_four_rank_bytes"] + 2 * 1024**3
    if shutil.disk_usage(destination.parent).free < required:
        raise RuntimeError(f"Insufficient disk space: require {required} bytes")
    destination.mkdir(mode=0o700)
    component = destination / "transformer"
    component.mkdir(mode=0o700)
    status = destination / "export-status.json"
    fingerprint = source_fingerprint(source)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names = sorted(recipe["tensors"])
    paths = [component / f"rank-{rank:05d}-of-00004.safetensors" for rank in range(4)]
    started = time.time()
    manifest = {
        "format": FORMAT,
        "component": "transformer",
        "world_size": 4,
        "source_fingerprint": fingerprint,
        "tp_size": 2,
        "ulysses_degree": 2,
        "global_rank_to_tp_rank": [0, 1, 0, 1],
        "source": str(source),
        "created_at": started,
        "serving_validated": False,
        "files": [],
    }
    atomic_json(destination / "slice-plan.json", recipe)
    try:
        with ExitStack() as stack:
            inputs = {
                name: stack.enter_context(
                    safe_open(source / name, framework="pt", device="cpu")
                )
                for name in sorted(set(index["weight_map"].values()))
            }
            writers = []
            for rank, path in enumerate(paths):
                header = {
                    "__metadata__": {
                        "format": FORMAT,
                        "component": "transformer",
                        "rank": str(rank),
                        "world_size": "4",
                        "source_fingerprint": fingerprint,
                        "tp_size": "2",
                        "ulysses_degree": "2",
                    }
                }
                offset = 0
                for name in names:
                    spec = recipe["tensors"][name]
                    header[name] = {
                        "dtype": spec["dtype"],
                        "shape": spec["local_shape"],
                        "data_offsets": [offset, offset + spec["bytes"]],
                    }
                    offset += spec["bytes"]
                encoded = json.dumps(header, separators=(",", ":")).encode()
                encoded += b" " * (-len(encoded) % 8)
                writer = stack.enter_context(path.with_suffix(".partial").open("xb"))
                writer.write(struct.pack("<Q", len(encoded)))
                writer.write(encoded)
                writers.append(writer)
            for number, name in enumerate(names, 1):
                tensor = inputs[index["weight_map"][name]].get_tensor(name)
                for rank in range(2):
                    local = slice_tensor(
                        name,
                        tensor,
                        rank=rank,
                        tp=2,
                        heads=config["num_attention_heads"],
                        head_dim=config["attention_head_dim"],
                    )
                    data = local.view(torch.uint8).numpy().tobytes()
                    if len(data) != recipe["tensors"][name]["bytes"]:
                        raise RuntimeError(f"Unexpected serialized size: {name}")
                    writers[rank].write(data)
                    writers[rank + 2].write(data)
                    del data, local
                del tensor
                if number == 1 or number % 10 == 0 or number == len(names):
                    atomic_json(
                        status,
                        {
                            "state": "exporting",
                            "tensors_done": number,
                            "tensors_total": len(names),
                            "elapsed_seconds": time.time() - started,
                        },
                    )
            for writer in writers:
                writer.flush()
                os.fsync(writer.fileno())
        for rank, path in enumerate(paths):
            part = path.with_suffix(".partial")
            atomic_json(
                status,
                {
                    "state": "verifying",
                    "rank": rank,
                    "elapsed_seconds": time.time() - started,
                },
            )
            with safe_open(part, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(names):
                    raise RuntimeError(f"Exported tensor coverage mismatch: {part}")
                for name in names:
                    if (
                        handle.get_slice(name).get_shape()
                        != recipe["tensors"][name]["local_shape"]
                    ):
                        raise RuntimeError(f"Exported shape mismatch: {name}")
            digest = file_sha256(part)
            part.replace(path)
            manifest["files"].append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": digest}
            )
        if source_fingerprint(source) != fingerprint:
            raise RuntimeError("Source metadata changed during export")
        atomic_json(component / ".complete.json", manifest)
        atomic_json(
            status,
            {
                "state": "complete",
                "serving_validated": False,
                "elapsed_seconds": time.time() - started,
                "files": manifest["files"],
            },
        )
    except BaseException as error:
        atomic_json(
            status,
            {
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.time() - started,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination)
