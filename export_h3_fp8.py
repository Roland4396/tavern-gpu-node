#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_MODEL = Path("/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3")
DEFAULT_SOURCE = DEFAULT_MODEL / "FL2VA" / "transformer"
DEFAULT_DESTINATION = DEFAULT_MODEL / "FL2VA" / "transformer-fp8"
DEFAULT_STAGING = Path("/home/worker/.cache/gpu-runtime/h3-fp8-staging")
READY_MARKER = ".complete.json"
BLOCK_SIZE = (128, 128)
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
EXPORT_FORMAT = "h3-fp8-block-128x128-v1"

QUANTIZED_SUFFIXES = (
    ".adaln_proj.linear.weight",
    ".attn.qkv_proj.weight",
    ".attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return False
    if name == "condition_proj.weight" or name == "final_layer.adaln_proj.linear.weight":
        return True
    return name.startswith(("blocks.", "token_refiner.blocks.")) and name.endswith(
        QUANTIZED_SUFFIXES
    )


def scale_name(name: str) -> str:
    return name.removesuffix(".weight") + ".weight_scale_inv"


def source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update((source / name).read_bytes())
    return digest.hexdigest()


def checkpoint_ready(destination: Path, source: Path | None = None) -> bool:
    try:
        marker = json.loads((destination / READY_MARKER).read_text(encoding="utf-8"))
        config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        index = json.loads(
            (destination / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    quantization = config.get("quantization_config") or {}
    if (
        marker.get("format") != "h3-fp8-block"
        or quantization.get("quant_method") != "fp8"
        or quantization.get("weight_block_size") != list(BLOCK_SIZE)
    ):
        return False
    if source is not None and marker.get("source_fingerprint") != source_fingerprint(source):
        return False
    files = set((index.get("weight_map") or {}).values())
    if not files or any(not (destination / name).is_file() for name in files):
        return False
    try:
        for name in files:
            with safe_open(destination / name, framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    return False
    except (OSError, ValueError):
        return False
    return True


def acquire_gpu_lease(lock_dir: Path, gpu: int) -> int:
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock_dir / f"gpu-{gpu}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block_n, block_k = BLOCK_SIZE
    rows, columns = weight.shape
    padded_rows = math.ceil(rows / block_n) * block_n
    padded_columns = math.ceil(columns / block_k) * block_k
    padded = torch.zeros(
        (padded_rows, padded_columns), dtype=torch.float32, device=weight.device
    )
    padded[:rows, :columns].copy_(weight)
    tiled = padded.view(
        padded_rows // block_n, block_n, padded_columns // block_k, block_k
    )
    scale = tiled.abs().amax(dim=(1, 3)).clamp_min_(torch.finfo(torch.float32).tiny)
    scale.div_(FP8_MAX)
    quantized = (
        (tiled / scale[:, None, :, None])
        .clamp_(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(padded_rows, padded_columns)[:rows, :columns]
        .contiguous()
    )
    return quantized, scale.contiguous()


def shard_keys_if_valid(path: Path, fingerprint: str) -> list[str] | None:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("h3_fp8_export_format") != EXPORT_FORMAT
                or metadata.get("h3_fp8_source_fingerprint") != fingerprint
            ):
                return None
            keys = list(handle.keys())
            return keys or None
    except (OSError, ValueError):
        return None


def publish_shard(staging_file: Path, destination_file: Path) -> None:
    temporary = destination_file.with_name(f".{destination_file.name}.partial")
    temporary.unlink(missing_ok=True)
    with staging_file.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        writer.flush()
    if temporary.stat().st_size != staging_file.stat().st_size:
        raise RuntimeError(f"published size mismatch: {destination_file.name}")
    temporary.replace(destination_file)


def convert_shard(
    source_file: Path, staging_file: Path, fingerprint: str
) -> tuple[dict[str, str], int, int]:
    temporary = staging_file.with_name(f".{staging_file.name}.partial")
    temporary.unlink(missing_ok=True)
    output: dict[str, torch.Tensor] = {}
    quantized_count = 0
    with safe_open(source_file, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if should_quantize(name, tensor):
                quantized, scale = block_fp8(tensor.to(device="cuda", dtype=torch.float32))
                output[name] = quantized.cpu()
                output[scale_name(name)] = scale.cpu()
                quantized_count += 1
                del quantized, scale
                torch.cuda.empty_cache()
            else:
                output[name] = tensor
    save_file(
        output,
        str(temporary),
        metadata={
            "format": "pt",
            "h3_fp8_export_format": EXPORT_FORMAT,
            "h3_fp8_source_fingerprint": fingerprint,
        },
    )
    temporary.replace(staging_file)
    weight_map = {name: staging_file.name for name in output}
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in output.values())
    del output
    gc.collect()
    torch.cuda.empty_cache()
    return weight_map, total_bytes, quantized_count


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def export(source: Path, destination: Path, staging: Path) -> None:
    if checkpoint_ready(destination, source):
        print(f"h3_fp8_checkpoint_ready destination={destination}", flush=True)
        return
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    (destination / READY_MARKER).unlink(missing_ok=True)

    source_index = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    source_files = sorted(set(source_index["weight_map"].values()))
    fingerprint = source_fingerprint(source)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    total_quantized = 0
    started = time.monotonic()
    for number, name in enumerate(source_files, 1):
        target = destination / name
        keys = shard_keys_if_valid(target, fingerprint)
        if keys:
                weight_map.update({key: name for key in keys})
                total_bytes += target.stat().st_size
                total_quantized += sum(
                    key.endswith(".weight_scale_inv") for key in keys
                )
                print(f"shard_cached index={number}/{len(source_files)} name={name}", flush=True)
                continue
        target.unlink(missing_ok=True)
        staging_file = staging / name
        staging_keys = shard_keys_if_valid(staging_file, fingerprint)
        if staging_keys:
            shard_map = {key: name for key in staging_keys}
            shard_bytes = staging_file.stat().st_size
            quantized_count = sum(
                key.endswith(".weight_scale_inv") for key in staging_keys
            )
            print(f"staging_cached index={number}/{len(source_files)} name={name}", flush=True)
        else:
            staging_file.unlink(missing_ok=True)
            shard_map, shard_bytes, quantized_count = convert_shard(
                source / name, staging_file, fingerprint
            )
        publish_shard(staging_file, target)
        if not shard_keys_if_valid(target, fingerprint):
            raise RuntimeError(f"published shard failed validation: {target}")
        staging_file.unlink()
        weight_map.update(shard_map)
        total_bytes += target.stat().st_size
        total_quantized += quantized_count
        print(
            f"shard_done index={number}/{len(source_files)} name={name} "
            f"quantized={quantized_count} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    expected_keys = set(source_index["weight_map"])
    missing = expected_keys - set(weight_map)
    if missing:
        raise RuntimeError(f"export is missing {len(missing)} source tensors")
    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(BLOCK_SIZE),
    }
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = quantization_config
    write_json(destination / "config.json", config)
    write_json(
        destination / "model.safetensors.index.json",
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
    )
    write_json(
        destination / READY_MARKER,
        {
            "block_size": list(BLOCK_SIZE),
            "created_at": int(time.time()),
            "format": "h3-fp8-block",
            "quantized_tensors": total_quantized,
            "source": str(source),
            "source_fingerprint": fingerprint,
        },
    )
    if not checkpoint_ready(destination, source):
        raise RuntimeError(f"published H3 FP8 checkpoint failed validation: {destination}")
    shutil.rmtree(staging)
    print(
        f"h3_fp8_checkpoint_ready destination={destination} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    descriptor = acquire_gpu_lease(
        Path("/home/worker/.local/state/gpu-runtime/gpu-locks"), args.gpu
    )
    print(f"acquired_gpu_lease gpu={args.gpu}", flush=True)
    try:
        export(args.source, args.destination, args.staging)
    finally:
        os.close(descriptor)
        print("released_gpu_lease", flush=True)


if __name__ == "__main__":
    main()
