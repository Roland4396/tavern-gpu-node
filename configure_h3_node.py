#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path


# LoRA remains supported when explicitly supplied by a workflow, but it is
# deliberately disabled for the backend defaults.  H3 is intentionally
# configured to use the native installation that was provisioned on the GPU
# hosts.  This avoids the source-checkout/PYTHONPATH override that was used
# while testing the H20 SubBlock build.  The native launcher always selects
# the validated offline FP8 checkpoint (and falls back to SGLang's explicit
# ``--quantization fp8`` flag when that checkpoint is not present).
H3_FL2VA_LORA = ""
H3_REF2VA_LORA = ""
# Keep these paths explicit: they are the pre-existing, node-local native
# installation and must not be replaced by a copied release or source tree.
H3_NATIVE_LAUNCHER = "/home/worker/.local/libexec/gpu-runtime/h3_launcher.py"
H3_NATIVE_PYTHON = "/home/worker/.venvs/sglang-h3/bin/python"
# Retain the release/source constants for the opt-in source mode below.  The
# default configuration never references them.
H3_RELEASE_ROOT = "/home/worker/.local/releases/h3/current"
H3_SOURCE_ROOT = "/home/worker/.local/releases/h3-source/current"
H3_RUNTIME_ROOT = "/home/worker/.local/releases/gpu-runtime-source/current"
H3_PYTHON = H3_NATIVE_PYTHON
SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
H3_ACCELERATION_PROFILES = frozenset({
    "lossless",
    "rh_lightning",
    "cache_conservative",
    "cache_stride",
    "subblock_075",
    "subblock_080",
    "subblock_075_cache_conservative",
    "subblock_075_cache_stride",
    "subblock_080_cache_conservative",
    "subblock_080_cache_stride",
})
H3_ACCELERATION_PROFILE = os.environ.get(
    "H3_ACCELERATION_PROFILE", "subblock_075_cache_stride"
).strip().lower()
if H3_ACCELERATION_PROFILE not in H3_ACCELERATION_PROFILES:
    raise RuntimeError(f"unsupported H3_ACCELERATION_PROFILE: {H3_ACCELERATION_PROFILE!r}")


H3_SPEC = {
    "gpu_count": 4,
    "allowed_gpus": list(range(8)),
    "idle_seconds": 1800,
    "startup_seconds": 1800,
    "stop_seconds": 120,
    "cwd": "/home/worker",
    "profile_path": "/home/worker/.local/profiles/gpu-runtime/h3/current",
    "command": ["/usr/bin/python3", H3_NATIVE_LAUNCHER],
    "env": {
        "HOME": "/home/worker",
        "USER": "worker",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "H3_MODEL_PATH": "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3",
        "H3_MODEL_VARIANT": "FL2VA",
        "H3_FP8_TRANSFORMER_PATH": "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/FL2VA/transformer-fp8",
        "H3_TRANSFORMER_WEIGHTS_PATH": "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/FL2VA/transformer",
        "H3_RANK_SHARD_ROOT": "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/FL2VA/rank-sharded-tp2-ulysses2",
        "H3_TRANSFORMER_QUANTIZATION": "fp8",
        "H3_TP_SIZE": "2",
        "H3_ULYSSES_DEGREE": "2",
        "H3_LORA_PATH": H3_FL2VA_LORA,
        "H3_LORA_ALPHA": "",
        "H3_LORA_SCALE": "1.0",
        "H3_LORA_MERGE_MODE": "dynamic",
        "H3_PYTHON": H3_NATIVE_PYTHON,
        # Keep H3 on the exact FlashAttention path by default. SageAttention
        # remains available only through an explicit override for A/B tests.
        "H3_ATTENTION_BACKEND": "fa",
        # Native mode deliberately has no source override or Cache-DiT knobs.
        # ``lossless`` is retained as metadata for the request shaper.
        "H3_ACCELERATION_PROFILE": H3_ACCELERATION_PROFILE,
        "H3_PORT": "30010",
    },
    "ready": {"type": "http", "url": "http://127.0.0.1:30010/ready"},
}

H3_REF_SPEC = {
    **H3_SPEC,
    "command": list(H3_SPEC["command"]),
    "env": {
        **H3_SPEC["env"],
        "H3_MODEL_PATH": "/home/worker/.cache/gpu-runtime/h3-ref-model/MiniMax-H3",
        "H3_MODEL_VARIANT": "Ref2VA",
        "H3_FP8_TRANSFORMER_PATH": "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/Ref2VA/transformer-fp8",
        "H3_TRANSFORMER_WEIGHTS_PATH": "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/Ref2VA/transformer",
        "H3_TRANSFORMER_QUANTIZATION": "fp8",
        "H3_TP_SIZE": "2",
        "H3_ULYSSES_DEGREE": "2",
        "H3_RANK_SHARD_ROOT": "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/Ref2VA/rank-sharded-tp2-ulysses2",
        "H3_LORA_PATH": H3_REF2VA_LORA,
        "H3_LORA_ALPHA": "",
        "H3_ATTENTION_BACKEND": "fa",
        "H3_ACCELERATION_PROFILE": H3_ACCELERATION_PROFILE,
        "H3_PORT": "30011",
    },
    "ready": {"type": "http", "url": "http://127.0.0.1:30011/ready"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--acceleration-profile",
        choices=sorted(H3_ACCELERATION_PROFILES),
        default=None,
        help="H3 worker profile; defaults to H3_ACCELERATION_PROFILE or subblock_075_cache_stride",
    )
    parser.add_argument("--torch-compile", choices=("auto", "on", "off"), default="auto",
                        help="Override only torch.compile; auto preserves the selected recipe")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--python", type=Path, default=None)
    args = parser.parse_args()
    profile = (
        args.acceleration_profile
        or os.environ.get("H3_ACCELERATION_PROFILE", "subblock_075_cache_stride")
    ).strip().lower()
    if profile not in H3_ACCELERATION_PROFILES:
        raise RuntimeError(f"unsupported H3_ACCELERATION_PROFILE: {profile!r}")
    # Keep the CPU Graph request shaper and both remote workers on one explicit
    # profile.  The command-line option makes Ansible/runtime activation
    # deterministic instead of relying on the controller process environment.
    h3_spec = copy.deepcopy(H3_SPEC)
    h3_ref_spec = copy.deepcopy(H3_REF_SPEC)
    for spec in (h3_spec, h3_ref_spec):
        spec["env"]["H3_ACCELERATION_PROFILE"] = profile
        if profile == "rh_lightning":
            spec["env"]["H3_ATTENTION_BACKEND"] = "sage_attn"
            spec["env"]["H3_SHARED_ATTENTION_BACKEND"] = "fa"
            spec["env"]["H3_ENABLE_TORCH_COMPILE"] = "1"
            spec["env"]["H3_COMPILE_CACHE_ROOT"] = "/home/worker/.cache/gpu-runtime/h3-compile"
        if args.torch_compile != "auto":
            spec["env"]["H3_ENABLE_TORCH_COMPILE"] = "1" if args.torch_compile == "on" else "0"

    source_mode = any(
        value is not None
        for value in (
            args.runtime_root,
            args.source_root,
            args.source_revision,
            args.python,
        )
    )
    if source_mode:
        runtime_root = (args.runtime_root or Path(H3_RUNTIME_ROOT)).resolve()
        source_root = (args.source_root or Path(H3_SOURCE_ROOT)).resolve()
        source_revision = str(args.source_revision or "").strip().lower()
        # Keep the venv entry-point path itself.  ``bin/python`` is normally a
        # symlink to the base interpreter; resolving it here silently discards
        # the virtual environment and makes the H3 worker miss its installed
        # runtime dependencies.
        python = Path(os.path.abspath(os.fspath(args.python or Path(H3_PYTHON))))
        if not SOURCE_REVISION_PATTERN.fullmatch(source_revision):
            raise RuntimeError(
                "--source-revision must be one exact 40-character Git commit"
            )
        launcher = runtime_root / "remote/h3_launcher.py"
        source_package = source_root / "python/sglang/__init__.py"
        source_manifest = source_root / "release-manifest.json"
        for required in (launcher, source_package, source_manifest, python):
            if not required.is_file():
                raise RuntimeError(f"required H3 deployment file is missing: {required}")
        for spec in (h3_spec, h3_ref_spec):
            spec["command"] = ["/usr/bin/python3", str(launcher)]
            spec["release_manifest"] = str(source_manifest)
            spec.pop("profile_path", None)
            spec["env"]["H3_PYTHON"] = str(python)
            spec["env"].pop("H3_RELEASE_MANIFEST", None)
            spec["env"]["H3_SOURCE_ROOT"] = str(source_root)
            spec["env"]["H3_SOURCE_REVISION"] = source_revision
            spec["env"]["PYTHONPATH"] = str(source_root / "python")
        h3_spec["ready"] = {"type": "http", "url": "http://127.0.0.1:30010/health"}
        h3_ref_spec["ready"] = {
            "type": "http",
            "url": "http://127.0.0.1:30011/health",
        }

    config = json.loads(args.config.read_text(encoding="utf-8"))
    workloads = config.setdefault("workloads", {})
    workloads["h3"] = h3_spec
    workloads["h3_ref"] = h3_ref_spec
    graph = workloads.get("graph")
    if isinstance(graph, dict):
        graph.setdefault("env", {})["GPU_ORCHESTRATOR_SOCKET"] = (
            "/home/worker/.local/state/gpu-runtime/orchestrator.sock"
        )
        # The CPU-only Comfy graph shapes H3 requests. Keep its profile in
        # lockstep with both remote H3 workers so cache profiles can omit the
        # explicit quality override that disables generic Cache-DiT.
        graph["env"]["H3_ACCELERATION_PROFILE"] = profile

    temporary = args.config.with_suffix(args.config.suffix + ".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(args.config)


if __name__ == "__main__":
    main()
