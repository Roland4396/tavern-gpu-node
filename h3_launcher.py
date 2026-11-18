#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

try:
    from h3_release_guard import verify_from_environment
except ModuleNotFoundError:  # Repository layout; release artifacts place it beside this file.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release"))
    from h3_release_guard import verify_from_environment


DEFAULT_MODEL = Path("/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3")
DEFAULT_PYTHON = Path("/home/worker/.venvs/sglang-h3/bin/python")
# SageAttention's H3 path has produced corrupted/noise frames on H20 in the
# deployed stack. Keep the exact FlashAttention path as the safe default;
# SageAttention remains an explicit opt-in override for future A/B tests.
DEFAULT_ATTENTION_BACKEND = "fa"
DEFAULT_SHARED_ATTENTION_BACKEND = "fa"
DEFAULT_LORA_CONFIG_ROOT = Path("/home/worker/.cache/gpu-runtime/h3-lora-configs")
LORA_MERGE_MODES = {"auto", "dynamic", "merge"}
FP8_READY_MARKER = ".complete.json"
TRANSFORMER_QUANTIZATION_MODES = {"fp8", "none"}
VARIANTS = {
    "FL2VA": "fl2va",
    "Ref2VA": "ref2va",
}
SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# H3's DiT stack is shared by FL2VA and Ref2VA.  Keep the acceleration policy
# in the launcher (rather than in a Diffusers ``--cache-dit-config`` file):
# native SGLang reads these environment variables before importing the
# pipeline.  ``lossless`` deliberately removes every Cache-DiT override so a
# stale node environment cannot silently change the numerical path.
_CACHE_CONSERVATIVE_ENV = {
    "SGLANG_CACHE_DIT_ENABLED": "1",
    "SGLANG_CACHE_DIT_FN": "1",
    "SGLANG_CACHE_DIT_BN": "0",
    "SGLANG_CACHE_DIT_WARMUP": "4",
    "SGLANG_CACHE_DIT_RDT": "0.04",
    "SGLANG_CACHE_DIT_MC": "1",
}
_CACHE_STRIDE_ENV = {
    "SGLANG_CACHE_DIT_ENABLED": "1",
    "SGLANG_CACHE_DIT_FN": "1",
    "SGLANG_CACHE_DIT_BN": "0",
    "SGLANG_CACHE_DIT_WARMUP": "4",
    "SGLANG_CACHE_DIT_RDT": "0.08",
    "SGLANG_CACHE_DIT_MC": "3",
}

H3_ACCELERATION_PROFILES = {
    "lossless": {},
    # RunningHub's public launcher enables native Cache-DiT defaults. Pin the
    # values from its SGLang f8cbf000 snapshot rather than inheriting a stale
    # SubBlock/cache-stride environment. This is approximate, not lossless.
    "rh_lightning": {
        "SGLANG_CACHE_DIT_ENABLED": "1",
        "SGLANG_CACHE_DIT_FN": "1",
        "SGLANG_CACHE_DIT_BN": "0",
        "SGLANG_CACHE_DIT_WARMUP": "4",
        "SGLANG_CACHE_DIT_RDT": "0.24",
        "SGLANG_CACHE_DIT_MC": "3",
    },
    # LMSYS' conservative H200 profile: compute the first block, warm up four
    # denoising steps, use a 4% residual-difference threshold, and allow only
    # one consecutive cached step.  This is the H20-safe starting point.
    "cache_conservative": _CACHE_CONSERVATIVE_ENV.copy(),
    # The stride profile keeps the same warmup and block policy but permits up
    # to three cached steps.  It is faster and intentionally opt-in because
    # it trades more temporal fidelity for throughput.
    "cache_stride": _CACHE_STRIDE_ENV.copy(),
    # LMSYS' SubBlock schedule: 64-token blocks, 4x4 query/key sub-blocks,
    # dense for the first ten denoising evaluations and for short sequences.
    # These profiles are intentionally opt-in until the SM90 CuTe source patch
    # is installed and validated on H20.
    "subblock_075": {},
    "subblock_080": {},
    "subblock_075_cache_conservative": _CACHE_CONSERVATIVE_ENV.copy(),
    "subblock_075_cache_stride": _CACHE_STRIDE_ENV.copy(),
    "subblock_080_cache_conservative": _CACHE_CONSERVATIVE_ENV.copy(),
    "subblock_080_cache_stride": _CACHE_STRIDE_ENV.copy(),
}

H3_SUBBLOCK_PROFILES = frozenset(
    name for name in H3_ACCELERATION_PROFILES if name.startswith("subblock_")
)
H3_SUBBLOCK_ATTENTION_CONFIG = {
    "skip_first_layers": 0,
    "skip_first_steps": 10,
    "min_seq_len": 4096,
    "n_k": 4,
    "n_q": 4,
}

# Remove all variables that can affect Cache-DiT, including secondary
# transformer/SCM knobs.  H3 currently has one packed transformer, but clearing
# the secondary set prevents inherited settings from affecting future model
# revisions or a shared worker process.
_CACHE_DIT_ENV_KEYS = (
    "SGLANG_CACHE_DIT_ENABLED",
    "SGLANG_CACHE_DIT_FN",
    "SGLANG_CACHE_DIT_BN",
    "SGLANG_CACHE_DIT_WARMUP",
    "SGLANG_CACHE_DIT_RDT",
    "SGLANG_CACHE_DIT_MC",
    "SGLANG_CACHE_DIT_TAYLORSEER",
    "SGLANG_CACHE_DIT_TS_ORDER",
    "SGLANG_CACHE_DIT_SCM_PRESET",
    "SGLANG_CACHE_DIT_SCM_COMPUTE_BINS",
    "SGLANG_CACHE_DIT_SCM_CACHE_BINS",
    "SGLANG_CACHE_DIT_SCM_POLICY",
    "SGLANG_CACHE_DIT_SECONDARY_FN",
    "SGLANG_CACHE_DIT_SECONDARY_BN",
    "SGLANG_CACHE_DIT_SECONDARY_WARMUP",
    "SGLANG_CACHE_DIT_SECONDARY_RDT",
    "SGLANG_CACHE_DIT_SECONDARY_MC",
    "SGLANG_CACHE_DIT_SECONDARY_TAYLORSEER",
    "SGLANG_CACHE_DIT_SECONDARY_TS_ORDER",
    "H3_SUBBLOCK_ENABLED",
)


def normalize_acceleration_profile(value: str | None) -> str:
    """Validate and canonicalize the H3 acceleration profile name."""

    # Use the previously validated speed profile by default.  ``lossless``
    # remains available as an explicit rollback when exact, uncached
    # numerical behavior is needed.
    profile = (value or "subblock_075_cache_stride").strip().lower()
    if profile not in H3_ACCELERATION_PROFILES:
        expected = ", ".join(sorted(H3_ACCELERATION_PROFILES))
        raise RuntimeError(
            f"unsupported H3_ACCELERATION_PROFILE: {profile!r}; expected one of {expected}"
        )
    return profile


def apply_acceleration_profile(env: dict[str, str], profile: str) -> str:
    """Apply one complete native Cache-DiT profile to a child environment.

    Profiles are intentionally whole-environment policies.  This avoids a
    partially configured Cache-DiT run when a service manager carries over an
    old variable, and makes ``lossless`` a reliable rollback path.
    """

    profile = normalize_acceleration_profile(profile)
    for key in _CACHE_DIT_ENV_KEYS:
        env.pop(key, None)
    env["H3_ACCELERATION_PROFILE"] = profile
    env["H3_SUBBLOCK_ENABLED"] = "1" if profile in H3_SUBBLOCK_PROFILES else "0"
    env.update(H3_ACCELERATION_PROFILES[profile])
    return profile


def attention_options_for_profile(
    profile: str,
    attention_backend: str,
    shared_attention_backend: str,
) -> tuple[str, str, str | None]:
    """Resolve the server-wide and component-local attention policy.

    SubBlock must be the global backend so the DiT selects it on its first
    forward.  The Qwen text encoder does not implement SubBlock, therefore it
    is explicitly kept on FlashAttention.  Existing lossless/Cache-DiT
    profiles retain the deployment's configured SageAttention/FA split.
    """

    profile = normalize_acceleration_profile(profile)
    if profile not in H3_SUBBLOCK_PROFILES:
        return (
            shared_attention_backend,
            f"transformer={attention_backend}",
            None,
        )
    suffix = "_".join(profile.split("_")[1:])
    sparsity = "0.80" if suffix.startswith("080") else "0.75"
    config = dict(H3_SUBBLOCK_ATTENTION_CONFIG)
    config["sparsity"] = float(sparsity)
    return (
        "subblock_sparse_attn",
        "text_encoder=fa",
        json.dumps(config, separators=(",", ":"), sort_keys=True),
    )


def execution_options(env: dict[str, str], profile: str) -> list[str]:
    """Keep compilation opt-in except for the explicit RunningHub recipe."""
    compile_enabled = env.get(
        "H3_ENABLE_TORCH_COMPILE", "1" if profile == "rh_lightning" else "0"
    ).strip().lower() in {"1", "true", "yes"}
    if compile_enabled:
        cache_root = Path(env.get(
            "H3_COMPILE_CACHE_ROOT", "/home/worker/.cache/gpu-runtime/h3-compile"
        ))
        for variable, directory in (
            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
            ("TRITON_CACHE_DIR", "triton"),
        ):
            path = cache_root / directory
            path.mkdir(parents=True, exist_ok=True)
            env[variable] = str(path)
    # The author candidate reuses a Qwen checkpoint folded over all four ranks,
    # independently of the DiT's TP2 x U2 grouping. Do not leave that layout to
    # the topology-dependent auto policy, or valid TP4 shards could be misused.
    encoder = (
        ["--encoder-parallel", "fold"]
        if env.get("H3_AUTHOR_CONTRACT") == "comfy_t8_v1" or env.get("H3_ORIGINAL_MANIFEST")
        else []
    )
    return encoder + ["--enable-torch-compile", "true" if compile_enabled else "false"]


def apply_author_contract(env: dict[str, str]) -> None:
    """Enable the separately selected author recipe, never silently inherit RH."""
    contract = env.get("H3_AUTHOR_CONTRACT", "").strip()
    if not contract:
        env.pop("SGLANG_H3_AUTHOR_CONTRACT", None)
        return
    if contract != "comfy_t8_v1":
        raise RuntimeError(f"Unknown H3_AUTHOR_CONTRACT: {contract}")
    expected = {
        "H3_MODEL_VARIANT": "fl2va",
        "H3_TRANSFORMER_QUANTIZATION": "none",
        "H3_ACCELERATION_PROFILE": "lossless",
        "H3_TP_SIZE": "2", "H3_ULYSSES_DEGREE": "2",
        "H3_ATTENTION_BACKEND": "sage_attn",
    }
    for key, value in expected.items():
        if str(env.get(key, "")).strip().lower() != value:
            raise RuntimeError(f"{contract} requires {key}={value}")
    env["SGLANG_H3_AUTHOR_CONTRACT"] = contract


def original_node_options(env: dict[str, str]) -> list[str]:
    """Opt-in ORIGINAL graph mode, separate from the older video-API adapter.

    Sage and LoRAs are selected by each original model branch, not globally by
    the launcher. Qwen/refiner stay FA and no cache skips original evaluations.
    """
    path = env.get("H3_ORIGINAL_MANIFEST", "").strip()
    if not path:
        env.pop("SGLANG_H3_ORIGINAL_MANIFEST", None)
        return []
    if env.get("H3_AUTHOR_CONTRACT") or env.get("H3_LORA_PATH"):
        raise RuntimeError("Original-node mode cannot inherit the older video adapter or startup LoRA")
    expected = {"H3_TRANSFORMER_QUANTIZATION": "none", "H3_ACCELERATION_PROFILE": "lossless",
                "H3_TP_SIZE": "2", "H3_ULYSSES_DEGREE": "2", "H3_ATTENTION_BACKEND": "fa",
                "H3_SHARED_ATTENTION_BACKEND": "fa"}
    for name, value in expected.items():
        if env.get(name) != value:
            raise RuntimeError(f"Original-node mode requires {name}={value}")
    manifest = json.loads(Path(path).read_text())
    endpoint = manifest["scheduler_endpoint"]
    if not endpoint.startswith("ipc:///"):
        raise RuntimeError("Original-node scheduler endpoint must be a private absolute Unix socket")
    socket_path = Path(endpoint.removeprefix("ipc://"))
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    env["SGLANG_H3_ORIGINAL_MANIFEST"] = str(Path(path).resolve())
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return ["--scheduler-ipc-path", str(socket_path)]


def verify_source_checkout(source_root: Path, expected_revision: str) -> str:
    """Fail closed unless the worker imports one clean, exact SGLang commit."""

    expected_revision = expected_revision.strip().lower()
    if not SOURCE_REVISION_PATTERN.fullmatch(expected_revision):
        raise RuntimeError(
            f"invalid H3_SOURCE_REVISION: {expected_revision!r}; expected 40 hex characters"
        )
    if not (source_root / "python/sglang/__init__.py").is_file():
        raise RuntimeError(f"H3 source checkout is incomplete: {source_root}")
    try:
        actual_revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot verify H3 source checkout {source_root}: {exc}") from exc
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"H3 source revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    if dirty:
        raise RuntimeError(f"H3 source checkout is dirty: {source_root}")
    return actual_revision


def fp8_checkpoint_ready(path: Path) -> bool:
    try:
        marker = json.loads((path / FP8_READY_MARKER).read_text(encoding="utf-8"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        index = json.loads(
            (path / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    quantization = config.get("quantization_config") or {}
    if (
        marker.get("format") != "h3-fp8-block"
        or quantization.get("quant_method") != "fp8"
        or quantization.get("weight_block_size") != [128, 128]
    ):
        return False
    files = set((index.get("weight_map") or {}).values())
    return bool(files) and all((path / name).is_file() for name in files)


def transformer_load_options(
    model: Path,
    variant: str,
    fp8_transformer: Path,
    quantization_mode: str,
    raw_transformer: Path | None = None,
) -> tuple[str, str]:
    """Resolve one explicit transformer weight path without silent fallback.

    The offline FP8 export is an optimization, not the source of truth.  The
    explicit ``none`` mode loads the original BF16 transformer so a corrupted
    FP8 conversion can be isolated on H20; it must never silently fall back to
    online FP8 quantization.
    """

    mode = (quantization_mode or "fp8").strip().lower()
    if mode not in TRANSFORMER_QUANTIZATION_MODES:
        expected = ", ".join(sorted(TRANSFORMER_QUANTIZATION_MODES))
        raise RuntimeError(
            "unsupported H3_TRANSFORMER_QUANTIZATION: "
            f"{mode!r}; expected one of {expected}"
        )
    if mode == "none":
        raw = raw_transformer or (model / variant / "transformer")
        if not raw.is_dir():
            raise RuntimeError(f"H3 raw transformer directory is missing: {raw}")
        return "--transformer-weights-path", str(raw)
    if fp8_checkpoint_ready(fp8_transformer):
        return "--transformer-weights-path", str(fp8_transformer)
    return "--quantization", "fp8"


def stage_lora_config(lora_path: Path, lora_alpha: int, root: Path) -> Path:
    adapter_dir = root / f"{lora_path.stem}-alpha-{lora_alpha}"
    adapter_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged_lora = adapter_dir / lora_path.name
    if staged_lora.is_symlink() and staged_lora.resolve() != lora_path.resolve():
        staged_lora.unlink()
    elif staged_lora.exists() and not staged_lora.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink LoRA staging path: {staged_lora}")
    if not staged_lora.exists():
        staged_lora.symlink_to(lora_path.resolve())
    config = adapter_dir / "adapter_config.json"
    temporary = config.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"lora_alpha": lora_alpha}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(config)
    return staged_lora


def main() -> int:
    acceleration_profile = normalize_acceleration_profile(
        os.environ.get("H3_ACCELERATION_PROFILE")
    )
    model = Path(os.environ.get("H3_MODEL_PATH", str(DEFAULT_MODEL)))
    variant = os.environ.get("H3_MODEL_VARIANT", "FL2VA").strip()
    variant = next(
        (name for name in VARIANTS if name.lower() == variant.lower()),
        variant,
    )
    if variant not in VARIANTS:
        raise RuntimeError(
            f"unsupported H3 model variant: {variant}; expected one of {sorted(VARIANTS)}"
        )
    python = Path(os.environ.get("H3_PYTHON", str(DEFAULT_PYTHON)))
    attention_backend = os.environ.get(
        "H3_ATTENTION_BACKEND",
        "sage_attn" if acceleration_profile == "rh_lightning" else DEFAULT_ATTENTION_BACKEND,
    ).strip()
    shared_attention_backend = os.environ.get(
        "H3_SHARED_ATTENTION_BACKEND", DEFAULT_SHARED_ATTENTION_BACKEND
    ).strip()
    launch_attention_backend, component_attention_backends, attention_backend_config = (
        attention_options_for_profile(
            acceleration_profile,
            attention_backend,
            shared_attention_backend,
        )
    )
    fp8_transformer = Path(
        os.environ.get(
            "H3_FP8_TRANSFORMER_PATH", str(model / variant / "transformer-fp8")
        )
    )
    transformer_quantization = os.environ.get(
        "H3_TRANSFORMER_QUANTIZATION", "fp8"
    )
    raw_transformer_value = os.environ.get("H3_TRANSFORMER_WEIGHTS_PATH", "").strip()
    raw_transformer = Path(raw_transformer_value) if raw_transformer_value else None
    complete = model / variant / ".complete.json"
    if variant == "FL2VA" and not complete.is_file():
        complete = model / ".complete.json"
    if not python.is_file():
        raise RuntimeError(f"H3 Python environment is missing: {python}")
    if not complete.is_file():
        raise RuntimeError(f"H3 model is incomplete: {complete}")

    source_root_value = os.environ.get("H3_SOURCE_ROOT", "").strip()
    source_revision = os.environ.get("H3_SOURCE_REVISION", "").strip()
    source_python = ""
    if source_root_value or source_revision:
        if not source_root_value or not source_revision:
            raise RuntimeError(
                "H3_SOURCE_ROOT and H3_SOURCE_REVISION must be configured together"
            )
        source_root = Path(source_root_value).resolve()
        verified_revision = verify_source_checkout(source_root, source_revision)
        source_python = str(source_root / "python")
        print(
            f"h3_source_verified source_commit={verified_revision} root={source_root}",
            flush=True,
        )

    lora_path_value = os.environ.get("H3_LORA_PATH", "").strip()
    lora_path = Path(lora_path_value) if lora_path_value else None
    lora_alpha_value = os.environ.get("H3_LORA_ALPHA", "").strip()
    lora_alpha = int(lora_alpha_value) if lora_alpha_value else None
    lora_scale = float(os.environ.get("H3_LORA_SCALE", "1.0"))
    lora_merge_mode = os.environ.get("H3_LORA_MERGE_MODE", "dynamic").strip()
    if not math.isfinite(lora_scale) or lora_scale <= 0:
        raise RuntimeError(f"H3_LORA_SCALE must be positive and finite: {lora_scale}")
    if lora_merge_mode not in LORA_MERGE_MODES:
        raise RuntimeError(
            f"unsupported H3_LORA_MERGE_MODE: {lora_merge_mode}; "
            f"expected one of {sorted(LORA_MERGE_MODES)}"
        )
    if lora_path is not None and not lora_path.is_file():
        raise RuntimeError(f"H3 LoRA is missing: {lora_path}")

    release = verify_from_environment()
    if release is not None:
        print(
            "h3_release_verified "
            f"release_id={release['release_id']} "
            f"source_commit={release.get('source_commit')} "
            f"expected_lora_layers={release.get('expected_lora_covered_layers')}/"
            f"{release.get('expected_lora_total_layers')}",
            flush=True,
        )
    if lora_alpha is not None:
        if lora_alpha <= 0:
            raise RuntimeError(f"H3_LORA_ALPHA must be positive: {lora_alpha}")
        if lora_path is None:
            raise RuntimeError("H3_LORA_ALPHA requires H3_LORA_PATH")
        lora_path = stage_lora_config(
            lora_path,
            lora_alpha,
            Path(os.environ.get("H3_LORA_CONFIG_ROOT", str(DEFAULT_LORA_CONFIG_ROOT))),
        )

    tp_size = int(os.environ.get("H3_TP_SIZE", "2"))
    ulysses_degree = int(os.environ.get("H3_ULYSSES_DEGREE", "2"))
    if tp_size < 1 or ulysses_degree < 1 or tp_size * ulysses_degree != 4:
        raise RuntimeError(
            "H3_TP_SIZE and H3_ULYSSES_DEGREE must be positive and multiply to 4"
        )

    command = [
        str(python),
        "-c",
        "from sglang.cli.main import main; main()",
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        str(model),
        "--model-variant",
        VARIANTS[variant],
        "--num-gpus",
        "4",
        "--tp-size",
        str(tp_size),
        "--ulysses-degree",
        str(ulysses_degree),
        "--performance-mode",
        "speed",
        "--warmup-mode",
        "off",
        "--attention-backend",
        launch_attention_backend,
        "--component-attention-backends",
        component_attention_backends,
        "--host",
        "127.0.0.1",
        "--port",
        os.environ.get("H3_PORT", "30010"),
    ]
    if attention_backend_config is not None:
        command.extend(["--attention-backend-config", attention_backend_config])
    weight_option, weight_value = transformer_load_options(
        model,
        variant,
        fp8_transformer,
        transformer_quantization,
        raw_transformer,
    )
    command.extend([weight_option, weight_value])
    if lora_path is not None:
        command.extend(
            [
                "--lora-path",
                str(lora_path),
                "--lora-scale",
                str(lora_scale),
                "--lora-merge-mode",
                lora_merge_mode,
            ]
        )
    env = os.environ.copy()
    env.setdefault("SGLANG_VIDEO_PROGRESS_DIR", "/home/worker/.local/state/gpu-runtime/h3-progress")
    # The rank-shard helper is part of the SGLang source tree and reads its
    # own namespace.  Keep the node configuration's H3_* setting as the
    # public interface, then pass it through to the worker process explicitly.
    rank_shard_root = env.get("H3_RANK_SHARD_ROOT", "").strip()
    if rank_shard_root:
        env["SGLANG_H3_RANK_SHARD_ROOT"] = rank_shard_root
    else:
        env.pop("SGLANG_H3_RANK_SHARD_ROOT", None)
    # Apply a complete profile after copying the service environment.  This
    # both enables the requested native Cache-DiT mode and removes stale
    # SGLANG_CACHE_DIT_* values inherited from a supervisor or shell.
    apply_acceleration_profile(env, acceleration_profile)
    apply_author_contract(env)
    command.extend(original_node_options(env))
    command.extend(execution_options(env, acceleration_profile))
    print(
        "h3_execution_config " + json.dumps({
            "profile": acceleration_profile,
            "tp": tp_size,
            "ulysses": ulysses_degree,
            "attention": launch_attention_backend,
            "component_attention": component_attention_backends,
            "compile": command[-1],
            "compile_cache": env.get("TORCHINDUCTOR_CACHE_DIR"),
            "cache_dit": {k: v for k, v in env.items() if k.startswith("SGLANG_CACHE_DIT_")},
            "startup_lora": str(lora_path) if lora_path else None,
        }, sort_keys=True),
        flush=True,
    )
    env["PATH"] = f"{python.parent}:{env.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"
    if source_python:
        existing_pythonpath = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = (
            f"{source_python}:{existing_pythonpath}"
            if existing_pythonpath
            else source_python
        )
    env["SGLANG_DIFFUSION_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
    env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = os.environ.get(
        "H3_DEEPGEMM_PRECOMPILE", "0"
    )
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=env)

    def forward(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
