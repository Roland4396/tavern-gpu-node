#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_MODEL = Path("/mnt/data/worker/.runtime/artifacts/text/qwen3.8-27b")
DEFAULT_PRESHARDED_PATH = Path(
    "/mnt/data/worker/.runtime/artifacts/text/qwen3.8-27b-tp2-presharded"
)
DEFAULT_PYTHON = "/home/worker/.venvs/qwen38-sglang-0.5.17/bin/python"
MODEL_ID = "qwen3.8-27b-local"
TP_SIZE = 2
PRESHARDED_FILE_BYTES = 4 * 1024**3
NATIVE_CONTEXT_LENGTH = 262_144
YARN_CONTEXT_LENGTH = 524_288
YARN_MODEL_OVERRIDE = {
    "text_config": {
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "yarn",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "factor": 2.0,
            "original_max_position_embeddings": NATIVE_CONTEXT_LENGTH,
        }
    }
}


def checkpoint_ready(path: Path) -> bool:
    metadata = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    )
    index_path = path / "model.safetensors.index.json"
    if not all((path / name).is_file() for name in metadata) or not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = {str(name) for name in index["weight_map"].values()}
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return bool(shards) and all(
        (path / shard).is_file() and (path / shard).stat().st_size > 0
        for shard in shards
    )


def build_command(
    model: Path = DEFAULT_MODEL,
    python: str = DEFAULT_PYTHON,
    *,
    mode: str = "native",
    port: int = 30_020,
    presharded_path: Path = DEFAULT_PRESHARDED_PATH,
    verify_on_load: bool = False,
    disable_cuda_graph: bool = False,
) -> list[str]:
    if not checkpoint_ready(model):
        raise RuntimeError(f"Qwen3.8 checkpoint is incomplete: {model}")
    if mode not in {"native", "yarn"}:
        raise RuntimeError(f"unsupported Qwen3.8 context mode: {mode}")

    context_length = YARN_CONTEXT_LENGTH if mode == "yarn" else NATIVE_CONTEXT_LENGTH
    command = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--served-model-name",
        MODEL_ID,
        "--tp",
        str(TP_SIZE),
        "--load-format",
        "presharded",
        "--model-loader-extra-config",
        json.dumps(
            {
                "presharded_path": str(presharded_path),
                "max_file_bytes": PRESHARDED_FILE_BYTES,
                "hash_num_threads": 8,
                "verify_on_load": verify_on_load,
                "enable_multithread_load": True,
                "num_threads": 8,
            },
            separators=(",", ":"),
        ),
        "--context-length",
        str(context_length),
        "--mem-fraction-static",
        "0.85",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--attention-backend",
        "flashinfer",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--chunked-prefill-size",
        "32768",
        "--max-prefill-tokens",
        "32768",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if mode == "yarn":
        command.extend(
            [
                "--json-model-override-args",
                json.dumps(YARN_MODEL_OVERRIDE, separators=(",", ":")),
            ]
        )
    if disable_cuda_graph:
        command.append("--disable-cuda-graph")
    return command


def main() -> None:
    model = Path(os.environ.get("QWEN38_MODEL_PATH", str(DEFAULT_MODEL)))
    presharded_path = Path(
        os.environ.get("QWEN38_PRESHARDED_PATH", str(DEFAULT_PRESHARDED_PATH))
    )
    python = os.environ.get("QWEN38_PYTHON", DEFAULT_PYTHON)
    mode = os.environ.get("QWEN38_CONTEXT_MODE", "native").strip().lower()
    port = int(os.environ.get("QWEN38_PORT", "30020"))
    verify_on_load = os.environ.get("QWEN38_VERIFY_ON_LOAD") == "1"
    disable_cuda_graph = os.environ.get("QWEN38_DISABLE_CUDA_GRAPH") == "1"
    command = build_command(
        model=model,
        python=python,
        mode=mode,
        port=port,
        presharded_path=presharded_path,
        verify_on_load=verify_on_load,
        disable_cuda_graph=disable_cuda_graph,
    )
    if mode == "yarn":
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    print(
        f"qwen38_launcher mode={mode} tp={TP_SIZE} "
        f"context={command[command.index('--context-length') + 1]} "
        f"model={model} presharded={presharded_path}",
        flush=True,
    )
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
