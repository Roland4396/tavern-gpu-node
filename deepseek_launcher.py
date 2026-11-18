#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_MODEL = Path("/mnt/data/worker/.runtime/artifacts/text/deepseek-v4-flash-0731")
DEFAULT_SHARDED = Path("/mnt/data/worker/.runtime/artifacts/text/deepseek-v4-flash-0731-tp4-sharded")
DEFAULT_PYTHON = "/home/worker/miniconda3/envs/runtime-core/bin/python"
DEFAULT_CUDA_HOME = "/usr/local/cuda"
MODEL_ID = "deepseek-v4-flash-local"
READY_MARKER = ".ready.json"
TP_SIZE = 4
CONTEXT_LENGTH = 204_800
MAX_TOTAL_TOKENS = 262_144
MAX_RUNNING_REQUESTS = 8
CUDA_GRAPH_BATCH_SIZES = (1, 2, 4, 8)


def checkpoint_ready(path: Path) -> bool:
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if "DeepseekV4ForCausalLM" not in config.get("architectures", []):
        return False
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    weight_files = {str(filename) for filename in weight_map.values() if filename}
    if not weight_files:
        return False
    required_metadata = ("tokenizer.json", "tokenizer_config.json", "encoding/encoding_dsv4.py")
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in (*weight_files, *required_metadata))


def sharded_checkpoint_ready(path: Path, tp_size: int = TP_SIZE) -> bool:
    try:
        marker = json.loads((path / READY_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if int(marker.get("tp_size") or 0) != tp_size:
        return False
    return all(
        any(part.is_file() and part.stat().st_size > 0 for part in path.glob(f"model-rank-{rank}-part-*.safetensors"))
        for rank in range(tp_size)
    )


def build_command(
    model: Path = DEFAULT_MODEL,
    sharded: Path = DEFAULT_SHARDED,
    python: str = DEFAULT_PYTHON,
) -> list[str]:
    if not checkpoint_ready(model):
        raise RuntimeError(f"DeepSeek V4 checkpoint is incomplete: {model}")
    use_sharded = sharded_checkpoint_ready(sharded)
    command = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(sharded if use_sharded else model),
        "--served-model-name",
        MODEL_ID,
        "--tp",
        str(TP_SIZE),
        "--moe-runner-backend",
        "marlin",
        "--tool-call-parser",
        "deepseekv4",
        "--context-length",
        str(CONTEXT_LENGTH),
        "--max-total-tokens",
        str(MAX_TOTAL_TOKENS),
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        "--mem-fraction-static",
        "0.80",
        "--cuda-graph-bs",
        *(str(size) for size in CUDA_GRAPH_BATCH_SIZES),
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
    ]
    if use_sharded:
        command.extend(["--load-format", "sharded_state"])
    return command


def main() -> None:
    model = Path(os.environ.get("DEEPSEEK_V4_MODEL_PATH", str(DEFAULT_MODEL)))
    sharded = Path(os.environ.get("DEEPSEEK_V4_SHARDED_PATH", str(DEFAULT_SHARDED)))
    python = os.environ.get("DEEPSEEK_PYTHON", DEFAULT_PYTHON)
    command = build_command(model=model, sharded=sharded, python=python)
    runtime_bin = str(Path(python).parent)
    cuda_home = os.environ.setdefault("CUDA_HOME", DEFAULT_CUDA_HOME)
    os.environ["PATH"] = os.pathsep.join((runtime_bin, str(Path(cuda_home) / "bin"), os.environ.get("PATH", "")))
    os.environ.setdefault("SGLANG_OPT_USE_TOPK_V2", "false")
    mode = "sharded_state" if "sharded_state" in command else "original"
    print(
        f"deepseek_launcher mode={mode} model_path={command[command.index('--model-path') + 1]} "
        f"tp={TP_SIZE} runner=marlin tool_parser=deepseekv4",
        flush=True,
    )
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
