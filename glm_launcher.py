#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_SOURCE = Path("/mnt/data/worker/.runtime/artifacts/text/engine/r20260722/a1")
DEFAULT_SHARDED = Path("/mnt/data/worker/.runtime/artifacts/text/engine/r20260722/tp8-sharded")
DEFAULT_PYTHON = "/home/worker/miniconda3/envs/runtime-core/bin/python"
DEFAULT_CHAT_TEMPLATE = Path(
    "/home/worker/.local/libexec/gpu-runtime/glm_chat_template_max.jinja"
)
READY_MARKER = ".ready.json"
TP_SIZE = 8
CUDA_GRAPH_BATCH_SIZES = (1, 2)


def sharded_checkpoint_ready(path: Path, tp_size: int = TP_SIZE) -> bool:
    marker = path / READY_MARKER
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if int(data.get("tp_size") or 0) != tp_size:
        return False
    return all(
        any(part.is_file() and part.stat().st_size > 0 for part in path.glob(f"model-rank-{rank}-part-*.safetensors"))
        for rank in range(tp_size)
    )


def build_command(
    source: Path = DEFAULT_SOURCE,
    sharded: Path = DEFAULT_SHARDED,
    python: str = DEFAULT_PYTHON,
    chat_template: Path = DEFAULT_CHAT_TEMPLATE,
) -> list[str]:
    use_sharded = sharded_checkpoint_ready(sharded)
    command = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(sharded if use_sharded else source),
        "--served-model-name",
        "glm-5.2-local",
        "--quantization",
        "w4afp8",
        "--disable-shared-experts-fusion",
        "--tp",
        str(TP_SIZE),
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--reasoning-parser",
        "glm45",
        "--tool-call-parser",
        "glm47",
        "--chat-template",
        str(chat_template),
        "--context-length",
        "409600",
        "--max-total-tokens",
        "409600",
        "--mem-fraction-static",
        "0.75",
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
    else:
        command.extend(
            [
                "--weight-loader-prefetch-checkpoints",
                "--model-loader-extra-config",
                '{"enable_multithread_load":false}',
            ]
        )
    return command


def main() -> None:
    source = Path(os.environ.get("GLM_SOURCE_PATH", str(DEFAULT_SOURCE)))
    sharded = Path(os.environ.get("GLM_SHARDED_PATH", str(DEFAULT_SHARDED)))
    python = os.environ.get("GLM_PYTHON", DEFAULT_PYTHON)
    command = build_command(source=source, sharded=sharded, python=python)
    mode = "sharded_state" if "sharded_state" in command else "coordinated_prefetch"
    print(f"glm_launcher mode={mode} model_path={command[command.index('--model-path') + 1]}", flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
