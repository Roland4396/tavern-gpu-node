#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_SHARED = Path(
    "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3"
)
DEFAULT_OVERLAY = Path(
    "/home/worker/.cache/gpu-runtime/h3-ref-model/MiniMax-H3"
)
ROOT_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "configuration.json",
    "model_index.json",
    "modular_model_index.json",
)
SHARED_COMPONENTS = (
    "audio_vae",
    "processor",
    "text_encoder",
    "tokenizer",
    "video_vae",
)


def link(target: Path, source: Path) -> None:
    if not source.exists():
        raise RuntimeError(f"H3 overlay source is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def prepare(shared: Path, overlay: Path) -> None:
    fl_complete = (shared / "FL2VA" / ".complete.json").is_file() or (
        shared / ".complete.json"
    ).is_file()
    if not fl_complete:
        raise RuntimeError(f"FL2VA is incomplete: {shared}")
    if not (shared / "Ref2VA" / ".complete.json").is_file():
        raise RuntimeError(f"Ref2VA is incomplete: {shared}")

    temporary = overlay.with_name(f".{overlay.name}.tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        for name in ROOT_FILES:
            link(temporary / name, shared / name)
        for name in SHARED_COMPONENTS:
            link(temporary / "Ref2VA" / name, shared / "FL2VA" / name)
        link(
            temporary / "Ref2VA" / "model_index.json",
            shared / "Ref2VA" / "model_index.json",
        )
        link(
            temporary / "Ref2VA" / "transformer",
            shared / "Ref2VA" / "transformer",
        )
        link(
            temporary / "Ref2VA" / ".complete.json",
            shared / "Ref2VA" / ".complete.json",
        )
        shutil.rmtree(overlay, ignore_errors=True)
        temporary.replace(overlay)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"h3_ref_overlay_ready path={overlay}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    args = parser.parse_args()
    prepare(args.shared, args.overlay)


if __name__ == "__main__":
    main()
