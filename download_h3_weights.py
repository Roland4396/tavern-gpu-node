#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API_ROOT = "https://modelscope.cn/api/v1/models/MiniMax/MiniMax-H3/repo/files"
DOWNLOAD_ROOT = "https://modelscope.cn/models/MiniMax/MiniMax-H3/resolve/master"
HUGGINGFACE_DOWNLOAD_ROOT = (
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main"
)
ROOT_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "configuration.json",
    "model_index.json",
    "modular_model_index.json",
}


def list_directory(root: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"Revision": "master", "Root": root})
    with urllib.request.urlopen(f"{API_ROOT}?{query}", timeout=60) as response:
        payload = json.load(response)
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope listing failed for {root!r}: {payload}")
    return list((payload.get("Data") or {}).get("Files") or [])


def build_manifest(variant: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    pending = [""]
    while pending:
        root = pending.pop()
        for item in list_directory(root):
            path = str(item.get("Path") or item.get("Name") or "")
            is_tree = str(item.get("Type") or "") == "tree"
            if is_tree:
                if path == variant or path.startswith(f"{variant}/"):
                    pending.append(path)
                continue
            if path.startswith(f"{variant}/") or ("/" not in path and path in ROOT_FILES):
                selected.append(
                    {
                        "path": path,
                        "size": int(item.get("Size") or 0),
                        "sha256": str(item.get("Sha256") or "").lower(),
                    }
                )
    return sorted(selected, key=lambda item: item["path"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, item: dict[str, Any], *, verify_hash: bool) -> bool:
    if not path.is_file() or path.stat().st_size != int(item["size"]):
        return False
    expected = str(item.get("sha256") or "")
    return not verify_hash or not expected or sha256(path) == expected


def relative_to_variant(path: str, variant: str) -> str | None:
    prefix = f"{variant}/"
    return path[len(prefix) :] if path.startswith(prefix) else None


def identify_reusable_files(
    manifest: list[dict[str, Any]],
    reuse_manifest: list[dict[str, Any]],
    destination: Path,
    variant: str,
    reuse_variant: str,
    staging: Path,
) -> tuple[set[str], int]:
    reusable = {
        relative: item
        for item in reuse_manifest
        if (relative := relative_to_variant(str(item["path"]), reuse_variant))
    }
    paths: set[str] = set()
    total = 0
    for item in manifest:
        relative = relative_to_variant(str(item["path"]), variant)
        source_item = reusable.get(relative or "")
        if (
            relative is None
            or relative.startswith("transformer/")
            or source_item is None
            or int(source_item["size"]) != int(item["size"])
            or not item.get("sha256")
            or source_item.get("sha256") != item.get("sha256")
        ):
            continue
        source = destination / str(source_item["path"])
        if not valid(source, source_item, verify_hash=False):
            continue
        (staging / str(item["path"])).unlink(missing_ok=True)
        paths.add(str(item["path"]))
        total += int(item["size"])
    return paths, total


def download_one(
    staging: Path, item: dict[str, Any], download_root: str = DOWNLOAD_ROOT
) -> str:
    relative = str(item["path"])
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid(target, item, verify_hash=True):
        return f"cached {relative}"
    url = f"{download_root.rstrip('/')}/{urllib.parse.quote(relative, safe='/')}"
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "30",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(target),
        url,
    ]
    subprocess.run(command, check=True)
    if not valid(target, item, verify_hash=True):
        raise RuntimeError(f"download verification failed: {relative}")
    return f"downloaded {relative}"


def copy_sequentially(staging: Path, destination: Path, item: dict[str, Any]) -> None:
    relative = str(item["path"])
    source = staging / relative
    target = destination / relative
    if valid(target, item, verify_hash=False):
        print(f"already published {relative}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    temporary.unlink(missing_ok=True)
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if temporary.stat().st_size != int(item["size"]):
        raise RuntimeError(f"published size mismatch: {relative}")
    temporary.replace(target)
    print(f"published {relative}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variant", choices=("FL2VA", "Ref2VA"), default="FL2VA")
    parser.add_argument(
        "--download-root",
        default=DOWNLOAD_ROOT,
        help="Base resolve URL; Hugging Face works with %s" % HUGGINGFACE_DOWNLOAD_ROOT,
    )
    parser.add_argument(
        "--reuse-variant",
        choices=("FL2VA", "Ref2VA"),
        help="Reuse byte-identical files from another complete variant in destination",
    )
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    args.staging.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.variant)
    if args.reuse_variant == args.variant:
        parser.error("--reuse-variant must differ from --variant")
    reused_paths: set[str] = set()
    if args.reuse_variant:
        reuse_complete = (
            args.destination / args.reuse_variant / ".complete.json"
        ).is_file() or (
            args.reuse_variant == "FL2VA"
            and (args.destination / ".complete.json").is_file()
        )
        if not reuse_complete:
            raise RuntimeError(
                f"reuse variant is incomplete: {args.reuse_variant}"
            )
        reused_paths, reused_bytes = identify_reusable_files(
            manifest,
            build_manifest(args.reuse_variant),
            args.destination,
            args.variant,
            args.reuse_variant,
            args.staging,
        )
        print(
            f"reused files={len(reused_paths)} bytes={reused_bytes} "
            f"variant={args.reuse_variant}",
            flush=True,
        )
    total = sum(int(item["size"]) for item in manifest)
    (args.staging / "download-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest files={len(manifest)} bytes={total}", flush=True)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending_downloads = [
            item
            for item in manifest
            if str(item["path"]) not in reused_paths
            and not valid(
                args.destination / str(item["path"]),
                item,
                verify_hash=False,
            )
        ]
        futures = {
            pool.submit(download_one, args.staging, item, args.download_root): item
            for item in pending_downloads
        }
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)
    print(f"downloads verified elapsed={time.monotonic() - started:.1f}s", flush=True)

    for item in manifest:
        if str(item["path"]) in reused_paths:
            print(f"reused {item['path']}", flush=True)
            continue
        copy_sequentially(args.staging, args.destination, item)
    variant_marker = args.destination / args.variant / ".complete.json"
    variant_marker.parent.mkdir(parents=True, exist_ok=True)
    variant_marker.write_text(
        json.dumps({"variant": args.variant, "files": len(manifest), "bytes": total}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"publish complete destination={args.destination}", flush=True)

    if not args.keep_staging:
        shutil.rmtree(args.staging)
        print("staging removed", flush=True)


if __name__ == "__main__":
    main()
