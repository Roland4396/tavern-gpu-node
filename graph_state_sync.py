#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tarfile
import time
from pathlib import Path


DATA_ROOT = Path(os.environ.get("GRAPH_DATA_ROOT", "/home/worker/.local/share/graph-stack/data"))
SHARED_ROOT = Path(os.environ.get("GRAPH_SHARED_STATE_ROOT", "/mnt/data/worker/.runtime/state/graph/v1"))
USER_DIR = DATA_ROOT / "user"
SNAPSHOT = SHARED_ROOT / "user-state.tar.gz"


def _purge_directory_contents(root: Path) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"refusing to purge unexpected Graph transient path: {root}")

    files = 0
    bytes_removed = 0
    for child in root.iterdir():
        if child.is_symlink():
            child.unlink()
            continue
        if child.is_dir():
            for descendant in child.rglob("*"):
                if descendant.is_file() and not descendant.is_symlink():
                    files += 1
                    bytes_removed += descendant.stat().st_size
            shutil.rmtree(child)
            continue
        if child.is_file():
            files += 1
            bytes_removed += child.stat().st_size
            child.unlink()
    return files, bytes_removed


def purge_transient_state() -> dict[str, int]:
    totals = {"files_removed": 0, "bytes_removed": 0}
    roots = [SHARED_ROOT / "input", SHARED_ROOT / "output", DATA_ROOT / "temp"]
    spool = os.environ.get("GRAPH_OUTPUT_SPOOL", "")
    if spool:
        roots.append(Path(spool))
    for root in roots:
        files, bytes_removed = _purge_directory_contents(root)
        totals["files_removed"] += files
        totals["bytes_removed"] += bytes_removed
    return totals


def ensure_file_layout() -> None:
    DATA_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    (DATA_ROOT / "temp").mkdir(mode=0o700, parents=True, exist_ok=True)
    SHARED_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in ("input", "output"):
        shared = SHARED_ROOT / name
        shared.mkdir(mode=0o700, parents=True, exist_ok=True)
        local = DATA_ROOT / name
        if local.is_symlink():
            if local.resolve() != shared.resolve():
                raise RuntimeError(f"unexpected {name} symlink: {local} -> {local.resolve()}")
            continue
        if local.exists():
            if not local.is_dir() or any(local.iterdir()):
                raise RuntimeError(f"refusing to replace non-empty Graph {name} directory: {local}")
            local.rmdir()
        local.symlink_to(shared, target_is_directory=True)


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"unsafe snapshot path: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"snapshot links are not allowed: {member.name}")
    archive.extractall(destination)


def restore() -> None:
    ensure_file_layout()
    purge_transient_state()
    if not SNAPSHOT.exists():
        USER_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        return
    staging = DATA_ROOT / f".user-restore-{os.getpid()}"
    backup = DATA_ROOT / f".user-backup-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    staging.mkdir(mode=0o700)
    with tarfile.open(SNAPSHOT, "r:gz") as archive:
        safe_extract(archive, staging)
    restored = staging / "user"
    if not restored.is_dir():
        raise RuntimeError("Graph snapshot does not contain user/")
    if USER_DIR.exists():
        USER_DIR.rename(backup)
    restored.rename(USER_DIR)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)


def archive_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if member.name.endswith(".lock") or member.name.endswith(("-shm", "-wal")):
        return None
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def publish() -> None:
    ensure_file_layout()
    USER_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    local_tmp = DATA_ROOT / f".user-state-{os.getpid()}.tar.gz"
    shared_tmp = SHARED_ROOT / f".user-state-{socket.gethostname()}-{os.getpid()}.tmp"
    try:
        with tarfile.open(local_tmp, "w:gz", compresslevel=1) as archive:
            archive.add(USER_DIR, arcname="user", recursive=True, filter=archive_filter)
        shutil.copyfile(local_tmp, shared_tmp)
        with shared_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(shared_tmp, SNAPSHOT)
        metadata = {
            "node": os.environ.get("GPU_RUNTIME_NODE") or socket.gethostname(),
            "published_at": time.time(),
            "snapshot_bytes": SNAPSHOT.stat().st_size,
        }
        metadata_tmp = SHARED_ROOT / f".metadata-{os.getpid()}.tmp"
        metadata_tmp.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        os.replace(metadata_tmp, SHARED_ROOT / "metadata.json")
    finally:
        local_tmp.unlink(missing_ok=True)
        shared_tmp.unlink(missing_ok=True)
        purge_transient_state()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"restore", "publish", "purge"}:
        raise SystemExit("usage: graph_state_sync.py restore|publish|purge")
    if sys.argv[1] == "restore":
        restore()
    elif sys.argv[1] == "publish":
        publish()
    else:
        print(json.dumps(purge_transient_state(), sort_keys=True))


if __name__ == "__main__":
    main()
