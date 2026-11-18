#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import signal
import socket
import socketserver
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


LOG = logging.getLogger("gpu-node-agent")
MAX_BODY_BYTES = 8 * 1024 * 1024
WORKLOAD_PATH = re.compile(r"^/workloads/([a-z0-9_-]+)/(ensure|stop|activity|publish)$")
H3_JOB_PATH = re.compile(
    r"^/h3/jobs/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(/artifact)?$"
)
H3_JOB_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
H3_LOCAL_OUTPUT_ROOT = Path("/home/worker/outputs")
H3_SHARED_INPUT_ROOT = Path("/mnt/data/worker/.runtime/state/h3/v1/input")
H3_SHARED_OUTPUT_ROOT = Path("/mnt/data/worker/.runtime/state/h3/v1/output")
H3_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = _canonical_json(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Content-Type": "application/json"} if body is not None else {}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method=method
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    try:
        decoded = json.loads(raw or b"{}")
    except (TypeError, ValueError):
        decoded = {"raw": (raw or b"").decode("utf-8", errors="replace")}
    return status, decoded if isinstance(decoded, dict) else {"data": decoded}


class H3JobRegistry:
    """Node-local durable execution facts for assigned H3 jobs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=5000")
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS h3_jobs (
                    queue_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    operation_id TEXT NOT NULL,
                    upstream_job_id TEXT,
                    workload TEXT NOT NULL CHECK (workload IN ('h3', 'h3_ref')),
                    request_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    lora_json TEXT NOT NULL,
                    reclaim_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_json TEXT,
                    output_path TEXT,
                    output_size INTEGER,
                    output_sha256 TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    submitted_at REAL,
                    terminal_at REAL
                );
                CREATE INDEX IF NOT EXISTS h3_local_workload_queue
                    ON h3_jobs(workload, status, queue_seq);
                CREATE TABLE IF NOT EXISTS h3_job_events (
                    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    previous_status TEXT,
                    status TEXT,
                    detail_json TEXT,
                    FOREIGN KEY(job_id) REFERENCES h3_jobs(job_id) ON DELETE CASCADE
                );
                PRAGMA user_version=1;
                """
            )
            # Native H3 installations before the durable idempotency contract
            # assign their own UUID to every POST /v1/videos request.  Keep
            # the caller's durable operation id as ``job_id`` and persist the
            # server-assigned id separately so polling/publishing can follow
            # the actual upstream job without creating a duplicate.
            columns = {
                str(row[1])
                for row in self._db.execute("PRAGMA table_info(h3_jobs)").fetchall()
            }
            if "upstream_job_id" not in columns:
                self._db.execute("ALTER TABLE h3_jobs ADD COLUMN upstream_job_id TEXT")
            self._db.execute("PRAGMA user_version=2")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _event_locked(
        self,
        job_id: str,
        previous: str | None,
        status: str | None,
        detail: Any = None,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO h3_job_events(job_id, created_at, previous_status, status, detail_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                job_id,
                time.time(),
                previous,
                status,
                None if detail is None else _canonical_json(detail),
            ),
        )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["payload"] = _decode_json(result.pop("payload_json"), {})
        result["lora_action"] = _decode_json(result.pop("lora_json"), {})
        result["reclaim"] = _decode_json(result.pop("reclaim_json"), [])
        result["error"] = _decode_json(result.pop("error_json"), None)
        return result

    def create(self, data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        job_id = str(data.get("job_id") or "")
        operation_id = str(data.get("operation_id") or "")
        workload = str(data.get("workload") or "")
        request_hash = str(data.get("request_hash") or "")
        payload = data.get("payload")
        lora_action = data.get("lora_action")
        reclaim = data.get("reclaim") or []
        if not H3_JOB_ID.fullmatch(job_id) or operation_id != job_id:
            raise ValueError("job_id and operation_id must be the same canonical UUID")
        if workload not in {"h3", "h3_ref"}:
            raise ValueError("invalid H3 workload")
        if not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            raise ValueError("request_hash must be SHA-256")
        if not isinstance(payload, dict) or not isinstance(lora_action, dict):
            raise ValueError("payload and lora_action must be objects")
        if not isinstance(reclaim, list) or any(not isinstance(item, str) for item in reclaim):
            raise ValueError("reclaim must be an array of workload names")
        now = time.time()
        with self._lock, self._db:
            existing = self._db.execute(
                "SELECT * FROM h3_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if existing is not None:
                job = self._row(existing)
                assert job is not None
                if job["request_hash"] != request_hash or job["workload"] != workload:
                    raise RuntimeError("H3 job id is already bound to a different request")
                return job, False
            self._db.execute(
                """
                INSERT INTO h3_jobs(
                    job_id, operation_id, workload, request_hash, payload_json,
                    lora_json, reclaim_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    job_id,
                    operation_id,
                    workload,
                    request_hash,
                    _canonical_json(payload),
                    _canonical_json(lora_action),
                    _canonical_json(reclaim),
                    now,
                    now,
                ),
            )
            self._event_locked(job_id, None, "queued", {"workload": workload})
            result = self.get(job_id)
            assert result is not None
            return result, True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(
                self._db.execute(
                    "SELECT * FROM h3_jobs WHERE job_id=?", (job_id,)
                ).fetchone()
            )

    def set_upstream_job_id(self, job_id: str, upstream_job_id: str) -> dict[str, Any]:
        if not H3_JOB_ID.fullmatch(upstream_job_id):
            raise ValueError("upstream_job_id must be a canonical UUID")
        with self._lock, self._db:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            existing = str(current.get("upstream_job_id") or "")
            if existing and existing != upstream_job_id:
                raise RuntimeError("H3 upstream job id cannot be changed after submission")
            if existing == upstream_job_id:
                return current
            self._db.execute(
                "UPDATE h3_jobs SET upstream_job_id=?, updated_at=? WHERE job_id=?",
                (upstream_job_id, time.time(), job_id),
            )
            self._event_locked(
                job_id,
                str(current["status"]),
                str(current["status"]),
                {"upstream_job_id": upstream_job_id},
            )
            result = self.get(job_id)
            assert result is not None
            return result

    def next_for_workload(self, workload: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT * FROM h3_jobs
                WHERE workload=? AND status NOT IN ('completed', 'failed', 'cancelled')
                ORDER BY queue_seq LIMIT 1
                """,
                (workload,),
            ).fetchone()
            return self._row(row)

    def active_workloads(self) -> set[str]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT DISTINCT workload FROM h3_jobs
                WHERE status NOT IN ('completed', 'failed', 'cancelled')
                """
            ).fetchall()
            return {str(row[0]) for row in rows}

    def transition(
        self,
        job_id: str,
        status: str,
        *,
        detail: Any = None,
        error: Any = None,
        output_path: str | None = None,
        output_size: int | None = None,
        output_sha256: str | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._db:
            current = self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            previous = str(current["status"])
            if previous in H3_TERMINAL_STATUSES:
                return current
            now = time.time()
            self._db.execute(
                """
                UPDATE h3_jobs
                SET status=?, error_json=?, output_path=COALESCE(?, output_path),
                    output_size=COALESCE(?, output_size),
                    output_sha256=COALESCE(?, output_sha256),
                    submitted_at=CASE WHEN ?='submitted' THEN COALESCE(submitted_at, ?) ELSE submitted_at END,
                    terminal_at=CASE WHEN ? IN ('completed','failed','cancelled') THEN COALESCE(terminal_at, ?) ELSE terminal_at END,
                    updated_at=?
                WHERE job_id=?
                """,
                (
                    status,
                    None if error is None else _canonical_json(error),
                    output_path,
                    output_size,
                    output_sha256,
                    status,
                    now,
                    status,
                    now,
                    now,
                    job_id,
                ),
            )
            if previous != status:
                self._event_locked(
                    job_id,
                    previous,
                    status,
                    detail if error is None else {"detail": detail, "error": error},
                )
            result = self.get(job_id)
            assert result is not None
            return result

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None or job["status"] in H3_TERMINAL_STATUSES:
            return job
        if job["status"] == "queued":
            return self.transition(job_id, "cancelled", detail={"phase": "queued"})
        return self.transition(job_id, "cancel_requested", detail={"phase": job["status"]})

    def cleanup_artifact(self, job_id: str, job_root: Path) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job["status"] not in H3_TERMINAL_STATUSES:
            raise RuntimeError("cannot clean a non-terminal H3 job")
        output = job.get("output_path")
        if output:
            try:
                path = Path(str(output)).resolve(strict=False)
                path.relative_to(H3_SHARED_OUTPUT_ROOT.resolve())
                path.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
        conditions = job.get("payload", {}).get("conditions")
        if isinstance(conditions, list):
            shared_input_root = H3_SHARED_INPUT_ROOT.resolve()
            for condition in conditions:
                uri = condition.get("uri") if isinstance(condition, dict) else None
                if not isinstance(uri, str) or not uri.startswith("file:"):
                    continue
                try:
                    parsed = urllib.parse.urlparse(uri)
                    source = Path(urllib.parse.unquote(parsed.path)).resolve(strict=False)
                    source.relative_to(shared_input_root)
                    source.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
        shutil.rmtree(job_root / job_id, ignore_errors=True)
        return job

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT status, COUNT(*) FROM h3_jobs GROUP BY status"
            ).fetchall()
            return {str(row[0]): int(row[1]) for row in rows}


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _process_start_ticks(pid: int) -> int:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])
    except (OSError, ValueError, IndexError):
        return 0


def _process_group_alive(pgid: int) -> bool:
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _unlink_socket(path: str) -> None:
    target = Path(path)
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"refusing to replace non-socket path: {target}")
    target.unlink()


def _unix_http_get(socket_path: str, path: str, timeout: float) -> tuple[int, bytes]:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect(socket_path)
        connection.sendall(
            f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
        )
        response = bytearray()
        while len(response) < 16 * 1024 * 1024:
            chunk = connection.recv(64 * 1024)
            if not chunk:
                break
            response.extend(chunk)
        head, _, body = bytes(response).partition(b"\r\n\r\n")
        first = head.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
        parts = first.split()
        return (int(parts[1]) if len(parts) >= 2 else 0), body
    finally:
        connection.close()


class WorkloadManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.node_name = str(config["node_name"])
        self.state_dir = Path(config["state_dir"])
        self.lock_dir = self.state_dir / "gpu-locks"
        self.log_dir = self.state_dir / "logs"
        self.workload_dir = self.state_dir / "workloads"
        self.h3_job_root = self.state_dir / "h3-jobs"
        for path in (
            self.state_dir,
            self.lock_dir,
            self.log_dir,
            self.workload_dir,
            self.h3_job_root,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.chmod(0o700)
        self.h3_registry = H3JobRegistry(self.state_dir / "h3-jobs.sqlite3")
        self.specs = dict(config.get("workloads") or {})
        self.states: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.lease_fds: dict[str, list[int]] = {}
        self.tickets: dict[str, dict[str, float]] = {}
        self.transitioning: set[str] = set()
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.gpu_memory_lock = threading.Lock()
        initial_gpu_memory = self._gpu_memory()
        self.gpu_memory_snapshot = dict(initial_gpu_memory or {})
        self.gpu_memory_updated_at = time.monotonic() if initial_gpu_memory is not None else 0.0
        self._recover_states()
        self.gpu_monitor = threading.Thread(target=self._gpu_monitor_loop, daemon=True, name="gpu-monitor")
        self.gpu_monitor.start()
        self.reaper = threading.Thread(target=self._reap_loop, daemon=True, name="idle-reaper")
        self.reaper.start()
        self.h3_runners: list[threading.Thread] = []
        for workload in ("h3", "h3_ref"):
            if workload not in self.specs:
                continue
            runner = threading.Thread(
                target=self._h3_job_loop,
                args=(workload,),
                daemon=True,
                name=f"{workload}-job-runner",
            )
            runner.start()
            self.h3_runners.append(runner)

    def close(self) -> None:
        self.stopping.set()
        for thread in [self.gpu_monitor, self.reaper, *self.h3_runners]:
            if thread.is_alive():
                thread.join(timeout=2)
        self.h3_registry.close()

    def _state_path(self, name: str) -> Path:
        return self.workload_dir / f"{name}.json"

    def _recover_states(self) -> None:
        for name in self.specs:
            try:
                state = json.loads(self._state_path(name).read_text(encoding="utf-8"))
                pid = int(state.get("pid") or 0)
                expected_start = int(state.get("process_start_ticks") or 0)
                if _pid_alive(pid) and expected_start and _process_start_ticks(pid) == expected_start:
                    state["ready"] = self._ready(self.specs[name])
                    self.states[name] = state
                    self.tickets[name] = {}
                    continue
            except (OSError, ValueError, TypeError):
                pass
            self._state_path(name).unlink(missing_ok=True)

    def _persist(self, name: str) -> None:
        state = self.states.get(name)
        if state is None:
            self._state_path(name).unlink(missing_ok=True)
        else:
            _atomic_json(self._state_path(name), state)

    def _cleanup_dead_locked(self) -> None:
        for name, state in list(self.states.items()):
            if name in self.transitioning:
                continue
            process = self.processes.get(name)
            if process is not None:
                alive = process.poll() is None
            else:
                alive = _pid_alive(int(state.get("pid") or 0))
            if alive:
                continue
            pid = int(state.get("pid") or 0)
            if pid > 1:
                self._terminate_pid(pid, timeout=0, process=process)
            LOG.info("workload_exited node=%s workload=%s pid=%s", self.node_name, name, state.get("pid"))
            self.states.pop(name, None)
            self.processes.pop(name, None)
            for descriptor in self.lease_fds.pop(name, []):
                os.close(descriptor)
            self.tickets.pop(name, None)
            self._persist(name)
            socket_path = str(self.specs[name].get("socket_path") or "")
            if socket_path:
                try:
                    _unlink_socket(socket_path)
                except RuntimeError:
                    LOG.exception("socket_cleanup_refused workload=%s", name)

    def _gpu_memory(self) -> dict[int, int] | None:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        result: dict[int, int] = {}
        for line in output.splitlines():
            try:
                index, used = (part.strip() for part in line.split(",", 1))
                result[int(index)] = int(used)
            except (ValueError, TypeError):
                continue
        return result

    def _gpu_monitor_loop(self) -> None:
        while not self.stopping.wait(2):
            memory = self._gpu_memory()
            if memory is None:
                continue
            with self.gpu_memory_lock:
                self.gpu_memory_snapshot = dict(memory)
                self.gpu_memory_updated_at = time.monotonic()

    def _cached_gpu_memory(self) -> tuple[dict[int, int], bool, float | None]:
        with self.gpu_memory_lock:
            memory = dict(self.gpu_memory_snapshot)
            updated_at = self.gpu_memory_updated_at
        if not updated_at:
            return memory, False, None
        age = max(0.0, time.monotonic() - updated_at)
        return memory, age <= 15, age

    def _leased_gpus(self) -> set[int]:
        leased: set[int] = set()
        for gpu in range(int(self.config.get("gpu_count") or 0)):
            descriptor = os.open(self.lock_dir / f"gpu-{gpu}.lock", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    leased.add(gpu)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        return leased

    def _acquire_gpus(self, spec: dict[str, Any]) -> tuple[list[int], list[int]]:
        count = int(spec.get("gpu_count") or 0)
        allowed = sorted({int(item) for item in spec.get("allowed_gpus", range(8))})
        if count <= 0:
            return [], []
        memory = self._gpu_memory()
        if memory is None:
            raise RuntimeError("GPU telemetry unavailable")
        for candidate in itertools.combinations(allowed, count):
            descriptors: list[int] = []
            try:
                for gpu in candidate:
                    descriptor = os.open(self.lock_dir / f"gpu-{gpu}.lock", os.O_RDWR | os.O_CREAT, 0o600)
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError:
                        os.close(descriptor)
                        raise
                    os.set_inheritable(descriptor, True)
                    descriptors.append(descriptor)
                if any(memory.get(gpu, 0) > 0 for gpu in candidate):
                    raise BlockingIOError("GPU has an unleased CUDA allocation")
                return list(candidate), descriptors
            except OSError:
                for descriptor in descriptors:
                    os.close(descriptor)
        raise RuntimeError(f"insufficient free GPUs: requested={count} allowed={allowed}")

    @staticmethod
    def _ready(spec: dict[str, Any]) -> bool:
        ready = dict(spec.get("ready") or {})
        kind = str(ready.get("type") or "")
        try:
            if kind == "http":
                url = str(ready["url"])
                with urllib.request.urlopen(url, timeout=3) as response:
                    return 200 <= response.status < 500
            if kind == "unix_http":
                status, _body = _unix_http_get(str(ready["socket"]), str(ready.get("path") or "/health"), 3)
                return 200 <= status < 500
            if kind == "socket":
                return Path(str(ready["socket"])).is_socket()
            if kind == "process":
                return True
        except (OSError, ValueError, urllib.error.URLError):
            return False
        return True

    def _wait_ready(self, name: str, pid: int) -> None:
        spec = self.specs[name]
        deadline = time.monotonic() + int(spec.get("startup_seconds") or 300)
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                raise RuntimeError(f"{name} exited during startup")
            if self._ready(spec):
                return
            time.sleep(1)
        raise TimeoutError(f"{name} readiness timed out")

    @staticmethod
    def _environment(spec: dict[str, Any], node_name: str, workload: str, gpus: list[int]) -> dict[str, str]:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in dict(spec.get("env") or {}).items()})
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
        env["GPU_RUNTIME_NODE"] = node_name
        env["GPU_RUNTIME_WORKLOAD"] = workload
        return env

    @staticmethod
    def _run_hook(
        spec: dict[str, Any],
        hook_name: str,
        env: dict[str, str],
        log_handle: Any,
    ) -> None:
        raw = spec.get(hook_name)
        if not raw:
            return
        hook = dict(raw)
        command = [str(item) for item in hook.get("command") or []]
        if not command:
            raise RuntimeError(f"{hook_name} command is empty")
        subprocess.run(
            command,
            cwd=str(hook.get("cwd") or spec.get("cwd") or "/home/worker"),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=int(hook.get("timeout_seconds") or 300),
            check=True,
        )

    def _terminate_pid(self, pid: int, timeout: int = 60, process: subprocess.Popen[bytes] | None = None) -> None:
        if process is not None and process.poll() is not None:
            process.wait()
        if not _process_group_alive(pid):
            return
        try:
            os.killpg(pid, signal.SIGINT)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + max(0, timeout)
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                process.wait()
            if not _process_group_alive(pid):
                return
            time.sleep(0.25)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def ensure(self, name: str, ticket: str = "") -> tuple[int, dict[str, Any]]:
        if name not in self.specs:
            return 404, {"error": f"unknown workload: {name}"}
        with self.lock:
            self._cleanup_dead_locked()
            if name in self.transitioning:
                return 409, {"error": f"{name} is stopping", "code": "stopping"}
            current = self.states.get(name)
            if current:
                if not bool(current.get("ready")):
                    return 409, {"error": f"{name} is starting", "code": "startup_in_progress", **current}
                if ticket:
                    self.tickets.setdefault(name, {})[ticket] = time.monotonic()
                current["last_activity"] = time.time()
                self._persist(name)
                return 200, {"status": "running", **current}
            spec = self.specs[name]
            try:
                gpus, descriptors = self._acquire_gpus(spec)
            except RuntimeError as exc:
                return 409, {"error": str(exc), "code": "gpu_busy"}
            env = self._environment(spec, self.node_name, name, gpus)
            log_path = self.log_dir / f"{name}.log"
            log_handle = log_path.open("ab", buffering=0)
            try:
                self._run_hook(spec, "pre_start", env, log_handle)
                socket_path = str(spec.get("socket_path") or "")
                if socket_path:
                    _unlink_socket(socket_path)
                process = subprocess.Popen(
                    [str(item) for item in spec["command"]],
                    cwd=str(spec.get("cwd") or "/home/worker"),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=tuple(descriptors),
                )
            except BaseException as exc:
                for descriptor in descriptors:
                    os.close(descriptor)
                log_handle.close()
                return 503, {"error": f"{type(exc).__name__}: {exc}", "code": "pre_start_or_spawn_failed"}
            log_handle.close()
            state = {
                "workload": name,
                "pid": process.pid,
                "process_start_ticks": _process_start_ticks(process.pid),
                "gpus": gpus,
                "started_at": time.time(),
                "last_activity": time.time(),
                "ready": False,
            }
            self.states[name] = state
            self.processes[name] = process
            self.lease_fds[name] = descriptors
            self.tickets[name] = {ticket: time.monotonic()} if ticket else {}
            self._persist(name)
        try:
            self._wait_ready(name, process.pid)
        except BaseException as exc:
            with self.lock:
                self._terminate_pid(process.pid, process=process)
                self.states.pop(name, None)
                self.processes.pop(name, None)
                for descriptor in self.lease_fds.pop(name, []):
                    os.close(descriptor)
                self.tickets.pop(name, None)
                self._persist(name)
            return 503, {"error": str(exc), "code": "startup_failed"}
        with self.lock:
            state = self.states.get(name)
            if not state or int(state.get("pid") or 0) != process.pid:
                return 503, {"error": f"{name} disappeared after startup"}
            state["ready"] = True
            state["last_activity"] = time.time()
            self._persist(name)
            LOG.info("workload_ready node=%s workload=%s pid=%d gpus=%s", self.node_name, name, process.pid, gpus)
            return 200, {"status": "running", **state}

    def stop(self, name: str, reason: str = "requested", *, only_if_idle: bool = False) -> tuple[int, dict[str, Any]]:
        if name not in self.specs:
            return 404, {"error": f"unknown workload: {name}"}
        with self.lock:
            self._cleanup_dead_locked()
            state = self.states.get(name)
            if not state:
                return 200, {"status": "stopped", "workload": name}
            if name in self.transitioning:
                return 409, {"error": f"{name} is stopping", "code": "stopping"}
            # The activity check and admission barrier share the same lock as
            # ensure/activity: a companion shutdown must not race a new request.
            if only_if_idle and (self.tickets.get(name) or self._busy(name) or not state.get("ready")):
                return 409, {"error": f"{name} has an active request", "code": "busy"}
            pid = int(state["pid"])
            process = self.processes.get(name)
            self.transitioning.add(name)
            gpus = [int(gpu) for gpu in state.get("gpus") or []]
        hook_error: BaseException | None = None
        self._terminate_pid(pid, int(self.specs[name].get("stop_seconds") or 60), process)
        spec = self.specs[name]
        log_handle = (self.log_dir / f"{name}.log").open("ab", buffering=0)
        try:
            self._run_hook(spec, "post_stop", self._environment(spec, self.node_name, name, gpus), log_handle)
        except BaseException as exc:
            hook_error = exc
            LOG.exception("post_stop_failed node=%s workload=%s", self.node_name, name)
        finally:
            log_handle.close()
        with self.lock:
            self.states.pop(name, None)
            self.processes.pop(name, None)
            for descriptor in self.lease_fds.pop(name, []):
                os.close(descriptor)
            self.tickets.pop(name, None)
            self.transitioning.discard(name)
            self._persist(name)
            socket_path = str(self.specs[name].get("socket_path") or "")
            if socket_path:
                try:
                    _unlink_socket(socket_path)
                except RuntimeError:
                    LOG.exception("socket_cleanup_refused workload=%s", name)
        LOG.info("workload_stopped node=%s workload=%s reason=%s", self.node_name, name, reason)
        companions = []
        for companion in spec.get("stop_idle_companions", []):
            code, detail = self.stop(companion, reason=f"{name}_stopped", only_if_idle=True)
            companions.append({"workload": companion, "status": code, "detail": detail})
        if hook_error is not None:
            return 503, {
                "status": "stopped",
                "workload": name,
                "reason": reason,
                "code": "post_stop_failed",
                "error": f"{type(hook_error).__name__}: {hook_error}",
            }
        return 200, {"status": "stopped", "workload": name, "reason": reason, "companions": companions}

    def activity(self, name: str, ticket: str, action: str) -> tuple[int, dict[str, Any]]:
        if name not in self.specs:
            return 404, {"error": f"unknown workload: {name}"}
        if not ticket:
            return 400, {"error": "ticket is required"}
        with self.lock:
            self._cleanup_dead_locked()
            state = self.states.get(name)
            if not state:
                return 409, {"error": f"{name} is not running"}
            if name in self.transitioning:
                return 409, {"error": f"{name} is stopping", "code": "stopping"}
            active = self.tickets.setdefault(name, {})
            if action in {"start", "heartbeat"}:
                active[ticket] = time.monotonic()
            elif action == "end":
                active.pop(ticket, None)
            else:
                return 400, {"error": f"invalid activity action: {action}"}
            state["last_activity"] = time.time()
            self._persist(name)
            return 200, {"status": "ok", "active_tickets": len(active)}

    def publish_h3_output(
        self,
        name: str,
        job_id: str,
        *,
        source_job_id: str | None = None,
        local_root: Path | None = None,
        shared_root: Path | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if name not in {"h3", "h3_ref"} or name not in self.specs:
            return 404, {"error": f"unsupported H3 workload: {name}"}
        if not H3_JOB_ID.fullmatch(job_id):
            return 400, {"error": "invalid H3 job id"}
        source_job_id = source_job_id or job_id
        if not H3_JOB_ID.fullmatch(source_job_id):
            return 400, {"error": "invalid upstream H3 job id"}

        local = (local_root or H3_LOCAL_OUTPUT_ROOT).resolve()
        shared = shared_root or H3_SHARED_OUTPUT_ROOT
        shared.mkdir(parents=True, exist_ok=True)
        shared = shared.resolve()
        try:
            source = (local / f"{source_job_id}.mp4").resolve(strict=True)
            source.relative_to(local)
        except (OSError, ValueError):
            return 404, {"error": "completed H3 output was not found"}
        if not source.is_file():
            return 404, {"error": "completed H3 output was not found"}

        destination = shared / f"{job_id}.mp4"
        staging = shared / (
            f".{source.name}.partial-{os.getpid()}-{threading.get_ident()}"
        )
        staging.unlink(missing_ok=True)
        try:
            source_size = source.stat().st_size
            if source_size <= 0:
                return 503, {"error": "completed H3 output is empty"}
            shutil.copyfile(source, staging)
            with staging.open("ab") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            copied_size = staging.stat().st_size
            if copied_size != source_size:
                return 503, {
                    "error": "H3 output publish was truncated",
                    "expected_bytes": source_size,
                    "copied_bytes": copied_size,
                }
            staging.replace(destination)
            source.unlink(missing_ok=True)
            return 200, {
                "status": "published",
                "file_path": str(destination),
                "file_paths": [str(destination)],
                "size_bytes": copied_size,
                "upstream_job_id": source_job_id,
            }
        except OSError as exc:
            LOG.exception(
                "h3_output_publish_failed node=%s workload=%s job=%s",
                self.node_name,
                name,
                job_id,
            )
            return 503, {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            staging.unlink(missing_ok=True)

    def _h3_public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": job["job_id"],
            "job_id": job["job_id"],
            "operation_id": job["operation_id"],
            "workload": job["workload"],
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }
        upstream_id = str(job.get("upstream_job_id") or job["job_id"])
        root = self.specs.get(job["workload"], {}).get("env", {}).get(
            "SGLANG_VIDEO_PROGRESS_DIR", str(self.state_dir / "h3-progress")
        )
        if H3_JOB_ID.fullmatch(upstream_id):
            try:
                progress = json.loads((Path(root) / f"{upstream_id}.json").read_text())
                if (
                    isinstance(progress, dict) and progress.get("job_id") == upstream_id
                    and type(progress.get("step")) is int and type(progress.get("total")) is int
                    and 0 <= progress["step"] <= progress["total"] <= 1000
                    and progress["total"] > 0
                    and progress.get("phase") in {"denoising", "decoding"}
                ):
                    result["sampling_progress"] = {k: progress[k] for k in (
                        "step", "total", "phase", "updated_at"
                    ) if k in progress}
            except (OSError, ValueError):
                pass
        if job.get("error") is not None:
            result["error"] = job["error"]
        if job.get("output_path"):
            result.update(
                {
                    "output_path": job["output_path"],
                    "file_path": job["output_path"],
                    "output_size": job.get("output_size"),
                    "size_bytes": job.get("output_size"),
                    "output_sha256": job.get("output_sha256"),
                    "sha256": job.get("output_sha256"),
                }
            )
        return result

    def submit_h3_job(self, data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            job, created = self.h3_registry.create(data)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        except RuntimeError as exc:
            return 409, {"error": str(exc)}
        LOG.info(
            "h3_job_accepted node=%s workload=%s job=%s created=%s",
            self.node_name,
            job["workload"],
            job["job_id"],
            created,
        )
        return (202 if created else 200), self._h3_public_job(job)

    def get_h3_job(self, job_id: str) -> tuple[int, dict[str, Any]]:
        job = self.h3_registry.get(job_id)
        if job is None:
            return 404, {"error": "H3 job not found"}
        return 200, self._h3_public_job(job)

    def cancel_h3_job(self, job_id: str) -> tuple[int, dict[str, Any]]:
        job = self.h3_registry.cancel(job_id)
        if job is None:
            return 404, {"error": "H3 job not found"}
        return 200, self._h3_public_job(job)

    def cleanup_h3_artifact(self, job_id: str) -> tuple[int, dict[str, Any]]:
        try:
            job = self.h3_registry.cleanup_artifact(job_id, self.h3_job_root)
        except RuntimeError as exc:
            return 409, {"error": str(exc)}
        if job is None:
            return 404, {"error": "H3 job not found"}
        return 200, {"status": "cleaned", "job_id": job_id}

    def cleanup_h3_session(self) -> tuple[int, dict[str, Any]]:
        """Remove only runtime-owned transient media; keep backups/user copies."""
        from graph_state_sync import _purge_directory_contents

        with self.lock:
            counts = self.h3_registry.counts()
            if any(count for status, count in counts.items() if status not in H3_TERMINAL_STATUSES):
                return 409, {"error": "H3 generation is active"}
            if self.states.get("graph") or any(
                self.tickets.get(name) or name in self.transitioning
                for name in ("graph", "h3", "h3_ref", "seedvr3")
            ):
                return 409, {"error": "media session is active"}
            totals = {"files_removed": 0, "bytes_removed": 0}
            try:
                for root in (H3_SHARED_INPUT_ROOT, H3_SHARED_OUTPUT_ROOT, self.h3_job_root):
                    files, size = _purge_directory_contents(root)
                    totals["files_removed"] += files
                    totals["bytes_removed"] += size
                # The legacy output folder also contains unrelated user work.
                # Delete H3's UUID-named MP4s only, never the entire directory.
                for path in H3_LOCAL_OUTPUT_ROOT.glob("*.mp4"):
                    if H3_JOB_ID.fullmatch(path.stem) and not path.is_symlink():
                        size = path.stat().st_size
                        path.unlink()
                        totals["files_removed"] += 1
                        totals["bytes_removed"] += size
            except (OSError, RuntimeError) as exc:
                LOG.exception("h3_session_cleanup_failed")
                return 503, {"error": str(exc), **totals}
            return 200, {"status": "cleaned", **totals}

    def _h3_port(self, workload: str) -> int:
        value = (self.specs[workload].get("env") or {}).get("H3_PORT")
        if value is None:
            value = "30010" if workload == "h3" else "30011"
        return int(value)

    def _h3_worker_alive(self, workload: str) -> bool:
        with self.lock:
            self._cleanup_dead_locked()
            state = self.states.get(workload)
            return bool(state and _pid_alive(int(state.get("pid") or 0)))

    def _materialize_h3_payload(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(_canonical_json(job["payload"]))
        conditions = payload.get("conditions")
        if not isinstance(conditions, list):
            return payload
        root = self.h3_job_root / job["job_id"] / "input"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        shared_root = H3_SHARED_INPUT_ROOT.resolve()
        for index, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                continue
            uri = condition.get("uri")
            if not isinstance(uri, str) or not uri.startswith("file:"):
                continue
            parsed = urllib.parse.urlparse(uri)
            source = Path(urllib.parse.unquote(parsed.path)).resolve(strict=True)
            source.relative_to(shared_root)
            if not source.is_file() or source.stat().st_size <= 0:
                raise RuntimeError(f"H3 condition is missing or empty: {source}")
            suffix = source.suffix if source.suffix else ".bin"
            target = root / f"{index:02d}{suffix}"
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.unlink(missing_ok=True)
            shutil.copyfile(source, temporary)
            with temporary.open("ab") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"H3 condition copy was truncated: {source.name}")
            temporary.replace(target)
            condition["uri"] = target.resolve().as_uri()
        return payload

    def _configure_h3_lora_local(
        self, workload: str, action: dict[str, Any]
    ) -> None:
        path = str(action.get("path") or "")
        payload = action.get("payload")
        if path not in {"/v1/set_lora", "/v1/unmerge_lora_weights"}:
            raise RuntimeError("invalid H3 LoRA action path")
        if not isinstance(payload, dict):
            raise RuntimeError("invalid H3 LoRA action payload")
        status, result = _http_json(
            "POST",
            f"http://127.0.0.1:{self._h3_port(workload)}{path}",
            payload,
            timeout=120,
        )
        if not 200 <= status < 300:
            raise RuntimeError(f"H3 LoRA configuration failed ({status}): {result}")

    @staticmethod
    def _validate_h3_video(path: Path) -> tuple[int, str]:
        if not path.is_file():
            raise RuntimeError(f"H3 output is missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"H3 output is empty: {path}")
        ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "H3 output failed ffprobe validation: "
                + result.stderr.strip()[:1000]
            )
        try:
            probe = json.loads(result.stdout)
            duration = float((probe.get("format") or {}).get("duration") or 0)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("H3 output returned invalid ffprobe metadata") from exc
        if duration <= 0:
            raise RuntimeError("H3 output has no positive media duration")
        return size, _sha256_file(path)

    def _publish_completed_h3_job(self, job: dict[str, Any]) -> dict[str, Any]:
        upstream_job_id = str(job.get("upstream_job_id") or job["job_id"])
        source = H3_LOCAL_OUTPUT_ROOT / f"{upstream_job_id}.mp4"
        destination = H3_SHARED_OUTPUT_ROOT / f"{job['job_id']}.mp4"
        if source.is_file():
            expected_size, expected_sha256 = self._validate_h3_video(source)
            status, published = self.publish_h3_output(
                job["workload"],
                job["job_id"],
                source_job_id=upstream_job_id,
            )
            if status != 200:
                raise RuntimeError(f"H3 output publish failed ({status}): {published}")
            destination = Path(str(published["file_path"]))
        elif destination.is_file():
            # The prior process may have completed the atomic rename and exited
            # before committing the SQLite transition.  Recover from the
            # durable shared artifact instead of misclassifying it as failure.
            expected_size, expected_sha256 = self._validate_h3_video(destination)
        else:
            raise RuntimeError(f"H3 output is missing: {source}")
        actual_size, actual_sha256 = self._validate_h3_video(destination)
        if actual_size != expected_size or actual_sha256 != expected_sha256:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "H3 published output checksum mismatch: "
                f"size={actual_size}/{expected_size} sha256={actual_sha256}/{expected_sha256}"
            )
        return self.h3_registry.transition(
            job["job_id"],
            "completed",
            detail={"published": True},
            output_path=str(destination),
            output_size=actual_size,
            output_sha256=actual_sha256,
        )

    def _cancel_running_h3_job(self, job: dict[str, Any]) -> dict[str, Any]:
        workload = str(job["workload"])
        upstream_job_id = str(job.get("upstream_job_id") or job["job_id"])
        try:
            status, result = _http_json(
                "DELETE",
                f"http://127.0.0.1:{self._h3_port(workload)}/v1/videos/{upstream_job_id}",
                None,
                timeout=15,
            )
        except (OSError, urllib.error.URLError) as exc:
            status, result = 503, {"error": f"{type(exc).__name__}: {exc}"}
        if not 200 <= status < 300 and self._h3_worker_alive(workload):
            self.stop(workload, reason=f"cancel_h3_job_{job['job_id']}")
        return self.h3_registry.transition(
            job["job_id"],
            "cancelled",
            detail={"upstream_status": status, "upstream": result},
        )

    def _run_h3_job_once(self, job: dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        workload = str(job["workload"])
        status = str(job["status"])
        if status == "cancel_requested":
            self._cancel_running_h3_job(job)
            return

        if status in {"queued", "starting_worker"}:
            self.h3_registry.transition(
                job_id, "starting_worker", detail={"node": self.node_name}
            )
            for victim in job.get("reclaim") or []:
                if victim in self.specs and victim != workload:
                    self.stop(victim, reason=f"reclaimed_for_{workload}_job")
            code, detail = self.ensure(workload, f"h3job-{job_id}")
            if code != 200:
                raise RuntimeError(f"H3 worker startup failed ({code}): {detail}")
            payload = self._materialize_h3_payload(job)
            self._configure_h3_lora_local(workload, job["lora_action"])
            self.h3_registry.transition(
                job_id, "submitted", detail={"worker_pid": detail.get("pid")}
            )
            try:
                upstream_status, result = _http_json(
                    "POST",
                    f"http://127.0.0.1:{self._h3_port(workload)}/v1/videos",
                    payload,
                    timeout=2700,
                    headers={"Idempotency-Key": job_id},
                )
            except (OSError, urllib.error.URLError) as exc:
                # The request may have been accepted before the response path
                # disappeared.  Keep the durable submitted state and reconcile
                # by stable job id instead of starting a second generation.
                LOG.warning(
                    "h3_submit_response_lost node=%s workload=%s job=%s error=%r",
                    self.node_name,
                    workload,
                    job_id,
                    exc,
                )
                return
            if upstream_status == 409 and str(result.get("code") or "") == "request_in_progress":
                return
            if not 200 <= upstream_status < 300:
                raise RuntimeError(
                    f"H3 SGLang submission failed ({upstream_status}): {result}"
                )
            upstream_job_id = str(result.get("id") or "").strip().lower()
            if not H3_JOB_ID.fullmatch(upstream_job_id):
                raise RuntimeError(f"H3 SGLang returned an invalid job id: {result}")
            if upstream_job_id != job_id:
                LOG.warning(
                    "h3_upstream_job_id_assigned node=%s workload=%s job=%s upstream_job=%s",
                    self.node_name,
                    workload,
                    job_id,
                    upstream_job_id,
                )
            self.h3_registry.set_upstream_job_id(job_id, upstream_job_id)
            return

        if status in {"submitted", "running"}:
            upstream_job_id = str(job.get("upstream_job_id") or job_id)
            if not self._h3_worker_alive(workload):
                self.h3_registry.transition(
                    job_id,
                    "failed",
                    error={"code": "worker_exited", "workload": workload},
                )
                return
            submitted_at = float(job.get("submitted_at") or time.time())
            timeout = float(self.config.get("h3_job_timeout_seconds") or 7200)
            if time.time() - submitted_at > timeout:
                try:
                    abort_status, _abort_result = _http_json(
                        "DELETE",
                        f"http://127.0.0.1:{self._h3_port(workload)}/v1/videos/{upstream_job_id}",
                        None,
                        timeout=15,
                    )
                except (OSError, urllib.error.URLError):
                    abort_status = 503
                if not 200 <= abort_status < 300 and self._h3_worker_alive(workload):
                    self.stop(workload, reason=f"timeout_h3_job_{job_id}")
                self.h3_registry.transition(
                    job_id,
                    "failed",
                    error={"code": "generation_timeout", "seconds": timeout},
                )
                return
            try:
                upstream_status, result = _http_json(
                    "GET",
                    f"http://127.0.0.1:{self._h3_port(workload)}/v1/videos/{upstream_job_id}",
                    None,
                    timeout=10,
                )
            except (OSError, urllib.error.URLError):
                return
            if upstream_status == 404:
                # A lost create response can briefly precede the store entry.
                # Wait for several polls before replaying the same idempotent
                # POST; idempotent SGLang reserves the key while the first
                # create runs.  Once an upstream id has been recorded, a 404
                # means that the server-side job disappeared and must not be
                # replaced by a second generation under a new UUID.
                if job.get("upstream_job_id"):
                    raise RuntimeError(
                        "H3 upstream job disappeared after submission: "
                        f"{upstream_job_id}"
                    )
                if time.time() - float(job.get("updated_at") or 0) < 10:
                    return
                payload = self._materialize_h3_payload(job)
                try:
                    create_status, create_result = _http_json(
                        "POST",
                        f"http://127.0.0.1:{self._h3_port(workload)}/v1/videos",
                        payload,
                        timeout=2700,
                        headers={"Idempotency-Key": job_id},
                    )
                except (OSError, urllib.error.URLError):
                    return
                if create_status == 409 and str(create_result.get("code") or "") == "request_in_progress":
                    return
                if not 200 <= create_status < 300:
                    raise RuntimeError(
                        f"H3 idempotent resubmission failed ({create_status}): {create_result}"
                    )
                returned_id = str(create_result.get("id") or "").strip().lower()
                if not H3_JOB_ID.fullmatch(returned_id):
                    raise RuntimeError(
                        f"H3 idempotent resubmission returned an invalid job id: {create_result}"
                    )
                self.h3_registry.set_upstream_job_id(job_id, returned_id)
                return
            if upstream_status != 200:
                return
            upstream_state = str(result.get("status") or "").lower()
            if upstream_state in {"queued", "submitted", "running", "in_progress"}:
                if status != "running":
                    self.h3_registry.transition(
                        job_id, "running", detail={"upstream_status": upstream_state}
                    )
                return
            if upstream_state == "completed":
                self.h3_registry.transition(job_id, "publishing")
                return
            if upstream_state in {"cancelled", "canceled", "deleted"}:
                self.h3_registry.transition(job_id, "cancelled", detail=result)
                return
            if upstream_state == "failed":
                self.h3_registry.transition(
                    job_id,
                    "failed",
                    error=result.get("error") or result,
                )
                return
            return

        if status == "publishing":
            self._publish_completed_h3_job(job)

    def _h3_job_loop(self, workload: str) -> None:
        while not self.stopping.wait(1):
            job = self.h3_registry.next_for_workload(workload)
            if job is None:
                continue
            try:
                self._run_h3_job_once(job)
            except BaseException as exc:
                LOG.exception(
                    "h3_job_failed node=%s workload=%s job=%s",
                    self.node_name,
                    workload,
                    job.get("job_id"),
                )
                current = self.h3_registry.get(str(job.get("job_id") or ""))
                if current and current["status"] not in H3_TERMINAL_STATUSES:
                    self.h3_registry.transition(
                        current["job_id"],
                        "failed",
                        error={"code": "execution_failed", "message": f"{type(exc).__name__}: {exc}"},
                    )

    def _busy(self, name: str) -> bool:
        if name in {"h3", "h3_ref"} and name in self.h3_registry.active_workloads():
            return True
        probe = dict(self.specs[name].get("busy") or {})
        if probe.get("type") != "unix_http_json":
            return False
        try:
            status, body = _unix_http_get(str(probe["socket"]), str(probe["path"]), 3)
            if status != 200:
                return False
            payload = json.loads(body)
            return any(bool(payload.get(field)) for field in probe.get("fields") or [])
        except (OSError, ValueError, TypeError):
            return False

    def status(self) -> dict[str, Any]:
        with self.lock:
            self._cleanup_dead_locked()
            workloads = {}
            occupied = self._leased_gpus()
            gpu_memory, gpu_telemetry_ok, gpu_telemetry_age = self._cached_gpu_memory()
            memory_occupied = {gpu for gpu, used_mib in gpu_memory.items() if used_mib > 0}
            if gpu_telemetry_ok:
                occupied.update(memory_occupied)
            else:
                occupied.update(range(int(self.config.get("gpu_count") or 8)))
            managed_gpus: set[int] = set()
            for name in self.specs:
                spec = self.specs[name]
                configured_gpu_count = int(spec.get("gpu_count") or 0)
                release = self._release_metadata(name)
                state = self.states.get(name)
                if state:
                    state_gpus = {int(gpu) for gpu in state.get("gpus", [])}
                    managed_gpus.update(state_gpus)
                    occupied.update(state_gpus)
                    workloads[name] = {
                        **state,
                        "status": "stopping" if name in self.transitioning else ("running" if state.get("ready") else "starting"),
                        "gpu_count": configured_gpu_count,
                        "active_tickets": len(self.tickets.get(name, {})),
                        "busy": self._busy(name),
                        **release,
                    }
                else:
                    workloads[name] = {
                        "workload": name,
                        "status": "stopped",
                        "gpu_count": configured_gpu_count,
                        "gpus": [],
                        **release,
                    }
            return {
                "status": "ok",
                "node": self.node_name,
                **self._stack_metadata(),
                "h3_jobs": self.h3_registry.counts(),
                "gpu_count": int(self.config.get("gpu_count") or 8),
                "occupied_gpus": sorted(occupied),
                "external_occupied_gpus": sorted(memory_occupied - managed_gpus),
                "gpu_memory_mib": gpu_memory,
                "gpu_telemetry_ok": gpu_telemetry_ok,
                "gpu_telemetry_age_seconds": round(gpu_telemetry_age, 3) if gpu_telemetry_age is not None else None,
                "workloads": workloads,
            }

    def _release_metadata(self, name: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        profile_path = str(self.specs[name].get("profile_path") or "")
        if profile_path:
            try:
                profile = Path(profile_path)
                resolved = profile.resolve(strict=True)
                result.update(
                    {
                        "profile_path": profile_path,
                        "profile_store_path": str(resolved),
                    }
                )
            except (OSError, RuntimeError) as exc:
                result["profile_error"] = f"{type(exc).__name__}: {exc}"
        path = str(self.specs[name].get("release_manifest") or "")
        if not path:
            return result
        try:
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
            release_id = str(manifest.get("release_id") or "")
            source_commit = str(manifest.get("source_commit") or "")
            if not release_id:
                raise ValueError("release_id is missing")
            result.update({
                "release_id": release_id,
                "release_source_commit": source_commit,
            })
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            result["release_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _stack_metadata(self) -> dict[str, Any]:
        path = str(self.config.get("stack_manifest") or "")
        if not path:
            return {}
        try:
            raw = Path(path).read_bytes()
            manifest = json.loads(raw)
            stack_id = str(manifest.get("stack_id") or "")
            if not stack_id:
                raise ValueError("stack_id is missing")
            return {
                "stack_id": stack_id,
                "stack_manifest": path,
                "stack_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            }
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            return {"stack_error": f"{type(exc).__name__}: {exc}"}

    def voices(self) -> dict[str, Any]:
        release = Path(str(self.config["tts_release_root"]))
        raw = json.loads((release / "catalog.json").read_text(encoding="utf-8"))
        result = []
        for entry in raw.get("voices", []):
            voice_id = str(entry.get("id") or "")
            metadata = json.loads((release / "roles" / voice_id / "metadata.json").read_text(encoding="utf-8"))
            result.append(
                {
                    "id": voice_id,
                    "name": str(metadata.get("display_name") or metadata.get("role") or voice_id),
                    "language": str(metadata.get("language") or "zh"),
                    "model": str(metadata.get("model_id") or "gpt-sovits-v2proplus"),
                    "model_version": str(metadata.get("model_version") or "v2ProPlus"),
                    "revision": str(entry.get("revision") or metadata.get("revision") or ""),
                }
            )
        return {"object": "list", "data": result, "revision": str(raw.get("revision") or "")}

    def _reap_loop(self) -> None:
        while not self.stopping.wait(5):
            now = time.time()
            stale: list[str] = []
            with self.lock:
                self._cleanup_dead_locked()
                monotonic_now = time.monotonic()
                for name, tickets in self.tickets.items():
                    for ticket, seen in list(tickets.items()):
                        if monotonic_now - seen > 120:
                            tickets.pop(ticket, None)
                for name, state in self.states.items():
                    if not state.get("ready"):
                        if self._ready(self.specs[name]):
                            state["ready"] = True
                            state["last_activity"] = now
                            self._persist(name)
                        elif now - float(state.get("started_at") or now) >= int(self.specs[name].get("startup_seconds") or 300):
                            stale.append(name)
                        continue
                    if self._busy(name):
                        state["last_activity"] = now
                        self._persist(name)
                        continue
                    idle = int(self.specs[name].get("idle_seconds") or 0)
                    if idle and not self.tickets.get(name) and now - float(state.get("last_activity") or now) >= idle:
                        stale.append(name)
            for name in stale:
                self.stop(name, reason="idle_expired")


class AgentServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, path: str, manager: WorkloadManager):
        self.manager = manager
        super().__init__(path, AgentHandler)


class AgentHandler(socketserver.StreamRequestHandler):
    server: AgentServer

    def _respond(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        reason = {
            200: "OK",
            201: "Created",
            202: "Accepted",
            400: "Bad Request",
            404: "Not Found",
            409: "Conflict",
            503: "Unavailable",
        }.get(status, "Error")
        self.wfile.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        )
        self.wfile.write(body)

    def handle(self) -> None:
        try:
            request_line = self.rfile.readline(8192).decode("latin-1").strip()
            method, target, _version = request_line.split(" ", 2)
            headers: dict[str, str] = {}
            while True:
                line = self.rfile.readline(8192)
                if line in {b"\r\n", b"\n", b""}:
                    break
                key, value = line.decode("latin-1").split(":", 1)
                headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length") or 0)
            if length < 0 or length > MAX_BODY_BYTES:
                self._respond(400, {"error": "invalid content length"})
                return
            body = self.rfile.read(length) if length else b"{}"
            data = json.loads(body) if body else {}
            path = target.split("?", 1)[0]
            manager = self.server.manager
            if method == "GET" and path == "/health":
                self._respond(200, {"status": "ok", "node": manager.node_name})
                return
            if method == "GET" and path == "/status":
                self._respond(200, manager.status())
                return
            if method == "GET" and path == "/catalog/voices":
                self._respond(200, manager.voices())
                return
            if method == "POST" and path == "/graph/media/sign":
                from graph_oss_media import sign_media
                try:
                    payload = sign_media(data)
                    self._respond(200, payload)
                except FileNotFoundError:
                    self._respond(404, {"error": "Graph image not found"})
                except ValueError:
                    self._respond(400, {"error": "invalid Graph media path"})
                except Exception as exc:
                    LOG.warning("graph_oss_sign_failed error_type=%s", type(exc).__name__)
                    self._respond(503, {"error": "OSS signing unavailable"})
                return
            if method == "DELETE" and path == "/h3/artifacts":
                status, payload = manager.cleanup_h3_session()
                self._respond(status, payload)
                return
            h3_match = H3_JOB_PATH.fullmatch(path)
            if h3_match:
                job_id, artifact_suffix = h3_match.groups()
                if method == "PUT" and artifact_suffix is None:
                    status, payload = manager.submit_h3_job(data)
                elif method == "GET" and artifact_suffix is None:
                    status, payload = manager.get_h3_job(job_id)
                elif method == "DELETE" and artifact_suffix is None:
                    status, payload = manager.cancel_h3_job(job_id)
                elif method == "DELETE" and artifact_suffix == "/artifact":
                    status, payload = manager.cleanup_h3_artifact(job_id)
                else:
                    status, payload = 404, {"error": "not found"}
                self._respond(status, payload)
                return
            match = WORKLOAD_PATH.fullmatch(path)
            if method == "POST" and match:
                name, action = match.groups()
                if action == "ensure":
                    status, payload = manager.ensure(name, str(data.get("ticket") or ""))
                elif action == "stop":
                    status, payload = manager.stop(name, str(data.get("reason") or "requested"),
                                                   only_if_idle=bool(data.get("only_if_idle")))
                elif action == "publish":
                    status, payload = manager.publish_h3_output(
                        name,
                        str(data.get("job_id") or ""),
                    )
                else:
                    status, payload = manager.activity(name, str(data.get("ticket") or ""), str(data.get("action") or ""))
                self._respond(status, payload)
                return
            self._respond(404, {"error": "not found"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._respond(400, {"error": str(exc)})
        except BaseException as exc:
            LOG.exception("request_failed")
            self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    # This executable is the long-running agent and intentionally has no
    # command-line status/query mode.  Reject arguments instead of silently
    # starting a second agent when an operator mistakenly runs e.g.
    # ``node_agent.py status --json``.  Status must be read through the
    # agent's Unix-socket HTTP endpoint.
    if len(sys.argv) != 1:
        print(
            "node_agent.py does not accept command-line arguments; "
            "query status through the configured Unix socket",
            file=sys.stderr,
        )
        raise SystemExit(2)
    config_path = Path(os.environ.get("GPU_NODE_CONFIG", "/home/worker/.config/gpu-runtime/node.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    manager = WorkloadManager(config)
    socket_path = Path(str(config["socket_path"]))
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if socket_path.exists():
        _unlink_socket(str(socket_path))
    os.umask(0o077)
    server = AgentServer(str(socket_path), manager)
    socket_path.chmod(0o600)

    def stop_server(_signum: int, _frame: object) -> None:
        manager.stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    LOG.info("agent_ready node=%s socket=%s", manager.node_name, socket_path)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        socket_path.unlink(missing_ok=True)
        manager.close()


if __name__ == "__main__":
    main()
