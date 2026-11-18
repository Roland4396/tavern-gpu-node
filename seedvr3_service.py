#!/home/worker/.venvs/deep-learning/bin/python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from seedvr3_benchmark import (
    OVERLAP_FRAMES,
    Worker,
    chunk_plan,
    execute_prompt,
    merge_outputs,
    physical_gpus,
    split_source,
    start_worker,
    terminate_workers,
    validate_worker_nodes,
    wait_ready,
)


LOG = logging.getLogger("seedvr3-service")
DEFAULT_SOCKET = Path("/home/worker/.local/state/gpu-runtime/seedvr3.sock")
DEFAULT_ROOT = Path("/home/worker/.cache/gpu-runtime/seedvr3-service")
DEFAULT_ALLOWED_INPUT_ROOT = Path("/mnt/data/worker/.runtime/state/graph/v1/input")
DEFAULT_JOB_TTL_SECONDS = 3600
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


@dataclass
class VideoMetadata:
    frames: int
    fps: float
    width: int
    height: int
    has_audio: bool


@dataclass
class UpscaleJob:
    job_id: str
    source: Path
    overlap_frames: int
    seed: int
    scale: float
    color_correction_method: str
    status: str = "queued"
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    output: Path | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.job_id,
            "object": "video.upscale",
            "status": self.status,
            "progress": self.progress,
            "created_at": int(self.created_at),
            "started_at": int(self.started_at) if self.started_at else None,
            "completed_at": int(self.completed_at) if self.completed_at else None,
            "error": {"message": self.error} if self.error else None,
        }
        if self.report is not None:
            result["report"] = self.report
        return result


def probe_video(path: Path) -> VideoMetadata:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("input does not contain a video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or float(rate) <= 0:
            raise ValueError("input video frame rate is unavailable")
        frames = int(stream.frames or 0)
        width = int(stream.width or 0)
        height = int(stream.height or 0)
        has_audio = bool(container.streams.audio)
        if not frames:
            frames = sum(1 for _frame in container.decode(stream))
    if frames <= 0 or width <= 0 or height <= 0:
        raise ValueError("input video metadata is incomplete")
    return VideoMetadata(frames, float(rate), width, height, has_audio)


class SeedVR3Service:
    def __init__(
        self,
        root: Path,
        allowed_input_root: Path,
        job_ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS,
    ) -> None:
        self.root = root
        self.worker_root = root / "workers"
        self.job_root = root / "jobs"
        self.allowed_input_root = allowed_input_root.resolve()
        self.job_ttl_seconds = job_ttl_seconds
        self.workers: list[Worker] = []
        self.jobs: dict[str, UpscaleJob] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.runner: asyncio.Task[None] | None = None
        self.cleaner: asyncio.Task[None] | None = None
        self.active_job_id: str | None = None
        self.stopping = False

    async def start(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.job_root.exists():
            shutil.rmtree(self.job_root)
        self.job_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        await self._start_workers()
        self.runner = asyncio.create_task(self._run_queue(), name="seedvr3-job-runner")
        self.cleaner = asyncio.create_task(self._cleanup_loop(), name="seedvr3-job-cleaner")

    async def close(self) -> None:
        self.stopping = True
        tasks = [task for task in (self.runner, self.cleaner) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.runner = None
        self.cleaner = None
        await asyncio.to_thread(terminate_workers, self.workers)
        self.workers.clear()

    async def _start_workers(self) -> None:
        gpus = physical_gpus()
        if self.worker_root.exists():
            shutil.rmtree(self.worker_root)
        self.workers = [
            start_worker(index, gpu, self.worker_root)
            for index, gpu in enumerate(gpus)
        ]
        try:
            await asyncio.gather(*(wait_ready(worker) for worker in self.workers))
            await asyncio.gather(*(validate_worker_nodes(worker) for worker in self.workers))
        except BaseException:
            await asyncio.to_thread(terminate_workers, self.workers)
            self.workers.clear()
            raise
        LOG.info("workers_ready gpus=%s", gpus)

    async def _ensure_workers(self) -> None:
        if len(self.workers) == 3 and all(worker.process.poll() is None for worker in self.workers):
            return
        if self.workers:
            await asyncio.to_thread(terminate_workers, self.workers)
            self.workers.clear()
        await self._start_workers()

    def validate_source(self, raw: Any) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("source_path is required")
        source = Path(raw).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("source_path is not a file")
        if source != self.allowed_input_root and self.allowed_input_root not in source.parents:
            raise ValueError(f"source_path must be below {self.allowed_input_root}")
        return source

    def submit(self, payload: dict[str, Any]) -> UpscaleJob:
        source = self.validate_source(payload.get("source_path"))
        overlap = int(payload.get("overlap_frames", OVERLAP_FRAMES))
        seed = int(payload.get("seed", 0))
        scale = float(payload.get("scale", 2.0))
        color = str(payload.get("color_correction_method", "none"))
        if not 1 <= overlap <= 32:
            raise ValueError("overlap_frames must be in [1, 32]")
        if not 0 <= seed <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError("seed is out of range")
        if scale != 2.0:
            raise ValueError("this SeedVR2 deployment supports scale=2.0")
        if color != "none":
            raise ValueError("this validated workflow supports color_correction_method=none")
        job_id = str(uuid.uuid4())
        job = UpscaleJob(job_id, source, overlap, seed, scale, color)
        self.jobs[job_id] = job
        self.queue.put_nowait(job_id)
        LOG.info("job_queued id=%s source=%s queue=%d", job_id, source, self.queue.qsize())
        return job

    async def _run_queue(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                self.queue.task_done()
                continue
            self.active_job_id = job_id
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.error = "service stopped"
                job.completed_at = time.time()
                raise
            except BaseException as exc:
                LOG.exception("job_failed id=%s", job_id)
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.completed_at = time.time()
            finally:
                self.active_job_id = None
                job.source.unlink(missing_ok=True)
                self.queue.task_done()

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(min(300, max(30, self.job_ttl_seconds // 4)))
            cutoff = time.time() - self.job_ttl_seconds
            expired = [
                job_id
                for job_id, job in self.jobs.items()
                if job.status in TERMINAL_STATES
                and (job.completed_at or job.created_at) <= cutoff
            ]
            for job_id in expired:
                self.delete(job_id)

    async def _run_job(self, job: UpscaleJob) -> None:
        await self._ensure_workers()
        job.status = "processing"
        job.started_at = time.time()
        job.progress = 1
        started = time.monotonic()
        metadata = await asyncio.to_thread(probe_video, job.source)
        chunks = chunk_plan(metadata.frames, job.overlap_frames)
        root = self.job_root / job.job_id
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        inputs = [
            worker.root / "input" / f"{job.job_id}-chunk-{worker.index}.mkv"
            for worker in self.workers
        ]
        copied_outputs: list[Path] = []
        try:
            await asyncio.to_thread(split_source, job.source, chunks, inputs, metadata.fps)
            job.progress = 5
            inference_started = time.monotonic()
            results = await asyncio.gather(*(
                execute_prompt(
                    worker,
                    input_path.name,
                    seed=job.seed + worker.index,
                    scale=job.scale,
                    color_correction_method=job.color_correction_method,
                )
                for worker, input_path in zip(self.workers, inputs, strict=True)
            ))
            inference_wall = time.monotonic() - inference_started
            job.progress = 90
            for result in results:
                original = Path(result.pop("output_path"))
                persisted = root / f"chunk-{result['worker']}.mp4"
                await asyncio.to_thread(shutil.copy2, original, persisted)
                original.unlink(missing_ok=True)
                copied_outputs.append(persisted)
                result["output"] = persisted.name
            output = root / "output.mp4"
            merge_wall = await asyncio.to_thread(
                merge_outputs,
                job.source,
                copied_outputs,
                output,
                chunks,
                metadata.fps,
            )
            for path in copied_outputs:
                path.unlink(missing_ok=True)
            job.output = output
            job.report = {
                "source_frames": metadata.frames,
                "fps": metadata.fps,
                "source_size": f"{metadata.width}x{metadata.height}",
                "output_size": f"{metadata.width * 2}x{metadata.height * 2}",
                "audio_preserved": metadata.has_audio,
                "chunks": [chunk.__dict__ | {"frames": chunk.frames} for chunk in chunks],
                "overlap_frames": job.overlap_frames,
                "workers": results,
                "inference_wall_s": inference_wall,
                "merge_wall_s": merge_wall,
                "total_wall_s": time.monotonic() - started,
            }
            job.status = "completed"
            job.progress = 100
            job.completed_at = time.time()
            LOG.info(
                "job_completed id=%s frames=%d inference_s=%.3f total_s=%.3f",
                job.job_id,
                metadata.frames,
                inference_wall,
                job.report["total_wall_s"],
            )
        finally:
            for path in inputs:
                path.unlink(missing_ok=True)
            if job.status != "completed":
                shutil.rmtree(root, ignore_errors=True)

    def delete(self, job_id: str) -> tuple[int, dict[str, Any]]:
        job = self.jobs.get(job_id)
        if job is None:
            return 404, {"error": "job not found"}
        if job.status not in TERMINAL_STATES:
            return 409, {"error": "active jobs cannot be deleted"}
        self.jobs.pop(job_id, None)
        shutil.rmtree(self.job_root / job_id, ignore_errors=True)
        return 200, {"id": job_id, "status": "deleted"}

    async def health(self, _request: web.Request) -> web.Response:
        workers_ready = len(self.workers) == 3 and all(
            worker.process.poll() is None and worker.socket.is_socket()
            for worker in self.workers
        )
        return web.json_response({
            "status": "ok" if workers_ready else "degraded",
            "workers_ready": sum(worker.process.poll() is None for worker in self.workers),
            "active_jobs": 1 if self.active_job_id else 0,
            "queued_jobs": self.queue.qsize(),
            "active_job_id": self.active_job_id,
        }, status=200 if workers_ready else 503)

    async def submit_http(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            job = self.submit(payload)
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(job.public(), status=202)

    async def status_http(self, request: web.Request) -> web.Response:
        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return web.json_response({"error": "job not found"}, status=404)
        return web.json_response(job.public())

    async def content_http(self, request: web.Request) -> web.StreamResponse:
        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return web.json_response({"error": "job not found"}, status=404)
        if job.status != "completed" or job.output is None or not job.output.is_file():
            return web.json_response({"error": f"job is {job.status}"}, status=409)
        return web.FileResponse(job.output, headers={"Content-Type": "video/mp4"})

    async def delete_http(self, request: web.Request) -> web.Response:
        status, body = self.delete(request.match_info["job_id"])
        return web.json_response(body, status=status)


def build_app(service: SeedVR3Service) -> web.Application:
    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/health", service.health)
    app.router.add_post("/v1/upscale", service.submit_http)
    app.router.add_get("/v1/upscale/{job_id}", service.status_http)
    app.router.add_get("/v1/upscale/{job_id}/content", service.content_http)
    app.router.add_delete("/v1/upscale/{job_id}", service.delete_http)
    app.on_startup.append(lambda _app: service.start())
    app.on_cleanup.append(lambda _app: service.close())
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent three-GPU SeedVR2 service")
    parser.add_argument("--socket", type=Path, default=Path(os.environ.get("SEEDVR3_SOCKET", DEFAULT_SOCKET)))
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("SEEDVR3_ROOT", DEFAULT_ROOT)))
    parser.add_argument(
        "--allowed-input-root",
        type=Path,
        default=Path(os.environ.get("SEEDVR3_ALLOWED_INPUT_ROOT", DEFAULT_ALLOWED_INPUT_ROOT)),
    )
    parser.add_argument(
        "--job-ttl-seconds",
        type=int,
        default=int(os.environ.get("SEEDVR3_JOB_TTL_SECONDS", DEFAULT_JOB_TTL_SECONDS)),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    service = SeedVR3Service(args.root, args.allowed_input_root, args.job_ttl_seconds)
    web.run_app(
        build_app(service),
        path=str(args.socket),
        access_log=None,
        print=None,
        shutdown_timeout=30,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raise SystemExit(main())
