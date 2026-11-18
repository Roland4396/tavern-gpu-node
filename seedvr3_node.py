from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import comfy.model_management
import folder_paths
import torch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP
from comfy_api.latest import Input, InputImpl, io


ORCHESTRATOR_SOCKET = os.environ.get(
    "GPU_ORCHESTRATOR_SOCKET",
    "/home/worker/.local/state/gpu-runtime/orchestrator.sock",
)
SHARED_INPUT_DIR = Path("/mnt/data/worker/.runtime/state/graph/v1/input/seedvr3")
SeedVR3Model = io.Custom("SEEDVR3_MODEL")
SeedVR3Settings = io.Custom("SEEDVR3_SETTINGS")
SeedVR3Job = io.Custom("SEEDVR3_JOB")
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "canceled"})


class SeedVR3LifecyclePatcher(ModelPatcher):
    """Represent the external three-GPU service in native Comfy model state."""

    def is_clone(self, other: object) -> bool:
        return isinstance(other, SeedVR3LifecyclePatcher)


@dataclass(frozen=True)
class SeedVR3ModelHandle:
    patcher: SeedVR3LifecyclePatcher


@dataclass(frozen=True)
class SeedVR3Options:
    overlap_frames: int
    color_correction_method: str


@dataclass(frozen=True)
class SeedVR3JobHandle:
    job_id: str


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def json_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    connection = UnixHTTPConnection(ORCHESTRATOR_SOCKET, timeout)
    body = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        result = {"raw": raw.decode("utf-8", errors="replace")}
    if not 200 <= response.status < 300:
        raise RuntimeError(f"SeedVR2 request failed ({response.status}): {result}")
    if not isinstance(result, dict):
        raise RuntimeError(f"SeedVR2 returned an invalid response: {result!r}")
    return result


def download_content(path: str, target: Path, timeout: float) -> None:
    connection = UnixHTTPConnection(ORCHESTRATOR_SOCKET, timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            detail = response.read(1024 * 1024).decode("utf-8", errors="replace")
            raise RuntimeError(f"SeedVR2 content download failed ({response.status}): {detail}")
        with target.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    finally:
        connection.close()


def make_seedvr3_model() -> SeedVR3ModelHandle:
    patcher = SeedVR3LifecyclePatcher(
        torch.nn.Module(),
        comfy.model_management.get_torch_device(),
        torch.device("cpu"),
        size=1,
    )
    loaded = False

    def on_load(
        _patcher: ModelPatcher,
        _device: torch.device,
        _lowvram_model_memory: int,
        _force_patch_weights: bool,
        _full_load: bool,
    ) -> None:
        nonlocal loaded
        if not loaded:
            json_request("POST", "/control/seedvr3/ensure", {}, timeout=300)
            loaded = True

    def on_detach(_patcher: ModelPatcher, _unpatch_all: bool) -> None:
        nonlocal loaded
        # The node agent owns idle release. A local Comfy model swap must not
        # evict three warm SeedVR2 workers between adjacent pipeline jobs.
        loaded = False

    patcher.add_callback(CallbacksMP.ON_LOAD, on_load)
    patcher.add_callback(CallbacksMP.ON_DETACH, on_detach)
    return SeedVR3ModelHandle(patcher)


def load_seedvr3_model(model: SeedVR3ModelHandle) -> None:
    if any(loaded is model.patcher for loaded in comfy.model_management.loaded_models()):
        return
    comfy.model_management.load_models_gpu([model.patcher])


class SeedVR3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR3Loader",
            display_name="Load SeedVR2 (3 GPU Persistent)",
            category="loaders/video",
            outputs=[SeedVR3Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(make_seedvr3_model())


class SeedVR3SettingsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR3Settings",
            display_name="SeedVR2 Upscale Settings",
            category="conditioning/video",
            inputs=[
                io.Combo.Input("scale", options=["2x"], default="2x"),
                io.Int.Input("overlap_frames", default=9, min=1, max=32, step=1),
                io.Combo.Input(
                    "color_correction_method",
                    options=["none"],
                    default="none",
                ),
            ],
            outputs=[SeedVR3Settings.Output(display_name="settings")],
        )

    @classmethod
    def execute(
        cls,
        scale: str,
        overlap_frames: int,
        color_correction_method: str,
    ) -> io.NodeOutput:
        if scale != "2x":
            raise ValueError("this SeedVR2 deployment supports 2x only")
        return io.NodeOutput(SeedVR3Options(int(overlap_frames), color_correction_method))


def submit_seedvr3_job(
    model: SeedVR3ModelHandle,
    video: Input.Video,
    settings: SeedVR3Options,
    seed: int,
) -> SeedVR3JobHandle:
    load_seedvr3_model(model)
    upload_id = str(uuid.uuid4())
    local_input_dir = Path(folder_paths.get_temp_directory()) / "seedvr3-input"
    local_input_dir.mkdir(parents=True, exist_ok=True)
    local_source = local_input_dir / f"{upload_id}.mp4"
    source = SHARED_INPUT_DIR / f"{upload_id}.mp4"
    submitted = False
    try:
        # PyAV cannot reliably finalize MP4 trailers directly on OSSFS.
        # Finalize on the local POSIX filesystem, then publish the complete file.
        video.save_to(str(local_source))
        if not local_source.is_file() or local_source.stat().st_size <= 0:
            raise RuntimeError("SeedVR2 input video serialization produced an empty file")
        SHARED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_source, source)
        with source.open("rb") as handle:
            os.fsync(handle.fileno())
        if source.stat().st_size != local_source.stat().st_size:
            raise RuntimeError("SeedVR2 shared input upload size mismatch")

        response = json_request(
            "POST",
            "/seedvr3/v1/upscale",
            {
                "source_path": str(source),
                "scale": 2.0,
                "overlap_frames": settings.overlap_frames,
                "color_correction_method": settings.color_correction_method,
                "seed": int(seed),
            },
            timeout=300,
        )
        job_id = str(response.get("id") or "")
        if not job_id:
            raise RuntimeError(f"SeedVR2 submission did not return a job id: {response}")
        submitted = True
        return SeedVR3JobHandle(job_id)
    finally:
        local_source.unlink(missing_ok=True)
        if not submitted:
            source.unlink(missing_ok=True)


def await_seedvr3_job(job: SeedVR3JobHandle) -> Input.Video:
    job_id = job.job_id
    if not job_id:
        raise ValueError("SeedVR2 job handle is empty")
    terminal = False
    output: Path | None = None
    try:
        deadline = time.monotonic() + 10800
        while time.monotonic() < deadline:
            status = json_request("GET", f"/seedvr3/v1/upscale/{job_id}", None, timeout=30)
            state = str(status.get("status") or "").lower()
            if state == "completed":
                terminal = True
                break
            if state in TERMINAL_STATES:
                terminal = True
                raise RuntimeError(f"SeedVR2 generation {state}: {status}")
            time.sleep(2)
        else:
            raise TimeoutError(f"SeedVR2 upscale timed out: {job_id}")

        output_dir = Path(folder_paths.get_temp_directory()) / "seedvr3"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{job_id}.mp4"
        download_content(
            f"/seedvr3/v1/upscale/{job_id}/content",
            output,
            timeout=1200,
        )
        return InputImpl.VideoFromFile(str(output))
    finally:
        if terminal:
            try:
                json_request("DELETE", f"/seedvr3/v1/upscale/{job_id}", None, timeout=30)
            except Exception:
                if output is not None:
                    output.unlink(missing_ok=True)
                raise


class SeedVR3Submit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR3Submit",
            display_name="SeedVR2 Submit (Pipeline)",
            category="sampling/video",
            inputs=[
                SeedVR3Model.Input("model"),
                io.Video.Input("video"),
                SeedVR3Settings.Input("settings"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[
                SeedVR3Job.Output("job"),
                io.Video.Output("video"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: SeedVR3ModelHandle,
        video: Input.Video,
        settings: SeedVR3Options,
        seed: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            submit_seedvr3_job(model, video, settings, seed),
            video,
        )


class SeedVR3Await(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR3Await",
            display_name="SeedVR2 Await (Pipeline)",
            category="sampling/video",
            inputs=[
                SeedVR3Job.Input("job"),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        job: SeedVR3JobHandle,
    ) -> io.NodeOutput:
        return io.NodeOutput(await_seedvr3_job(job))


class SeedVR3Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR3Sampler",
            display_name="SeedVR2 Video Upscale (3 GPU)",
            category="sampling/video",
            inputs=[
                SeedVR3Model.Input("model"),
                io.Video.Input("video"),
                SeedVR3Settings.Input("settings"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        model: SeedVR3ModelHandle,
        video: Input.Video,
        settings: SeedVR3Options,
        seed: int,
    ) -> io.NodeOutput:
        job = submit_seedvr3_job(model, video, settings, seed)
        return io.NodeOutput(await_seedvr3_job(job))
