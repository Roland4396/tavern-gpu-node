from __future__ import annotations

import http.client
import hashlib
import json
import logging
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.utils
from server import PromptServer
from comfy_execution.utils import get_executing_context
import folder_paths
import numpy as np
import torch
import torchaudio
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import CallbacksMP
from comfy_api.latest import ComfyExtension, Input, InputImpl, io
from PIL import Image
from typing_extensions import override


LOG = logging.getLogger(__name__)


ORCHESTRATOR_SOCKET = os.environ.get(
    "GPU_ORCHESTRATOR_SOCKET",
    "/home/worker/.local/state/gpu-runtime/orchestrator.sock",
)
SHARED_INPUT_DIR = Path("/mnt/data/worker/.runtime/state/h3/v1/input")
H3_SHARED_OUTPUT_DIR = os.environ.get(
    "H3_SHARED_OUTPUT_DIR",
    "/mnt/data/worker/.runtime/state/h3/v1/output",
)
H3_LORA_ROOT = Path(
    "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/loras"
)
H3_BACKEND_DEFAULT_LORA = "Backend default"
H3_DEFAULT_STEPS = 50
H3ConditionBundle = io.Custom("H3_CONDITION_BUNDLE")
H3FL2VAModel = io.Custom("H3_FL2VA_MODEL")
H3Ref2VAModel = io.Custom("H3_REF2VA_MODEL")
H3FL2VAConditioning = io.Custom("H3_FL2VA_CONDITIONING")
H3Ref2VAConditioning = io.Custom("H3_REF2VA_CONDITIONING")
H3_ASPECT_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "auto"]
H3_ACCELERATION_PROFILES = frozenset({
    "lossless",
    "rh_lightning",
    "cache_conservative",
    "cache_stride",
    "subblock_075",
    "subblock_080",
    "subblock_075_cache_conservative",
    "subblock_075_cache_stride",
    "subblock_080_cache_conservative",
    "subblock_080_cache_stride",
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class H3LifecyclePatcher(ModelPatcher):
    """Expose an external SGLang worker through ComfyUI's model lifecycle."""

    def is_clone(self, other: object) -> bool:
        return isinstance(other, H3LifecyclePatcher)


@dataclass(frozen=True)
class H3LoRAAdapter:
    name: str
    strength: float


@dataclass(frozen=True)
class H3ModelHandle:
    workload: str
    patcher: H3LifecyclePatcher
    lora_adapters: tuple[H3LoRAAdapter, ...] = ()
    # The Comfy graph and the external SGLang worker are separate processes.
    # Capture the deployment profile in the model handle so request shaping
    # follows the same profile that the worker was launched with.
    acceleration_profile: str = "subblock_075_cache_stride"


@dataclass(frozen=True)
class H3FL2VACondition:
    prompt: str
    aspect_ratio: str
    duration_seconds: float
    first_frame: Input.Image | None = None
    last_frame: Input.Image | None = None


@dataclass(frozen=True)
class H3Ref2VACondition:
    prompt: str
    aspect_ratio: str
    duration_seconds: float
    references: tuple[dict[str, Any], ...] = ()


def make_h3_model(workload: str) -> H3ModelHandle:
    if workload not in {"h3", "h3_ref"}:
        raise ValueError(f"unsupported H3 workload: {workload}")

    patcher = H3LifecyclePatcher(
        torch.nn.Module(),
        comfy.model_management.get_torch_device(),
        torch.device("cpu"),
        size=1,
    )
    def on_load(
        _patcher: ModelPatcher,
        _device: torch.device,
        _lowvram_model_memory: int,
        _force_patch_weights: bool,
        _full_load: bool,
    ) -> None:
        # The loader is intentionally CPU-only.  A real generation is queued
        # durably by run_video_job(), and the owning node-agent starts the GPU
        # worker in the background.  Model loading must never hold an FRP-backed
        # HTTP request open for the worker's cold-start duration.
        return None

    def on_detach(_patcher: ModelPatcher, _unpatch_all: bool) -> None:
        return None

    patcher.add_callback(CallbacksMP.ON_LOAD, on_load)
    patcher.add_callback(CallbacksMP.ON_DETACH, on_detach)
    acceleration_profile = os.environ.get(
        "H3_ACCELERATION_PROFILE", "subblock_075_cache_stride"
    ).strip().lower()
    if acceleration_profile not in H3_ACCELERATION_PROFILES:
        expected = ", ".join(sorted(H3_ACCELERATION_PROFILES))
        raise ValueError(
            f"unsupported H3_ACCELERATION_PROFILE: {acceleration_profile!r}; expected one of {expected}"
        )
    return H3ModelHandle(
        workload=workload,
        patcher=patcher,
        acceleration_profile=acceleration_profile,
    )


def load_h3_model(model: H3ModelHandle) -> None:
    if any(loaded is model.patcher for loaded in comfy.model_management.loaded_models()):
        return
    comfy.model_management.load_models_gpu([model.patcher])


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
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    connection = UnixHTTPConnection(ORCHESTRATOR_SOCKET, timeout)
    body = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    request_headers = {"Content-Type": "application/json"} if body is not None else {}
    request_headers.update(headers or {})
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        result = {"raw": raw.decode("utf-8", errors="replace")}
    if not 200 <= response.status < 300:
        raise RuntimeError(f"H3 request failed ({response.status}): {result}")
    if not isinstance(result, dict):
        raise RuntimeError(f"H3 returned an invalid response: {result!r}")
    return result


def copy_shared_content(
    status: dict[str, Any],
    job_id: str,
    target: Path,
    *,
    shared_root: Path | None = None,
) -> bool:
    root = (shared_root or Path(H3_SHARED_OUTPUT_DIR)).resolve()
    candidates = status.get("file_paths")
    raw_path = (
        candidates[0]
        if isinstance(candidates, list) and candidates
        else status.get("file_path")
    )
    if not isinstance(raw_path, str) or not raw_path:
        return False

    source = Path(raw_path)
    try:
        resolved = source.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return False
    if resolved.name != f"{job_id}.mp4" or not resolved.is_file():
        return False

    staging = target.with_name(f".{target.name}.part")
    staging.unlink(missing_ok=True)
    try:
        source_size = resolved.stat().st_size
        if source_size <= 0:
            raise RuntimeError(f"H3 shared output is empty: {resolved}")
        shutil.copyfile(resolved, staging)
        with staging.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        copied_size = staging.stat().st_size
        if copied_size != source_size:
            raise RuntimeError(
                "H3 shared output copy was truncated: "
                f"expected {source_size} bytes, copied {copied_size}"
            )
        expected_hash = status.get("sha256") or status.get("output_sha256")
        if expected_hash:
            copied_hash = _sha256_file(staging)
            if copied_hash != expected_hash:
                raise RuntimeError(
                    "H3 shared output checksum mismatch: "
                    f"expected {expected_hash}, copied {copied_hash}"
                )
        staging.replace(target)
        return True
    finally:
        staging.unlink(missing_ok=True)


def copy_shared_content_with_retries(
    status: dict[str, Any],
    job_id: str,
    target: Path,
    *,
    attempts: int = 45,
) -> None:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            if copy_shared_content(status, job_id, target):
                return
            last_error = RuntimeError("published H3 artifact is not visible yet")
        except (OSError, RuntimeError) as exc:
            last_error = exc
        target.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(min(2 + attempt // 10, 6))
    assert last_error is not None
    raise RuntimeError(
        f"H3 shared artifact remained unavailable after {attempts} attempts: {last_error}"
    ) from last_error


def publish_condition(local: Path, target: Path) -> Path:
    SHARED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.part")
    staging.unlink(missing_ok=True)
    try:
        if not local.is_file() or local.stat().st_size <= 0:
            raise RuntimeError(f"H3 condition staging produced an empty file: {local.name}")
        local_size = local.stat().st_size
        if target.is_file():
            # The path is operation-scoped and created by atomic rename.  Reuse
            # it on Graph replay because video/audio encoders may produce
            # byte-different containers for the same logical input.
            if target.stat().st_size > 0:
                return target
            target.unlink(missing_ok=True)
        local_hash = _sha256_file(local)
        shutil.copyfile(local, staging)
        with staging.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if staging.stat().st_size != local_size:
            raise RuntimeError(f"H3 condition upload size mismatch: {target.name}")
        if _sha256_file(staging) != local_hash:
            raise RuntimeError(f"H3 condition upload checksum mismatch: {target.name}")
        staging.replace(target)
        return target
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    finally:
        local.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)


def condition_staging_path(identifier: str, suffix: str) -> Path:
    staging = Path(folder_paths.get_temp_directory()) / "h3-condition-staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging / f"{identifier}{suffix}"


def save_condition(image: Input.Image, identifier: str) -> Path:
    local = condition_staging_path(identifier, ".png")
    target = SHARED_INPUT_DIR / f"{identifier}.png"
    pixels = image[0].detach().cpu().clamp(0, 1).numpy()
    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(local, format="PNG")
    return publish_condition(local, target)


def save_video_condition(video: Input.Video, identifier: str) -> Path:
    local = condition_staging_path(identifier, ".mp4")
    target = SHARED_INPUT_DIR / f"{identifier}.mp4"
    duration = float(video.get_duration())
    source = video
    if duration > 15.001:
        source = video.as_trimmed(
            start_time=max(0.0, duration - 15.0),
            duration=15.0,
            strict_duration=False,
        )
        if source is None:
            raise ValueError(f"unable to trim H3 reference video from {duration:.3f} seconds")
    source.save_to(str(local))
    return publish_condition(local, target)


def save_audio_condition(audio: Input.Audio, identifier: str) -> Path:
    local = condition_staging_path(identifier, ".wav")
    target = SHARED_INPUT_DIR / f"{identifier}.wav"
    waveform = audio["waveform"][0].detach().cpu()
    torchaudio.save(str(local), waveform, int(audio["sample_rate"]), format="wav")
    return publish_condition(local, target)


class H3ProgressReporter:
    """Forward real sampler counts through Comfy's standard progress hook."""

    def __init__(self, total: int, node_id: str, prompt_id: str, operation_id: str):
        self.total = max(1, total)
        self.step = 0
        context = get_executing_context()
        self.node_id = node_id or (context.node_id if context else "")
        self.prompt_id = prompt_id or (context.prompt_id if context else "")
        self.operation_id = operation_id
        self.last = None
        self.bar = comfy.utils.ProgressBar(self.total, node_id=self.node_id or None)
        self.update({"status": "submitting"})

    def update(self, status: dict[str, Any]) -> None:
        progress = status.get("sampling_progress") or {}
        step, total = progress.get("step"), progress.get("total")
        if type(step) is int and type(total) is int and 0 <= step <= total and total > 0:
            self.step, self.total = step, total
        state = str(status.get("status") or "submitting")
        phase = str(status.get("phase") or progress.get("phase") or state)
        key = (self.step, self.total, phase)
        if key == self.last:
            return
        self.last = key
        self.bar.update_absolute(self.step, self.total)
        LOG.info("H3 progress job=%s phase=%s step=%d/%d", self.operation_id, phase, self.step, self.total)
        if self.node_id and PromptServer.instance is not None:
            PromptServer.instance.send_sync("h3_progress", {
                "node": self.node_id, "prompt_id": self.prompt_id,
                "operation_id": self.operation_id, "phase": phase,
                "value": self.step, "max": self.total,
            }, PromptServer.instance.client_id)


def run_video_job(
    route: str,
    payload: dict[str, Any],
    *,
    operation_id: str,
    graph_prompt_id: str = "",
    graph_node_id: str = "",
) -> io.NodeOutput:
    job_id = operation_id
    prefix = route.removesuffix("/v1/videos")
    Path(H3_SHARED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    durable_payload = dict(payload)
    durable_payload["_h3_operation_id"] = operation_id
    progress = H3ProgressReporter(
        max(1, int(payload.get("num_inference_steps", 51)) - 1),
        graph_node_id, graph_prompt_id, operation_id,
    )
    if graph_prompt_id:
        durable_payload["_graph_prompt_id"] = graph_prompt_id
    if graph_node_id:
        durable_payload["_graph_node_id"] = graph_node_id

    deadline = time.monotonic() + 7200
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            submitted = json_request(
                "POST",
                route,
                durable_payload,
                timeout=30,
                headers={"Idempotency-Key": operation_id},
            )
            returned_id = str(submitted.get("id") or "")
            if returned_id != operation_id:
                raise RuntimeError(
                    f"H3 submission returned a mismatched job id: {submitted}"
                )
            progress.update(submitted)
            break
        except RuntimeError as exc:
            if "H3 request failed (400)" in str(exc) or "H3 request failed (409)" in str(exc):
                raise
            LOG.warning(
                "H3 durable submission transport failure job=%s attempt=%d: %s",
                operation_id,
                attempt,
                exc,
            )
        except (OSError, http.client.HTTPException) as exc:
            LOG.warning(
                "H3 durable submission transport failure job=%s attempt=%d: %s",
                operation_id,
                attempt,
                exc,
            )
        time.sleep(min(2**min(attempt, 5), 30))
    else:
        raise TimeoutError(f"H3 submission could not reach the durable queue: {job_id}")

    status: dict[str, Any] = {}
    transport_attempt = 0
    while time.monotonic() < deadline:
        try:
            status = json_request("GET", f"{prefix}/v1/videos/{job_id}", None, timeout=30)
            transport_attempt = 0
            state = str(status.get("status") or "").lower()
            progress.update({**status, "phase": "copying"} if state == "completed" else status)
            if state == "completed":
                break
            if state in {"failed", "cancelled", "canceled"}:
                raise RuntimeError(f"H3 generation {state}: {status}")
        except RuntimeError as exc:
            if "H3 generation failed" in str(exc) or "H3 generation cancelled" in str(exc) or "H3 generation canceled" in str(exc):
                raise
            transport_attempt += 1
            LOG.warning(
                "H3 status transport unavailable job=%s attempt=%d: %s",
                job_id,
                transport_attempt,
                exc,
            )
        except (OSError, http.client.HTTPException) as exc:
            transport_attempt += 1
            LOG.warning(
                "H3 status transport unavailable job=%s attempt=%d: %s",
                job_id,
                transport_attempt,
                exc,
            )
        time.sleep(min(2 + transport_attempt, 30))
    else:
        raise TimeoutError(f"H3 generation timed out: {job_id}")

    output_dir = Path(folder_paths.get_temp_directory()) / "h3-sglang"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{job_id}.mp4"
    if not status.get("file_path") and not status.get("file_paths"):
        published: dict[str, Any] = {}
        try:
            published = json_request(
                "POST",
                f"{prefix}/v1/videos/{job_id}/publish",
                {},
                timeout=600,
            )
        except RuntimeError as exc:
            LOG.warning(
                "H3 job %s could not be published through shared storage: %s",
                job_id,
                exc,
            )
        status.update(published)
    progress.update({**status, "phase": "copying"})
    copy_shared_content_with_retries(status, job_id, output)
    progress.update({**status, "phase": "completed", "sampling_progress": {
        "step": progress.total, "total": progress.total,
    }})
    return io.NodeOutput(InputImpl.VideoFromFile(str(output)))


def _prompt(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("MiniMax H3 prompt must not be empty")
    return value


def _operation_id(value: str) -> str:
    value = str(value or "").strip().lower()
    if not value:
        return str(uuid.uuid4())
    try:
        normalized = str(uuid.UUID(value))
    except ValueError as exc:
        raise ValueError("H3 operation id must be a canonical UUID") from exc
    if normalized != value:
        raise ValueError("H3 operation id must be a canonical UUID")
    return normalized


def _h3_lora_options(root: Path = H3_LORA_ROOT) -> list[str]:
    try:
        names = sorted(
            {
                path.name
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() == ".safetensors"
            },
            key=str.casefold,
        )
    except OSError:
        names = []
    return [H3_BACKEND_DEFAULT_LORA, *names]


def _apply_lora(
    model: H3ModelHandle,
    strength: float,
    lora_name: str = H3_BACKEND_DEFAULT_LORA,
) -> H3ModelHandle:
    strength = float(strength)
    lora_name = str(lora_name)
    adapters = tuple(item for item in model.lora_adapters if item.name != lora_name)
    if strength > 0:
        adapters += (H3LoRAAdapter(name=lora_name, strength=strength),)
    return replace(model, lora_adapters=adapters)


def _lora_payload(model: H3ModelHandle) -> dict[str, Any]:
    adapters = [
        {"lora_name": adapter.name, "strength": adapter.strength}
        for adapter in model.lora_adapters
    ]
    if not adapters:
        return {"enabled": False}
    if len(adapters) == 1:
        # Preserve compatibility with orchestrators deployed before multi-LoRA.
        return {"enabled": True, **adapters[0]}
    return {"enabled": True, "adapters": adapters}


def _scheduler_grid_points(num_transformer_evaluations: int) -> int:
    # MiniMaxH3Scheduler includes terminal sigma zero in this count.
    return int(num_transformer_evaluations) + 1


def _quality_for_current_deployment(quality: str) -> str:
    if quality in {"lossless", "high"}:
        # Legacy workflows may contain high, which requires strict 4xH200.
        return "lossless"
    raise ValueError(f"unsupported MiniMax H3 quality: {quality}")


def _quality_is_explicitly_supported(profile: str) -> bool:
    """Whether the request must carry a quality field.

    SGLang's native MiniMax-H3 stage treats an explicit ``quality`` field as a
    user override. In particular, ``quality=lossless`` disables the generic
    Cache-DiT path even when the worker environment enables it. Cache profiles
    therefore intentionally omit the field and let the process-wide profile
    select the path.
    """

    profile = str(profile).strip().lower()
    if profile not in H3_ACCELERATION_PROFILES:
        expected = ", ".join(sorted(H3_ACCELERATION_PROFILES))
        raise ValueError(
            f"unsupported H3_ACCELERATION_PROFILE: {profile!r}; expected one of {expected}"
        )
    return profile == "lossless"


def _payload(
    model: H3ModelHandle,
    prompt: str,
    task: str,
    conditions: list[dict[str, Any]],
    aspect_ratio: str,
    duration_seconds: float,
    steps: int,
    quality: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": prompt,
        "seconds": int(duration_seconds),
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": 768,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": float(duration_seconds),
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": _scheduler_grid_points(steps),
        "_h3_turbo_lora": _lora_payload(model),
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": int(seed),
    }
    # Validate legacy/UI quality even when a manually tuned cache profile
    # suppresses the request field. This keeps malformed requests fail-closed
    # while preserving the official Cache-DiT rule that omission, rather than
    # ``quality=lossless``, enables the process-wide cache policy.
    normalized_quality = _quality_for_current_deployment(quality)
    if _quality_is_explicitly_supported(model.acceleration_profile):
        payload["quality"] = normalized_quality
    return payload


class MiniMaxH3FL2VALoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FL2VALoader",
            display_name="Load MiniMax H3 FL2VA (SGLang)",
            category="model/loaders",
            description="Loads the FL2VA partition used by T2VA and exact first/last-frame generation.",
            outputs=[H3FL2VAModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(make_h3_model("h3"))


class MiniMaxH3Ref2VALoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Ref2VALoader",
            display_name="Load MiniMax H3 Ref2VA (SGLang)",
            category="model/loaders",
            description="Loads the Ref2VA partition used for semantic image, video, and audio references.",
            outputs=[H3Ref2VAModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(make_h3_model("h3_ref"))


class MiniMaxH3FL2VALoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FL2VALoRA",
            display_name="Apply MiniMax H3 LoRA (FL2VA)",
            category="model/loaders",
            description="Appends one LoRA to the model. Chain this node to combine any number of LoRAs.",
            inputs=[
                H3FL2VAModel.Input("model"),
                io.Float.Input("strength", default=0.0, min=0.0, max=2.0, step=0.05),
                io.Combo.Input(
                    "lora_name",
                    options=_h3_lora_options(),
                    default=H3_BACKEND_DEFAULT_LORA,
                ),
            ],
            outputs=[H3FL2VAModel.Output(display_name="model")],
        )

    @classmethod
    def execute(
        cls,
        model: H3ModelHandle,
        strength: float,
        lora_name: str = H3_BACKEND_DEFAULT_LORA,
    ) -> io.NodeOutput:
        return io.NodeOutput(_apply_lora(model, strength, lora_name))


class MiniMaxH3Ref2VALoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Ref2VALoRA",
            display_name="Apply MiniMax H3 LoRA (Ref2VA)",
            category="model/loaders",
            description="Appends one LoRA to the model. Chain this node to combine any number of LoRAs.",
            inputs=[
                H3Ref2VAModel.Input("model"),
                io.Float.Input("strength", default=0.0, min=0.0, max=2.0, step=0.05),
                io.Combo.Input(
                    "lora_name",
                    options=_h3_lora_options(),
                    default=H3_BACKEND_DEFAULT_LORA,
                ),
            ],
            outputs=[H3Ref2VAModel.Output(display_name="model")],
        )

    @classmethod
    def execute(
        cls,
        model: H3ModelHandle,
        strength: float,
        lora_name: str = H3_BACKEND_DEFAULT_LORA,
    ) -> io.NodeOutput:
        return io.NodeOutput(_apply_lora(model, strength, lora_name))


class MiniMaxH3FL2VAConditionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FL2VACondition",
            display_name="MiniMax H3 Text / First-Last Frame Conditioning",
            category="model/conditioning/minimax",
            description="Text-only T2VA, or FL2VA when an exact first frame, last frame, or both are connected.",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input("aspect_ratio", options=H3_ASPECT_RATIOS, default="16:9"),
                io.Float.Input("duration_seconds", default=5.0, min=4.0, max=15.0, step=1.0),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                H3ConditionBundle.Input("condition_bundle", optional=True),
            ],
            outputs=[H3FL2VAConditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        duration_seconds: float,
        first_frame: Input.Image | None = None,
        last_frame: Input.Image | None = None,
        condition_bundle: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
        bundle = condition_bundle or {}
        return io.NodeOutput(H3FL2VACondition(
            prompt=_prompt(prompt),
            aspect_ratio=aspect_ratio,
            duration_seconds=float(duration_seconds),
            first_frame=first_frame if first_frame is not None else bundle.get("first_frame"),
            last_frame=last_frame if last_frame is not None else bundle.get("last_frame"),
        ))


class MiniMaxH3Ref2VAConditionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        image_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9
        )
        video_template = io.Autogrow.TemplatePrefix(
            input=io.Video.Input("ref_video"), prefix="ref_video_", min=0, max=3
        )
        audio_template = io.Autogrow.TemplatePrefix(
            input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3
        )
        return io.Schema(
            node_id="MiniMaxH3Ref2VACondition",
            display_name="MiniMax H3 Reference Conditioning",
            category="model/conditioning/minimax",
            description="Semantic references for identity, style, motion, scene, or sound. Images are not endpoint frames.",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input("aspect_ratio", options=H3_ASPECT_RATIOS, default="16:9"),
                io.Float.Input("duration_seconds", default=5.0, min=4.0, max=15.0, step=1.0),
                H3ConditionBundle.Input("condition_bundle", optional=True),
                io.Autogrow.Input("ref_images", template=image_template, optional=True),
                io.Autogrow.Input("ref_videos", template=video_template, optional=True),
                io.Autogrow.Input("ref_audios", template=audio_template, optional=True),
            ],
            outputs=[H3Ref2VAConditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        duration_seconds: float,
        condition_bundle: dict[str, Any] | None = None,
        ref_images: io.Autogrow.Type | None = None,
        ref_videos: io.Autogrow.Type | None = None,
        ref_audios: io.Autogrow.Type | None = None,
    ) -> io.NodeOutput:
        references = list((condition_bundle or {}).get("references") or [])
        references.extend(
            {"type": "image", "data": value, "asset_id": name}
            for name, value in (ref_images or {}).items() if value is not None
        )
        references.extend(
            {"type": "video", "data": value, "asset_id": name}
            for name, value in (ref_videos or {}).items() if value is not None
        )
        references.extend(
            {"type": "audio", "data": value, "asset_id": name}
            for name, value in (ref_audios or {}).items() if value is not None
        )
        if not references:
            raise ValueError("Ref2VA requires at least one image, video, or audio reference")

        image_count = sum(item["type"] == "image" for item in references)
        video_count = sum(item["type"] in {"video", "video_audio"} for item in references)
        audio_count = sum(item["type"] in {"audio", "video_audio"} for item in references)
        if len(references) > 12 or image_count > 9 or video_count > 3 or audio_count > 3:
            raise ValueError("Ref2VA reference count exceeds H3 limits")
        return io.NodeOutput(H3Ref2VACondition(
            prompt=_prompt(prompt),
            aspect_ratio=aspect_ratio,
            duration_seconds=float(duration_seconds),
            references=tuple(references),
        ))


class MiniMaxH3FL2VASampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FL2VASampler",
            display_name="MiniMax H3 FL2VA Sampler (SGLang)",
            category="sampling/video",
            inputs=[
                H3FL2VAModel.Input("model"),
                H3FL2VAConditioning.Input("conditioning"),
                io.Int.Input(
                    "steps", default=H3_DEFAULT_STEPS, min=1, max=100, step=1
                ),
                io.Combo.Input("quality", options=["lossless"], default="lossless"),
                io.Int.Input("seed", default=1101, min=0, max=0x7FFFFFFFFFFFFFFF, control_after_generate=True),
                io.String.Input("_h3_operation_id", default="", optional=True),
                io.String.Input("_graph_prompt_id", default="", optional=True),
                io.String.Input("_graph_node_id", default="", optional=True),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        model: H3ModelHandle,
        conditioning: H3FL2VACondition,
        steps: int,
        quality: str,
        seed: int,
        _h3_operation_id: str = "",
        _graph_prompt_id: str = "",
        _graph_node_id: str = "",
    ) -> io.NodeOutput:
        if model.workload != "h3":
            raise ValueError("FL2VA sampler requires an FL2VA model")
        load_h3_model(model)
        operation_id = _operation_id(_h3_operation_id)
        conditions: list[dict[str, Any]] = []
        if conditioning.first_frame is not None:
            path = save_condition(conditioning.first_frame, f"{operation_id}-first")
            conditions.append({"type": "image", "uri": path.as_uri(), "role": "keyframe", "frame_index": 0})
        if conditioning.last_frame is not None:
            path = save_condition(conditioning.last_frame, f"{operation_id}-last")
            conditions.append({"type": "image", "uri": path.as_uri(), "role": "keyframe", "frame_index": -1})
        payload = _payload(
            model,
            conditioning.prompt,
            "fl2va" if conditions else "t2va",
            conditions,
            conditioning.aspect_ratio,
            conditioning.duration_seconds,
            steps,
            quality,
            seed,
        )
        return run_video_job(
            "/h3/v1/videos",
            payload,
            operation_id=operation_id,
            graph_prompt_id=str(_graph_prompt_id or ""),
            graph_node_id=str(_graph_node_id or ""),
        )


class MiniMaxH3Ref2VASampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Ref2VASampler",
            display_name="MiniMax H3 Ref2VA Sampler (SGLang)",
            category="sampling/video",
            inputs=[
                H3Ref2VAModel.Input("model"),
                H3Ref2VAConditioning.Input("conditioning"),
                io.Int.Input(
                    "steps", default=H3_DEFAULT_STEPS, min=1, max=100, step=1
                ),
                io.Combo.Input("quality", options=["lossless"], default="lossless"),
                io.Int.Input("seed", default=1101, min=0, max=0x7FFFFFFFFFFFFFFF, control_after_generate=True),
                io.String.Input("_h3_operation_id", default="", optional=True),
                io.String.Input("_graph_prompt_id", default="", optional=True),
                io.String.Input("_graph_node_id", default="", optional=True),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        model: H3ModelHandle,
        conditioning: H3Ref2VACondition,
        steps: int,
        quality: str,
        seed: int,
        _h3_operation_id: str = "",
        _graph_prompt_id: str = "",
        _graph_node_id: str = "",
    ) -> io.NodeOutput:
        if model.workload != "h3_ref":
            raise ValueError("Ref2VA sampler requires a Ref2VA model")
        load_h3_model(model)
        operation_id = _operation_id(_h3_operation_id)
        conditions: list[dict[str, Any]] = []
        for index, item in enumerate(conditioning.references, start=1):
            kind = item["type"]
            data = item["data"]
            if kind == "image":
                path = save_condition(data, f"{operation_id}-ref-image-{index}")
            elif kind in {"video", "video_audio"}:
                path = save_video_condition(data, f"{operation_id}-ref-video-{index}")
            elif kind == "audio":
                path = save_audio_condition(data, f"{operation_id}-ref-audio-{index}")
            else:
                raise ValueError(f"unsupported Ref2VA reference type: {kind}")
            condition = {"type": kind, "uri": path.as_uri(), "role": "reference"}
            if item.get("start_time_seconds") is not None:
                condition["start_time_seconds"] = float(item["start_time_seconds"])
            conditions.append(condition)
        payload = _payload(
            model,
            conditioning.prompt,
            "ref2va",
            conditions,
            conditioning.aspect_ratio,
            conditioning.duration_seconds,
            steps,
            quality,
            seed,
        )
        return run_video_job(
            "/h3-ref/v1/videos",
            payload,
            operation_id=operation_id,
            graph_prompt_id=str(_graph_prompt_id or ""),
            graph_node_id=str(_graph_node_id or ""),
        )


class MiniMaxH3SGLangExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MiniMaxH3FL2VALoader,
            MiniMaxH3Ref2VALoader,
            MiniMaxH3FL2VALoRA,
            MiniMaxH3Ref2VALoRA,
            MiniMaxH3FL2VAConditionNode,
            MiniMaxH3Ref2VAConditionNode,
            MiniMaxH3FL2VASampler,
            MiniMaxH3Ref2VASampler,
        ]


async def comfy_entrypoint() -> MiniMaxH3SGLangExtension:
    return MiniMaxH3SGLangExtension()
