# SPDX-License-Identifier: GPL-3.0-or-later
"""Private, same-host original-node IPC. Never encodes images as video.

Only small descriptors cross SGLang's existing scheduler socket. Raw tensors
use a per-call file shared by the CPU Graph and all worker ranks. Model objects
stay resident on the workers. This module does not start or activate services.
"""
from dataclasses import dataclass
from pathlib import Path
import io
import os
import pickle
import re
import sys
import threading
import time
import uuid

from h3_original_router import H3RemoteValue, OUTPUT_TYPES


@dataclass(frozen=True)
class OriginalNodeRequest:
    session: str
    call: str
    action: str
    node_type: str = ""
    payload: object = None


@dataclass
class OriginalNodeReply:
    output: object = None
    error: str | None = None


def decode_scheduler_reply(data):
    """Read the scheduler envelope without importing its inference environment.

    Original-node replies contain scalar descriptors only; tensors are loaded
    separately by TensorCodec. Importing SGLang's OutputBatch in Graph would
    pull in SGLang's Torch/CUDA extension dependencies across Python runtimes.
    """
    class ReplyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in {
                ("sglang.multimodal_gen.runtime.pipelines_core.schedule_batch", "OutputBatch"),
                (__name__, "OriginalNodeReply"),
            }:
                return OriginalNodeReply
            raise pickle.UnpicklingError(f"Unexpected original-node reply type: {module}.{name}")
    return ReplyUnpickler(io.BytesIO(data)).load()


def call_directory(root, session, call):
    if not all(re.fullmatch(r"[0-9a-f]{32}", s) for s in (session, call)):
        raise ValueError("Invalid original-node session/call identifier")
    return Path(root) / session / call


def is_native_definition(value, native):
    """Comfy discovery loads extras under file-path module names as well.

    Accept the exact class definition from that same source file, not arbitrary
    duck-typed objects or a different custom node with a coincidentally matching
    class name. The transported noise remains its original algorithm and seed.
    """
    actual = type(value)
    if actual is native:
        return True
    if actual.__qualname__ != native.__qualname__:
        return False
    left = getattr(sys.modules.get(actual.__module__), "__file__", None)
    right = getattr(sys.modules.get(native.__module__), "__file__", None)
    return bool(left and right and Path(left).resolve() == Path(right).resolve())


class TensorCodec:
    """Lossless CPU tensors plus explicit native Comfy value types.

    No unpickling arbitrary node/model instances across Torch environments.
    torch.load is restricted to a tensor list, with weights_only=True.
    """
    def __init__(self, directory):
        self.directory = Path(directory)

    def pack(self, value, filename):
        import torch
        from comfy.nested_tensor import NestedTensor
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
        import comfy.samplers
        tensors = []

        def visit(v):
            if v is None or type(v) in (str, int, float, bool):
                return ("scalar", v)
            if isinstance(v, H3RemoteValue):
                return ("handle", v.session, v.object_id, v.kind)
            if isinstance(v, NestedTensor):
                return ("nested", [visit(t) for t in v.unbind()])
            if isinstance(v, torch.Tensor):
                index = len(tensors)
                tensors.append(v.detach().cpu())
                return ("tensor", index)
            if type(v) is dict:
                return ("dict", [(visit(k), visit(x)) for k, x in v.items()])
            if type(v) in (tuple, list):
                return ("tuple" if type(v) is tuple else "list", [visit(x) for x in v])
            if is_native_definition(v, Noise_RandomNoise):
                return ("noise_random", v.seed)
            if is_native_definition(v, Noise_EmptyNoise):
                return ("noise_empty",)
            if type(v) is comfy.samplers.KSAMPLER:
                for name in comfy.samplers.SAMPLER_NAMES:
                    native = comfy.samplers.sampler_object(name)
                    if native.sampler_function is v.sampler_function:
                        return ("sampler", name, visit(v.extra_options), visit(v.inpaint_options))
            raise TypeError(f"Unsupported original-node IPC value: {type(v).__module__}.{type(v).__name__}")

        tree = visit(value)
        if tensors:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.directory / filename
            # Private call directory, no input-derived filename or public URL.
            with path.open("xb") as stream:
                os.chmod(path, 0o600)
                torch.save(tensors, stream)
        return {"tree": tree, "has_tensors": bool(tensors)}

    def unpack(self, payload, filename, *, resolve_handle=None):
        import torch
        from comfy.nested_tensor import NestedTensor
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
        import comfy.samplers
        tensors = (torch.load(self.directory / filename, map_location="cpu", weights_only=True, mmap=True)
                   if payload["has_tensors"] else [])

        def visit(v):
            tag = v[0]
            if tag == "scalar": return v[1]
            if tag == "handle":
                handle = H3RemoteValue(*v[1:])
                return resolve_handle(handle) if resolve_handle else handle
            if tag == "tensor": return tensors[v[1]]
            if tag == "nested": return NestedTensor([visit(t) for t in v[1]])
            if tag == "dict": return {visit(k): visit(x) for k, x in v[1]}
            if tag == "list": return [visit(x) for x in v[1]]
            if tag == "tuple": return tuple(visit(x) for x in v[1])
            if tag == "noise_random": return Noise_RandomNoise(v[1])
            if tag == "noise_empty": return Noise_EmptyNoise()
            if tag == "sampler":
                sampler = comfy.samplers.sampler_object(v[1])
                sampler.extra_options, sampler.inpaint_options = visit(v[2]), visit(v[3])
                return sampler
            raise ValueError(f"Unknown original-node wire type: {tag}")
        return visit(payload["tree"])


class OriginalObjectStore:
    # Conditioning can contain native PackedLayout/control objects. It never
    # needs materializing on Graph: its original consumer is BasicGuider.
    opaque_kinds = frozenset({"MODEL", "CLIP", "VAE", "GUIDER", "SAMPLER", "CONDITIONING"})

    def __init__(self, session):
        self.session, self.objects = session, {}

    def resolve(self, handle):
        if handle.session != self.session:
            raise ValueError("Original-node handle belongs to another worker session")
        value, kind = self.objects[handle.object_id]
        if kind != handle.kind:
            raise ValueError("Original-node handle type mismatch")
        return value

    def export(self, call, kinds, values):
        if len(values) != len(kinds):
            raise ValueError("Original-node output arity mismatch")
        result = []
        for slot, (kind, value) in enumerate(zip(kinds, values)):
            if kind in self.opaque_kinds:
                key = f"{call}:{slot}"  # identical across every rank
                self.objects[key] = (value, kind)
                result.append(H3RemoteValue(self.session, key, kind))
            else:
                result.append(value)
        return tuple(result)


class ZmqOriginalTransport:
    """Reuse the SGLang scheduler's private IPC endpoint, not a new HTTP API.

    A timeout/cancel never retries inference. On cancellation the client waits
    for the ranks to leave the operation before removing shared input files.
    """
    def __init__(self, endpoint, root, *, timeout_seconds=1800, poll_hook=None):
        if not endpoint.startswith("ipc://"):
            raise ValueError("Original-node transport requires a private Unix IPC endpoint")
        self.endpoint, self.root = endpoint, Path(root)
        self.timeout_seconds, self.poll_hook = timeout_seconds, poll_hook

    def __call__(self, request):
        import zmq
        import pickle
        directory = call_directory(self.root, request.session, request.call)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        deadline = time.monotonic() + self.timeout_seconds
        cancelled = None
        with zmq.Context() as context, context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self.endpoint)
            socket.send(pickle.dumps(request))
            while not socket.poll(200):
                if self.poll_hook is not None and cancelled is None:
                    try: self.poll_hook(directory)
                    except Exception as exc:
                        cancelled = exc
                        (directory / "cancel").touch(mode=0o600)
                if time.monotonic() >= deadline:
                    (directory / "cancel").touch(mode=0o600)
                    raise TimeoutError("Original SGLang node did not finish; cancellation requested, inference NOT retried")
            reply = decode_scheduler_reply(socket.recv())
        (directory / "terminal").touch(mode=0o600)
        if cancelled is not None:
            raise cancelled
        if self.poll_hook is not None:
            # Short operations can finish before the first 200 ms poll. Still
            # forward their final native progress event before unlinking it.
            self.poll_hook(directory)
        if reply.error:
            raise RuntimeError(reply.error)
        return reply.output


class OriginalBackendClient:
    """Schema-preserving router backend; constructing it does not wake GPUs."""
    def __init__(self, transport, root, assets, *, activity_context=None):
        self.transport, self.root, self.assets = transport, Path(root), assets
        self.capabilities = dict(OUTPUT_TYPES)
        self.session = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.opened = False
        self.activity_context = activity_context
        self.worker_epoch = None

    def owns_asset(self, node_type, name):
        return name in self.assets.get(node_type, {})

    def cache_fingerprint(self):
        # Read-only status: cached UI/validation must never wake the worker.
        current = getattr(self.activity_context, "current_epoch", None)
        if current is None:
            return self.session
        epoch = current()
        # The first prompt can fingerprint before ensure cold-starts H3.
        # A later stopped process must not match that initial cold cache key.
        return epoch if epoch is not None else ("stopped", self.worker_epoch)

    def invoke(self, node_type, inputs):
        from contextlib import nullcontext
        import shutil
        with self.lock, (self.activity_context() if self.activity_context else nullcontext()) as epoch:
            if epoch is not None and epoch != self.worker_epoch:
                self.worker_epoch = epoch
                self.session = uuid.uuid4().hex
                self.opened = False
            if not self.opened:
                result = self.transport(OriginalNodeRequest(self.session, uuid.uuid4().hex, "open"))
                if result["assets"] != self.assets or result["capabilities"] != self.capabilities:
                    raise RuntimeError("Graph and original SGLang worker manifests differ")
                self.opened = True
            call = uuid.uuid4().hex
            directory = call_directory(self.root, self.session, call)
            codec = TensorCodec(directory)
            try:
                payload = codec.pack(inputs, "inputs.pt")
                reply = self.transport(OriginalNodeRequest(self.session, call, "invoke", node_type, payload))
                return codec.unpack(reply, "outputs.pt")
            finally:
                # A timed-out worker may still be reading these inputs. Its
                # process-level cleanup owns abandoned calls, not this client.
                if not (directory / "cancel").exists() or (directory / "terminal").exists():
                    shutil.rmtree(directory, ignore_errors=True)

    def close(self):
        with self.lock:
            if self.opened:
                self.transport(OriginalNodeRequest(self.session, uuid.uuid4().hex, "close"))
                self.opened = False
