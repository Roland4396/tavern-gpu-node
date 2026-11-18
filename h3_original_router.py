"""Opt-in execution routing behind ORIGINAL Comfy node classes.

No graph conversion, no replacement public node names/schemas, no local GPU
fallback for a remote H3 value. Installation requires a complete backend; this
module on its own is not a functioning SGLang backend and is not auto-enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
from typing import Any, Protocol


@dataclass(frozen=True)
class H3RemoteValue:
    """Opaque backend object, not an MP4 disguised as a tensor/latent."""
    session: str
    object_id: str
    kind: str


OUTPUT_TYPES = {
    "UNETLoader": ("MODEL",), "CLIPLoader": ("CLIP",), "VAELoader": ("VAE",),
    "LoraLoaderModelOnly": ("MODEL",),
    "MiniMaxH3MemoryEfficientSageAttentionPatch": ("MODEL",),
    "MiniMaxH3AudioConditioningT8": ("CONDITIONING", "LATENT", "AUDIO", "STRING", "STRING", "STRING"),
    "MiniMaxH3DualClockSamplerT8": ("MODEL", "SAMPLER", "SIGMAS"),
    "BasicGuider": ("GUIDER",), "BasicScheduler": ("SIGMAS",),
    "SamplerCustomAdvanced": ("LATENT", "LATENT"),
    "MiniMaxH3AVDecodeT8": ("IMAGE", "AUDIO", "LATENT", "LATENT"),
    "VAEEncode": ("LATENT",), "VAEDecode": ("IMAGE",),
    "VAEEncodeAudio": ("LATENT",), "VAEDecodeAudio": ("AUDIO",),
    "PT_H3ConcatAVLatent": ("LATENT",),
    "ImageResizeKJv2": ("IMAGE", "INT", "INT", "MASK"),
}
LOADER_FIELDS = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name"}


class OriginalBackend(Protocol):
    capabilities: dict[str, tuple[str, ...]]

    def owns_asset(self, node_type: str, name: str) -> bool: ...
    def invoke(self, node_type: str, inputs: dict[str, Any]) -> tuple: ...


def has_remote(value):
    if isinstance(value, H3RemoteValue):
        return True
    if isinstance(value, dict):
        return any(has_remote(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(has_remote(v) for v in value)
    return False


def should_route(node_type, inputs, backend):
    field = LOADER_FIELDS.get(node_type)
    if field:
        return backend.owns_asset(node_type, inputs.get(field, ""))
    # VFX is model computation even when fed a materialized CPU IMAGE. Scope
    # this routing to the specific requested algorithm, not all image resizing.
    if node_type == "ImageResizeKJv2":
        return inputs.get("upscale_method") == "nvidia_rtx_vsr" or has_remote(inputs)
    return has_remote(inputs)


def install_routes(registry, backend, *, node_output_factory=None):
    """Atomically install scoped wrappers; caller installs AFTER node discovery.

    Classes retain their original INPUT_TYPES/define_schema, category, sockets,
    output types and public registry keys. Non-H3 inputs call original methods.
    A missing operation aborts BEFORE modifying the registry; never silently
    starts Comfy model inference as a fallback. Native output/ZIP/save nodes are
    intentionally untouched and receive real CPU IMAGE/AUDIO tensors.
    """
    missing = [key for key in OUTPUT_TYPES if key not in registry]
    unsupported = [key for key, outputs in OUTPUT_TYPES.items()
                   if tuple(backend.capabilities.get(key, ())) != outputs]
    if missing or unsupported:
        raise RuntimeError(f"Original H3 routing incomplete: missing nodes={missing}, missing backend operations={unsupported}")
    replacements = {}
    for key, output_types in OUTPUT_TYPES.items():
        original = registry[key]
        # Detect v3 from its schema API, not FUNCTION: v3 aliases can name a
        # normalized dispatch method rather than the original execute method.
        v3 = hasattr(original, "define_schema")
        method_name = "execute" if v3 else original.FUNCTION
        raw = inspect.getattr_static(original, method_name)
        fn = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
        sig = inspect.signature(fn)
        implicit = not isinstance(raw, staticmethod)
        schema = original.GET_NODE_INFO_V1() if v3 else None
        actual_outputs = tuple(schema["output"]) if v3 else tuple(original.RETURN_TYPES)
        if actual_outputs != output_types:
            raise RuntimeError(f"Original output contract changed for {key}: {actual_outputs}")

        def make_dispatch(key, original, fn, sig, implicit, v3, output_types):
            @functools.wraps(fn)
            def dispatch(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                inputs = dict(bound.arguments)
                if implicit:
                    inputs.pop(next(iter(sig.parameters)))
                # Flatten a legacy **kwargs parameter without altering named inputs.
                for name, parameter in sig.parameters.items():
                    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                        inputs.update(inputs.pop(name, {}))
                if not should_route(key, inputs, backend):
                    return fn(*args, **kwargs)
                result = backend.invoke(key, inputs)
                if not isinstance(result, tuple) or len(result) != len(output_types):
                    raise RuntimeError(f"Backend output arity mismatch for {key}")
                if v3:
                    if node_output_factory is None:
                        from comfy_api.latest import io
                        return io.NodeOutput(*result)
                    return node_output_factory(*result)
                return result
            return dispatch

        wrapper = make_dispatch(key, original, fn, sig, implicit, v3, output_types)
        descriptor = classmethod(wrapper) if isinstance(raw, classmethod) else staticmethod(wrapper) if isinstance(raw, staticmethod) else wrapper
        replacements[key] = type(original.__name__, (original,), {
            "__module__": original.__module__, method_name: descriptor,
            "_h3_original_class": original,
        })
        if key in LOADER_FIELDS and hasattr(backend, "cache_fingerprint"):
            # A process-local worker object cannot survive its process. Only
            # routed H3 loaders gain this dependency; native image loaders keep
            # their existing fingerprint behavior.
            fingerprint_name = "fingerprint_inputs" if v3 else "IS_CHANGED"
            previous = getattr(original, fingerprint_name, None)
            def make_fingerprint(key, previous):
                def fingerprint(cls, **inputs):
                    if backend.owns_asset(key, inputs.get(LOADER_FIELDS[key], "")):
                        return ("h3-original-worker", backend.cache_fingerprint())
                    return previous(**inputs) if previous else False
                return classmethod(fingerprint)
            setattr(replacements[key], fingerprint_name, make_fingerprint(key, previous))
    prior = {key: registry[key] for key in replacements}
    registry.update(replacements)
    return prior


def restore_routes(registry, originals):
    for key, original in originals.items():
        current = registry.get(key)
        if getattr(current, "_h3_original_class", None) is not original:
            raise RuntimeError(f"Cannot overwrite a later registration of {key}")
    registry.update(originals)
