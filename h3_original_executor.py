# SPDX-License-Identifier: GPL-3.0-or-later
"""Worker-local execution of original Comfy nodes using SGLang components.

This is not an HTTP/video workflow replacement. It must run on every SGLang
rank in the same operation order. RPC/object transport and production admission
are separate; importing this module never installs routes or starts a worker.
"""
from contextlib import contextmanager, nullcontext
import hashlib

from h3_original_router import LOADER_FIELDS, OUTPUT_TYPES


BRANCH_KEY = "h3_sglang_original_branch"


def branch_of(patcher):
    return patcher.model_options.get(BRANCH_KEY, {"loras": (), "kj_sage": False})


def clone_branch(patcher, *, lora=None, kj_sage=None):
    result = patcher.clone()
    old = branch_of(patcher)
    result.model_options[BRANCH_KEY] = {
        "loras": tuple(old["loras"]) + ((lora,) if lora is not None else ()),
        "kj_sage": old["kj_sage"] if kj_sage is None else kj_sage,
    }
    return result


class SGLangOriginalExecutor:
    """Execute with an already-loaded SGLang pipeline and original node registry.

    ``assets`` maps original loader names to explicit component identifiers,
    e.g. {"VAELoader": {"...fp16.safetensors": "video_vae"}}. It is supplied
    by the validated deployment manifest, never guessed from a filename.
    The registry must be the ORIGINAL classes, before installing proxy routes.
    """
    def __init__(self, pipeline, registry, assets, *, device, sequence_parallel_size,
                 kj_helper, lora_resolver, vsr_context=None, request_context=nullcontext):
        self.pipeline, self.registry, self.assets = pipeline, registry, assets
        self.device, self.sequence_parallel_size = device, sequence_parallel_size
        self.kj_helper, self.lora_resolver, self.vsr_context = kj_helper, lora_resolver, vsr_context
        self.request_context = request_context
        missing = set(OUTPUT_TYPES) - set(registry)
        if missing:
            raise ValueError(f"Original node registry incomplete: {sorted(missing)}")
        self.capabilities = dict(OUTPUT_TYPES)
        if vsr_context is None:
            self.capabilities.pop("ImageResizeKJv2")
        if kj_helper is None:
            self.capabilities.pop("MiniMaxH3MemoryEfficientSageAttentionPatch")
        self._facades = {}

    def owns_asset(self, node_type, name):
        return name in self.assets.get(node_type, {})

    @contextmanager
    def _component(self, name, *, phase=None, dtype=None):
        from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import ComponentUse
        manager = self.pipeline.component_residency_manager
        use = ComponentUse(stage_name="OriginalComfy", component_name=name, phase=phase,
                           target_dtype=dtype, allow_prefetch=False, memory_intensive=True)
        with manager.use_component(use, self.pipeline.get_module(name)) as component:
            # VAE attention needs the same SGLang forward context as Qwen/DiT.
            # Native-node dispatch does not enter Pipeline.forward for us.
            with self._forward_context():
                yield component

    @staticmethod
    def _forward_context():
        from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
        return set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=None)

    @contextmanager
    def _text_context(self):
        with self._component("text_encoder"):
            yield

    def _load(self, kind, inputs):
        import torch
        name = inputs[LOADER_FIELDS[kind]]
        component = self.assets[kind][name]
        if component in self._facades:
            return self._facades[component]
        if kind == "UNETLoader":
            if component != "transformer" or inputs.get("weight_dtype", "default") != "default":
                raise ValueError("Original BF16 worker requires its manifest transformer and default loader dtype")
            from h3_comfy_model_bridge import make_comfy_model_patcher
            result = make_comfy_model_patcher(self.pipeline.get_module(component), device=self.device,
                sequence_parallel_size=self.sequence_parallel_size, context_factory=self._forward_context)
        elif kind == "CLIPLoader":
            if component != "text_encoder":
                raise ValueError("CLIP manifest must select H3 text_encoder")
            from h3_comfy_clip_bridge import SGLangComfyH3Clip
            result = SGLangComfyH3Clip(self.pipeline.get_module(component),
                self.pipeline.get_module("tokenizer"), context_factory=self._text_context)
        else:
            if component not in ("video_vae", "audio_vae"):
                raise ValueError("VAE manifest must select an H3 VAE component")
            from h3_comfy_vae_bridge import make_comfy_vae_bridge
            dtype = torch.float16 if component == "video_vae" else torch.float32
            result = make_comfy_vae_bridge(self.pipeline.get_module(component), component.removesuffix("_vae"),
                device=self.device, dtype=dtype,
                execution_context=lambda direction: self._component(component, phase=direction, dtype=dtype))
        self._facades[component] = result
        return result

    def _native(self, node_type, inputs):
        cls = self.registry[node_type]
        if hasattr(cls, "define_schema"):
            return tuple(cls.execute(**inputs).result)
        return tuple(getattr(cls(), cls.FUNCTION)(**inputs))

    @contextmanager
    def _sampling_scope(self, patcher):
        from h3_comfy_attention_bridge import kj_sage_scope
        branch = branch_of(patcher)
        with self._component("transformer") as transformer:
            try:
                # Restore from immutable base snapshots, never subtract an
                # already-rounded LoRA delta. Stage 2 starts from #17's branch.
                if self.pipeline.lora_initialized:
                    self.pipeline.unmerge_lora_weights(target="transformer")
                if branch["loras"]:
                    paths, strengths = zip(*branch["loras"])
                    names = [hashlib.sha256(path.encode()).hexdigest() for path in paths]
                    self.pipeline.set_lora(list(names), lora_path=list(paths), target="transformer",
                        strength=list(strengths), merge_mode="merge", cache_merged=False)
                with kj_sage_scope(transformer, self.kj_helper) if branch["kj_sage"] else nullcontext():
                    yield
            finally:
                if self.pipeline.lora_initialized:
                    self.pipeline.unmerge_lora_weights(target="transformer")

    def invoke(self, node_type, inputs):
        with self.request_context():
            return self._invoke(node_type, inputs)

    def _invoke(self, node_type, inputs):
        if node_type not in self.capabilities:
            raise ValueError(f"Original worker cannot execute {node_type}")
        if node_type in LOADER_FIELDS:
            if not self.owns_asset(node_type, inputs[LOADER_FIELDS[node_type]]):
                raise ValueError("Asset is not in the worker's validated model manifest")
            return (self._load(node_type, inputs),)
        if node_type == "LoraLoaderModelOnly":
            strength = float(inputs["strength_model"])
            if strength == 0:
                return (inputs["model"],)
            path = str(self.lora_resolver(inputs["lora_name"]))
            return (clone_branch(inputs["model"], lora=(path, strength)),)
        if node_type == "MiniMaxH3MemoryEfficientSageAttentionPatch":
            return (clone_branch(inputs["model"], kj_sage=True),)
        if node_type == "SamplerCustomAdvanced":
            with self._sampling_scope(inputs["guider"].model_patcher):
                return self._native(node_type, inputs)
        if node_type == "ImageResizeKJv2":
            with self.vsr_context():
                return self._native(node_type, inputs)
        return self._native(node_type, inputs)
