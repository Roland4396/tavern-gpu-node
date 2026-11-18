# SPDX-License-Identifier: GPL-3.0-or-later
"""Native Comfy FLOW_AV/sampler/ModelPatcher facade, SGLang-owned DiT.

This allocates no second transformer. Original Comfy control nodes may clone
and patch the facade's schedule without mutating the SGLang weights. LoRA and
attention selections still require explicit per-branch worker dispatch; they
must not be fed to Comfy's local weight loader.
"""


def make_comfy_model_patcher(model, *, device, sequence_parallel_size=1,
                            context_factory=None):
    import torch
    from comfy import model_base, model_patcher, supported_models
    from h3_comfy_dit_bridge import make_comfy_dit_bridge

    config = supported_models.MiniMaxH3({
        "image_model": "minimax_h3",
        "disable_unet_model_creation": True,
        "dtype": torch.bfloat16,
    })
    # BaseModel's native MiniMaxH3 behavior retains packed audio scaling,
    # conditioning/masks, T8 patch wrappers and the sampler x0/velocity protocol.
    base = model_base.MiniMaxH3(config, device=torch.device(device))
    base.diffusion_model = make_comfy_dit_bridge(model,
        sequence_parallel_size=sequence_parallel_size, context_factory=context_factory)
    return model_patcher.ModelPatcher(base, load_device=torch.device(device),
                                     offload_device=torch.device("cpu"))
