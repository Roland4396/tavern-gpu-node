# SPDX-License-Identifier: GPL-3.0-or-later
"""Original Comfy H3 VAE control algorithms over SGLang-owned neural modules.

No weights are loaded here. In particular, do not use SGLang's default sampled
posterior/video pipeline: original Comfy uses posterior mean, its own temporal
padding/overlap, spatial tile order, and raw float IMAGE/AUDIO tensors.
"""
from contextlib import nullcontext
import math
from types import SimpleNamespace


def make_comfy_vae_bridge(model, kind, *, device, dtype, execution_context=None,
                         output_device="cpu"):
    """The caller owns component residency, precision, and distributed groups.

    ``execution_context(direction)`` must make the SGLang component ready at
    the given device/dtype. This adapter never moves the borrowed weights.
    Neural modules are deliberately NOT registered as children of the facade.
    """
    import torch
    from comfy import sd
    from comfy.ldm.minimax import vae as video_native
    from comfy.ldm.minimax import audio_vae as audio_native

    if kind not in ("video", "audio"):
        raise ValueError("H3 VAE kind must be video or audio")

    class VideoStage(video_native.MiniMaxH3VideoVAE):
        def __init__(self):
            torch.nn.Module.__init__(self)
            object.__setattr__(self, "_sglang_model", model)
            # Original Core's constructor geometry, without constructing a
            # second CNN/36-layer transformer or allocating duplicate weights.
            self.vae_ratio = 16
            self.vae_ratio_t = 4
            self.clip_length = 17
            self.token_drop = 3
            self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
            self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
            self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
            self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
            self.tiling = True
            self.tile_size = 256
            self.tile_overlap_min = 64
            self.decoder = SimpleNamespace(out_channels=3)
            self.register_buffer("latents_mean", torch.tensor(video_native.LATENTS_MEAN))
            self.register_buffer("latents_std", torch.tensor(video_native.LATENTS_STD))
            self.register_buffer("pixel_mean", torch.tensor(video_native.IMAGENET_MEAN).view(1, 3, 1, 1, 1))
            self.register_buffer("pixel_std", torch.tensor(video_native.IMAGENET_STD).view(1, 3, 1, 1, 1))

        def _encode_moments(self, x):
            # Raw SGLang encoder + quant_conv, not encode_base's temporal logic.
            return self._sglang_model.encode(x)

        def _decode_pixels(self, z):
            # Raw SGLang post_quant_conv + transformer decoder. The inherited
            # original code controls tiling, frame overlap and finalization.
            return self._sglang_model.decode(z)

    class AudioStage(audio_native.MiniMaxH3AudioVAE):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.sample_rate = self.output_sample_rate = 32000
            self.hop_length = self.samples_per_latent = 800
            self.latents_per_second = 40
            for name in ("encoder", "pre_block", "mean_proj", "dec_in_proj", "decoder"):
                object.__setattr__(self, name, getattr(model, name))
            arch = model.config.arch_config
            for name in ("latents_mean", "latents_std"):
                values = getattr(arch, name)
                if values is None or len(values) != 32:
                    raise ValueError(f"H3 audio VAE requires checkpoint {name}")
                self.register_buffer(name, torch.tensor(values))

    class ComfyVAE:
        # Invoke the upstream crop/channel-padding code verbatim.
        vae_encode_crop_pixels = sd.VAE.vae_encode_crop_pixels

        def __init__(self):
            self.first_stage_model = VideoStage() if kind == "video" else AudioStage()
            self.latent_channels = 24 if kind == "video" else 32
            self.latent_dim = 3 if kind == "video" else 2
            self.output_channels = 3 if kind == "video" else 2
            self.crop_input = kind == "video"  # T8 explicitly disables audio crop.
            self.pad_channel_value = None if kind == "video" else "replicate"
            self.audio_sample_rate = 32000
            self.device = torch.device(device)
            self.output_device = torch.device(output_device)
            self.vae_dtype = dtype
            self.handles_tiling = True
            self.downscale_ratio = ((lambda n: max(1, (n - 5) // 17 * 5 + 2) if n > 1 else 1), 16, 16) if kind == "video" else 800
            self.upscale_ratio = ((lambda n: max(1, (n - 2) // 5 * 17 + 5)), 16, 16) if kind == "video" else 800

        def spacial_compression_encode(self):
            return 16 if kind == "video" else 800

        def spacial_compression_decode(self):
            return self.spacial_compression_encode()

        def _context(self, direction):
            return execution_context(direction) if execution_context else nullcontext()

        @torch.inference_mode()
        def encode(self, pixel_samples):
            x = self.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
            if kind == "video":
                if x.ndim < 5:
                    x = x.movedim(1, 0).unsqueeze(0)
                # Preserve native order: process_input, dtype cast, then the
                # inherited stage moves only temporal clips to the device.
                x = (x * 2.0 - 1.0).to(self.vae_dtype)
            else:
                x = x.to(device=self.device, dtype=self.vae_dtype)
            with self._context("encode"):
                result = (self.first_stage_model.encode(x, device=self.device)
                          if kind == "video" else self.first_stage_model.encode(x))
            # sd.VAE.encode copies each result into a preallocated contiguous
            # buffer. Keep that layout too, not only its values: stereo/latent
            # permutation otherwise changes downstream reduction order.
            return result.to(device=self.output_device, dtype=torch.float32).contiguous()

        @torch.inference_mode()
        def decode(self, samples):
            x = samples.to(device=self.device, dtype=self.vae_dtype)
            with self._context("decode"):
                if kind == "video":
                    out = torch.empty(self.first_stage_model.decode_output_shape(x.shape),
                                      device=self.output_device, dtype=torch.float32)
                    self.first_stage_model.decode(x, output_buffer=out)
                else:
                    out = self.first_stage_model.decode(x).to(device=self.output_device, dtype=torch.float32)
            return out.movedim(1, -1)

        def encode_tiled(self, pixels, **kwargs):
            # Original H3 owns fixed internal tile sizes, ignores fallback hints.
            return self.encode(pixels)

        def decode_tiled(self, samples, **kwargs):
            return self.decode(samples)

    return ComfyVAE()
