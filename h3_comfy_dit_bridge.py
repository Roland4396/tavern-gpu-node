# SPDX-License-Identifier: GPL-3.0-or-later
"""Original Comfy H3 tensor/sampler contract over a SGLang DiT.

The conditioning, packing, clock and mask logic follows ComfyUI 187eda8
comfy/ldm/minimax/model.py (GPL-3.0). Neural projection, text refinement,
attention, MLP and output layers are executed by the supplied SGLang model.
This is a worker-side component, NOT a replacement workflow or HTTP video API.
No model weights are loaded here. Installation/serving is deliberately separate.
"""
from __future__ import annotations

from contextlib import nullcontext
import math


def make_comfy_dit_bridge(model, *, sequence_parallel_size=1, context_factory=None):
    """Borrow an already-loaded SGLang transformer without registering its weights.

    SGLang owns placement and TP/SP groups. Comfy must not move/reload the borrowed
    transformer. A caller must select the immutable branch's LoRAs before using
    this adapter, and provide SGLang's forward context when running real kernels.
    The original outer forward retains FLOW_AV audio carry and T8 wrappers.
    """
    import torch
    from comfy.ldm.minimax import model as native

    if sequence_parallel_size < 1:
        raise ValueError("sequence_parallel_size must be positive")

    class SGLangComfyH3(native.MiniMaxH3Model):
        def __init__(self):
            torch.nn.Module.__init__(self)
            # Do not make the borrowed model a child Module: ModelPatcher must
            # never migrate the SGLang rank's weights behind its manager's back.
            object.__setattr__(self, "_sglang_model", model)
            self.hidden_size = int(model.config.hidden_size)
            self.latents_dim = int(model.config.latents_dim)
            self.audio_latents_dim = int(model.config.audio_latents_dim)
            self.patch_size = tuple(model.config.patch_size)
            self.sigma_shift_video = 12.0
            self.sigma_shift_audio = 3.0
            self.dtype = torch.bfloat16

        def preprocess_text_embeds(self, text_states):
            if text_states.shape[-1] == self.hidden_size:
                return text_states
            size = text_states.shape[1]
            cu = torch.tensor([0, size, size], device=text_states.device, dtype=torch.int32)
            with context_factory() if context_factory else nullcontext():
                return self._sglang_model.refine_prompt_embeds(
                    text_states[0], cu, device=text_states.device,
                ).unsqueeze(0)

        def prepare_call(self, x, timestep, context, transformer_options=None,
                         minimax_payload=None, denoise_mask=None, audio_denoise_mask=None):
            options = transformer_options or {}
            if options.get("patches_replace", {}).get("dit"):
                raise ValueError("Comfy per-block replacements need an explicit SGLang implementation")
            video_x, audio_x = x[:2]
            original_shape = tuple(video_x.shape)
            video_x = native.comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
            if video_x.shape[0] != 1 or audio_x.shape[0] != 1:
                raise ValueError("MiniMax H3 supports batch size 1")
            payload = minimax_payload or {}
            device = video_x.device
            latent_t, lat_h, lat_w = video_x.shape[2:]
            audio_t = audio_x.shape[-1]
            text_len = context.shape[1]
            layout = payload.get("layout")
            if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
                layout = native.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                             keyframes=payload.get("keyframes"), refs=payload.get("refs"))

            shift_v = float(options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
            shift_a = float(options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            t_v = float(1.0 - sigma_v)
            t_a = float(1.0 - native.time_shift_sigma(sigma_v, shift_v, shift_a))
            vis_aug = float(payload.get("visual_cond_noise_aug", native.VISUAL_COND_TIMESTEP))
            aud_aug = float(payload.get("audio_cond_noise_aug", native.AUDIO_COND_TIMESTEP))
            segment_time = {"text": t_v, "video": t_v, "audio": t_a,
                            "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                            "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
            row_times = {}
            if denoise_mask is not None:
                m = native.mask_row_values(denoise_mask[0, 0].float(), latent_t, lat_h, lat_w)
                if m is not None:
                    rows = (1.0 - m * sigma_v.to(m.device)).clamp(max=max(t_v, native.VISUAL_COND_TIMESTEP))
                    if rows.unique().numel() == 1:
                        segment_time["video"] = float(rows[0])
                    else:
                        row_times["video"] = rows
            if audio_denoise_mask is not None:
                m = audio_denoise_mask[0, 0].float().reshape(-1)
                if not bool((m >= 1.0 - 1e-3).all()):
                    rows = (1.0 - m * (1.0 - t_a)).clamp(max=max(t_a, native.AUDIO_COND_TIMESTEP))
                    if rows.unique().numel() == 1:
                        segment_time["audio"] = float(rows[0])
                    else:
                        row_times["audio"] = rows

            levels = {t_v, t_a} | {segment_time[k] for _, _, k in layout.segments}
            for rows in row_times.values():
                levels.update(rows.unique().tolist())
            levels = sorted(levels)
            time_index = {t: i for i, t in enumerate(levels)}
            tag_by_kind = {"text": 1, "video": 0, "audio": 2, "cond": 0,
                           "ref_img": 0, "cond_audio": 2, "ref_audio": 2}
            # Padding is a separate attention sequence, never visible to real
            # tokens. It makes the full length divisible by the actual SP group.
            multiple = math.lcm(128, sequence_parallel_size)
            seq_len = (layout.seq_len + multiple - 1) // multiple * multiple
            inverse = torch.full((seq_len,), time_index[t_v], dtype=torch.long, device=device)
            tags = torch.ones(seq_len, dtype=torch.long, device=device)
            for start, stop, kind in layout.segments:
                if kind in row_times:
                    rows = row_times[kind]
                    unique = rows.unique()
                    ids = torch.tensor([time_index[t] for t in unique.tolist()], dtype=torch.long, device=rows.device)
                    inverse[start:stop] = ids[torch.searchsorted(unique, rows)].to(device)
                else:
                    inverse[start:stop] = time_index[segment_time[kind]]
                if kind == "text" and payload.get("text_token_tags") is not None:
                    text_tags = payload["text_token_tags"].reshape(-1)
                    if text_tags.numel() != stop - start:
                        raise ValueError("Text token tags do not match the original text span")
                    tags[start:stop] = text_tags.to(device)
                else:
                    tags[start:stop] = tag_by_kind[kind]

            video_rows = native.patchify_video(video_x.float(), self.patch_size)
            audio_rows = native.pack_audio(audio_x.float())
            img_update = layout.img_update.to(device)
            audio_update = layout.audio_update.to(device)
            video_cond = self._cond_video_rows(payload, device)
            audio_cond = self._cond_audio_rows(payload, device)
            img_pos, audio_pos = layout.img_pos.to(device), layout.audio_pos.to(device)
            packed_video = torch.zeros((1, seq_len, video_rows.shape[-1]), device=device, dtype=torch.float32)
            packed_audio = torch.zeros((1, seq_len, audio_rows.shape[-1]), device=device, dtype=torch.float32)
            packed_video[0, img_pos[img_update]] = video_rows
            packed_audio[0, audio_pos[audio_update]] = audio_rows
            if video_cond is not None:
                packed_video[0, img_pos[~img_update]] = video_cond
            if audio_cond is not None:
                packed_audio[0, audio_pos[~audio_update]] = audio_cond
            positions = torch.zeros((seq_len, 3), dtype=torch.float32, device=device)
            positions[:layout.seq_len] = layout.position_ids.float().to(device)
            cu = (0, layout.seq_len, seq_len)
            refiner_cu = (0, text_len, text_len)
            call = {
                "x": packed_video, "audio_x": packed_audio,
                "img_position_ids": positions.unsqueeze(0),
                "unique_timesteps": torch.tensor(levels, dtype=torch.float32, device=device),
                "inverse_indices": inverse, "token_tags": tags,
                "update_mask": img_update, "skip_mask_out_condition": True,
                "prompt_embeds": context[0],
                "img_pos_info": {"position_ids": img_pos},
                "audio_pos_info": {"position_ids": audio_pos},
                "text_pos_info": {"position_ids": torch.arange(text_len, device=device)},
                "img_pos_for_infer_output_info": {"position_ids": img_pos[img_update]},
                "packed_seq_params": {"cu_seqlens_q": torch.tensor(cu, dtype=torch.int32, device=device),
                                      "cu_seqlens_q_host": cu, "max_seqlen_q": layout.seq_len},
                "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor(refiner_cu, dtype=torch.int32, device=device),
                                              "cu_seqlens_q_host": refiner_cu, "max_seqlen_q": text_len},
            }
            if context.shape[-1] == self.hidden_size:
                call["refined_prompt_embeds_length"] = text_len
            return call, (original_shape, tuple(video_x.shape), audio_update)

        def _forward(self, x, timestep, context, transformer_options=None, minimax_payload=None,
                     denoise_mask=None, audio_denoise_mask=None, **kwargs):
            call, (original_shape, padded_shape, audio_update) = self.prepare_call(
                x, timestep, context, transformer_options, minimax_payload, denoise_mask, audio_denoise_mask,
            )
            with context_factory() if context_factory else nullcontext():
                video, audio = self._sglang_model(**call)
            t, h, w = padded_shape[2:]
            video = native.unpatchify_video(video, t, h // 2, w // 2, self.latents_dim, self.patch_size)
            video = video[:, :, :original_shape[2], :original_shape[3], :original_shape[4]]
            audio = native.unpack_audio(audio[audio_update])
            # Native Comfy returns negative velocity; SGLang returns positive
            # flow direction. The inherited outer forward applies audio carry.
            return [-video.to(x[0].dtype), -audio.to(x[1].dtype)]

    return SGLangComfyH3()
