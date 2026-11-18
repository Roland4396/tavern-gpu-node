# SPDX-License-Identifier: GPL-3.0-or-later
"""Original MiniMax Comfy presentation and pixels, SGLang Qwen computation.

Uses the unmodified pinned Core tokenizer, image/video preprocessing, MRoPE
positions and token tags. No PIL/PNG round trip, prompt rewriting or chat template.
The caller owns SGLang's encoder-fold TP group and component residency scope.
"""
from contextlib import nullcontext


class SGLangComfyH3Clip:
    def __init__(self, text_encoder, tokenizer, *, context_factory=None):
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
        self.encoder = text_encoder
        self.context_factory = context_factory

        class RawTokens:
            # H3's upstream tokenizer always requests disable_weights=True.
            # Reuse SGLang's already loaded vocabulary, not a new download.
            def tokenize_with_weights(self, text, **kwargs):
                if not kwargs.get("disable_weights"):
                    raise ValueError("H3 presentation requires raw, unweighted text tokens")
                return [[(token, 1.0) for token in tokenizer.encode(text, add_special_tokens=False)]]

        self.presentation = MiniMaxH3Tokenizer.__new__(MiniMaxH3Tokenizer)
        self.presentation.qwen3vl_32b = RawTokens()

    def tokenize(self, text, **kwargs):
        return self.presentation.tokenize_with_weights(text, **kwargs)

    def prepare_encoder_inputs(self, tokens):
        import torch
        from comfy.text_encoders.minimax import process_video_block, token_tags_from_embeds_info
        from comfy.text_encoders.qwen_vl import process_qwen2vl_images, qwen2vl_mrope_position_ids
        batches = tokens["qwen3vl_32b"]
        if len(batches) != 1:
            raise ValueError("Original H3 presentation requires one token batch")
        ids, patches, grids, embed_info = [], [], [], []
        for entry in batches[0]:
            item, weight = entry[:2]
            if weight != 1.0:
                raise ValueError("Unexpected weighted H3 token")
            if isinstance(item, dict):
                if item.get("type") != "image":
                    raise ValueError("Unsupported original H3 embedding entry")
                if item.get("minimax_video_block", False):
                    pixels, grid = process_video_block(item["data"])
                else:
                    pixels, grid = process_qwen2vl_images(item["data"], patch_size=16,
                        image_mean=[.5, .5, .5], image_std=[.5, .5, .5])
                count = int(grid.prod()) // 4
                embed_info.append({"type": "image", "index": len(ids), "size": count, "extra": {"grid": grid}})
                ids.extend([151655] * count)
                patches.append(pixels); grids.append(grid)
            else:
                ids.append(int(item))
        call = {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        if patches:
            call["pixel_values"] = torch.cat(patches)
            call["image_grid_thw"] = torch.cat(grids)
            # Preserve Comfy's exact media-position logic, instead of invoking
            # a different processor's default image/video timeline policy.
            call["position_ids"] = qwen2vl_mrope_position_ids(embed_info, len(ids), "cpu").unsqueeze(1).long()
        tags = token_tags_from_embeds_info(len(ids), embed_info)
        return call, tags

    def encode_from_tokens_scheduled(self, tokens):
        import torch
        call, tags = self.prepare_encoder_inputs(tokens)
        with self.context_factory() if self.context_factory else nullcontext():
            device = self.encoder.device
            call = {k: v.to(device, dtype=torch.bfloat16 if k == "pixel_values" else v.dtype)
                    for k, v in call.items()}
            with torch.inference_mode():
                hidden = self.encoder(**call).last_hidden_state
                projection = getattr(self.encoder, "conditioning_projection", None)
                if projection is not None:
                    hidden = projection(hidden)
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != tags.numel():
                raise ValueError("SGLang Qwen output does not match the original token presentation")
            hidden = hidden.detach().cpu()
        return [[hidden, {"minimax_token_tags": tags}]]
