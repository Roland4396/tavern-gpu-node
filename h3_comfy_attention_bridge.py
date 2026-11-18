# SPDX-License-Identifier: GPL-3.0-or-later
"""Scope the ORIGINAL KJ Sage kernel to SGLang's DiT block attention.

The original node patches DiT blocks only, not the text token refiner. Keep
SGLang's TP/Ulysses collectives; run KJ's full, unmodified helper on live tokens
after sequence/head exchange. Never include SP padding in quantization means.
"""
from contextlib import contextmanager


class KJPackedAttention:
    def __init__(self, helper):
        self.helper = helper

    def forward_varlen(self, query, key, value, *, cu_seqlens, max_seqlen,
                       cu_seqlens_host=None):
        import torch
        bounds = tuple(cu_seqlens_host) if cu_seqlens_host is not None else tuple(cu_seqlens.tolist())
        if len(bounds) != 3 or bounds != (0, max_seqlen, query.shape[0]):
            raise ValueError("Original H3 KJ attention expects one live sequence and its isolated padding")
        used = max_seqlen
        live = self.helper([query[:used].unsqueeze(0), key[:used].unsqueeze(0),
                            value[:used].unsqueeze(0)], query.dtype)[0]
        output = torch.zeros_like(query)
        output[:used] = live
        return output


@contextmanager
def kj_sage_scope(model, original_helper):
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    # Resolve the normal refiner/backends first, then patch ONLY DiT blocks as
    # the original Comfy node does. Restore even when a request raises/cancels.
    model._resolve_attention_backend_once()
    prior = []
    try:
        for block in model.blocks:
            attn = block.attn
            prior.append((attn, attn._attention_impl, attn._attention_backend_enum))
            attn._attention_impl = KJPackedAttention(original_helper)
            attn._attention_backend_enum = AttentionBackendEnum.SAGE_ATTN
        yield
    finally:
        for attn, impl, backend in prior:
            attn._attention_impl, attn._attention_backend_enum = impl, backend
