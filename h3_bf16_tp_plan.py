"""Offline DiT TP slice plan for the original (non-pruned) H3 checkpoint.

Planning uses safetensors headers only. This is not a tensor exporter and never
activates a worker. The rules mirror SGLang b2a437a's MiniMaxH3DiTModel: TP divides
projections, Ulysses replicates each TP shard. QKV and gate/up must not be cut
as a single ordinary column-parallel matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

_BLOCK = r"(?:blocks\.\d+|token_refiner\.blocks\.\d+)"


def tensor_rule(name: str) -> str:
    if re.fullmatch(_BLOCK + r"\.attn\.qkv_proj\.weight", name):
        return "grouped_qkv"
    if re.fullmatch(_BLOCK + r"\.mlp\.fc1\.weight", name):
        return "merged_gate_up"
    if re.fullmatch(_BLOCK + r"\.(?:attn\.out_proj|mlp\.fc2)\.weight", name):
        return "row"
    if name == "time_embedder.proj_out.weight":
        return "row"
    if re.fullmatch(
        r"(?:video_patch_proj|audio_patch_proj|condition_proj|time_embedder\.proj_in|final_layer\.(?:video_out|audio_out))\.(?:weight|bias)",
        name,
    ):
        return "column"
    if re.fullmatch(
        r"(?:blocks\.\d+|final_layer)\.adaln_proj\.linear\.(?:weight|bias)", name
    ):
        return "column"
    if re.fullmatch(_BLOCK + r"\.(?:attn\.[qk]_norm|norm[12])\.weight", name):
        return "replicated"
    if name in {
        "rope.inv_freq",
        "time_embedder.proj_out.bias",
        "token_refiner.final_norm.weight",
        "final_layer.norm.weight",
    }:
        return "replicated"
    raise ValueError(f"No reviewed BF16 TP rule for {name!r}")


def local_shape(
    name: str, shape: list[int], *, tp: int, heads: int, head_dim: int
) -> list[int]:
    if tp not in (1, 2, 4) or heads % tp:
        raise ValueError("TP must be 1/2/4 and divide attention heads")
    rule = tensor_rule(name)
    out = list(shape)
    if rule == "replicated":
        return out
    dim = 1 if rule == "row" else 0
    if len(shape) <= dim or shape[dim] % tp:
        raise ValueError(f"Cannot shard {name} shape={shape} over TP={tp}")
    if rule == "grouped_qkv" and shape[0] != heads * 3 * head_dim:
        raise ValueError(f"Unexpected grouped QKV shape for {name}: {shape}")
    if rule == "merged_gate_up" and shape[0] % (2 * tp):
        raise ValueError(f"Gate/up widths must each divide TP: {name}")
    out[dim] //= tp
    return out


def slice_tensor(name, tensor, *, rank, tp, heads, head_dim):
    """CPU verification/export primitive; does not import or initialize CUDA."""
    import torch

    if not 0 <= rank < tp:
        raise ValueError("TP rank out of range")
    expected = local_shape(
        name, list(tensor.shape), tp=tp, heads=heads, head_dim=head_dim
    )
    rule = tensor_rule(name)
    if rule == "grouped_qkv":
        grouped = tensor.reshape(heads, 3, head_dim, *tensor.shape[1:])
        count = heads // tp
        selected = grouped[rank * count : (rank + 1) * count]
        out = torch.cat(
            [selected[:, i].reshape(-1, *tensor.shape[1:]) for i in range(3)], dim=0
        )
    elif rule == "merged_gate_up":
        half = tensor.shape[0] // 2
        count = half // tp
        out = torch.cat(
            [
                tensor[part * half + rank * count : part * half + (rank + 1) * count]
                for part in range(2)
            ],
            dim=0,
        )
    elif rule in ("column", "row"):
        dim = 1 if rule == "row" else 0
        count = tensor.shape[dim] // tp
        out = tensor.narrow(dim, rank * count, count)
    else:
        out = tensor
    assert list(out.shape) == expected
    return out.contiguous()


def plan(config: dict, headers: dict, *, tp: int = 2, ulysses: int = 2) -> dict:
    if (
        config.get("_class_name") != "MiniMaxH3DiTModel"
        or config.get("adaln_curve_grid")
        or config.get("checkpoint_uses_diffusers_layout")
    ):
        raise ValueError("This plan is only for the original grouped-QKV H3 checkpoint")
    if tp != 2 or ulysses != 2:
        raise ValueError(
            "This preparation is pinned to the approved TP2 x Ulysses2 layout"
        )
    specs = {}
    total = 0
    for name, entry in sorted(headers.items()):
        if name == "__metadata__":
            continue
        dtype = entry["dtype"]
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"Not a BF16/F32 source: {name} {dtype}")
        shape = local_shape(
            name,
            entry["shape"],
            tp=tp,
            heads=config["num_attention_heads"],
            head_dim=config["attention_head_dim"],
        )
        size = math.prod(shape) * (2 if dtype == "BF16" else 4)
        specs[name] = {
            "rule": tensor_rule(name),
            "source_shape": entry["shape"],
            "local_shape": shape,
            "dtype": dtype,
            "bytes": size,
        }
        total += size
    if not specs:
        raise ValueError("Empty checkpoint")
    return {
        "status": "plan_only_not_exported",
        "component": "transformer",
        "tp_size": tp,
        "ulysses_degree": ulysses,
        "world_size": tp * ulysses,
        "global_rank_to_tp_rank": [r % tp for r in range(tp * ulysses)],
        "tensor_count": len(specs),
        "bytes_per_rank": total,
        "v1_four_rank_bytes": total * tp * ulysses,
        "tensors": specs,
    }


def read_source(source: Path):
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    headers = {}
    for name in sorted(set(index["weight_map"].values())):
        with (source / name).open("rb") as f:
            count = struct.unpack("<Q", f.read(8))[0]
            data = json.loads(f.read(count))
        for key, entry in data.items():
            if key == "__metadata__":
                continue
            if key in headers or index["weight_map"].get(key) != name:
                raise ValueError(f"Inconsistent source index for {key}")
            headers[key] = entry
    if set(headers) != set(index["weight_map"]):
        raise ValueError("Source shards do not cover the complete index")
    return config, headers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(plan(*read_source(args.source)), indent=2))
