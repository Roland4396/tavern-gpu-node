#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


OLD = '''        if hasattr(weight, "_weight_loader"):
            weight._weight_loader = _weight_loader
        else:
            weight.weight_loader = _weight_loader
'''

NEW = '''        if hasattr(weight, "_weight_loader"):
            weight._weight_loader = _weight_loader
        else:
            weight.weight_loader = _weight_loader

        # Serialized block-FP8 checkpoints carry one scale row per 128 output
        # rows. H3's on-disk Q/K/V rows are interleaved per head, so their scale
        # rows require the same grouped-to-QKV permutation as the weights.
        scale = getattr(self.qkv_proj, "weight_scale_inv", None)
        if scale is not None:
            base_scale_loader = scale.weight_loader

            def _scale_loader(param: torch.Tensor, loaded_scale: torch.Tensor) -> None:
                reordered = _reorder_grouped_qkv_to_qkv(
                    loaded_scale,
                    num_query_groups=arch.num_attention_heads,
                    heads_per_group=1,
                    head_dim=1,
                )
                base_scale_loader(param, reordered)

            if hasattr(scale, "_weight_loader"):
                scale._weight_loader = _scale_loader
            else:
                scale.weight_loader = _scale_loader
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    source = args.path.read_text(encoding="utf-8")
    if NEW in source:
        return
    if source.count(OLD) != 1:
        raise RuntimeError(f"unexpected MiniMax H3 QKV loader source: {args.path}")
    args.path.write_text(source.replace(OLD, NEW), encoding="utf-8")


if __name__ == "__main__":
    main()
