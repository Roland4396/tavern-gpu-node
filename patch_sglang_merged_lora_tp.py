#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


OLD = '''    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[:, start_idx:end_idx, :]
'''

NEW = '''    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        if B.dim() == 2:
            # Some adapters store an already-fused B matrix. Each logical
            # projection must be sharded independently to match the packed
            # base-layer layout (for example Q/K/V or gate/up).
            shards = []
            offset = 0
            for output_size, shard_size in zip(
                self.base_layer.output_sizes,
                self.base_layer.output_partition_sizes,
            ):
                start_idx = offset + tp_rank * shard_size
                shards.append(B[start_idx : start_idx + shard_size, :])
                offset += output_size
            return torch.cat(shards, dim=0)

        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[:, start_idx:end_idx, :]
'''


def patch_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if NEW in source:
        return
    if source.count(OLD) != 1:
        raise RuntimeError(f"unexpected SGLang merged LoRA source: {path}")
    path.write_text(source.replace(OLD, NEW), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    patch_source(args.path)


if __name__ == "__main__":
    main()
