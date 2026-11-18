#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TEXT_IMPORT_OLD = '''from sglang.multimodal_gen.runtime.loader.fsdp_load import shard_model
from sglang.multimodal_gen.runtime.loader.utils import (
'''
TEXT_IMPORT_NEW = '''from sglang.multimodal_gen.runtime.loader.fsdp_load import shard_model
from sglang.multimodal_gen.runtime.loader.h3_rank_sharded_checkpoint import (
    maybe_export_rank_shard,
    resolve_rank_shard,
)
from sglang.multimodal_gen.runtime.loader.utils import (
'''

TEXT_FILE_OLD = '''        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"

        use_safetensors = False
'''
TEXT_FILE_NEW = '''        if os.path.isfile(model_name_or_path):
            if not model_name_or_path.endswith(".safetensors"):
                raise RuntimeError(
                    f"Rank-sharded text encoder must be safetensors: {model_name_or_path}"
                )
            return os.path.dirname(model_name_or_path), [model_name_or_path], True

        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"

        use_safetensors = False
'''

TEXT_LOAD_OLD = '''            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self._get_all_weights(
                    model,
                    model_path,
                    to_cpu=should_offload,
                )
            )

            if should_offload:
'''
TEXT_LOAD_NEW = '''            weights_to_load = {name for name, _ in model.named_parameters()}
            source_model_path = model_path
            rank_shard = resolve_rank_shard("text_encoder", source_model_path)
            if rank_shard is not None:
                logger.info("Loading text_encoder rank shard from %s", rank_shard)
                model._rank_sharded_checkpoint = True
                model_path = str(rank_shard)
            loaded_weights = model.load_weights(
                self._get_all_weights(
                    model,
                    model_path,
                    to_cpu=should_offload,
                )
            )
            maybe_export_rank_shard("text_encoder", model, source_model_path)

            if should_offload:
'''

FSDP_IMPORT_OLD = '''from collections import Counter, defaultdict
from collections.abc import Callable, Generator
from itertools import chain
from typing import Any
'''
FSDP_IMPORT_NEW = '''from collections import Counter, defaultdict
from collections.abc import Callable, Generator
from itertools import chain
from pathlib import Path
from typing import Any
'''

FSDP_HELPER_IMPORT_OLD = '''from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
'''
FSDP_HELPER_IMPORT_NEW = '''from sglang.multimodal_gen.runtime.loader.h3_rank_sharded_checkpoint import (
    maybe_export_rank_shard,
    resolve_rank_shard,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
'''

FSDP_LOAD_OLD = '''    # 2. load model from disk
    weight_iterator = safetensors_weights_iterator(weight_dir_list)
    preprocess_loaded_state_dict = getattr(model, "preprocess_loaded_state_dict", None)
'''
FSDP_LOAD_NEW = '''    # 2. load model from disk
    source_weight_dir = Path(weight_dir_list[0]).parent
    rank_shard = resolve_rank_shard("transformer", source_weight_dir)
    pre_sharded_checkpoint = rank_shard is not None
    if rank_shard is not None:
        logger.info("Loading transformer rank shard from %s", rank_shard)
        weight_dir_list = [str(rank_shard)]
    weight_iterator = safetensors_weights_iterator(weight_dir_list)
    preprocess_loaded_state_dict = getattr(model, "preprocess_loaded_state_dict", None)
'''

FSDP_EXPORT_OLD = '''    if bnb_quant_states:
        attach_bitsandbytes_4bit_quant_states(
            dict(model.named_parameters()), bnb_quant_states
        )

    # 3. postprocessing
'''
FSDP_EXPORT_NEW = '''    if bnb_quant_states:
        attach_bitsandbytes_4bit_quant_states(
            dict(model.named_parameters()), bnb_quant_states
        )
    maybe_export_rank_shard("transformer", model, source_weight_dir)

    # 3. postprocessing
'''

FSDP_CALL_OLD = '''        strict=strict,
        cpu_offload=load_cpu_offload,
        param_names_mapping=param_names_mapping_fn,
    )
'''
FSDP_CALL_NEW = '''        strict=strict,
        cpu_offload=load_cpu_offload,
        param_names_mapping=param_names_mapping_fn,
        pre_sharded_checkpoint=pre_sharded_checkpoint,
    )
'''

FSDP_SIGNATURE_OLD = '''    cpu_offload: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
) -> _IncompatibleKeys:
'''
FSDP_SIGNATURE_NEW = '''    cpu_offload: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
    pre_sharded_checkpoint: bool = False,
) -> _IncompatibleKeys:
'''

FSDP_WEIGHT_LOADER_OLD = '''            weight_loader = (
                getattr(actual_param, "weight_loader", None)
                if actual_param is not None
                else None
            )
'''
FSDP_WEIGHT_LOADER_NEW = '''            weight_loader = (
                getattr(actual_param, "weight_loader", None)
                if actual_param is not None and not pre_sharded_checkpoint
                else None
            )
'''

ENCODER_LOAD_OLD = '''            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
'''
ENCODER_LOAD_NEW = '''            weight_loader = (
                default_weight_loader
                if getattr(self, "_rank_sharded_checkpoint", False)
                else getattr(param, "weight_loader", default_weight_loader)
            )
            try:
'''


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f"unexpected source while patching {path}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("helper", type=Path)
    args = parser.parse_args()

    loader = (
        args.source_root
        / "python/sglang/multimodal_gen/runtime/loader"
    )
    shutil.copy2(args.helper, loader / "h3_rank_sharded_checkpoint.py")

    text_loader = loader / "component_loaders/text_encoder_loader.py"
    replace_once(text_loader, TEXT_IMPORT_OLD, TEXT_IMPORT_NEW)
    replace_once(text_loader, TEXT_FILE_OLD, TEXT_FILE_NEW)
    replace_once(text_loader, TEXT_LOAD_OLD, TEXT_LOAD_NEW)

    fsdp_loader = loader / "fsdp_load.py"
    replace_once(fsdp_loader, FSDP_IMPORT_OLD, FSDP_IMPORT_NEW)
    replace_once(fsdp_loader, FSDP_HELPER_IMPORT_OLD, FSDP_HELPER_IMPORT_NEW)
    replace_once(fsdp_loader, FSDP_LOAD_OLD, FSDP_LOAD_NEW)
    replace_once(fsdp_loader, FSDP_CALL_OLD, FSDP_CALL_NEW)
    replace_once(fsdp_loader, FSDP_SIGNATURE_OLD, FSDP_SIGNATURE_NEW)
    source = fsdp_loader.read_text(encoding="utf-8")
    if FSDP_WEIGHT_LOADER_NEW not in source:
        if source.count(FSDP_WEIGHT_LOADER_OLD) != 2:
            raise RuntimeError(f"unexpected weight-loader source in {fsdp_loader}")
        fsdp_loader.write_text(
            source.replace(FSDP_WEIGHT_LOADER_OLD, FSDP_WEIGHT_LOADER_NEW),
            encoding="utf-8",
        )
    replace_once(fsdp_loader, FSDP_EXPORT_OLD, FSDP_EXPORT_NEW)

    encoder = (
        args.source_root
        / "python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py"
    )
    replace_once(encoder, ENCODER_LOAD_OLD, ENCODER_LOAD_NEW)


if __name__ == "__main__":
    main()
