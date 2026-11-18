from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .h3_sglang_node import (
    MiniMaxH3FL2VAConditionNode,
    MiniMaxH3FL2VALoader,
    MiniMaxH3FL2VALoRA,
    MiniMaxH3FL2VASampler,
    MiniMaxH3Ref2VAConditionNode,
    MiniMaxH3Ref2VALoader,
    MiniMaxH3Ref2VALoRA,
    MiniMaxH3Ref2VASampler,
)
from .preview_video_node import PreviewVideo
from .h3_chatu8_node import (
    MiniMaxH3Chatu8Asset,
    MiniMaxH3Chatu8SGLang,
    MiniMaxH3Chatu8SaveVideo,
)
from .h3_author_node import (
    MiniMaxH3AuthorSGLang,
    MiniMaxH3AuthorAVComponents,
    MiniMaxH3AuthorSaveVideo,
)
from .seedvr3_node import (
    SeedVR3Await,
    SeedVR3Loader,
    SeedVR3Sampler,
    SeedVR3SettingsNode,
    SeedVR3Submit,
)
from .sequence_execution_node import (
    AISequenceDynamicWiringNode,
    H3BundleSequenceNode,
    H3SequenceSettingsNode,
    H3VideoLastFrameNode,
    SequenceVideoConcatNode,
)
from .sequence_nodes import (
    AISequenceClipSelectorNode,
    AISequencePlanParserNode,
    AISequenceRulesPromptNode,
    AISequenceTaskPromptNode,
    H3AssetCatalogNode,
    H3AudioAssetNode,
    H3ConditionResolverNode,
    H3ImageAssetNode,
    H3VideoAssetNode,
    StreamChatCompletionNode,
)


class GpuRuntimeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            H3ImageAssetNode,
            H3VideoAssetNode,
            H3AudioAssetNode,
            H3AssetCatalogNode,
            AISequenceRulesPromptNode,
            AISequenceTaskPromptNode,
            StreamChatCompletionNode,
            AISequencePlanParserNode,
            AISequenceClipSelectorNode,
            H3ConditionResolverNode,
            H3SequenceSettingsNode,
            H3BundleSequenceNode,
            AISequenceDynamicWiringNode,
            H3VideoLastFrameNode,
            SequenceVideoConcatNode,
            MiniMaxH3FL2VALoader,
            MiniMaxH3Ref2VALoader,
            MiniMaxH3FL2VALoRA,
            MiniMaxH3Ref2VALoRA,
            MiniMaxH3FL2VAConditionNode,
            MiniMaxH3Ref2VAConditionNode,
            MiniMaxH3FL2VASampler,
            MiniMaxH3Ref2VASampler,
            SeedVR3Loader,
            SeedVR3SettingsNode,
            SeedVR3Submit,
            SeedVR3Await,
            SeedVR3Sampler,
            PreviewVideo,
            MiniMaxH3Chatu8Asset,
            MiniMaxH3Chatu8SGLang,
            MiniMaxH3Chatu8SaveVideo,
            MiniMaxH3AuthorSGLang,
            MiniMaxH3AuthorAVComponents,
            MiniMaxH3AuthorSaveVideo,
        ]


async def comfy_entrypoint() -> GpuRuntimeExtension:
    return GpuRuntimeExtension()


WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
