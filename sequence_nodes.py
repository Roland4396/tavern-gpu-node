from __future__ import annotations

import http.client
import json
import os
import socket
from pathlib import Path
from typing import Any

from comfy_api.latest import ComfyExtension, Input, io
from typing_extensions import override


ORCHESTRATOR_SOCKET = os.environ.get(
    "GPU_ORCHESTRATOR_SOCKET",
    "/home/worker/.local/state/gpu-runtime/orchestrator.sock",
)

H3Asset = io.Custom("H3_ASSET")
H3AssetCatalog = io.Custom("H3_ASSET_CATALOG")
H3ConditionBundle = io.Custom("H3_CONDITION_BUNDLE")

DEFAULT_SEQUENCE_SYSTEM_PROMPT = (
    Path(__file__).with_name("sequence_system_prompt.md").read_text(encoding="utf-8").strip()
)


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def orchestrator_json(
    method: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    connection = UnixHTTPConnection(ORCHESTRATOR_SOCKET, timeout)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    try:
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        result = {"raw": raw.decode("utf-8", errors="replace")}
    if not 200 <= response.status < 300:
        raise RuntimeError(f"Local planner request failed ({response.status}): {result}")
    if not isinstance(result, dict):
        raise RuntimeError(f"Local planner returned an invalid response: {result!r}")
    return result


def _asset_duration(asset: dict[str, Any]) -> float | None:
    kind = asset["type"]
    data = asset["data"]
    if kind in {"video", "video_audio"}:
        return float(data.get_duration())
    if kind == "audio":
        waveform = data["waveform"]
        return float(waveform.shape[-1]) / float(data["sample_rate"])
    return None


def catalog_as_json(assets: dict[str, dict[str, Any]]) -> str:
    public = []
    for asset in assets.values():
        item = {
            "asset_id": asset["asset_id"],
            "type": asset["type"],
            "description": asset["description"],
        }
        duration = _asset_duration(asset)
        if duration is not None:
            item["duration_sec"] = round(duration, 3)
        public.append(item)
    return json.dumps(public, ensure_ascii=False, indent=2)


class H3ImageAssetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ImageAsset",
            display_name="H3 Image Asset",
            category="H3/sequence/assets",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("asset_id", default="character_01"),
                io.String.Input("description", multiline=True, default=""),
            ],
            outputs=[H3Asset.Output(display_name="asset")],
        )

    @classmethod
    def execute(cls, image: Input.Image, asset_id: str, description: str) -> io.NodeOutput:
        asset_id = asset_id.strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        return io.NodeOutput({
            "asset_id": asset_id,
            "type": "image",
            "description": description.strip(),
            "data": image,
        })


class H3VideoAssetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VideoAsset",
            display_name="H3 Video Asset",
            category="H3/sequence/assets",
            inputs=[
                io.Video.Input("video"),
                io.String.Input("asset_id", default="video_01"),
                io.Combo.Input("reference_type", options=["video_audio", "video"], default="video_audio"),
                io.String.Input("description", multiline=True, default=""),
            ],
            outputs=[H3Asset.Output(display_name="asset")],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        asset_id: str,
        reference_type: str,
        description: str,
    ) -> io.NodeOutput:
        asset_id = asset_id.strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        duration = float(video.get_duration())
        if not 2 <= duration <= 15.001:
            raise ValueError(f"H3 reference video duration must be 2-15 seconds, got {duration:.3f}")
        return io.NodeOutput({
            "asset_id": asset_id,
            "type": reference_type,
            "description": description.strip(),
            "data": video,
        })


class H3AudioAssetNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3AudioAsset",
            display_name="H3 Audio Asset",
            category="H3/sequence/assets",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("asset_id", default="voice_01"),
                io.String.Input("description", multiline=True, default=""),
            ],
            outputs=[H3Asset.Output(display_name="asset")],
        )

    @classmethod
    def execute(cls, audio: Input.Audio, asset_id: str, description: str) -> io.NodeOutput:
        asset_id = asset_id.strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        asset = {
            "asset_id": asset_id,
            "type": "audio",
            "description": description.strip(),
            "data": audio,
        }
        duration = _asset_duration(asset)
        if duration is None or not 2 <= duration <= 15.001:
            raise ValueError(f"H3 reference audio duration must be 2-15 seconds, got {duration}")
        return io.NodeOutput(asset)


class H3AssetCatalogNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            input=H3Asset.Input("asset"), prefix="asset_", min=0, max=12
        )
        return io.Schema(
            node_id="H3AssetCatalog",
            display_name="H3 Asset Catalog",
            category="H3/sequence/assets",
            inputs=[io.Autogrow.Input("assets", template=template, optional=True)],
            outputs=[
                H3AssetCatalog.Output(display_name="asset_catalog"),
                io.String.Output(display_name="catalog_json"),
            ],
        )

    @classmethod
    def execute(cls, assets: io.Autogrow.Type | None = None) -> io.NodeOutput:
        indexed: dict[str, dict[str, Any]] = {}
        for asset in (assets or {}).values():
            if asset is None:
                continue
            asset_id = asset["asset_id"]
            if asset_id in indexed:
                raise ValueError(f"duplicate H3 asset_id: {asset_id}")
            indexed[asset_id] = asset
        return io.NodeOutput({"assets": indexed}, catalog_as_json(indexed))


class AISequenceRulesPromptNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AISequenceRulesPrompt",
            display_name="AI Sequence Rules Prompt",
            category="H3/sequence/AI",
            inputs=[io.String.Input("system_prompt", multiline=True, default=DEFAULT_SEQUENCE_SYSTEM_PROMPT)],
            outputs=[io.String.Output(display_name="system_prompt")],
        )

    @classmethod
    def execute(cls, system_prompt: str) -> io.NodeOutput:
        return io.NodeOutput(system_prompt)


class AISequenceTaskPromptNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AISequenceTaskPrompt",
            display_name="AI Sequence Task Prompt",
            category="H3/sequence/AI",
            inputs=[
                io.String.Input("source_text", multiline=True, default=""),
                io.String.Input("asset_catalog", multiline=True, default="[]"),
                io.Combo.Input("source_type", options=["auto", "prose", "comic", "script"], default="auto"),
                io.Float.Input("minimum_clip_seconds", default=5, min=1, max=15, step=1),
                io.Float.Input("maximum_clip_seconds", default=15, min=5, max=15, step=1),
                io.String.Input("additional_direction", multiline=True, default=""),
            ],
            outputs=[
                io.String.Output(display_name="user_prompt"),
                io.String.Output(display_name="asset_catalog"),
            ],
        )

    @classmethod
    def execute(
        cls,
        source_text: str,
        asset_catalog: str,
        source_type: str,
        minimum_clip_seconds: float,
        maximum_clip_seconds: float,
        additional_direction: str,
    ) -> io.NodeOutput:
        if not source_text.strip():
            raise ValueError("source_text is required")
        if minimum_clip_seconds > maximum_clip_seconds:
            raise ValueError("minimum_clip_seconds cannot exceed maximum_clip_seconds")
        prompt = (
            f"Plan the following {source_type} source as semantic scenes and H3 generation clips.\n"
            f"Allowed clip duration range: {minimum_clip_seconds:g} to "
            f"{maximum_clip_seconds:g} seconds.\n"
            "There is no preferred or fixed clip duration. Choose each clip's duration "
            "independently from its dialogue timing, action density, camera complexity, "
            "and complete narrative beat. Use the shortest duration that can present the "
            "clip clearly without compression.\n"
            f"Additional direction: {additional_direction.strip() or 'None'}\n\n"
            f"Asset catalog:\n{asset_catalog.strip() or '[]'}\n\n"
            f"Source:\n{source_text.strip()}\n\n"
            "Mandatory final frame-chain check:\n"
            "- Every clip MUST use generation_branch=base and reference_assets=[].\n"
            "- Only the first project clip may select an external first_frame_asset_id.\n"
            "- Every later clip sets first_frame_asset_id=null because the executor injects "
            "the immediately preceding H3 clip's exact last frame as its first frame.\n"
            "- Every later h3_prompt follows the I2VA base contract and describes motion "
            "forward from that injected Picture 1.\n"
            "- Inside every h3_prompt, immediately pair each <d> block with one exact "
            "bottom-center native subtitle in English double quotes; H3 generates it "
            "synchronously with the voice. Do not emit subtitle timing metadata.\n"
            "- Use the dialogue-density budget when allocating utterances and clip "
            "durations; move excess complete utterances to later clips.\n"
            "- Use only the three base prompt sections. Correct every violation before "
            "producing <content>."
        )
        return io.NodeOutput(prompt, asset_catalog)


class StreamChatCompletionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PrivateStreamChatCompletion",
            display_name="Stream Chat Completion",
            category="H3/sequence/AI",
            inputs=[
                io.String.Input("system_prompt", force_input=True),
                io.String.Input("user_prompt", force_input=True),
                io.String.Input("model", default="deepseek-v4-flash-local"),
                io.Float.Input("temperature", default=0.2, min=0, max=2, step=0.05),
                io.Int.Input("max_tokens", default=12000, min=256, max=65536, step=256),
                io.Int.Input(
                    "request_seed",
                    default=1,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="assistant_text"),
                io.String.Output(display_name="raw_response"),
            ],
        )

    @classmethod
    def execute(
        cls,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        request_seed: int,
    ) -> io.NodeOutput:
        del request_seed
        result = orchestrator_json(
            "POST",
            "/planner/v1/chat/completions",
            {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            },
            timeout=1800,
        )
        choices = result.get("choices") or []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Stream response does not contain assistant text: {result}")
        return io.NodeOutput(content, json.dumps(result, ensure_ascii=False, indent=2))


class AISequencePlanParserNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AISequencePlanParser",
            display_name="AI Sequence Plan Parser",
            category="H3/sequence/AI",
            inputs=[
                io.String.Input("assistant_text", force_input=True),
                io.String.Input("asset_catalog", force_input=True),
                io.Float.Input("minimum_clip_seconds", default=5, min=1, max=15, step=1),
                io.Float.Input("maximum_clip_seconds", default=15, min=5, max=30, step=1),
            ],
            outputs=[io.String.Output(display_name="normalized_plan")],
        )

    @classmethod
    def execute(
        cls,
        assistant_text: str,
        asset_catalog: str,
        minimum_clip_seconds: float,
        maximum_clip_seconds: float,
    ) -> io.NodeOutput:
        result = orchestrator_json(
            "POST",
            "/planner/v1/validate",
            {
                "response_text": assistant_text,
                "asset_catalog": asset_catalog,
                "min_clip_seconds": minimum_clip_seconds,
                "max_clip_seconds": maximum_clip_seconds,
            },
            timeout=30,
        )
        return io.NodeOutput(json.dumps(result["plan"], ensure_ascii=False, indent=2))


class AISequenceClipSelectorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AISequenceClipSelector",
            display_name="AI Sequence Clip Selector",
            category="H3/sequence/AI",
            inputs=[
                io.String.Input("normalized_plan", force_input=True),
                io.Int.Input("scene_number", default=1, min=1, max=10000),
                io.Int.Input("clip_number", default=1, min=1, max=10000),
            ],
            outputs=[
                io.String.Output(display_name="clip_json"),
                io.String.Output(display_name="h3_prompt"),
                io.Float.Output(display_name="duration_seconds"),
                io.Boolean.Output(display_name="use_ref2va"),
            ],
        )

    @classmethod
    def execute(cls, normalized_plan: str, scene_number: int, clip_number: int) -> io.NodeOutput:
        plan = json.loads(normalized_plan)
        try:
            clip = plan["scenes"][scene_number - 1]["clips"][clip_number - 1]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"plan does not contain scene {scene_number}, clip {clip_number}"
            ) from exc
        return io.NodeOutput(
            json.dumps(clip, ensure_ascii=False, indent=2),
            clip["h3_prompt"],
            float(clip["target_duration_sec"]),
            clip["generation_branch"] == "ref2va",
        )


class H3ConditionResolverNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ConditionResolver",
            display_name="H3 Condition Resolver",
            category="H3/sequence/conditioning",
            inputs=[
                io.String.Input("clip_json", force_input=True),
                H3AssetCatalog.Input("asset_catalog"),
                io.Video.Input("continuation_video", optional=True),
            ],
            outputs=[
                H3ConditionBundle.Output(display_name="base_conditions"),
                H3ConditionBundle.Output(display_name="ref2va_conditions"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip_json: str,
        asset_catalog: dict[str, Any],
        continuation_video: Input.Video | None = None,
    ) -> io.NodeOutput:
        clip = json.loads(clip_json)
        assets = asset_catalog.get("assets") or {}

        def require(asset_id: str, expected: str | None = None) -> dict[str, Any]:
            if asset_id.startswith("generated:"):
                expected_id = f"generated:{clip.get('parent_clip_id')}"
                if asset_id != expected_id:
                    raise ValueError(
                        f"generated continuation {asset_id!r} does not match {expected_id!r}"
                    )
                if continuation_video is None:
                    raise ValueError(
                        f"clip requires generated continuation video {asset_id!r}, but none is connected"
                    )
                if expected and expected != "video_audio":
                    raise ValueError(f"generated continuation {asset_id!r} must be video_audio")
                return {
                    "asset_id": asset_id,
                    "type": "video_audio",
                    "description": "generated previous clip continuation",
                    "data": continuation_video,
                }
            asset = assets.get(asset_id)
            if asset is None:
                raise ValueError(f"clip references an unconnected asset_id: {asset_id}")
            if expected and asset["type"] != expected:
                raise ValueError(
                    f"asset {asset_id!r} must be {expected}, got {asset['type']}"
                )
            return asset

        first_id = clip.get("first_frame_asset_id")
        last_id = clip.get("last_frame_asset_id")
        first = require(first_id, "image") if first_id else None
        last = require(last_id, "image") if last_id else None
        references = []
        seen: set[str] = set()
        for spec in clip.get("reference_assets") or []:
            asset_id = spec["asset_id"]
            asset = require(asset_id, "video_audio" if asset_id.startswith("generated:") else None)
            references.append({**asset, "purpose": spec.get("purpose") or "reference"})
            seen.add(asset_id)
        if clip.get("generation_branch") == "ref2va":
            for asset, purpose in ((first, "opening_visual_anchor"), (last, "ending_visual_anchor")):
                if asset is not None and asset["asset_id"] not in seen:
                    references.append({**asset, "purpose": purpose})
        base = {
            "first_frame": first["data"] if first else None,
            "last_frame": last["data"] if last else None,
        }
        ref = {"references": references}
        return io.NodeOutput(base, ref)


class SequenceNodesExtension(ComfyExtension):
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
        ]


async def comfy_entrypoint() -> SequenceNodesExtension:
    return SequenceNodesExtension()
