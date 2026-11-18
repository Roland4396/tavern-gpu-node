"""chatu8's media-slot contract -> the existing four-GPU H3 job service.

No T8 model patches or sampler replacement is installed. This adapter uses the
deployed SGLang scheduler/profile and exposes a separate Comfy workflow.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from . import h3_sglang_node as h3
from .preview_video_node import PreviewVideo


Chatu8Asset = io.Custom("H3_CHATU8_ASSET")
TURBO_V4 = "minimax_h3_turbo_v4_step600_ema.safetensors"
MEDIA_TAG = re.compile(r"<\s*(Picture|Image|Audio)\s*(\d+)\s*>", re.IGNORECASE)


def empty_slot(kind: str, enabled: int, filename: str) -> bool:
    # chatu8 3.1.2 marks its synthetic first black image / silent audio as
    # enabled even when the manifest contains no media. These exact reserved
    # upload names come from uploadBlank{Png,Audio}ToComfyUIWithCache.
    placeholders = {"image": "blank_placeholder.png", "audio": "blank_placeholder.wav"}
    return enabled == 2 or Path(filename).name == placeholders.get(kind)


def uploaded_path(filename: str) -> Path:
    root = Path(folder_paths.get_input_directory()).resolve()
    path = Path(folder_paths.get_annotated_filepath(filename)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("chatu8 media must be an uploaded ComfyUI input file")
    if not path.is_file():
        raise FileNotFoundError(f"chatu8 uploaded media not found: {filename}")
    return path


def stage_asset(filename: str) -> Path:
    """Publish input bytes, not tensors/base64, where either GPU node can read."""
    source = uploaded_path(filename)
    root = h3.SHARED_INPUT_DIR
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"chatu8-{h3._sha256_file(source)}{source.suffix.lower()}"
    if not destination.is_file():
        fd, temporary = tempfile.mkstemp(prefix=".chatu8-", dir=root)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination


def prepare_request(prompt: str, mode: str, assets: dict | None):
    """Preserve slot identity even when Picture 2 or Audio 1 is unconnected."""
    active = [item for item in (assets or {}).values() if item is not None]
    active.sort(key=lambda item: (item["kind"] != "image", item["slot"]))
    slots = {"image": {}, "audio": {}}
    for item in active:
        kind, slot = item["kind"], item["slot"]
        limit = 9 if kind == "image" else 3
        if kind not in slots or not 1 <= slot <= limit or slot in slots[kind]:
            raise ValueError("Invalid or duplicate chatu8 media slot")
        slots[kind][slot] = len(slots[kind]) + 1
    if mode == "first_frame":
        if set(slots["image"]) != {1} or slots["audio"]:
            raise ValueError(
                "首帧工作流只接受图片1；多图或音频参考请选参考工作流。"
                "当前 SGLang 不支持精确首帧与音频参考混合，不能静默丢弃素材。"
            )
        task, route = "fl2va", "/h3/v1/videos"
    elif mode == "reference":
        task, route = ("ref2va", "/h3-ref/v1/videos") if active else ("t2va", "/h3/v1/videos")
    else:
        raise ValueError(f"Unsupported chatu8 H3 mode: {mode}")

    def remap(match):
        kind = "audio" if match[1].lower() == "audio" else "image"
        slot = int(match[2])
        if slot not in slots[kind]:
            raise ValueError(f"提示词引用了未启用的素材：{match[0]}")
        label = "Audio" if kind == "audio" else "Picture"
        return f"<{label} {slots[kind][slot]}>"

    prompt = MEDIA_TAG.sub(remap, h3._prompt(prompt))
    return prompt, task, route, active


def short_edge_for(megapixels: float, aspect_ratio: str) -> int:
    # SGLang's adapt_shape_v1 has a 768*1344 soft area budget. Do not pretend
    # an arbitrarily large MP setting is honored while the backend caps it.
    megapixels = float(megapixels)
    if not math.isfinite(megapixels) or not 0.1 <= megapixels <= 1.032192:
        raise ValueError("当前 H3 后端支持 0.1–1.032192 百万像素；更大尺寸需独立超分工作流。")
    if aspect_ratio not in h3.H3_ASPECT_RATIOS or aspect_ratio == "auto":
        raise ValueError("chatu8 H3 requires an explicit supported aspect ratio")
    width, height = map(float, aspect_ratio.split(":"))
    return max(32, round(math.sqrt(megapixels * 1_000_000 * min(width, height) / max(width, height)) / 32) * 32)


class MiniMaxH3Chatu8Asset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Chatu8Asset", display_name="chatu8 H3 Uploaded Asset",
            category="video/chatu8",
            inputs=[
                io.Combo.Input("kind", options=["image", "audio"]),
                io.Int.Input("slot", default=1, min=1, max=9),
                io.Int.Input("enabled", default=2, min=1, max=2,
                             tooltip="chatu8: 1 = connected, 2 = empty. Empty placeholders are never read."),
                io.String.Input("filename", default=""),
            ],
            outputs=[Chatu8Asset.Output("asset")],
        )

    @classmethod
    def fingerprint_inputs(cls, kind, slot, enabled, filename):
        if empty_slot(kind, enabled, filename):
            return "disabled"
        stat = uploaded_path(filename).stat()
        return (stat.st_size, stat.st_mtime_ns)

    @classmethod
    def execute(cls, kind, slot, enabled, filename):
        if empty_slot(kind, enabled, filename):
            return io.NodeOutput(None)
        if enabled != 1 or kind not in {"image", "audio"} or not 1 <= slot <= (9 if kind == "image" else 3):
            raise ValueError("Invalid chatu8 media slot")
        uploaded_path(filename)
        return io.NodeOutput({"kind": kind, "slot": slot, "filename": filename})


class MiniMaxH3Chatu8SGLang(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Chatu8SGLang", display_name="chatu8 H3 · 4-GPU SGLang",
            category="video/chatu8",
            description="Uses existing SGLang workers. This is not T8's beta sampler or Comfy INT8 model.",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("mode", options=["reference", "first_frame"], default="reference"),
                io.Combo.Input("aspect_ratio", options=[x for x in h3.H3_ASPECT_RATIOS if x != "auto"], default="16:9"),
                io.Float.Input("megapixels", default=0.9, min=0.1, max=1.032192, step=0.01),
                io.Float.Input("duration_seconds", default=4, min=4, max=15, step=1),
                io.Int.Input("steps", default=4, min=1, max=100),
                io.Int.Input("seed", default=1101, min=0, max=0x1FFFFFFFFFFFFF, control_after_generate=True),
                io.Combo.Input("lora_name", options=h3._h3_lora_options(), default=TURBO_V4),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.1),
                io.Autogrow.Input("assets", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=Chatu8Asset.Input("asset"), prefix="asset_", min=0, max=12)),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(cls, prompt, mode, aspect_ratio, megapixels, duration_seconds,
                steps, seed, lora_name, lora_strength, assets=None):
        prompt, task, route, active = prepare_request(prompt, mode, assets)
        short_edge = short_edge_for(megapixels, aspect_ratio)
        if not 4 <= float(duration_seconds) <= 15 or not 1 <= int(steps) <= 100:
            raise ValueError("H3 requires 4–15 seconds and 1–100 steps")
        strength = float(lora_strength)
        if not (strength == 0 or 0.1 <= strength <= 2):
            raise ValueError("H3 LoRA strength must be 0 or 0.1–2.0")
        if strength and lora_name not in h3._h3_lora_options():
            raise ValueError(f"H3 LoRA not installed: {lora_name}")
        conditions = []
        for item in active:
            condition = {"type": item["kind"], "uri": stage_asset(item["filename"]).as_uri(), "role": "reference"}
            if task == "fl2va":
                condition.update(role="keyframe", frame_index=0)
            conditions.append(condition)
        model = h3.make_h3_model("h3_ref" if task == "ref2va" else "h3")
        model = h3._apply_lora(model, strength, lora_name)
        payload = h3._payload(model, prompt, task, conditions, aspect_ratio,
                              duration_seconds, steps, "lossless", seed)
        payload["target"]["short_edge"] = short_edge
        h3.load_h3_model(model)
        context = h3.get_executing_context()
        h3.LOG.info("chatu8 H3 route=%s assets=%d steps=%d profile=%s", route, len(active), steps, model.acceleration_profile)
        return h3.run_video_job(route, payload, operation_id=h3._operation_id(""),
                                graph_prompt_id=context.prompt_id if context else "",
                                graph_node_id=context.node_id if context else "")


class MiniMaxH3Chatu8SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Chatu8SaveVideo", display_name="Save H3 Video for chatu8",
            category="video/chatu8", is_output_node=True,
            inputs=[io.Video.Input("video")], outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(cls, video):
        result = PreviewVideo.execute(video)
        native = result.ui.as_dict()
        # chatu8 3.1.2 polls gifs first, and mistakes native images for PNGs.
        # Keep the native preview too, so the same workflow remains usable in Graph.
        return io.NodeOutput(video, ui={**native, "gifs": [
            {**item, "format": "video/mp4"} for item in native["images"]
        ]})
