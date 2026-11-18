from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import folder_paths
from comfy_api.latest import Input, InputImpl, io
from comfy_execution.graph_utils import GraphBuilder

H3_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
MAX_SEED = 0x7FFFFFFFFFFFFFFF
H3_DEFAULT_STEPS = 50

H3AssetCatalog = io.Custom("H3_ASSET_CATALOG")
H3FL2VAModel = io.Custom("H3_FL2VA_MODEL")
H3SequenceSettings = io.Custom("H3_SEQUENCE_SETTINGS")
SeedVR3Model = io.Custom("SEEDVR3_MODEL")
SeedVR3Settings = io.Custom("SEEDVR3_SETTINGS")


def parse_h3_prompt_bundle(bundle_text: str) -> dict[str, Any]:
    """Parse the fixed 15-second bundle format without changing prompt text."""
    text = str(bundle_text or "")
    duration_match = re.search(
        r"^SEGMENT_DURATION_SECONDS:\s*(\d+(?:\.\d+)?)\s*$", text, re.MULTILINE
    )
    if duration_match is None or float(duration_match.group(1)) != 15:
        raise ValueError("H3 bundle must declare SEGMENT_DURATION_SECONDS: 15")

    block_pattern = re.compile(
        r"^--- SEGMENT (\d+) ---\n(.*?)^--- END SEGMENT \1 ---$",
        re.MULTILINE | re.DOTALL,
    )
    blocks = list(block_pattern.finditer(text))
    if not blocks:
        raise ValueError("H3 bundle contains no segments")

    scenes: list[dict[str, Any]] = []
    scene: dict[str, Any] | None = None
    previous_clip_id: str | None = None
    for expected_id, match in enumerate(blocks, start=1):
        segment_id = int(match.group(1))
        if segment_id != expected_id:
            raise ValueError(
                f"H3 bundle segment IDs must be contiguous; expected {expected_id:03d}, got {segment_id:03d}"
            )
        body = match.group(2)
        continuity_match = re.search(r"^CONTINUITY:\s*(\S+)\s*$", body, re.MULTILINE)
        duration_match = re.search(r"^DURATION_SECONDS:\s*(\S+)\s*$", body, re.MULTILINE)
        first_frame_match = re.search(r"^FIRST_FRAME:\s*(\S+)\s*$", body, re.MULTILINE)
        if not all((continuity_match, duration_match, first_frame_match)):
            raise ValueError(f"segment {segment_id:03d} is missing continuity metadata")
        continuity = continuity_match.group(1)
        duration = float(duration_match.group(1))
        first_frame = first_frame_match.group(1)
        if continuity not in {"START", "NEW", "CONTINUOUS"}:
            raise ValueError(f"segment {segment_id:03d} has unsupported CONTINUITY: {continuity}")
        if duration != 15:
            raise ValueError(f"segment {segment_id:03d} must be exactly 15 seconds")
        expected_first = "PREVIOUS_LAST_FRAME" if continuity == "CONTINUOUS" else "NONE"
        if first_frame != expected_first:
            raise ValueError(
                f"segment {segment_id:03d} has FIRST_FRAME={first_frame}; expected {expected_first} for {continuity}"
            )
        prompt_match = re.search(r"^PROMPT_BEGIN\n(.*?)\nPROMPT_END$", body, re.MULTILINE | re.DOTALL)
        if prompt_match is None or not prompt_match.group(1).strip():
            raise ValueError(f"segment {segment_id:03d} has an empty prompt")

        if continuity in {"START", "NEW"}:
            scene = {
                "scene_id": f"scene_{len(scenes) + 1:03d}",
                "scene_summary": f"Bundle scene beginning at segment {segment_id:03d}",
                "canonical_reference_ids": [],
                "clips": [],
            }
            scenes.append(scene)
            previous_clip_id = None
        if scene is None:
            raise ValueError(f"segment {segment_id:03d} starts without START or NEW")

        clip_id = f"scene_{len(scenes):03d}_clip_{len(scene['clips']) + 1:03d}"
        scene["clips"].append({
            "clip_id": clip_id,
            "parent_clip_id": previous_clip_id,
            "scene_continues": True,
            "continue_from_previous": previous_clip_id is not None,
            "continuation_reason": (
                "Direct continuation from the preceding segment exact last frame."
                if previous_clip_id
                else ""
            ),
            "generation_branch": "base",
            "target_duration_sec": 15.0,
            "first_frame_asset_id": None,
            "last_frame_asset_id": None,
            "reference_assets": [],
            "h3_prompt": prompt_match.group(1),
            "beats": [f"Complete bundle segment {segment_id:03d}."],
            "reserved_beats": [],
        })
        previous_clip_id = clip_id

    for item in scenes:
        item["clips"][-1]["scene_continues"] = False
    return {
        "schema_version": "1.0",
        "project_summary": "H3 fixed 15-second prompt bundle",
        "chain_across_scene_boundaries": False,
        "scenes": scenes,
    }


class H3SequenceSettingsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SequenceSettings",
            display_name="H3 Sequence Generation Settings",
            category="H3/sequence/generation",
            inputs=[
                io.Combo.Input("aspect_ratio", options=H3_ASPECT_RATIOS, default="16:9"),
                io.Int.Input(
                    "steps", default=H3_DEFAULT_STEPS, min=1, max=100, step=1
                ),
                io.Combo.Input("quality", options=["lossless"], default="lossless"),
                io.Int.Input(
                    "seed",
                    default=1101,
                    min=0,
                    max=MAX_SEED,
                    control_after_generate=True,
                ),
                # Keep this appended after seed and its control widget so older
                # saved workflows retain their existing widget positions.
                io.Boolean.Input("enable_seedvr2", default=False),
            ],
            outputs=[H3SequenceSettings.Output(display_name="settings")],
        )

    @classmethod
    def execute(
        cls,
        aspect_ratio: str,
        steps: int,
        quality: str,
        seed: int,
        enable_seedvr2: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput({
            "aspect_ratio": aspect_ratio,
            "steps": int(steps),
            "quality": quality,
            "enable_seedvr2": bool(enable_seedvr2),
            "seed": int(seed),
        })


class H3BundleSequenceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3BundleSequence",
            display_name="H3 TXT Bundle Sequence",
            category="H3/sequence/generation",
            description=(
                "Runs a fixed 15-second H3 prompt bundle locally. START/NEW begin a "
                "fresh frame chain; CONTINUOUS inherits the previous exact last frame."
            ),
            inputs=[
                io.String.Input("bundle_text", multiline=True, force_input=True),
                H3FL2VAModel.Input("fl2va_model"),
                H3SequenceSettings.Input("h3_settings"),
            ],
            outputs=[io.Video.Output("video")],
            enable_expand=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(
        cls,
        bundle_text: str,
        fl2va_model: Any,
        h3_settings: dict[str, Any],
    ) -> io.NodeOutput:
        plan = parse_h3_prompt_bundle(bundle_text)
        graph = GraphBuilder()
        sequence = graph.node(
            "AISequenceDynamicWiring",
            normalized_plan=json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
            asset_catalog={"assets": {}},
            fl2va_model=fl2va_model,
            h3_settings=h3_settings,
        )
        return io.NodeOutput(sequence.out(0), expand=graph.finalize())


class SequenceVideoConcatNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SequenceVideoConcat",
            display_name="Sequence Video Concatenate (Lossless)",
            category="H3/sequence/video",
            inputs=[io.Video.Input("video_1")],
            outputs=[io.Video.Output("video")],
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, video_1: Input.Video, **kwargs: Any) -> io.NodeOutput:
        videos: dict[int, Input.Video] = {1: video_1}
        for name, value in kwargs.items():
            if not name.startswith("video_") or not name[6:].isdigit():
                raise ValueError(f"unexpected sequence concat input: {name}")
            videos[int(name[6:])] = value
        expected = list(range(1, len(videos) + 1))
        if sorted(videos) != expected:
            raise ValueError(f"sequence concat inputs must be contiguous: expected {expected}")

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to concatenate sequence clips")

        run_id = uuid.uuid4().hex
        staging = Path(folder_paths.get_temp_directory()) / f"sequence-concat-{run_id}"
        output_dir = Path(folder_paths.get_temp_directory()) / "sequence-output"
        output = output_dir / f"{run_id}.mp4"
        staging.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            paths: list[Path] = []
            for index in expected:
                path = staging / f"clip-{index:04d}.mp4"
                videos[index].save_to(str(path))
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"sequence clip {index} serialization produced an empty file")
                paths.append(path)

            manifest = staging / "clips.ffconcat"
            manifest.write_text(
                "ffconcat version 1.0\n"
                + "".join(f"file '{path.as_posix()}'\n" for path in paths),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(manifest),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                output.unlink(missing_ok=True)
                detail = result.stderr.strip()[-4000:]
                raise RuntimeError(f"sequence video concatenation failed: {detail}")
            if not output.is_file() or output.stat().st_size <= 0:
                raise RuntimeError("sequence video concatenation returned an empty file")
            return io.NodeOutput(InputImpl.VideoFromFile(str(output)))
        finally:
            shutil.rmtree(staging, ignore_errors=True)


class H3VideoLastFrameNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VideoLastFrame",
            display_name="H3 Video Last Frame",
            category="H3/sequence/video",
            inputs=[
                io.Video.Input("video"),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
    ) -> io.NodeOutput:
        frame_rate = float(video.get_frame_rate())
        if frame_rate <= 0:
            raise ValueError("H3 video frame rate must be positive")
        duration = float(video.get_duration())
        window = min(duration, max(2.0 / frame_rate, 0.1))
        tail = video.as_trimmed(
            start_time=-window,
            duration=window,
            strict_duration=False,
        )
        components = (tail or video).get_components()
        if components.images.shape[0] < 1:
            raise RuntimeError("H3 video does not contain a decodable last frame")
        return io.NodeOutput(components.images[-1:].contiguous())


class AISequenceDynamicWiringNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AISequenceDynamicWiring",
            display_name="AI Sequence Dynamic Wiring",
            category="H3/sequence/generation",
            description="Expands an AI plan into ordered last-frame-linked FL2VA clips, optional upscale, and concat nodes.",
            inputs=[
                io.String.Input("normalized_plan", force_input=True),
                H3AssetCatalog.Input("asset_catalog"),
                H3FL2VAModel.Input("fl2va_model"),
                H3SequenceSettings.Input("h3_settings"),
                SeedVR3Model.Input("seedvr3_model", optional=True),
                SeedVR3Settings.Input("seedvr3_settings", optional=True),
            ],
            outputs=[io.Video.Output("video")],
            enable_expand=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(
        cls,
        normalized_plan: str,
        asset_catalog: dict[str, Any],
        fl2va_model: Any,
        h3_settings: dict[str, Any],
        seedvr3_model: Any = None,
        seedvr3_settings: Any = None,
    ) -> io.NodeOutput:
        plan = json.loads(normalized_plan)
        scenes = plan.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("sequence plan contains no scenes")

        aspect_ratio = str(h3_settings.get("aspect_ratio") or "")
        steps = int(h3_settings.get("steps") or 0)
        quality = str(h3_settings.get("quality") or "")
        enable_seedvr2 = bool(h3_settings.get("enable_seedvr2", False))
        base_seed = int(h3_settings.get("seed") or 0)
        if aspect_ratio not in H3_ASPECT_RATIOS:
            raise ValueError(f"unsupported H3 sequence aspect ratio: {aspect_ratio}")
        if not 1 <= steps <= 100:
            raise ValueError("H3 sequence steps must be in [1, 100]")
        if quality != "lossless":
            raise ValueError(f"unsupported H3 sequence quality: {quality}")
        if enable_seedvr2 and (seedvr3_model is None or seedvr3_settings is None):
            raise ValueError(
                "SeedVR2 is enabled, but its model or settings input is not connected"
            )

        graph = GraphBuilder()
        pending_upscales: list[list[Any]] = []
        raw_clips: list[list[Any]] = []
        previous_last_frame: list[Any] | None = None
        ordinal = 0
        chain_across_scene_boundaries = bool(
            plan.get("chain_across_scene_boundaries", True)
        )

        for scene_index, scene in enumerate(scenes, start=1):
            if scene_index > 1 and not chain_across_scene_boundaries:
                previous_last_frame = None
            clips = scene.get("clips") if isinstance(scene, dict) else None
            if not isinstance(clips, list) or not clips:
                raise ValueError(f"sequence scene {scene_index} contains no clips")
            for clip_index, clip in enumerate(clips, start=1):
                if not isinstance(clip, dict):
                    raise ValueError(f"sequence scene {scene_index} clip {clip_index} is invalid")
                ordinal += 1
                branch = str(clip.get("generation_branch") or "")
                if branch != "base":
                    raise ValueError(
                        f"sequence scene {scene_index} clip {clip_index} must use the base branch"
                    )

                prefix = f"clip-{ordinal:04d}"
                resolver = graph.node(
                    "H3ConditionResolver",
                    id=f"{prefix}-resolve",
                    clip_json=json.dumps(clip, ensure_ascii=False, separators=(",", ":")),
                    asset_catalog=asset_catalog,
                )

                duration = float(clip["target_duration_sec"])
                prompt = str(clip["h3_prompt"])
                condition_inputs: dict[str, Any] = {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration_seconds": duration,
                    "condition_bundle": resolver.out(0),
                }
                if previous_last_frame is not None:
                    # The execution-derived frame always wins over a planner
                    # supplied opening frame after the first project clip.
                    condition_inputs["first_frame"] = previous_last_frame
                condition = graph.node(
                    "MiniMaxH3FL2VACondition",
                    id=f"{prefix}-base-condition",
                    **condition_inputs,
                )
                sampler_inputs: dict[str, Any] = {
                    "model": fl2va_model,
                    "conditioning": condition.out(0),
                    "steps": steps,
                    "quality": quality,
                    "seed": (base_seed + ordinal - 1) & MAX_SEED,
                }
                sampler = graph.node(
                    "MiniMaxH3FL2VASampler",
                    id=f"{prefix}-base-sample",
                    **sampler_inputs,
                )

                raw_video = sampler.out(0)
                raw_clips.append(raw_video)
                if enable_seedvr2:
                    submitted = graph.node(
                        "SeedVR3Submit",
                        id=f"{prefix}-upscale-submit",
                        model=seedvr3_model,
                        video=raw_video,
                        settings=seedvr3_settings,
                        seed=(base_seed + ordinal - 1) & MAX_SEED,
                    )
                    pending_upscales.append(submitted.out(0))
                last_frame = graph.node(
                    "H3VideoLastFrame",
                    id=f"{prefix}-last-frame",
                    video=raw_video,
                )
                previous_last_frame = last_frame.out(0)

        if not raw_clips:
            raise ValueError("sequence plan contains no clips")

        output_clips = raw_clips
        if enable_seedvr2:
            output_clips = []
            for index, job in enumerate(pending_upscales, start=1):
                completed = graph.node(
                    "SeedVR3Await",
                    id=f"clip-{index:04d}-upscale-await",
                    job=job,
                )
                output_clips.append(completed.out(0))

        if len(output_clips) == 1:
            final_video = output_clips[0]
        else:
            concat = graph.node(
                "SequenceVideoConcat",
                id="final-concat",
                **{
                    f"video_{index}": value
                    for index, value in enumerate(output_clips, start=1)
                },
            )
            final_video = concat.out(0)
        return io.NodeOutput(final_video, expand=graph.finalize())
