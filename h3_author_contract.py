"""First-stage author contract: CPU request shaping only.

Tag handling ported from T8 6063faf, h3_t8/prompt_tags.py. No prompt rewriting.
Heavy text/media encoding, diffusion and decoding stay in the SGLang worker.
"""

from __future__ import annotations
import re
import math

AUTHOR_CONTRACT = "comfy_t8_v1"

MEDIA_TAG_RE = re.compile(
    r"<\s*(Image|Picture|Video|Audio)\s*(\d+)\s*>|"
    r"(?<![\w<])(Image|Picture|Video|Audio)\s*#?\s*(\d+)\b(?!\s*>)",
    re.IGNORECASE,
)


def canonicalize_media_tags(prompt: str) -> str:
    def replacement(match: re.Match) -> str:
        media_type = (match.group(1) or match.group(3)).lower()
        ordinal = int(match.group(2) or match.group(4))
        official_type = (
            "Picture" if media_type in {"image", "picture"} else media_type.title()
        )
        return f"<{official_type} {ordinal}>"

    return MEDIA_TAG_RE.sub(replacement, prompt or "")


def prepare_prompt(
    prompt: str,
    counts: dict[str, int],
    source_audio_ordinal: int = 0,
    prompt_primary_audio_ordinal: int = 1,
    strict: bool = True,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    fatal_warnings: list[str] = []
    limits = {
        "picture": max(0, int(counts.get("pictures", 0))),
        "video": max(0, int(counts.get("videos", 0))),
        "audio": max(0, int(counts.get("audios", 0))),
    }

    def add_warning(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    def replacement(match: re.Match) -> str:
        explicit = match.group(1) is not None
        media_type = (match.group(1) or match.group(3)).lower()
        original_ordinal = int(match.group(2) or match.group(4))
        official_type = (
            "Picture" if media_type in {"image", "picture"} else media_type.title()
        )
        limit_key = official_type.lower()
        limit = limits[limit_key]
        ordinal = original_ordinal

        if (
            limit_key == "audio"
            and source_audio_ordinal > 0
            and prompt_primary_audio_ordinal > 0
            and ordinal == int(prompt_primary_audio_ordinal)
        ):
            ordinal = int(source_audio_ordinal)

        if ordinal == 0 and limit >= 1:
            ordinal = 1
            add_warning(
                f"{match.group(0)} used legacy zero-based numbering and was mapped to "
                f"<{official_type} 1>"
            )

        if 1 <= ordinal <= limit:
            return f"<{official_type} {ordinal}>"

        if limit == 1 and ordinal > 1:
            add_warning(
                f"{match.group(0)} was mapped to <{official_type} 1> because exactly one "
                f"{limit_key} is connected"
            )
            return f"<{official_type} 1>"

        message = f"<{official_type} {ordinal}> is not connected; available {limit_key} count is {limit}"
        add_warning(message)
        if strict and (explicit or limit > 0):
            fatal_warnings.append(message)
            return f"<{official_type} {ordinal}>"

        # When no media of this type exists, plain prose such as "Video 2" is
        # common in generated prompts and is not necessarily a reference request.
        # Once at least one same-type item is connected, an ambiguous ordinal still
        # fails in strict mode. Non-strict explicit tags are demoted to prose so
        # Qwen never receives a dangling reference tag.
        return match.group(0) if not explicit else f"{official_type} {ordinal}"

    normalized = MEDIA_TAG_RE.sub(replacement, prompt or "")
    if fatal_warnings:
        raise ValueError(
            "MiniMax H3 prompt media tag validation failed: "
            + "; ".join(dict.fromkeys(fatal_warnings))
            + ". Connect the referenced media, correct the ordinal, or disable "
            "strict_prompt_tags to treat it as plain prompt text."
        )
    return normalized, warnings


def author_request(
    model_payload,
    *,
    prompt,
    width,
    height,
    length,
    steps,
    video_shift,
    audio_shift,
    seed,
    conditions,
    task_type="ref2va",
):
    """No replacement of selected base weights, LoRAs, canvas, or update count."""
    for name, value in (("width", width), ("height", height)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 32
        ):
            raise ValueError(f"{name} must be a positive multiple of 32")
    if isinstance(length, bool) or not isinstance(length, int) or length < 5:
        raise ValueError("length must be a positive H3 frame count")
    frames = length + (5 - length) % 17
    if not 4 <= frames / 24 <= 15.1:
        raise ValueError(
            "author adapter currently supports 4–15 second first-stage requests"
        )
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("steps must be between 1 and 100 updates")
    if any(not math.isfinite(float(v)) or v <= 0 for v in (video_shift, audio_shift)):
        raise ValueError("video/audio shifts must be positive and finite")
    task = "ref2va" if conditions else "t2va"
    if task_type not in {task, "auto"}:
        raise ValueError(f"{task_type} does not match connected reference media")
    counts = {
        "pictures": sum(c["type"] == "image" for c in conditions),
        "audios": sum(c["type"] == "audio" for c in conditions),
        "videos": 0,
    }
    if counts["pictures"] > 9 or counts["audios"] > 3:
        raise ValueError("H3 supports at most 9 images and 3 standalone audios")
    if any(
        c["type"] not in {"image", "audio"} or c["role"] != "reference"
        for c in conditions
    ):
        raise ValueError(
            "this first-stage contract accepts reference images/audios only"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("H3 prompt must not be empty")
    conditioned_prompt, _ = prepare_prompt(prompt, counts, strict=True)
    result = dict(model_payload)
    result.update(
        prompt=conditioned_prompt,
        task=task,
        conditions=conditions,
        target={
            "width": width,
            "height": height,
            "short_edge": min(width, height),
            "aspect_ratio": "auto",
            "duration_seconds": min(15.0, frames / 24),
        },
        seconds=int(min(15, frames / 24)),
        num_inference_steps=steps + 1,
        output_compression=62,  # imageio quality6.2 -> H264 CRF19, yuv420p
        flow_shift=float(video_shift),
        audio_flow_shift=float(audio_shift),
        seed=seed,
        quality="lossless",
        sigma_schedule="comfy_beta",
        noise_layout="comfy_av",
        conditioning_profile="comfy_t8_match",
    )
    return result
