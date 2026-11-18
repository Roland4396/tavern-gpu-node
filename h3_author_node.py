"""Separate author-workflow adapter; never replaces the existing H3 node IDs."""

from __future__ import annotations

from comfy_api.latest import io

from . import h3_sglang_node as h3
from .h3_author_contract import author_request


class MiniMaxH3AuthorSGLang(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AuthorSGLang",
            display_name="H3 Author First Stage · SGLang TP2×U2",
            category="video/chatu8",
            description="Requires the separate BF16 author-contract deployment. All model computation is remote SGLang.",
            inputs=[
                h3.H3FL2VAModel.Input("model"),
                io.String.Input("prompt", default="", multiline=True),
                io.Int.Input("width", default=832, min=32, step=32),
                io.Int.Input("height", default=480, min=32, step=32),
                io.Int.Input("length", default=107, min=5),
                io.Int.Input("steps", default=6, min=1, max=100),
                io.Float.Input("shift_video", default=12.0, min=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01),
                io.Int.Input(
                    "seed",
                    default=1101,
                    min=0,
                    max=0x1FFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Combo.Input(
                    "task_type", options=["ref2va", "t2va", "auto"], default="ref2va"
                ),
                *[
                    io.Image.Input(f"ref_image_{i}", optional=True)
                    for i in range(1, 10)
                ],
                *[io.Audio.Input(f"ref_audio_{i}", optional=True) for i in range(1, 4)],
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        model,
        prompt,
        width,
        height,
        length,
        steps,
        shift_video,
        shift_audio,
        seed,
        task_type="ref2va",
        **references,
    ):
        if model.workload != "h3":
            raise ValueError(
                "Author workflow requires the FL2VA base, not the Ref2VA checkpoint"
            )
        operation_id = h3._operation_id("")
        # Construct/validate the entire request before staging media or starting a job.
        active = [
            (kind, i, references.get(f"ref_{kind}_{i}"))
            for kind, limit in (("image", 9), ("audio", 3))
            for i in range(1, limit + 1)
            if references.get(f"ref_{kind}_{i}") is not None
        ]
        conditions = [
            {"type": kind, "role": "reference", "uri": "file:///pending"}
            for kind, _, _ in active
        ]
        payload = author_request(
            {
                "model": "MiniMaxAI/MiniMax-H3",
                "num_outputs_per_prompt": 1,
                "_h3_turbo_lora": h3._lora_payload(model),
            },
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            steps=steps,
            video_shift=shift_video,
            audio_shift=shift_audio,
            seed=seed,
            conditions=conditions,
            task_type=task_type,
        )
        for condition, (kind, slot, value) in zip(conditions, active):
            identifier = f"{operation_id}-{kind}-{slot}"
            if kind == "image":
                path = h3.save_condition(value, identifier)
            else:
                # FLOAT WAV preserves the original waveform; PCM16 would alter it.
                import soundfile

                local = h3.condition_staging_path(identifier, ".wav")
                soundfile.write(
                    str(local),
                    value["waveform"][0].detach().cpu().T.numpy(),
                    int(value["sample_rate"]),
                    subtype="FLOAT",
                )
                path = h3.publish_condition(
                    local, h3.SHARED_INPUT_DIR / f"{identifier}.wav"
                )
            condition["uri"] = path.as_uri()
        h3.load_h3_model(model)
        context = h3.get_executing_context()
        return h3.run_video_job(
            "/h3/v1/videos",
            payload,
            operation_id=operation_id,
            graph_prompt_id=context.prompt_id if context else "",
            graph_node_id=context.node_id if context else "",
        )


class MiniMaxH3AuthorAVComponents(io.ComfyNode):
    """CPU demux for the author's existing VHS writer; no second VAE/model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AuthorAVComponents",
            display_name="H3 Video → VHS Image/Audio (CPU)",
            category="video/chatu8",
            inputs=[io.Video.Input("video")],
            outputs=[io.Image.Output("images"), io.Audio.Output("audio")],
        )

    @classmethod
    def execute(cls, video):
        components = video.get_components()
        if components.audio is None:
            raise ValueError("H3 author output is missing its generated audio track")
        return io.NodeOutput(components.images, components.audio)


class MiniMaxH3AuthorSaveVideo(io.ComfyNode):
    """Publish the worker's single CRF19 encode without a second lossy encode."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AuthorSaveVideo",
            display_name="Save H3 Author MP4 (no re-encode)",
            category="video/chatu8",
            is_output_node=True,
            inputs=[
                io.Video.Input("video"),
                io.String.Input("filename_prefix", default="MiniMaxH3/stable_4v4a"),
            ],
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(cls, video, filename_prefix="MiniMaxH3/stable_4v4a"):
        import os
        import shutil
        import tempfile
        import folder_paths
        from comfy_api.latest import Types

        width, height = video.get_dimensions()
        directory, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        os.makedirs(directory, exist_ok=True)
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        filename = f"{name}_{counter:05}_.mp4"
        # Finalize/remux locally, never write an MP4 trailer on OSSFS.
        with tempfile.TemporaryDirectory(
            prefix="h3-author-save-", dir=folder_paths.get_temp_directory()
        ) as tmp:
            local = os.path.join(tmp, filename)
            video.save_to(local, format=Types.VideoContainer("mp4"), codec="h264")
            fd, partial = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".partial", dir=directory
            )
            os.close(fd)
            try:
                shutil.copyfile(local, partial)
                os.replace(partial, os.path.join(directory, filename))
            finally:
                if os.path.exists(partial):
                    os.unlink(partial)
        result = {
            "filename": filename,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/mp4",
        }
        return io.NodeOutput(
            video, ui={"gifs": [result], "images": [result], "animated": (True,)}
        )
