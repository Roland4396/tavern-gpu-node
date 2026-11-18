import os
import shutil
import tempfile

import folder_paths
from comfy_api.latest import ComfyExtension, Input, Types, io, ui
from typing_extensions import override


class PreviewVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewVideo",
            display_name="Preview Video",
            category="video",
            description=(
                "Displays a video in the UI and keeps it available across browser refreshes "
                "until the Graph workload is released."
            ),
            inputs=[io.Video.Input("video", tooltip="The video to preview.")],
            is_output_node=True,
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        width, height = video.get_dimensions()
        output_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "preview/ComfyUI",
            folder_paths.get_output_directory(),
            width,
            height,
        )
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension('auto')}"
        output_path = os.path.join(output_dir, file)
        os.makedirs(output_dir, exist_ok=True)

        # The shared Graph output path is an object-storage FUSE mount.  PyAV
        # can write the MP4 packets there, but its final trailer write fails on
        # that filesystem.  Encode on the local filesystem first, then publish
        # the finished file into the shared output directory.
        temp_root = folder_paths.get_temp_directory()
        os.makedirs(temp_root, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="preview-", dir=temp_root) as work_dir:
            local_path = os.path.join(work_dir, file)
            video.save_to(
                local_path,
                format=Types.VideoContainer("auto"),
                codec="auto",
            )

            staging_fd, staging_path = tempfile.mkstemp(
                prefix=f".{file}.", suffix=".partial", dir=output_dir
            )
            os.close(staging_fd)
            try:
                shutil.copyfile(local_path, staging_path)
                os.replace(staging_path, output_path)
            finally:
                try:
                    os.unlink(staging_path)
                except FileNotFoundError:
                    pass
        return io.NodeOutput(
            video,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )


class PreviewVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PreviewVideo]


async def comfy_entrypoint() -> PreviewVideoExtension:
    return PreviewVideoExtension()
