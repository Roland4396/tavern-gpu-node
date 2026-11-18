# SPDX-License-Identifier: GPL-3.0-or-later
"""Run original node operations in SGLang's existing all-rank scheduler order."""
from contextlib import contextmanager
import json
import logging
import traceback

from h3_original_ipc import OriginalNodeRequest, OriginalObjectStore, TensorCodec, call_directory


class OriginalNodeDispatcher:
    def __init__(self, executor, root, *, rank=0, collect_errors=None):
        self.executor, self.root, self.rank = executor, root, rank
        self.collect_errors = collect_errors or (lambda error: [error])
        self.sessions = {}

    def _collect(self, action):
        value, error = None, None
        try:
            value = action()
        except Exception as exc:
            # Do not put prompts, media, file contents or exception arguments
            # into scheduler logs / protocol errors. Keep only class + phase.
            error = type(exc).__name__
            frames = [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]
            logging.error("original_node_failure rank=%s type=%s frames=%s", self.rank, error, frames)
        errors = self.collect_errors(error)
        if any(errors):
            raise RuntimeError(f"Original-node operation failed on ranks: {errors}")
        return value

    @contextmanager
    def _progress(self, directory):
        import comfy.utils
        old = comfy.utils.PROGRESS_BAR_HOOK

        def hook(value, total, preview=None, **kwargs):
            # A single rank observes the filesystem, then shares cancellation
            # before any rank enters the next model collective.
            cancel = directory.joinpath("cancel").exists() if self.rank == 0 else False
            if any(self.collect_errors("cancelled" if cancel else None)):
                raise InterruptedError("Original-node request cancelled")
            if self.rank == 0:
                temporary = directory / "progress.tmp"
                temporary.write_text(json.dumps({"value": int(value), "max": int(total)}))
                temporary.replace(directory / "progress.json")
        comfy.utils.set_progress_bar_global_hook(hook)
        try: yield
        finally: comfy.utils.set_progress_bar_global_hook(old)

    def handle(self, request):
        directory = call_directory(self.root, request.session, request.call)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if request.action == "open":
            self.sessions.setdefault(request.session, OriginalObjectStore(request.session))
            return {"capabilities": self.executor.capabilities, "assets": self.executor.assets}
        if request.action == "close":
            self.sessions.pop(request.session, None)
            return {"closed": True}
        if request.action != "invoke":
            raise ValueError("Unknown original-node operation")

        def prepare():
            store = self.sessions[request.session]
            kinds = self.executor.capabilities[request.node_type]
            codec = TensorCodec(directory)
            inputs = codec.unpack(request.payload, "inputs.pt", resolve_handle=store.resolve)
            return store, kinds, codec, inputs

        store, kinds, codec, inputs = self._collect(prepare)
        with self._progress(directory):
            outputs = self._collect(lambda: self.executor.invoke(request.node_type, inputs))
        exported = self._collect(lambda: store.export(request.call, kinds, outputs))
        # Only reply rank writes large CPU outputs. Other ranks still retain
        # model/conditioning objects and join the same completion barrier.
        result = self._collect(lambda: codec.pack(exported, "outputs.pt") if self.rank == 0 else None)
        return result


def install_scheduler_handler(scheduler, executor_factory):
    """Explicit opt-in hook. SGLang recv_reqs already fans out to TP/SP ranks."""
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group

    if scheduler.server_args.dp_size != 1:
        raise ValueError("Original Comfy object sessions require one SGLang DP replica")
    dispatcher = None
    group = get_world_group().cpu_group

    def collect(error):
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        return errors

    def handler(requests):
        nonlocal dispatcher
        if dispatcher is None:
            error = None
            try:
                executor, root = executor_factory(scheduler.worker)
            except Exception as exc:
                error = type(exc).__name__
                frames = [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]
                logging.error("original_registry_failure rank=%s type=%s frames=%s", scheduler.gpu_id, error, frames)
            errors = collect(error)
            if any(errors):
                return OutputBatch(error=f"Original-node worker initialization failed: {errors}")
            dispatcher = OriginalNodeDispatcher(executor, root, rank=scheduler.gpu_id, collect_errors=collect)
        return OutputBatch(output=dispatcher.handle(requests[0]))

    scheduler.request_handlers[OriginalNodeRequest] = handler
