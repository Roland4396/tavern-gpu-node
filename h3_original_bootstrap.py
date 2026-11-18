# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit deployment hooks; no activation without an original-node manifest."""
import asyncio
from contextlib import contextmanager, nullcontext
import http.client
import json
from pathlib import Path
import socket
import sys
import threading
import uuid
from types import SimpleNamespace


@contextmanager
def original_residency_request(pipeline, server_args):
    """Use SGLang's normal residency lifecycle without running its video graph.

    Pipeline.forward normally initializes this manager. Original Comfy nodes
    bypass forward intentionally, but must not bypass component ownership.
    Their component order is dynamic, so there is no speculative stage plan.
    """
    import torch
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import get_global_component_residency_manager
    manager = get_global_component_residency_manager(pipeline, server_args)
    pipeline.component_residency_manager = manager
    pipeline.executor.component_residency_manager = manager
    # Native Comfy nodes can normalize a decoded inference tensor in-place
    # (e.g. vae_decode_audio). no_grad alone does not permit that operation.
    with torch.inference_mode():
        manager.begin_request([], SimpleNamespace(is_warmup=False), server_args)
        try:
            yield
        finally:
            manager.finish_request()


def worker_executor_factory(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())

    def build(worker):
        import torch
        from h3_original_executor import SGLangOriginalExecutor
        args = worker.server_args
        if (args.num_gpus, args.tp_size, args.ulysses_degree, args.dp_size) != (4, 2, 2, 1):
            raise ValueError("Original workflow deployment requires TP2 x Ulysses2 on four GPUs")
        sys.path.insert(0, manifest["core"])
        import nodes
        import server
        from h3_original_router import OUTPUT_TYPES
        loop = asyncio.new_event_loop()
        if getattr(server.PromptServer, "instance", None) is None:
            server.PromptServer(loop)  # imports only; no HTTP/WS listener

        async def load():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
            for path in manifest["worker_node_paths"]:
                if not await nodes.load_custom_node(path, module_parent="custom_nodes"):
                    raise RuntimeError(f"Original package import failed: {Path(path).name}")
        loop.run_until_complete(load())
        registry = {name: nodes.NODE_CLASS_MAPPINGS[name] for name in OUTPUT_TYPES}
        # The registered original class identifies its actually imported module;
        # no AST extraction or second copy of KJ quantization logic.
        sage_class = registry["MiniMaxH3MemoryEfficientSageAttentionPatch"]
        kj_module = sys.modules[sage_class.__module__]
        helper = kj_module._sageattn_int8_fp8_nhd

        def resolve_lora(name):
            path = Path(manifest["loras"][name])
            if not path.is_file():
                raise FileNotFoundError("Manifest LoRA is unavailable")
            return path

        # Private VFX libraries are selected BEFORE process startup via the
        # runtime manifest. Loading here verifies binding availability only;
        # admission still requires original-node GPU acceptance.
        vsr = None
        if manifest.get("enable_original_vsr"):
            import nvvfx
            vsr = nullcontext
        executor = SGLangOriginalExecutor(worker.pipeline, registry, manifest["assets"],
            device=torch.device("cuda", worker.local_rank), sequence_parallel_size=args.sp_degree,
            kj_helper=helper, lora_resolver=resolve_lora, vsr_context=vsr,
            request_context=lambda: original_residency_request(worker.pipeline, args))
        executor._registry_event_loop = loop
        return executor, manifest["ipc_root"]
    return build


def install_manifest_catalogue(folder_paths, manifest):
    """Advertise explicitly routed assets without fake weight files or downloads.

    Filenames retained in the original canvas are aliases; actual component
    precision is declared by the deployment manifest, not guessed from names.
    Other model lists and their native loading paths remain unchanged.
    """
    original = folder_paths.get_filename_list
    aliases = {}
    for category, node in (("diffusion_models", "UNETLoader"), ("unet", "UNETLoader"),
                           ("text_encoders", "CLIPLoader"), ("clip", "CLIPLoader"),
                           ("vae", "VAELoader")):
        aliases[category] = set(manifest["assets"].get(node, {}))
    aliases["loras"] = set(manifest.get("loras", {}))

    def filenames(category):
        values = original(category)
        if category not in aliases:
            return values
        return sorted(set(values) | aliases[category])
    folder_paths.get_filename_list = filenames
    return original


async def load_graph_packages(nodes, manifest):
    """Load the pinned original packages, including their frontend extensions."""
    for path in manifest.get("graph_node_paths", ()):
        if not await nodes.load_custom_node(path, module_parent="custom_nodes"):
            raise RuntimeError(f"Original Graph package import failed: {Path(path).name}")


class AgentActivity:
    """Use the existing node-agent GPU lease and request heartbeat."""
    def __init__(self, socket_path, workload):
        self.socket_path, self.workload = socket_path, workload

    def _request(self, method, path, data=None, timeout=30):
        connection = http.client.HTTPConnection("localhost", timeout=timeout)
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(timeout)
        connection.sock.connect(self.socket_path)
        try:
            connection.request(method, path, json.dumps(data) if data is not None else None,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            result = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(f"H3 node agent {method} {path} failed: HTTP {response.status}")
            return result
        finally:
            connection.close()

    def _post(self, action, data, timeout=30):
        return self._request("POST", f"/workloads/{self.workload}/{action}", data, timeout)

    @staticmethod
    def epoch(state):
        if state.get("status") == "running":
            return (state["pid"], state["process_start_ticks"])
        return None

    def current_epoch(self):
        status = self._request("GET", "/status", timeout=5)
        return self.epoch(status["workloads"][self.workload])

    @contextmanager
    def __call__(self):
        ticket, stopped, failure = uuid.uuid4().hex, threading.Event(), []
        state = self._post("ensure", {"ticket": ticket}, timeout=1900)

        def heartbeat():
            while not stopped.wait(30):
                try: self._post("activity", {"ticket": ticket, "action": "heartbeat"})
                except Exception as exc:
                    failure.append(exc)
                    return
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield self.epoch(state)
        finally:
            stopped.set()
            thread.join(timeout=35)
            self._post("activity", {"ticket": ticket, "action": "end"})
        if failure:
            raise RuntimeError("H3 request heartbeat failed") from failure[0]


def install_graph_startup(manifest_path, *, activity_context=None):
    """Call before main.py; register proxies only AFTER original discovery."""
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    from h3_original_ipc import OriginalBackendClient, ZmqOriginalTransport
    from h3_original_router import install_routes
    manifest = json.loads(Path(manifest_path).read_text())
    original_init = nodes.init_extra_nodes

    async def init(*args, **kwargs):
        result = await original_init(*args, **kwargs)
        await load_graph_packages(nodes, manifest)
        import folder_paths
        import comfy.model_management
        import comfy.utils
        last_progress = [None]

        def poll(directory):
            comfy.model_management.throw_exception_if_processing_interrupted()
            path = directory / "progress.json"
            if path.is_file():
                progress = json.loads(path.read_text())
                marker = (directory, progress["value"], progress["max"])
                if marker != last_progress[0]:
                    comfy.utils.ProgressBar(progress["max"]).update_absolute(progress["value"])
                    last_progress[0] = marker

        transport = ZmqOriginalTransport(manifest["scheduler_endpoint"], manifest["ipc_root"], poll_hook=poll)
        client = OriginalBackendClient(transport, manifest["ipc_root"], manifest["assets"],
            activity_context=activity_context if activity_context is not None else
                AgentActivity(manifest["agent_socket"], manifest["workload"]))
        install_routes(nodes.NODE_CLASS_MAPPINGS, client)
        install_manifest_catalogue(folder_paths, manifest)
        nodes._h3_original_backend = client
        if manifest.get("output_publish_root"):
            import server
            from graph_output_publish import install_output_publisher
            nodes._h3_output_publisher = install_output_publisher(
                server.PromptServer.instance, folder_paths.get_output_directory(),
                manifest["output_publish_root"])
        return result
    nodes.init_extra_nodes = init
