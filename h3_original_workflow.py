"""Read-only coverage audit for the original Comfy graph, including bypasses.

This module deliberately has no workflow writer/compiler. It never puts prompt
text, media names, output prefixes, or arbitrary widget values into its report.
Get/Set references are resolved for inspection, not rewritten in the graph.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


MODEL_NODES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
    "MiniMaxH3MemoryEfficientSageAttentionPatch", "MiniMaxH3AudioConditioningT8",
    "MiniMaxH3DualClockSamplerT8", "BasicGuider", "BasicScheduler",
    "SamplerCustomAdvanced", "MiniMaxH3AVDecodeT8", "VAEEncode", "VAEDecode",
    "VAEEncodeAudio", "VAEDecodeAudio", "PT_H3ConcatAVLatent",
}
OUTPUT_NODES = {"VHS_VideoCombine", "solarL_SaveImagesToZip", "SaveImage"}
CONTROL_NODES = {
    "LoadImage", "LoadAudio", "PrimitiveFloat", "ComfyMathExpression",
    "ResolutionSelector", "Switch", "RandomNoise", "KSamplerSelect",
    "GetNode", "SetNode", "easy cleanGpuUsed", "ImageFromBatch", "ImageScale",
}


def audit_original(graph: dict) -> dict:
    """Inventory EVERY node; never prune an inactive or unconnected branch."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate node IDs")
    links = {link[0]: link for link in graph["links"]}
    if len(links) != len(graph["links"]):
        raise ValueError("Duplicate link IDs")
    setters = {}
    for node in nodes.values():
        if node["type"] == "SetNode":
            name = tuple(node.get("widgets_values", []))
            setters.setdefault(name, []).append(node["id"])

    def input_source(node_id, input_name):
        inp = next(i for i in nodes[node_id].get("inputs", []) if i["name"] == input_name)
        if inp.get("link") is None:
            return None
        link = links[inp["link"]]
        return resolve(link[1], link[2], set())

    def resolve(node_id, slot, seen):
        if node_id in seen:
            raise ValueError("Cycle in virtual wiring")
        node = nodes[node_id]
        if node["type"] not in {"GetNode", "SetNode"}:
            return [node_id, slot]
        seen = seen | {node_id}
        if node["type"] == "GetNode":
            choices = setters.get(tuple(node.get("widgets_values", [])), [])
            if len(choices) != 1:
                raise ValueError(f"Ambiguous/missing SetNode for GetNode {node_id}")
            return resolve(choices[0], 0, seen)
        connected = [p for p in node.get("inputs", []) if p.get("link") is not None]
        if len(connected) != 1:
            raise ValueError(f"SetNode {node_id} must have one source")
        link = links[connected[0]["link"]]
        return resolve(link[1], link[2], seen)

    rows = []
    for node in nodes.values():
        kind = node["type"]
        inputs = []
        for index, inp in enumerate(node.get("inputs", [])):
            link_id = inp.get("link")
            if link_id is None:
                continue
            link = links[link_id]
            if link[3:5] != [node["id"], index] or link[1] not in nodes:
                raise ValueError(f"Broken link {link_id}")
            inputs.append({"name": inp["name"], "source": [link[1], link[2]],
                           "resolved_source": resolve(link[1], link[2], set())})
        route = (
            "sglang_model_boundary" if kind in MODEL_NODES else
            "native_output" if kind in OUTPUT_NODES else
            "native_control" if kind in CONTROL_NODES else
            "sglang_vfx_boundary" if kind == "ImageResizeKJv2" else "unclassified"
        )
        rows.append({"id": node["id"], "type": kind, "mode": node.get("mode", 0),
                     "route_required": route, "inputs": inputs,
                     "outputs": [{"slot": i, "type": o["type"]}
                                 for i, o in enumerate(node.get("outputs", []))]})
    # Trace model ancestry without treating graph order as model state. In this
    # workflow stage 2 branches BEFORE LoRAs and BEFORE the dual-clock patch.
    chains = []
    for node in nodes.values():
        if node["type"] != "SamplerCustomAdvanced":
            continue
        guide = input_source(node["id"], "guider")
        if not guide or nodes[guide[0]]["type"] != "BasicGuider":
            raise ValueError("Sampler has an unsupported guider in coverage audit")
        current = input_source(guide[0], "model")
        chain, seen = [], set()
        while current:
            n = nodes[current[0]]
            if n["id"] in seen:
                raise ValueError("Cycle in model lineage")
            seen.add(n["id"])
            row = {"id": n["id"], "type": n["type"]}
            if n["type"] == "LoraLoaderModelOnly":
                row["strength"] = n["widgets_values"][1]
            chain.append(row)
            current = input_source(n["id"], "model") if any(
                p["name"] == "model" for p in n.get("inputs", [])
            ) else None
        chains.append({"sampler_id": node["id"], "model_chain": list(reversed(chain)),
                       "sigmas_source": input_source(node["id"], "sigmas"),
                       "latent_source": input_source(node["id"], "latent_image")})
    return {
        "scope": "original graph including disabled branches; audit is not implementation",
        "node_count": len(rows), "mode_counts": dict(Counter(r["mode"] for r in rows)),
        "nodes": sorted(rows, key=lambda r: r["id"]), "sampling_chains": chains,
        "unclassified": [r["id"] for r in rows if r["route_required"] == "unclassified"],
    }


def audit_file(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    return {"source_sha256": hashlib.sha256(raw).hexdigest(),
            **audit_original(json.loads(raw))}
