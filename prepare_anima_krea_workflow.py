#!/usr/bin/env python3
"""Adapt the locally uploaded workflow without exporting its prompt text."""
import argparse
import copy
import hashlib
import json
import uuid
from pathlib import Path


def adapt(source):
    result = copy.deepcopy(source)
    result['id'] = str(uuid.uuid4())
    result['revision'] = 0
    replacements = {
        3: 'miaomiaoHarem_29BBETA10.safetensors',
        34: 'moodyKrea2Mix_v6seAsianFever4K_3071483.safetensors',
        25: 'qwen3vl_4b_bf16.safetensors',
        28: '4xNomosWebPhoto_RealPLKSR.safetensors',
    }
    by_id = {node['id']: node for node in result['nodes']}
    for node_id, filename in replacements.items():
        by_id[node_id]['widgets_values'][0] = filename
    for node in result['nodes']:
        if 'Lora' in node['type']:
            node['mode'] = 4  # Keep editable LoRA nodes, bypass their execution.
        if node['type'] == 'Text Multiline':
            # Current WAS adds a boolean widget; do not reinterpret prompt braces.
            node['widgets_values'] = [node['widgets_values'][0], False]
    sampler = by_id[7]
    assert sampler['type'] == 'KSampler'
    sampler['widgets_values'][2:4] = [32, 4.0]
    removed = set()
    for inp in sampler.get('inputs', []):
        if inp['name'] == 'cfg' and inp.get('link') is not None:
            removed.add(inp['link'])
            inp['link'] = None
    result['links'] = [edge for edge in result['links'] if edge[0] not in removed]
    for node in result['nodes']:
        for output in node.get('outputs', []):
            if output.get('links') is not None:
                output['links'] = [link for link in output['links'] if link not in removed]
    return result


def prompt_widgets(workflow):
    return {
        node['id']: node['widgets_values'][0]
        for node in workflow['nodes']
        if node['type'] in ('Text Multiline', 'CLIPTextEncode')
        and node.get('widgets_values')
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing adapted workflow')
    raw = args.source.read_bytes()
    source = json.loads(raw)
    result = adapt(source)
    assert prompt_widgets(source) == prompt_widgets(result), 'Prompt text changed'
    assert args.source.read_bytes() == raw, 'Original workflow changed'
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'nodes': len(result['nodes']),
                      'prompts_preserved': True, 'lora': 'bypassed',
                      'original_sha256': hashlib.sha256(raw).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
