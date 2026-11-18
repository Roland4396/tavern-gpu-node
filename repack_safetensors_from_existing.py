#!/usr/bin/env python3
"""Repack existing shard bytes into a pinned canonical safetensors artifact.

No tensor math, dtype changes, Torch, or GPU initialization. The complete output
must match the published SHA-256 before it can enter the shared model library.
"""
import argparse
import hashlib
import json
import os
import struct
import urllib.request
from contextlib import ExitStack
from pathlib import Path


def read_header(handle):
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError('Incomplete safetensors length')
    length = struct.unpack('<Q', prefix)[0]
    if not 0 < length <= 8 * 1024 * 1024:
        raise ValueError('Unexpected header size')
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError('Incomplete safetensors header')
    return prefix + raw, json.loads(raw)


def source_name(name, mapping):
    if mapping == 'qwen3vl':
        return name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else 'model.' + name
    return name


def repack(source, canonical_header, target, mapping, expected_hash, expected_size):
    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    header = json.loads(canonical_header[8:])
    tensors = sorted(((k, v) for k, v in header.items() if k != '__metadata__'), key=lambda kv: kv[1]['data_offsets'][0])
    with ExitStack() as stack:
        handles, headers = {}, {}
        plan = []
        position = 0
        for name, info in tensors:
            key = source_name(name, mapping)
            shard = index[key]
            if Path(shard).name != shard:
                raise ValueError('Shard name must be a plain filename')
            if shard not in handles:
                handles[shard] = stack.enter_context((source / shard).open('rb'))
                raw, decoded = read_header(handles[shard])
                headers[shard] = (len(raw), decoded)
            base, decoded = headers[shard]
            original = decoded[key]
            begin, end = info['data_offsets']
            old_begin, old_end = original['data_offsets']
            if info['dtype'] != original['dtype'] or info['shape'] != original['shape']:
                raise ValueError('Shape/dtype mismatch: ' + name)
            if begin != position or end - begin != old_end - old_begin:
                raise ValueError('Non-contiguous or mismatched tensor range: ' + name)
            plan.append((name, handles[shard], base + old_begin, end - begin))
            position = end
        if len(canonical_header) + position != expected_size:
            raise ValueError('Canonical size mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with target.open('wb') as out:
            out.write(canonical_header)
            digest.update(canonical_header)
            for number, (name, handle, offset, count) in enumerate(plan):
                handle.seek(offset)
                while count:
                    block = handle.read(min(count, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError('Truncated source tensor: ' + name)
                    count -= len(block)
                    out.write(block)
                    digest.update(block)
                if number % 50 == 0:
                    print(json.dumps({'tensors': number + 1, 'total': len(plan), 'bytes': out.tell()}), flush=True)
            out.flush()
            os.fsync(out.fileno())
        actual = digest.hexdigest()
        if actual != expected_hash:
            raise ValueError('Canonical SHA-256 mismatch; artifact was not published: ' + actual)
        return actual


def publish(source, destination, expected_hash):
    partial = destination.with_name(destination.name + '.repack.part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open('rb') as inp, partial.open('wb') as out:
        while block := inp.read(8 * 1024 * 1024):
            out.write(block)
            digest.update(block)
        out.flush()
        os.fsync(out.fileno())
    if digest.hexdigest() != expected_hash:
        raise ValueError('Publication checksum mismatch')
    os.replace(partial, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    args.stage.mkdir(parents=True, exist_ok=True)
    for item in json.loads(args.manifest.read_text()):
        destination = Path(item['destination'])
        if destination.exists():
            digest = hashlib.sha256()
            with destination.open('rb') as handle:
                while block := handle.read(8 * 1024 * 1024):
                    digest.update(block)
            if destination.stat().st_size == item['size'] and digest.hexdigest() == item['sha256']:
                print('already verified ' + destination.name, flush=True)
                continue
            raise ValueError('Existing destination differs; refusing to overwrite: ' + str(destination))
        request = urllib.request.Request(item['header_url'], headers={'Range': 'bytes=0-1048575'})
        with urllib.request.urlopen(request, timeout=60) as response:
            canonical, _ = read_header(response)
        temporary = args.stage / (destination.name + '.part')
        status = args.stage / 'repack-status.json'
        status.write_text(json.dumps({'state': 'repacking', 'file': destination.name}))
        repack(Path(item['source']), canonical, temporary, item.get('mapping', 'identity'), item['sha256'], item['size'])
        status.write_text(json.dumps({'state': 'publishing', 'file': destination.name}))
        publish(temporary, destination, item['sha256'])
        temporary.unlink()
        print(json.dumps({'verified': str(destination), 'sha256': item['sha256'], 'size': item['size']}), flush=True)
    (args.stage / 'repack-status.json').write_text(json.dumps({'state': 'complete'}))


if __name__ == '__main__':
    main()
