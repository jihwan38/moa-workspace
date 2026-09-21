#!/usr/bin/env python3
"""Compare the diagnostic registry to the measured original host, read-only."""
import hashlib
import json
import struct
import sys
from pathlib import Path


def inspect(path):
    blob = Path(path).read_bytes()
    assert blob[:6] == b'\x7fELF\x02\x01'
    shoff = struct.unpack_from('<Q', blob, 40)[0]
    esize, count, names_index = struct.unpack_from('<HHH', blob, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', blob, shoff+i*esize) for i in range(count)]
    h = headers[names_index]
    names = blob[h[4]:h[4]+h[5]]
    sections = {names[h[0]:].split(b'\0', 1)[0].decode():h for h in headers}
    h = sections['furiosa_kernels']
    registry = blob[h[4]:h[4]+h[5]]
    pos = 0
    def u64():
        nonlocal pos
        value = struct.unpack_from('<Q', registry, pos)[0]
        pos += 8
        return value
    entries = {}
    for _ in range(u64()):
        n = u64()
        name = registry[pos:pos+n].decode()
        pos += n
        key = [u64() for _ in range(u64())]
        n = u64()
        offset = h[4]+pos
        image = registry[pos:pos+n]
        pos += n
        assert not key and len(image) == n and name not in entries
        entries[name] = {'bytes':n, 'sha256':hashlib.sha256(image).hexdigest(),
                         'elf_offset':offset, 'host_occurrences':blob.count(image)}
    assert pos == len(registry)
    return {'path':str(path), 'bytes':len(blob), 'sha256':hashlib.sha256(blob).hexdigest(), 'registry':entries}


control, original = inspect(sys.argv[1]), inspect(sys.argv[2])
name = 'furiosa_opt_gemma4::ops::sliding_attention_output'
assert set(control['registry']) == {name}
assert original['sha256'] == '942c1958f8e643a4625e111e312e469263598e5da2efeebcaf74fb08f7a5336c'
for field in ['bytes', 'sha256']:
    assert control['registry'][name][field] == original['registry'][name][field]
assert control['registry'][name]['host_occurrences'] == 1
print(json.dumps({'control':control, 'original_measured_host':original,
                  'o_image_matches_measured_original':True, 'binary_executed':False}, indent=2))
