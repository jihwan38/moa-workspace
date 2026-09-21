#!/usr/bin/env python3
"""Check SDK 0.8.1 registry entries in existing ELF64 little-endian binaries.

This reads artifacts only; it does not launch a model or an NPU. The installed
SDK serializes Vec<(fn path, Vec<u64>, image bytes)> using bincode 1.3.3's
fixed-width little-endian wire format in the furiosa_kernels ELF section.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

if not __debug__:
    raise RuntimeError('Registry verification requires assertions; do not use Python optimization mode.')


OPS = {
    'embed_token', 'sliding_project_qkv', 'sliding_attention',
    'sliding_attention_output', 'decoder_feedforward', 'full_project_qkv',
    'full_attention_first_page', 'full_attention_page', 'full_attention_output',
    'final_norm_and_logits',
}


def fingerprint(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def registry(binary):
    data = binary.read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01', 'Requires ELF64 little-endian'
    shoff = struct.unpack_from('<Q', data, 40)[0]
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 58)
    assert shentsize == 64 and 0 < shstrndx < shnum
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize)
                for i in range(shnum)]
    names = sections[shstrndx]
    strings = data[names[4]:names[4] + names[5]]
    found = [s for s in sections
             if strings[s[0]:].split(b'\x00', 1)[0] == b'furiosa_kernels']
    assert len(found) == 1
    section = found[0]
    assert section[4] + section[5] <= len(data)
    wire = data[section[4]:section[4] + section[5]]
    pos = 0

    def take(size):
        nonlocal pos
        assert 0 <= size <= len(wire) - pos
        value = wire[pos:pos + size]
        pos += size
        return value

    def u64():
        return struct.unpack('<Q', take(8))[0]

    count = u64()
    assert count <= 1000
    entries, order = {}, []
    for _ in range(count):
        path = take(u64()).decode('utf-8')
        key_count = u64()
        assert key_count < 1000
        key = tuple(u64() for _ in range(key_count))
        image = take(u64())
        assert image and (path, key) not in entries
        entries[path, key] = image
        order.append((path, key))
    assert pos == len(wire), 'Trailing or unparsed registry bytes'
    assert order == sorted(order), 'Registry order differs from SDK contract'
    return entries, {'binary': fingerprint(data),
                     'registry_section': fingerprint(wire),
                     'registry_offset': section[4]}



def main():
    c=Path(__file__).resolve().parent
    r=c.parents[3]
    a=r/'moa-dev/experiments/artifacts/model-control-qkv-original-e55'
    suite=r/'moa-dev/experiments/artifacts/model-prefix-suite-e55'
    target=('furiosa_opt_gemma4::ops::sliding_project_qkv',())
    controls,control_info=registry(a/'probe_model_prefix_suite')
    original,original_info=registry(suite/'original/probe_model_prefix_suite')
    e55,e55_info=registry(suite/'e55/probe_model_prefix_suite')
    assert original_info['binary']['sha256']=='d0ae91f22f2b272fcf2fb77462c02bbc5b2a8ec5eaf79e0dc614bb62ce61c4e5'
    assert e55_info['binary']['sha256']=='52bd86d125e34da224738bf3a26a0672f8cefbcfa720b376ccf3a890c51afd57'
    expected={(f'furiosa_opt_gemma4::ops::{name}',()) for name in OPS}
    assert set(controls)==set(original)==set(e55)==expected
    assert controls[target]==original[target], 'Target QKV image differs from original'
    entries={}
    for (name,key),image in controls.items():
        assert key==()
        assert image==(a/'kernels'/(name+'.bin')).read_bytes(),name
        if (name,key)!=target: assert image==e55[name,key],('Non-target differs from E55',name)
        entries[name]={'control':fingerprint(image),'original':fingerprint(original[name,key]),
            'e55':fingerprint(e55[name,key]),'equal_original':image==original[name,key],
            'equal_e55':image==e55[name,key]}
    changed=[name for name,key in controls if controls[name,key]!=e55[name,key]]
    assert changed==[target[0]],changed
    print(json.dumps({'status':'EXACT_QKV_ORIGINAL_AND_OTHER_NINE_E55_IMAGES_PASS',
        'checked_at_utc':datetime.now(timezone.utc).isoformat(),'control_host':control_info,
        'reference_hosts':{'original':original_info,'e55':e55_info},'entry_count':len(controls),
        'target':target[0],'target_byte_identical_original':True,
        'other_nine_byte_identical_e55':True,'changed_from_e55':changed,'entries':entries,
        'scope':'ELF registry and standalone image identity; source integrity is checked separately.',
        'npu_executed':False,'logits_compared':False},indent=2))

if __name__=='__main__':
    main()
