#!/usr/bin/env python3
"""Check SDK 0.8.1 registry entries in existing ELF64 little-endian binaries.

This reads artifacts only; it does not launch a model or an NPU. The installed
SDK serializes Vec<(fn path, Vec<u64>, image bytes)> using bincode 1.3.3's
fixed-width little-endian wire format in the furiosa_kernels ELF section.
"""
import argparse
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
STAGE1 = {
    'furiosa_opt_gemma4::ops::sliding_project_qkv':
        '5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e',
    'furiosa_opt_gemma4::ops::sliding_attention_output':
        '359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b',
    'furiosa_opt_gemma4::ops::decoder_feedforward':
        '61ad0c50a03902de9208447951086df930669ded080c94e4c637bffac3c414b6',
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
    parser=argparse.ArgumentParser(description='Read and verify the prefix-suite hosts; no NPU execution.')
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    root=args.workspace
    artifact=root/'moa-dev/experiments/artifacts/model-prefix-suite-e55'
    refs={
      'original':root/'moa-dev/experiments/artifacts/original/test_kernels',
      'e55':root/'moa-dev/experiments/artifacts/e55-fused-qkv/target/release/test_kernels',
    }
    inventories={}; report={}
    for variant in ('original','e55'):
        host=artifact/variant/'probe_model_prefix_suite'
        entries,info=registry(host)
        expected={(f'furiosa_opt_gemma4::ops::{name}',()) for name in OPS}
        assert set(entries)==expected, (variant,set(entries))
        reference,reference_info=registry(refs[variant])
        assert set(reference)=={(name,()) for name in STAGE1}
        images={}
        for (name,key),image in entries.items():
            assert key==()
            assert image==(artifact/variant/'kernels'/(name+'.bin')).read_bytes(),name
            images[name]=fingerprint(image)
            if (name,()) in reference: assert reference[name,()]==image,(variant,name)
        if variant=='e55':
            for name,sha in STAGE1.items():assert images[name]['sha256']==sha,name
        previous_path = root / 'moa-dev/experiments/artifacts/model-fixed-prefix-e55' / variant / 'probe_model_logits'
        previous, previous_info = registry(previous_path)
        assert entries == previous, (variant, 'ten images differ from frozen fixed-prefix host')
        info['fixed_prefix_reference_path'] = str(previous_path.relative_to(root))
        info['fixed_prefix_reference_host'] = previous_info
        info['all_ten_images_equal_fixed_prefix_host'] = True
        inventories[variant]=entries
        report[variant]={'host_path':str(host.relative_to(root)),**info,'entry_count':len(entries),'entries':images,
                         'stage1_reference_path':str(refs[variant].relative_to(root)),
                         'stage1_reference_host':reference_info,'three_stage1_images_byte_identical':True}
    changed=[name for name,key in inventories['original'] if inventories['original'][name,key]!=inventories['e55'][name,key]]
    print(json.dumps({'status':'ELF_REGISTRY_IDENTITY_ONLY_PASS','checked_at_utc':datetime.now(timezone.utc).isoformat(),
      'hosts':report,'images_changed_from_original_to_e55':sorted(changed),
      'all_ten_text_kernels_present':True,'unexpected_images':False,
      'registry_fully_consumed_and_sorted':True,'npu_run':False,'logits_accuracy_validated':False},indent=2))

if __name__=='__main__':
    main()
