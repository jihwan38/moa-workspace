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
EXPECTED = {f'furiosa_opt_gemma4::ops::{name}' for name in OPS} | {
    'furiosa_opt_gemma4::ops_vision::patch_embed',
    'furiosa_opt_gemma4::ops_vision::add_position_and_norm',
    'furiosa_opt_gemma4::ops_vision::project_to_text_embedding',
    'furiosa_opt_gemma4::ops_audio::audio_project_frame',
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--stage1-host', type=Path, required=True)
    args = parser.parse_args()
    hosts = {}
    inventories = {}
    for name in ('gemma4', 'server'):
        entries, info = registry(args.artifact / name)
        assert set(entries) == {(path, ()) for path in EXPECTED}, set(entries)
        inventory = {}
        for (path, key), image in entries.items():
            assert image == (args.artifact / 'kernels' / (path + '.bin')).read_bytes(), path
            inventory[path] = {'generic_key': list(key), **fingerprint(image)}
            if path in STAGE1:
                assert inventory[path]['sha256'] == STAGE1[path], path
        hosts[name] = {**info, 'entries': inventory, 'entry_count': len(entries)}
        inventories[name] = entries
    assert inventories['gemma4'] == inventories['server']
    stage1, stage1_info = registry(args.stage1_host)
    assert set(stage1) == {(path, ()) for path in STAGE1}
    for key, image in stage1.items():
        assert inventories['gemma4'][key] == image
    print(json.dumps({
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'ELF_KERNEL_REGISTRY_CHECK_PASS',
        'hosts': hosts, 'stage1_host': stage1_info,
        'stage1_registry_entries': len(stage1),
        'both_model_hosts_have_all_expected_entries': True,
        'gemma4_and_server_registry_images_identical': True,
        'stage1_three_images_identical_to_measured_e55': True,
        'all_entries_have_empty_generic_key': True,
        'registry_fully_consumed_and_sorted': True,
        'unused_vision_layernorm_absent': True,
        'npu_run': False, 'full_model_accuracy_validated': False,
        'scope': 'ELF section parsing and exact device image identity only. '
                 'Does not test runtime device loading, weights, tokenizer, '
                 'logits, generation accuracy, throughput, or official scoring.',
    }, indent=2))


if __name__ == '__main__':
    main()
