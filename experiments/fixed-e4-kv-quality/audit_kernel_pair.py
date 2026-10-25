#!/usr/bin/env python3
"""Read-only ELF/kernel-cache audit. Never invokes objcopy on frozen artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import struct


def sha(b):
    return hashlib.sha256(b).hexdigest()


def section(path, wanted='furiosa_kernels'):
    data = Path(path).read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01'
    offset = struct.unpack_from('<Q', data, 40)[0]
    entry_size, count, names_index = struct.unpack_from('<HHH', data, 58)
    entries = [struct.unpack_from('<IIQQQQIIQQ', data, offset+i*entry_size) for i in range(count)]
    names_header = entries[names_index]
    names = data[names_header[4]:names_header[4]+names_header[5]]
    for e in entries:
        end = names.find(b'\0', e[0])
        if names[e[0]:end].decode() == wanted:
            return data[e[4]:e[4]+e[5]]
    raise ValueError(f'{wanted} absent from {path}')


NAMES = [
 'ops::sliding_project_qkv', 'ops_cache_pools::sliding_project_qkv',
 'ops_head_batch::sliding_heads_b16', 'ops_head_b16_blocked::sliding_heads_b16',
 'ops_head_r64::sliding_heads_r64', 'ops_head_r64_blocked::sliding_heads_r64',
 'ops_head_wide::sliding_heads_p128', 'ops_head_wide_blocked::sliding_heads_p128',
 'ops_head_wide_tile::sliding_heads_from_wide_tile',
 'ops_prefill_head_attn_commit::sliding_fused_first_p128',
 'ops_prefill_head_attn_commit::sliding_fused_p128', 'ops_b16_prefix::sliding',
 'ops_prefill_head_attn_commit::headed_outputs',
]
BINS = ['test_fixed_e4_head_codec', 'compare_fixed_e4_kv', 'compare_fixed_e4_packed']


def audit(off_run, on_run, off_cache, on_cache):
    sections, manifests, result = {}, {}, {'sections': {}, 'required_entries': []}
    for mode, run in [('off', off_run), ('on', on_run)]:
        manifests[mode] = json.loads((run/'run.json').read_text())
        sections[mode] = {}
        for binary in BINS:
            meta = manifests[mode]['built_artifacts']['target/release/'+binary]
            path = Path(meta['path'])
            assert sha(path.read_bytes()) == meta['sha256'], f'frozen ELF hash: {path}'
            b = section(path)
            sections[mode][binary] = b
            result['sections'].setdefault(binary, {})[mode] = {'bytes': len(b), 'sha256': sha(b), 'elf_sha256': meta['sha256']}
    for binary in BINS:
        assert sections['off'][binary] != sections['on'][binary], f'stale entire kernel section: {binary}'
    for name in NAMES:
        stem = 'furiosa_opt_gemma4::'+name
        a, b = (off_cache/(stem+'.bin')).read_bytes(), (on_cache/(stem+'.bin')).read_bytes()
        assert a != b, f'expected distinct precision kernel: {name}'
        ah, bh = (off_cache/(stem+'.hash')).read_bytes(), (on_cache/(stem+'.hash')).read_bytes()
        embeds = {mode: {binary: sec.find(blob) for binary, sec in sections[mode].items()} for mode, blob in [('off', a), ('on', b)]}
        for mode in ['off', 'on']:
            if name.endswith('::headed_outputs'):
                assert embeds[mode]['test_fixed_e4_head_codec'] >= 0
            else:
                assert embeds[mode]['compare_fixed_e4_kv'] >= 0, (mode, name, 'not in model ELF')
                assert embeds[mode]['compare_fixed_e4_packed'] >= 0, (mode, name, 'not in packed ELF')
        result['required_entries'].append({'name': name, 'off_bin_bytes': len(a), 'on_bin_bytes': len(b), 'off_sha256': sha(a), 'on_sha256': sha(b), 'stored_fingerprints_identical': ah == bh, 'off_fingerprint_sha256': sha(ah), 'on_fingerprint_sha256': sha(bh), 'embedding_offsets': embeds})
    same, changed, missing = [], [], []
    for p in sorted(off_cache.glob('*.bin')):
        other = on_cache/p.name
        if not other.exists():
            missing.append(p.name)
        elif p.read_bytes() == other.read_bytes():
            same.append(p.name)
        else:
            changed.append(p.name)
    result.update({'same_cached_bins': same, 'changed_cached_bins': changed, 'absent_from_old_cache': missing, 'off_run': str(off_run), 'on_run': str(on_run), 'read_only': True})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--off-run', required=True, type=Path); p.add_argument('--on-run', required=True, type=Path)
    p.add_argument('--off-cache', required=True, type=Path); p.add_argument('--on-cache', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = audit(a.off_run, a.on_run, a.off_cache, a.on_cache)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('KERNEL_PAIR_AUDIT', len(result['required_entries']), 'required entries; unchanged bins', len(result['same_cached_bins']), 'changed bins', len(result['changed_cached_bins']))
