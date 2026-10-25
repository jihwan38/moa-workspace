#!/usr/bin/env python3
"""Read-only, deterministic source/body/layout checks; not a device oracle."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = 'cd331f3e02bb2926c7f16fdff19d166ad35adf04'
def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()
def old(path):
    return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT, text=True)
def read(path):
    return (ROOT/path).read_text()
def body(src, name):
    start = src.index('{', src.index('fn '+name+'('))
    depth = 0
    for pos in range(start, len(src)):
        depth += (src[pos] == '{') - (src[pos] == '}')
        if depth == 0:
            return src[start:pos+1]
    raise AssertionError(name)
ffn_path = 'src/ops_ffn_projection_clusters.rs'
ffn, original = read(ffn_path), old(ffn_path)
assert ffn.replace('pub(crate) fn global_dm(', 'fn global_dm(').replace(
    'pub(crate) fn geglu_tile_into(', 'fn geglu_tile_into(') == original
bridge = read('src/ops_int4_ffn_bridges.rs')
helper_bodies = {}
for name in ['global_dm', 'scale_tile', 'geglu_tile_into', 'stage_geglu_quantize',
             'stage_upgate', 'stage_down', 'ffn_core_stage']:
    assert body(ffn,name) == body(original,name)
    helper_bodies[name] = digest(body(ffn,name))
control_paths = [
    'src/ops_norm_blocked_chain_storegroup.rs', 'src/ops_norm_quarters.rs',
    'src/ops_norm_blocked_residual.rs', 'src/ops_norm_quarter_post.rs',
    'src/ops_group_ffn_projection_storegroup.rs', 'src/ops_int4_group960_projection.rs',
    'src/host/prefill_group.rs', 'src/host/load.rs']
control_hashes = {}
for path in control_paths:
    assert read(path) == old(path), path
    control_hashes[path] = digest(read(path))
norm = read(control_paths[0])
original_norm = body(norm,'post_attention')
start = original_norm.index('    let input = normalize')
end = original_norm.index('    let (quantized, scales)')
prefix = original_norm[start:end]
assert prefix in body(bridge, 'prefix_from_shards')
original_geglu = body(ffn,'stage_geglu_quantize')
start = original_geglu.index('    let up=joined.view()')
end = original_geglu.index('        let amax:')
geglu_prefix = original_geglu[start:end]
assert geglu_prefix in body(bridge,'geglu_bf16')
for kind in ['sliding','full']:
    before = body(norm,f'post_attention_{kind}_stage').split('    post_attention(')[0]
    after = body(bridge,f'prefix_{kind}_bf16').split('    prefix_from_shards(')[0]
    after = after.replace('NormCluster','Cluster').replace('NormShards','Shards').replace('NormLocal','Local')
    assert after == before, kind
# These are complete coordinate sets, independent of SDK named-axis inference.
norm_rows = [128*q+64*c+2*r+j for c in range(2) for q in range(4)
             for r in range(32) for j in range(2)]
geglu_rows = [256*c+64*chunk+4*r+j for c in range(2) for chunk in range(4)
              for r in range(16) for j in range(4)]
assert sorted(norm_rows) == sorted(geglu_rows) == list(range(512))
assert [256*c+64*chunk+p for c in range(2) for chunk in range(4)
        for p in range(64)] == list(range(512))
assert 'quantize_shards(' not in bridge and 'f8e4m3' not in bridge
assert body(bridge,'prefix_from_shards').index('output.view_mut()') < body(bridge,'prefix_from_shards').index('store(device, updated, residual)')
report = {
    'base_commit': BASE,
    'scope': 'Source/body and coordinate proof only; no runtime parity or performance proof.',
    'original_ffn_file_sha256': digest(original),
    'visibility_only_ffn_file_sha256': digest(ffn),
    'bridge_file_sha256': digest(bridge),
    'unchanged_function_body_sha256': helper_bodies,
    'unchanged_control_sha256': control_hashes,
    'copied_norm_arithmetic_prefix_sha256': digest(prefix),
    'copied_geglu_pre_amax_prefix_sha256': digest(geglu_prefix),
    'both_original_loader_prefixes_equal': True,
    'norm_and_geglu_each_cover_512_rows_once': True,
    'new_bridge_contains_no_e4_quantizer': True,
    'canonical_bf16_write_bytes': {'input':512*3840*2,'hidden':512*15360*2},
}
print(json.dumps(report,indent=2,sort_keys=True))
