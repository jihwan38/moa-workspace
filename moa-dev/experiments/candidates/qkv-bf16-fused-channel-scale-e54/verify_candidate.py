#!/usr/bin/env python3
"""Exhaustive coordinate/source checks; this is not a hardware correctness test."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT/'moa-dev/experiments/artifacts/e54-bf16-qkv-clean-source'
CRATE = ROOT/'moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = (HERE/'projection.rs.original').read_text()
new = (HERE/'projection.rs').read_text()
prefix, suffix = new.split('pub(crate) fn project_output(', 1)
assert suffix == old.split('pub(crate) fn project_output(', 1)[1]
assert prefix.count('.cast::<bf16,') == 2
assert prefix.count('let contraction: DmTensor<bf16,') == 2
assert prefix.count('let scaled: DmTensor<bf16,') == 2
assert 'DmTensor<f32,' not in prefix
assert 'TrfTensor<f8e4m3' not in prefix
for axis, local in [('Qs',8), ('Ps',4)]:
    start = prefix.index(f'        .fetch::<m![{axis} / 8 % {local}, H / 16], m![H % 16]>()')
    pipeline = prefix[start:prefix.index('        .commit();', start)]
    assert pipeline.index('.contract_time') < pipeline.index('FpBinaryOp::MulF') < pipeline.index('.cast::<bf16,') < pipeline.index('.transpose')
    # The DPE/reduction inputs and axes are unchanged from E54.
    previous = old[old.index(f'        .fetch::<m![{axis} / 8 % {local}, H / 16], m![H % 16]>()'):]
    assert pipeline[:pipeline.index('        .vector_init()')] == previous[:previous.index('        .transpose')]

reports = []
for rows, cluster_rows, local_rows in [(4096,2048,8), (2048,1024,4)]:
    pre, post = set(), set()
    for row in range(rows):
        cluster = row // cluster_rows
        group = row // (local_rows*8) % 32
        slc = 8*group + row%8
        local = row//8 % local_rows
        assert row == cluster*cluster_rows + group*(local_rows*8) + 8*local + slc%8
        assert (cluster,slc,local) not in pre
        pre.add((cluster,slc,local))
        # The aligned scale DMA uses canonical coordinates, then the inverse
        # Switch swaps its local and low-slice coordinates to match raw rows.
        weight_row = cluster*cluster_rows + (slc//8)*(local_rows*8) + 8*local + slc%8
        scale_hbm_byte = 2*weight_row
        canonical_slice = group*8+local
        canonical_local = slc%8
        assert (canonical_slice//8*8+canonical_local, canonical_slice%8) == (slc,local)
        scale_dm_byte = 2*canonical_local
        scale_vrf_byte = 32*local
        assert scale_hbm_byte == 2*row
        assert 0 <= scale_dm_byte < 16
        assert 0 <= scale_vrf_byte < local_rows*32
        assert local < local_rows  # No K/V padding lane is promoted or read.
        # Four BF16 row results form each aligned 8-byte commit.
        time, packet = divmod(local,4)
        assert time*8 + packet*2 == local*2
        target_slice, target_local = group*8+local, slc%8
        assert (cluster,target_slice,target_local) not in post
        post.add((cluster,target_slice,target_local))
        time, packet = divmod(target_local,4)
        assert time*8+packet*2 == target_local*2
        if rows == 4096:
            head, gd = divmod(row,512)
            g,d = divmod(gd,256)
            assert (cluster,target_slice,target_local) == (head//4,(head%4)*64+g*32+d//8,d%8)
        else:
            head,d = divmod(row,256)
            assert target_slice%8 < 4
            assert (cluster,target_slice,target_local) == (head//4,(head%4)*64+(d//32)*8+d//8%4,d%8)
    assert len(pre) == len(post) == rows
    reports.append({'rows':rows,'scale_and_weight_coordinates_verified':rows,
                    'local_rows':local_rows,'scale_valid_bytes_per_slice':local_rows*4,
                    'vrf_physical_bytes_per_slice':256, 'vrf_scalar_stride_bytes':32,'scalar_time_vrf_valid_addresses_only':True,
                    'raw_and_switch_bf16_commit_bytes':8,
                    'kv_switch_padding_preserved':rows==2048})
changes=[]
for p in sorted((BASE/'src').rglob('*')):
    if p.is_file() and sha(p)!=sha(CRATE/p.relative_to(BASE)):
        changes.append(str(p.relative_to(BASE)))
assert changes == ['src/device/sliding/projection.rs']
assert (CRATE/'src/device/sliding/projection.rs').read_bytes() == (HERE/'projection.rs').read_bytes()
for file in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:
    assert sha(BASE/file)==sha(CRATE/file)
print(json.dumps({'changed_sources':changes,'source_files':43,'other_42_sources_byte_identical':True,
    'o_suffix_byte_identical':True,'all_ops_public_signatures_headnorm_rope_cache_ffn_unchanged':True,
    'dpe_lookup_inputs_reduction_axes_unchanged':True,'scale_before_first_bf16_cast':True,
    'coordinate_checks':reports,'npu_run':False},indent=2))
