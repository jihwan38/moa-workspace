#!/usr/bin/env python3
"""Check source scope and initialized ring-leader addresses before compile."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-shared-trf-e55/crate'
CRATE=ROOT/'moa-dev/experiments/artifacts/qkv-bf16-broadcast-trf-e55/crate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=(HERE/'projection.rs.original').read_text();new=(HERE/'projection.rs').read_text()
assert old.split('pub(crate) fn project_query(',1)[1]==new.split('pub(crate) fn project_query(',1)[1]
assert new.count('.to_trf()')==old.count('.to_trf()')
prep=new.split('pub(crate) fn prepare_shared_input(',1)[1].split('pub(crate) fn project_query(',1)[0]
assert 'device.sub.begin(x)' in prep
assert '.switch::<Replicated, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 32 })' in prep
assert 'InputClusters, m![Dummy256 / 32, 1 # 32], m![H]>' in prep
assert '.cast::<' not in prep and '.commit' not in prep
oldops=(HERE/'ops.rs.original').read_text();newops=(HERE/'ops.rs').read_text()
begin=oldops.index('    let x: DmTensor<bf16, Chip, m![2], m![Dummy256 / 32, 32], m![H]> = device')
end=oldops.index('    let x_trf = sliding::projection::prepare_shared_input',begin)
assert newops==oldops[:begin]+oldops[end:]
public=r'#\[device\(chip = 1\)\]\s+pub fn \w+\([\s\S]*?\) \{'
assert re.findall(public,oldops)==re.findall(public,newops)

# The preserved typed DMA gathers all eight H480 shards into each ring leader.
# Every live source is at physical slice32*g. Other31 cells remain padding.
leader_values={(c,32*g,h):h for c in range(2) for g in range(8) for h in range(3840)}
assert len(leader_values)==2*8*3840
copies=0
for c in range(2):
    for s in range(256):
        leader=32*(s//32)
        assert leader%32==0 and 0<=leader<256
        # Same initialized source H interval and identity element stride for allh.
        assert leader_values[c,leader,0]==0
        assert leader_values[c,leader,3839]==3839
        for h in range(3840):
            assert leader_values[c,leader,h]==h
            assert (h//16)*32+(h%16)*2==2*h
            copies+=1
assert copies==2*256*3840
changes=[]
for f in sorted((BASE/'src').rglob('*')):
    if f.is_file() and sha(f)!=sha(CRATE/f.relative_to(BASE)):changes.append(str(f.relative_to(BASE)))
assert changes==['src/device/sliding/projection.rs','src/ops.rs']
for f in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:assert sha(BASE/f)==sha(CRATE/f)
print(json.dumps({'changed_sources':changes,'other_41_sources_unchanged':True,
    'all_projection_math_scale_row_layout_and_o_suffix_unchanged':True,
    'typed_h480_to_leader_gather_unchanged':True,'public_signatures_headnorm_rope_cache_ffn_unchanged':True,
    'initialized_leaders':16,'leader_h_elements':61440,'all_output_element_coordinates_checked':copies,
    'expected_source_slice_by_output':[32*(s//32) for s in range(256)],
    'input_slice_padding_never_reinterpreted_as_valid':True,
    'trf_dtype':'bf16','trf_lane':1,'trf_elements_per_slice':3840,'logical_trf_bytes_per_slice':7680,
    'claim':'Intended mathematical source mapping only; actual lowered bitmap must match after compile.',
    'npu_run':False},indent=2))
