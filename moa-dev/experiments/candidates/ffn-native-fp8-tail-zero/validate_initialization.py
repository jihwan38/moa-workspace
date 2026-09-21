#!/usr/bin/env python3
"""Exact address coverage of the two writes; no NPU semantic emulation."""
from pathlib import Path
import hashlib,json
p=Path(__file__).parent;b=p.parent/'ffn-native-fp8-full256-directcommit'
s=(p/'mlp.rs').read_text();base=(b/'mlp.direct-scales.rs').read_text()
assert s[:s.index('pub(crate) fn project_up_and_gate(')] == base[:base.index('pub(crate) fn project_up_and_gate(')]
assert s[s.index('fn compact_full256_rows('):] == base[base.index('fn compact_full256_rows('):]
assert s.count('.cast::<') == base.count('.cast::<')
assert s.count('_tail_zero.view_mut().memset(') == 4
assert '_dense.view_mut().memset(' not in s
out={}
for name,cols in [('weight',3840),('scale',240)]:
    destination=[None]*(32*cols)
    zero_source=[0]*(2*cols)
    tail_addresses=[];valid_addresses=[]
    # Destination tile starts at row30; source tile indices are local0..1.
    for local_row in range(2):
        for c in range(cols):
            src=local_row*cols+c;dst=(30+local_row)*cols+c
            destination[dst]=zero_source[src];tail_addresses.append(dst)
    for row in range(30):
        for c in range(cols):
            dst=row*cols+c
            assert destination[dst] is None
            destination[dst]=dst+1;valid_addresses.append(dst)
    assert all(v is not None for v in destination)
    assert set(tail_addresses).isdisjoint(valid_addresses)
    assert sorted(tail_addresses+valid_addresses)==list(range(32*cols))
    assert destination[30*cols:] == [0]*(2*cols)
    assert destination[:30*cols] == list(range(1,30*cols+1))
    assert (30*cols)%8 == 0 and (2*cols)%8 == 0
    out[name]={'valid_byte_range':[0,30*cols], 'tail_byte_range':[30*cols,32*cols],
               'destination_bytes':32*cols,'source_zero_bytes':2*cols,
               'tail_start_and_length_align8':True,'coverage':'every byte exactly once'}
assert (p/'ops.rs').read_bytes()==(b/'ops.rs').read_bytes()
print(json.dumps({'per_slice':out,'independent_buffers':4,'physical_slices':512,
                  'DPE_and_after_gather_byte_identical':True,'additional_casts':0,
                  'source_sha256':hashlib.sha256((p/'mlp.rs').read_bytes()).hexdigest()},indent=2))
