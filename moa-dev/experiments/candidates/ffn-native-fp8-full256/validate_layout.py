#!/usr/bin/env python3
"""Coordinate, actual-zero staging model, and source invariants; no NPU emulation."""
from pathlib import Path
import json

H, BLOCKS, ROWS = 3840, 240, 15360
coordinates=[]
reconstructed=[]
for cluster in range(2):
    gathered=[]
    for local_row in range(30):
        for sl in range(256):
            row = 7680*cluster + 256*local_row + sl
            coordinates.append(row)
            gathered.append(row)
            # HBM row-major reshape [cluster,row30,slice256,H] is wire-identical.
            for h in (0, 15, 16, H-1):
                assert (((cluster*30+local_row)*256+sl)*H+h) == row*H+h
            for block in (0, BLOCKS-1):
                assert (((cluster*30+local_row)*256+sl)*BLOCKS+block) == row*BLOCKS+block
    assert gathered == list(range(7680*cluster,7680*(cluster+1)))
    reconstructed.extend(gathered)
    for sl in range(256):
        # Model the complete live32 fill then copy of first30 source rows.
        weights=[[0]*H for _ in range(32)]
        scales=[[0]*BLOCKS for _ in range(32)]
        for r in range(30):
            value=7680*cluster+256*r+sl+1
            weights[r][:]=[value]*H
            scales[r][:]=[value]*BLOCKS
        assert all(v == 0 for row in weights[30:] for v in row)
        assert all(v == 0 for row in scales[30:] for v in row)
        assert all(weights[r][h] == 7680*cluster+256*r+sl+1
                   for r in range(30) for h in (0,15,16,H-1))
assert sorted(coordinates) == list(range(ROWS))
assert reconstructed == list(range(ROWS))
# Gather commits four adjacent slice outputs: four BF16 cells = eight bytes.
for r in range(30):
    for sl4 in range(64):
        group=[r*256+sl4*4+i for i in range(4)]
        assert group == list(range(group[0],group[0]+4))
base=Path(__file__).parent.parent/'ffn-native-fp8-scale-contraction'
current=Path(__file__).parent
b=(base/'mlp.rs').read_text();s=(current/'mlp.rs').read_text()
assert s.split('pub(crate) fn feedforward(',1)[1] == b.split('pub(crate) fn feedforward(',1)[1]
def helper(t):
    return t[t.index('fn project_rows_with_scale_contraction('):t.index('pub(crate) fn project_up_and_gate(')]
normalized=helper(s).replace('FfnRow / 4','L / 4 % 8').replace('FfnRow % 4','L % 4').replace('FfnRow','L % 32')
assert normalized == helper(b)
assert s.count('.memset(const { f8e4m3::from_f32(0.0) }') == 4
assert s.count('.cast::<') == b.count('.cast::<')
assert (current/'ops.rs').read_bytes() == (base/'ops.rs').read_bytes()
print(json.dumps({'valid_rows':ROWS,'physical_slices_per_cluster':256,
                  'live_rows_per_slice':32,'original_rows_per_slice':30,
                  'actual_zero_dummy_rows':2*256*2,
                  'weight_zero_tail_cells':2*256*2*H,
                  'scale_zero_tail_cells':2*256*2*BLOCKS,
                  'gather_commit_bytes':8,'no_extra_cast':True,
                  'DPE_helper_unchanged_after_axis_rename':True,
                  'feedforward_and_safe_down_suffix_unchanged':True},indent=2))
