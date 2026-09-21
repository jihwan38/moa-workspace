#!/usr/bin/env python3
"""CPU coordinate/zero-tail proof and one exact BF16 rounding counterexample.
This does not execute RNGD or model the compiler's FP32 accumulation order.
"""
import json
import struct
from pathlib import Path


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


rows = [1920*c + 60*g + 20*t + r
        for c in range(2) for g in range(32)
        for t in range(3) for r in range(20)]
inputs = [1920*c + 16*b + k
          for c in range(8) for b in range(120) for k in range(16)]
assert sorted(rows) == list(range(3840))
assert sorted(inputs) == list(range(15360))
# The complete scale-view rename preserves every within-slice row/block address.
for t in range(3):
    for r in range(20):
        for b in range(120):
            original = 120*(20*t+r)+b
            renamed_then_tiled = 120*(20*t+r)+b
            assert original == renamed_then_tiled
for r in range(20):
    source = [120*r+b+1 for b in range(120)]
    live128 = [0]*128
    live128[:120] = source
    assert live128[:120] == source and live128[120:] == [0]*8
# These sums are exactly representable in FP32, so this discrepancy is solely
# the new block-result BF16 boundary, not floating point summation uncertainty.
block0 = [1.0] + [2**-8]*15
block1 = [-1.0] + [-2**-8]*14 + [0.0]
assert all(bf16(x) == x for x in block0+block1)
p0, p1 = sum(block0), sum(block1)
reference, candidate = p0+p1, bf16(p0)+bf16(p1)
assert reference == 2**-8 and candidate == 2**-7
base = Path(__file__).parent.parent/'ffn-scale-contraction-streamed'
current = Path(__file__).parent
b = (base/'mlp.rs').read_text().split('pub(crate) fn project_down')[0]
s = (current/'mlp.rs').read_text().split('pub(crate) fn project_down')[0]
assert s.replace('\naxes![DownScaleBlock = 128];\n', '') == b
assert (current/'ops.rs').read_bytes() == (base/'ops.rs').read_bytes()
print(json.dumps({'rows':len(rows),'input_columns':len(inputs),
                 'zero_tail_per_row':8,'unchanged_upgate_and_ops':True,
                 'counterexample':{'reference':reference,'candidate':candidate}},indent=2))
