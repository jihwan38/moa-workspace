#!/usr/bin/env python3
"""Exhaust finite source-code pairs; prove only product BF16 representability."""
import json, math, struct
from pathlib import Path

def e2m1(code):
    return math.copysign([0.,.5,1.,1.5,2.,3.,4.,6.][code&7],-1. if code&8 else 1.)
def e4m3(code):
    exp=(code>>3)&15; mant=code&7
    if exp==15 and mant==7: return None
    value=mant*2.**-9 if exp==0 else (1.+mant/8.)*2.**(exp-7)
    return math.copysign(value,-1. if code&128 else 1.)

pairs=0; failures=[]; maxabs=0.; min_nonzero=float("inf")
for a in range(16):
    for b in range(256):
        y=e4m3(b)
        if y is None: continue
        z=e2m1(a)*y; bits=struct.unpack("<I",struct.pack("<f",z))[0]
        pairs+=1; maxabs=max(maxabs,abs(z))
        if z:min_nonzero=min(min_nonzero,abs(z))
        if bits&65535:failures.append([a,b,bits])
assert pairs==4064 and not failures
result=dict(finite_code_pairs=pairs,bf16_product_failures=failures,max_abs=maxabs,min_nonzero_abs=min_nonzero,
            scope="Only finite f4e2m1 times finite f8e4m3 product is exactly BF16 representable, including signedzero. Does not prove any changed DivF, F8cast, layout, device sequencer, or complete model.",
            consequence="The original prepack BF16 rounding immediately after finite F4*E4 multiplication is numerically redundant. Keep original separate DivF by retained row power2 and final E4 cast; verify device allweights before integration.")
if __name__=="__main__":
    print(json.dumps(result,indent=2))
