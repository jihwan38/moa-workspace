#!/usr/bin/env python3
"""Independent finite E5M2 RNE model for these eight diagnostic inputs only."""
import json, math, struct
from pathlib import Path

def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]
def activation(case, k):
    if case == 2: return {(0,0):1.,(0,1):15/256,(1,0):-1.,(1,1):-14/256}.get((k//16%2,k%16),0.)
    if case == 3: return 0.
    if case == 4: return 1. if k==0 else -1. if k==1 else 2**-22
    if case == 5: return .53125 if k==0 else -.53125 if k==1 else 2**-23
    if case == 6: return {0:1.,15:15/256,16:-1.,31:-14/256,32:.5,63:1/16,64:-.25,3838:1/128,3839:-1/256}.get(k,0.)
    if case == 7: return 128. if k==0 else -128. if k==1 else 1/16 if k%16==15 else -1/32 if k%16==0 else 0.
    return (k*7%17-8)/32

# Finite, non-negative values, ordered by the unsigned E5M2 encoding.
VALUES=[(math.ldexp(c&3,-16) if c>>2==0 else math.ldexp(1+(c&3)/4,(c>>2)-15),c) for c in range(124)]
def quantize(x):
    assert math.isfinite(x) and abs(x)<=57344
    value,code=min(VALUES,key=lambda v:(abs(v[0]-abs(x)),v[1]&1))
    return -value if x<0 else value

def audit():
    cases=[]
    for case in range(8):
        xs=[activation(case,k) for k in range(3840)]
        scale=math.ldexp(1.,math.floor(math.log2(max(max(map(abs,xs)),2**-112)))-14)
        values=[]
        for x in sorted(set(xs)):
            scaled=f32(x/scale);hi=quantize(scaled);lo=quantize(f32(scaled-hi));third=quantize(f32(scaled-f32(hi+lo)))
            restored=(hi+lo+third)*scale
            assert restored==x,(case,x,restored)
            assert f32(x)==x
            values.append({'input':x,'scaled':scaled,'terms':[hi,lo,third],'restored':restored})
        cases.append({'case':case,'inputs':3840,'runtime_scale':scale,'unique_values':values,'all_values_exactly_restored':True})
    return {'valid':True,'cases':cases,'input_elements_checked':8*3840,'scope':'Independent E5M2 RNE CPU model of selected dyadic inputs; no all-BF16 equivalence claim or hardware result. F64 dot oracle executes in the probe host.','rounding':'FP32 at division/subtraction/addition boundaries, finite E5M2 nearest ties to even.'}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args();r=audit()
    if a.out:
        with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))
