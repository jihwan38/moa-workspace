#!/usr/bin/env python3
"""Independent F64 mathematical oracle; no device implementation imported."""
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np
H,Q=3840,4096
NAMES=['signed_zero','isolated_k255','isolated_k256','isolated_k4095','multi_k_boundaries','stage_row_boundaries','tiny_correlated_cancellation','scale_before_bf16']

def checksum(raw):
    words=np.frombuffer(raw,dtype='<u4').astype(np.uint64)
    return int((words*(2*np.arange(len(words),dtype=np.uint64)+1)).sum(dtype=np.uint64)) & 0xffffffff

def bf(values):
    v=np.asarray(values,dtype='<f4').view('<u4')
    return ((v+np.uint32(0x7fff)+((v>>16)&1))>>16).astype('<u2')

def from_bf(bits):
    return (bits.astype('<u4')<<16).view('<f4').astype(np.float64)

def weights(case):
    r=np.arange(H,dtype=np.uint32)[:,None]; k=np.arange(Q,dtype=np.uint32)[None,:]
    code=(0x38+((r*3+r//60+r//120+r//1920+k+k//256)%16)) | (((r+k//256)%2)<<7)
    if case==5:
        code=np.where(np.isin(r%120,[0,58,59,60,61,118,119]),code,0)
    elif case==6:
        code=np.broadcast_to(np.where(k%256<2,0x38,0x77),(H,Q))
    elif case==7:
        code=np.broadcast_to(0x38+8*(r%2),(H,Q))
    return np.ascontiguousarray(code,dtype=np.uint8)

def decode_fp8(code):
    # E4M3 finite decode, independently from SDK lookup table.
    exp=(code.astype(np.int32)>>3)&15; mant=code&7
    mag=np.where(exp==0,np.ldexp(mant.astype(np.float64),-9),np.ldexp(1+mant/8.,exp-7))
    assert not np.any((exp==15)&(mant==7))
    return np.where(code&128,-mag,mag)

def generate(out):
    out.mkdir(parents=True,exist_ok=True); records=[]; payload=bytearray(b'C09PRE01')
    rows=np.arange(H)
    for case,name in enumerate(NAMES):
        x=np.zeros(Q,dtype=np.float64)
        scales=np.ldexp(1.+(rows%4)/8.,-(rows//4%5))
        if case==0: x[1::2]=-0.
        elif case==1: x[255]=1.
        elif case==2: x[256]=-.5
        elif case==3: x[4095]=1.5
        elif case in (4,5):
            x[[0,255,256,511,512,2047,2048,4095]]=[.125,-.25,.5,-.75,1.,-.5,.25,-.125]
        elif case==6:
            x[:]=2.**-22; x[0::256]=1.; x[1::256]=-1.
        else:
            x[0]=1.;x[1]=2.**-8
            scales=np.array([1.5,1.25,.5,0.,2.**-10,3.,.75,2.],dtype=np.float64)[rows%8]
        xb,sb=bf(x),bf(scales); codes=weights(case)
        # All tested products/sums are exact dyadics with <24 significant bits;
        # F64 scalar mathematical sum does not assume the NPU reduction topology.
        products=decode_fp8(codes)*from_bf(xb)[None,:]
        dot=products.sum(axis=1,dtype=np.float64)
        scaled=dot*from_bf(sb)
        assert np.array_equal(dot,dot.astype(np.float32).astype(np.float64))
        assert np.array_equal(scaled,scaled.astype(np.float32).astype(np.float64))
        expected=bf(scaled)
        blobs=[xb.tobytes(),sb.tobytes(),codes.tobytes()]
        checks=[checksum(b) for b in blobs]
        payload.extend(struct.pack('<III',*checks));payload.extend(blobs[0]);payload.extend(blobs[1]);payload.extend(expected.tobytes())
        counterfactual=bf(from_bf(bf(dot))*from_bf(sb))
        rec={'case':case,'name':name,'checksums':dict(zip(['x','scale','weight'],checks)),
             'input_sha256':dict(zip(['x','scale','weight'],[hashlib.sha256(b).hexdigest() for b in blobs])),
             'expected_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),
             'nonzero_expected':int(np.count_nonzero(expected&0x7fff)),
             'premature_bf16_dot_cast_differences':int(np.count_nonzero(expected!=counterfactual)),
             'min':float(scaled.min()),'max':float(scaled.max()),
             'samples':{str(r):{'dot_f64':float(dot[r]),'scaled_f64':float(scaled[r]),'expected_bits':int(expected[r])} for r in [0,58,59,60,61,118,119,120,1918,1919,1920,1921,3839]}}
        records.append(rec)
    assert records[7]['premature_bf16_dot_cast_differences']>0
    (out/'probe-fixture.bin').write_bytes(payload)
    (out/'probe-oracle.json').write_text(json.dumps({'oracle':'F64 mathematical dot of stored BF16 and decoded E4M3; F64 scale; BF16 RNE. Exact F32-representable dyadic cases. ±zero equivalent.','cases':records,'fixture':{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}},indent=2)+'\n')
if __name__=='__main__': generate(Path(sys.argv[1]))
