#!/usr/bin/env python3
"""Stored BF16/E4 mathematical truth; no candidate quantizer or reduction imported."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
COLS=3840
NAMES=['signed_zero','isolated_k959','isolated_k960','isolated_k3839','multi_k_boundaries','row_group_boundaries','tiny_correlated_cancellation','scale_before_bf16','global_scale_fourth_term_active']
def checksum(raw):
 w=np.frombuffer(raw,dtype='<u4').astype(np.uint64)
 return int((w*(2*np.arange(len(w),dtype=np.uint64)+1)).sum(dtype=np.uint64))&0xffffffff
def bf(v):
 u=np.asarray(v,dtype='<f4').view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
def unbf(v):return (v.astype('<u4')<<16).view('<f4').astype(np.float64)
def weights(case,rows,which):
 r=np.arange(rows,dtype=np.uint32)[:,None];k=np.arange(COLS,dtype=np.uint32)[None,:]; local=rows//128
 code=(0x38+((r*3+r//local+r//(rows//2)+k+k//960+which)%16))|(((r+k//960+which)%2)<<7)
 if case==5:code=np.where(np.isin(r%local,[0,1,local-2,local-1]),code,0)
 elif case==6:code=np.broadcast_to(np.where(k%960<2,0x38,0x77),(rows,COLS))
 elif case==7:code=np.broadcast_to(0x38+8*((r+which)%2),(rows,COLS))
 elif case==8:code=np.where(k<2,0x38,0x77|(((r//(rows//2)+r%2+which)%2)<<7))
 return np.ascontiguousarray(code,dtype=np.uint8)
def decode(c):
 e=(c.astype(np.int32)>>3)&15;m=c&7;assert not np.any((e==15)&(m==7))
 v=np.where(e==0,np.ldexp(m.astype(np.float64),-9),np.ldexp(1+m/8.,e-7));return np.where(c&128,-v,v)
def generate(out):
 out.mkdir(parents=True,exist_ok=False);payload=bytearray(b'E138RAW1');records=[]
 for case,name in enumerate(NAMES):
  x=np.zeros(COLS,dtype=np.float64)
  if case==0:x[1::2]=-0.
  elif case==1:x[959]=1.
  elif case==2:x[960]=-.5
  elif case==3:x[3839]=1.5
  elif case in (4,5):x[[0,959,960,1919,1920,2879,2880,3839]]=[.125,-.25,.5,-.75,1.,-.5,.25,-.125]
  elif case==6:x[:]=2.**-22;x[0::960]=1.;x[1::960]=-1.
  elif case==7:x[0]=1.;x[1]=2.**-8
  elif case==8:x[:]=129.*2.**-29;x[0]=1.;x[1]=-1.
  xb=bf(x);scales=[];expected=[];codes=[];summaries=[]
  for which,rows in enumerate([4096,2048,2048]):
   r=np.arange(rows);s=np.ldexp(1.+((r+which)%4)/8.,-(r//4%5))
   if case==7:s=np.array([1.5,1.25,.5,0.,2.**-10,3.,.75,2.])[((r+which)%8)]
   if case==8:s=np.ldexp(np.ones(rows),-((r+which)%5))
   sb=bf(s);w=weights(case,rows,which);dot=(decode(w)*unbf(xb)[None,:]).sum(axis=1,dtype=np.float64);scaled=dot*unbf(sb)
   assert np.array_equal(dot,dot.astype(np.float32).astype(np.float64)),(case,which,'dot')
   assert np.array_equal(scaled,scaled.astype(np.float32).astype(np.float64)),(case,which,'scaled')
   want=bf(scaled);assert np.all(np.isfinite(unbf(want)))
   wrong=bf(unbf(bf(dot))*unbf(sb));samples=[0,1,rows//128-1,rows//128,rows//2-1,rows//2,rows-1]
   summary={'rows':rows,'expected_sha256':hashlib.sha256(want.tobytes()).hexdigest(),'premature_cast_differences':int(np.count_nonzero(want!=wrong)),'samples':{str(t):{'raw_f64':float(dot[t]),'scaled_f64':float(scaled[t]),'expected_bits':int(want[t])} for t in samples}}
   if case==8:
    assert np.all(np.abs(dot)==3838*240*129*2.**-29)
    summary.update(tiny_normalized_terms=[0,0,128,32],common_scale=2.**-7,scope='Selected BF16 post-pre-RMS boundary input; not asserted reachable from a public PRNG seed.')
   scales.append(sb.tobytes());expected.append(want.tobytes());codes.append(w.tobytes());summaries.append(summary)
  blobs=[xb.tobytes(),*scales,*codes];keys=['x','q_scale','k_scale','v_scale','q_weight','k_weight','v_weight'];checks=[checksum(v) for v in blobs]
  payload.extend(struct.pack('<7I',*checks));payload.extend(blobs[0]);payload.extend(b''.join(scales));payload.extend(b''.join(expected))
  records.append({'case':case,'name':name,'checksums':dict(zip(keys,checks)),'input_sha256':dict(zip(keys,[hashlib.sha256(v).hexdigest() for v in blobs])),'outputs':dict(zip(['q','k','v'],summaries))})
 assert all(v['premature_cast_differences']>0 for v in records[7]['outputs'].values())
 (out/'probe-fixture.bin').write_bytes(payload)
 (out/'probe-oracle.json').write_text(json.dumps({'oracle':'F64 dot of stored BF16 and E4; F64 channel scale; exact F32 representability checked; BF16 RNE. ±zero equivalent. No candidate arithmetic imported.','cases':records,'fixture':{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}},indent=2)+'\n')
if __name__=='__main__':generate(Path(sys.argv[1]))
