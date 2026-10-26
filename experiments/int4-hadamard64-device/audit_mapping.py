#!/usr/bin/env python3
"""Read-only logical fetch/half mapping and frozen RN32 vector comparison.
This proves enumerated coordinates, not SDK lowering, allocation or device parity.
"""
from pathlib import Path
import hashlib,json,struct
ROOT=Path(__file__).resolve().parents[2]
def f(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def rn(x):return f(bits(x))
def mask(k):
 z=((k^0x243f6a8885a308d3)+0x9e3779b97f4a7c15)&((1<<64)-1)
 z=((z^(z>>30))*0xbf58476d1ce4e5b9)&((1<<64)-1)
 z=((z^(z>>27))*0x94d049bb133111eb)&((1<<64)-1)
 z^=z>>31
 return (z>>63)<<31

def low_coordinates(stride):
 if stride in (1,2,4,8):
  return [16*t+(2*lane if stride==1 else 4*(lane//2)+lane%2 if stride==2 else 8*(lane//4)+lane%4 if stride==4 else lane) for t in range(60) for lane in range(8)]
 if stride==16:return [32*t+8*q+lane for t in range(30) for q in range(2) for lane in range(8)]
 if stride==32:return [64*t+8*q+lane for t in range(15) for q in range(4) for lane in range(8)]
 raise AssertionError(stride)

def main():
 stages=[]
 for stride in (1,2,4,8,16,32):
  lo=low_coordinates(stride);hi=[k+stride for k in lo]
  assert len(lo)==len(set(lo))==480 and sorted(lo+hi)==list(range(960))
  assert all(k//stride%2==0 and k//64==(k+stride)//64 for k in lo)
  stages.append({'stride':stride,'pairs_per_token_per960':len(lo),'full_coverage':True,'unzip_group_f32_elements':8})
 seen=set()
 for c in range(2):
  for g in range(4):
   for r in range(64):
    for j in range(4):
     p=256*c+4*r+j
     for k in range(960):seen.add((p,960*g+k))
 assert len(seen)==512*3840
 vectors=ROOT/'experiments/int4-hadamard64-diagnostic/independent-proof/vectors.json'
 v=json.loads(vectors.read_text());checked=0
 for row in v['vectors']:
  a=[f(b^mask(row['base_channel']+i)) for i,b in enumerate(row['input_f32_bits'])]
  for stride in (1,2,4,8,16,32):
   old=a[:]
   for k in low_coordinates(stride)[:32]:
    assert k<64
    a[k]=rn(old[k]+old[k+stride]);a[k+stride]=rn(old[k]-old[k+stride])
  a=[rn(x*.125) for x in a]
  # Exact field names are frozen by the root independent proof.
  assert [bits(x) for x in a]==row['rotated_f32_bits']
  rounded=[]
  for x in a:
   b=bits(x);rounded.append(((b+0x7fff+((b>>16)&1))>>16)&0xffff)
  assert rounded==row['rotated_bf16_bits']
  checked+=1
 return {'source_sha256':hashlib.sha256((ROOT/'src/ops_int4_hadamard64.rs').read_bytes()).hexdigest(),'stages':stages,'mapped_input_values':len(seen),'frozen_vectors':checked,'vector_file_sha256':hashlib.sha256(vectors.read_bytes()).hexdigest(),'scope':'CPU logical coordinate/sign/RN32 comparison only; not SDK lowering/device execution'}
if __name__=='__main__':print(json.dumps(main(),indent=2))
