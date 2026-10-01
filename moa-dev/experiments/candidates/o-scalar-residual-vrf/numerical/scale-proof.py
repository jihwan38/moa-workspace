#!/usr/bin/env python3
"""Finite BF16 scale-producer identity, scalar F32 RNE model; no NPU/source mutation."""
import argparse,json,struct,hashlib
from pathlib import Path
f=lambda u:struct.unpack('<f',struct.pack('<I',u))[0]
b=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
f32=lambda x:f(b(x))
floor_x=2.0**-119;floor_s=2.0**-126
rows=[];out=[]
for bits in range(65536):
 if (bits&0x7f80)==0x7f80:continue
 x=f(bits<<16);absolute=f((bits&0x7fff)<<16)
 old=f32(f(b(max(absolute,floor_x))&0x7f800000)*2.0**-7)
 exponent=f((bits<<16)&0x7f800000)
 new=max(f32(exponent*2.0**-7),floor_s)
 flushed=max(0.0 if 0<abs(f32(exponent*2.0**-7))<2.0**-126 else f32(exponent*2.0**-7),floor_s)
 assert b(old)==b(new)==b(flushed),(bits,old,new,flushed)
 out.append(struct.pack('<I',b(new)))
 if bits in [0,0x8000,1,0x8001,0x007f,0x0080,0x03ff,0x0400,0x047f,0x0480,0x3f80,0x7f7f,0xff7f]:rows.append({'bf16_bits':f'{bits:04x}','x':x,'scale':new,'scale_bits':f'{b(new):08x}'})
# With signs removed, scale is nondecreasing across every finite magnitude.
positive=[max(f32(f((v<<16)&0x7f800000)*2.0**-7),floor_s) for v in range(0x7f80)]
assert all(a<=z for a,z in zip(positive,positive[1:]))
result={'valid':True,'finite_bf16_cases':len(out),'mismatches':0,'signed_zero_subnormal_floor_max_sign_checked':True,'f32_rne_and_intermediate_subnormal_flush_variants_equal':True,'finite_positive_scale_map_monotone':True,'output_scale_sequence_sha256':hashlib.sha256(b''.join(out)).hexdigest(),'old':'floorPow2(max(abs(x),2^-119))*2^-7','new':'max(f32((F32bits(x)&0x7f800000))*2^-7,2^-126)','group_identity':'Monotone max-homomorphism: max over K256 then K16 commutes with the pointwise map; every nonempty finite-BF16 K4096 group yields identical final scale bits.','sdk_path':'F32fetch -> Tag -> Logic(mask exponent/sign) -> Narrow -> Mul0(2^-7) -> ReduceAccTree Max(K256) -> Widen -> ClipMax(2^-126) -> InterSliceMax(K16) -> commit','scope':'Scalar proof for finite stored BF16 source and this scale only, not arbitrary F32/NPU pipeline correctness or performance. Four residual terms and restoration coefficients remain unchanged.','added_work':'Mul0 runs on131072 input values across512slices instead of32 valid group scalars. Hardware timing cannot be inferred from this count.','removed_work':'Two Main launches and intermediate local-max/global-max DM round trips; source scan and scalar distribution/load remain until separately replaced.','examples':rows}
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
original=json.loads((HERE/'e147-scale-proof.json').read_text())
assert {k:v for k,v in original.items() if k not in ('group_identity','sdk_path','added_work')}=={k:v for k,v in result.items() if k not in ('group_identity','sdk_path','added_work')}, 'Scalar exhaustive proof differs from E147'
result['original_e147_result']={'path':'e147-scale-proof.json','sha256':hashlib.sha256((HERE/'e147-scale-proof.json').read_bytes()).hexdigest()}
result['geometry']={'clusters':2,'row_groups_per_cluster':16,'columns_per_group':16,'bf16_values_per_column':256,'source_values_per_group':4096}
path=HERE/'scale-proof.json'
if args.check:assert json.loads(path.read_text())==result, 'Frozen finite-BF16 scale result changed'
else:
 assert not path.exists(), 'Refuse overwrite; use --check'
 path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['valid','finite_bf16_cases','mismatches','finite_positive_scale_map_monotone']}))
