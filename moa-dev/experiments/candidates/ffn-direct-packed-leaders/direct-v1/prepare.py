#!/usr/bin/env python3
"""Unselected source-only direct CL0 packed -> sixteen leaders reserve."""
from pathlib import Path
import argparse,json,hashlib,difflib
P=Path(__file__).resolve().parent
SOURCE=Path('/home/jihwan38/moa-ffn-sharded-scale-replicated/moa-dev/experiments/candidates/ffn-sharded-scale-replicated/replicated-v1')
EXPECTED={'ops.rs':'59a4ab6414d362876255caa8f332a01a567efd5ded0afa2d43da7ffe34e3c61e','rmsnorm.rs':'e2aa2efffd7cd6e257772c679b8655f42041963ed568688693842a466546bfc2','mlp.rs':'838bc4018756ac2096012ccc407fc97294029d9a642dcfa64a77ebbeaf234256'}
def dg(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--source',type=Path,default=SOURCE);a=ap.parse_args()
 base={n:(a.source/n).read_bytes() for n in EXPECTED}
 for n,b in base.items():assert dg(b)['sha256']==EXPECTED[n],n
 out=dict(base);s=base['mlp.rs'].decode()
 old='''fn broadcast_ffn_quantized(
    device: &mut Device,
    x: &DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]>,'''
 new=old.replace('Chip, m![2], FfnQuantSlices','Chip, Cluster, FfnQuantSlices')
 assert s.count(old)==1 and s.count('broadcast_ffn_quantized(')==2
 s=s.replace(old,new,1)
 old='''    // Replicate only prepared terms to the two cluster-local H480 producers.
    // 2*3*3840 = 23040 logical destination bytes; original ring32 seed and
    // full512-slice broadcast remain exactly the same function and mappings.
    let packed: DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]> = packed.to_dm(&mut device.tdma);
'''
 new='''    // The only broadcast helper now reads the original CL0 H480 packed shards
    // directly into sixteen full-H leaders. This removes the23040B intermediate,
    // but CL1 seed destinations require92160B across clusters instead of11520B.
'''
 assert s.count(old)==1;s=s.replace(old,new,1);out['mlp.rs']=s.encode()
 # Inverse transform proves there are no implicit arithmetic/ring edits.
 restored=s.replace(new,old,1).replace('''fn broadcast_ffn_quantized(
    device: &mut Device,
    x: &DmTensor<f8e5m2, Chip, Cluster, FfnQuantSlices, m![FfnTerm, H % 480]>,''', '''fn broadcast_ffn_quantized(
    device: &mut Device,
    x: &DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]>,''',1)
 assert restored==base['mlp.rs'].decode()
 contract={'status':'unselected_source_only_not_compiled_not_measured','base':'Frozen E154 replicated-v1; adoption status is a separate root decision','source_inputs':{n:dg(b) for n,b in base.items()},'prepared_sources':{n:dg(b) for n,b in out.items()},'changed':['private broadcast helper input Cluster now CL0 only','remove lateCL2 packed intermediate typedDMA'], 'only_private_helper_callers':1,'unchanged':['ops/RMS fullbytes','quantizer and scale8 arithmetic/restore512','three exact480B copies','seed destination16 leaders x3x3840E5','Main ring32','up/gate native and all downstream precision/weights/scales'], 'logical_bytes':{'source_packed':11520,'removed_late_packed_destination':23040,'seed_destination_unchanged':184320,'ring_destination_unchanged':5898240,'old_crosscluster_CL1_packed':11520,'new_crosscluster_CL1_seed_destination':92160,'crosscluster_destination_multiplier':8,'crosscluster_destination_increase':80640},'descriptor_gates':['all16leader output vectors exact term-major3xH3840','sourceCL0only8validH480shards; no padded source cells interpreted as useful','source fullDm notpartialtermview; offsets/extent cover only11520usefulpacked bytes','8CL1leaders receive correct fullH order across clusters','unchanged actualMainring32 plus original512scalarrestore','whole SRAM/VRF lifetime/overlap with changedDMA dependency'], 'prior':'M06 v4/v5 used same directCL0packed-to16leader type and exactcompiled but bothwhole timed out. This is source-level precedent, not successful NPU proof or proven cause. E154 retains separate terms/boundedcopies and original preRMS math.', 'selection':'Only parent may select after E154 outcome and remainingdeadline. This script performs no compile or NPU and predicts no speedup.'}
 files={**out,'source.diff':''.join(difflib.unified_diff(base['mlp.rs'].decode().splitlines(True),s.splitlines(True),fromfile='frozen-E154/mlp.rs',tofile='reserve/mlp.rs')).encode(),'source-contract.json':(json.dumps(contract,indent=2)+'\n').encode()}
 for n,b in files.items():
  if a.check:assert(P/n).read_bytes()==b,n
  else:(P/n).write_bytes(b)
 print(json.dumps({'passed':True,'source_only':True,'prepared_sources':contract['prepared_sources']}))
if __name__=='__main__':main()
