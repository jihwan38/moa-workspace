#!/usr/bin/env python3
"""Source-only E151 pre-RMS shard producer preparation; no builds or jobs."""
from pathlib import Path
import argparse,hashlib,json,difflib
P=Path(__file__).resolve().parent
R=Path('/home/jihwan38/moa-workspace')
def dg(b):return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def block(s,name):
 start=s.index(name);opening=s.index('{',start);depth=1;pos=opening+1
 while depth:
  depth+=(s[pos]=='{')-(s[pos]=='}');pos+=1
 return start,pos,s[start:pos]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');a=ap.parse_args()
 paths={'ops.rs':'src/ops.rs','rmsnorm.rs':'src/device/shared/rmsnorm.rs','mlp.rs':'src/device/shared/mlp.rs'}
 original={k:(R/'furiosa-opt-gemma4-12B'/v).read_text() for k,v in paths.items()}
 assert dg(original['mlp.rs'].encode())['sha256']=='e412ef9422a17cba47b94bbda8955cf45be98825e7dd3577efa09cab6d50f893','E151 source required'
 out=dict(original)
 # New pre-FF helper retains normalize's original arithmetic and F32 boundary;
 # only its final pure routing/cast stream becomes a local cast on each shard.
 rms=original['rmsnorm.rs'];start,end,fn=block(rms,'pub(crate) fn normalize<')
 helper=fn.replace('pub(crate) fn normalize<','pub(crate) fn normalize_ffn_input_sharded<',1)
 helper=helper.replace(') -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {',') -> DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]> {',1)
 old='''        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 8], m![H % 8]>()'''
 new='''        .fetch::<m![H / 8 % 60], m![H % 8]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()'''
 assert helper.count(old)==1;helper=helper.replace(old,new)
 out['rmsnorm.rs']=rms[:end]+'\n\n// Pre-FF-only ownership replacement; original RMS arithmetic is unchanged.\n'+helper+rms[end:]
 ops=original['ops.rs'];start,end,op=block(ops,'pub fn decoder_feedforward(')
 old='let x = shared::rmsnorm::normalize(device, &residual, pre_ff_rms_weight);'
 new='let x = shared::rmsnorm::normalize_ffn_input_sharded(device, &residual, pre_ff_rms_weight);'
 assert op.count(old)==1;out['ops.rs']=ops[:start]+op.replace(old,new)+ops[end:]
 mlp=original['mlp.rs']
 start,end,fn=block(mlp,'pub(crate) fn feedforward(')
 marker='    let down = feedforward_sharded(device, x,'
 assert marker in fn;fn=fn.replace(marker,'    let x: DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]> = x.to_dm(&mut device.tdma);\n'+marker,1)
 mlp=mlp[:start]+fn+mlp[end:]
 start,end,fn=block(mlp,'pub(crate) fn feedforward_sharded(')
 old='    x: DmTensor<bf16, Chip, Cluster, Slice, m![H]>,'
 assert fn.count(old)==1;fn=fn.replace(old,'    x: DmTensor<bf16, Chip, Cluster, DownEpilogueSlices, m![H % 480]>,',1)
 old='''    let x: DmTensor<bf16, Chip, m![2], Slice, m![H]> = x.to_dm(&mut device.tdma);
    let (x, x_scale) = prepare_ffn_fp8_split(device, &x);'''
 assert fn.count(old)==1;fn=fn.replace(old,'    let (x, x_scale) = prepare_ffn_fp8_split_sharded(device, &x);',1)
 mlp=mlp[:start]+fn+mlp[end:]
 start,end,fn=block(mlp,'pub(crate) fn prepare_ffn_fp8_split(')
 signature=fn[:fn.index('{')]
 wrapper=signature+'''{
    // Preserve the existing private boundary-probe interface. The production
    // path calls the sharded helper directly and does not execute this DMA.
    let x: DmTensor<bf16, Chip, Cluster, FfnQuantSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    prepare_ffn_fp8_split_sharded(device, &x)
}'''
 sh=fn.replace('pub(crate) fn prepare_ffn_fp8_split(','fn prepare_ffn_fp8_split_sharded(',1)
 sh=sh.replace('    x: &DmTensor<bf16, Chip, m![2], Slice, m![H]>,','    x: &DmTensor<bf16, Chip, Cluster, FfnQuantSlices, m![H % 480]>,',1)
 sh=sh.replace('''    // Amax on the canonical input avoids an expensive inter-slice reduction.
    // i32 reinterpret makes the absolute-value mask a non-NaN immediate.
    let amax: DmTensor<f32, Chip, m![2], Slice, m![1 # 8]>''','''    // Scan the eight actual cluster0 H480 shards; global finite Max preserves
    // the original amax. Keep the original floor before exponent extraction.
    let local_amax: DmTensor<f32, Chip, Cluster, FfnQuantSlices, m![1 # 8]>''',1)
 sh=sh.replace('m![H / 16], m![H % 16]','m![H / 16 % 30], m![H % 16]',1).replace('m![H / 8], m![H % 8]','m![H / 8 % 60], m![H % 8]',1).replace('m![H / 4], m![H % 4]','m![H / 4 % 120], m![H % 4]',1)
 anchor='    // Exponent mask produces'
 glob='''    let amax: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = device.main
        .begin(local_amax.view())
        .fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Slice, m![1]>(InterSliceReduceOpF32::Max)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
'''
 assert sh.count(anchor)==1;sh=sh.replace(anchor,glob+anchor,1)
 sh=sh.replace('let scale: DmTensor<f32, Chip, m![2], Slice','let scale: DmTensor<f32, Chip, Cluster, Slice',1)
 sh=sh.replace('Chip, m![2], FfnQuantSlices','Chip, Cluster, FfnQuantSlices')
 # x is already the final BF16 producer layout; remove all early copy/repartition.
 old='    let x: DmTensor<bf16, Chip, Cluster, FfnQuantSlices, m![H % 480]> = x.to_dm(&mut device.tdma);\n'
 assert sh.count(old)==1;sh=sh.replace(old,'',1)
 old='    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))'
 new='''    // Replicate only prepared terms to the two cluster-local H480 producers.
    // 2*3*3840 = 23040 logical destination bytes; original ring32 seed and
    // full512-slice broadcast remain exactly the same function and mappings.
    let packed: DmTensor<f8e5m2, Chip, m![2], FfnQuantSlices, m![FfnTerm, H % 480]> = packed.to_dm(&mut device.tdma);
    (broadcast_ffn_quantized(device, &packed), scale.to_dm(&mut device.tdma))'''
 assert sh.count(old)==1;sh=sh.replace(old,new,1)
 out['mlp.rs']=mlp[:start]+wrapper+'\n\n'+sh+mlp[end:]
 # Arithmetic-critical production bodies must remain byte-identical.
 for name in ['project_rows_with_scale_contraction','project_up_and_gate','geglu','prepare_down_fp8_split','project_down_native_half','project_down_tile','finish_down','broadcast_ffn_quantized']:
  token='fn '+name+'(';assert block(out['mlp.rs'],token)[2]==block(original['mlp.rs'],token)[2],name
 orig_before=fn[fn.index('    let hi:'):fn.index('    (broadcast_ffn_quantized')]
 new_before=sh[sh.index('    let hi:'):sh.index('    // Replicate only prepared')]
 assert new_before.replace('Chip, Cluster, FfnQuantSlices','Chip, m![2], FfnQuantSlices')==orig_before,'quantizer/copies arithmetic changed'
 report={'status':'prepared_source_only_not_compiled','base':'E151','source_inputs':{paths[k]:dg(v.encode()) for k,v in original.items()},'prepared_sources':{paths[k]:dg(v.encode()) for k,v in out.items()},'producer':'cluster0, physical slices0..7, H480 per slice','changes':['preFF helper preserves original normalize arithmetic but casts BF16 locally instead of final F32 Broadcast1 gather','production FFN takes H480 shards; no early canonicalBF16 cluster replication/repartition','local finiteMax then canonical interMax; original floor/exponent/scale/Subsum/three480Subcopies','late packed23040B cluster copy followed by original184320B seed and5898240B ring32 output'],'unchanged_functions':['project_rows_with_scale_contraction','project_up_and_gate','geglu','prepare_down_fp8_split','project_down_native_half','project_down_tile','finish_down','broadcast_ffn_quantized'],'new_stream_costs':{'global_inter_max_stream':1,'late_packed_cluster_destination_bytes':23040,'original_ring_seed_destination_bytes_retained':184320,'original_ring_broadcast_destination_bytes_retained':5898240},'scope':'Source/API-supported proposal only; original M06v4 timeout is unresolved and is not attributed to repaired DMA. Actual whole lowering, bounds, register lifetimes and NPU must be checked anew.'}
 files={k:v.encode() for k,v in out.items()};files['source.diff']=''.join(''.join(difflib.unified_diff(original[k].splitlines(True),out[k].splitlines(True),fromfile='E151/'+paths[k],tofile='prepared/'+paths[k])) for k in paths).encode();files['source-contract.json']=(json.dumps(report,indent=2)+'\n').encode()
 for n,b in files.items():
  p=P/n
  if a.check:assert p.read_bytes()==b,n
  else:p.write_bytes(b)
 print(json.dumps({'passed':True,'source_only':True,'prepared':report['prepared_sources']}))
if __name__=='__main__':main()
