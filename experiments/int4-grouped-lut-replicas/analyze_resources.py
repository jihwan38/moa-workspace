#!/usr/bin/env python3
"""Read-only source/resource replay; no standalone execution or timing inference."""
import contextlib,io,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'int4-grouped-activation-quant'
sys.path.insert(0,str(PRIOR))
import analyze_resource as schedule_tools
import analyze_metadata_store as provenance
import audit_source

def sha(p):return schedule_tools.sha(p)
def load(p):return json.loads(p.read_text())
def main():
 text=io.StringIO()
 with contextlib.redirect_stdout(text):audit_source.main()
 proof=json.loads(text.getvalue());assert proof==load(HERE/'source-proof.json')
 records=[]
 for path in sorted((HERE/'runs').iterdir()):
  r,h=load(path/'run.json'),load(path/'source-hashes.json')
  assert r['commit'].startswith('78054937') and r['exit_code']==0 and r['result']=='command_succeeded'
  assert (path/'source.diff').read_bytes()==b''
  provenance.git_sources(r['commit'],h)
  for name,digest in h.items():assert sha(ROOT/name)==digest,name
  if 'resource' in r['name']:
   assert r['fresh_package']['contents_changed'] is False
   # record_run captures build_inputs_unchanged only for ELF capture; this resource
   # run instead binds every current input directly to its frozen Git manifest above.
   out=next(v.split('=',1)[1]for v in r['command']if v.startswith('FURIOSA_OPT_OUT_DIR='))
   assert '/int4-grouped-lut-replicas/' in out
  records.append({'name':r['name'],'commit':r['commit'],'source_inputs':len(h),'command':r['command'],'exit_code':0,'recorded_status':r['status'],'recorded_files':{p.name:sha(p)for p in sorted(path.iterdir())}})
 results={};timeline_details={}
 for which,where,files in [('baseline',PRIOR,['input-store-schedule.json','hidden-store-schedule.json']),('replicated',HERE,['input-schedule.json','hidden-schedule.json'])]:
  schedule_tools.HERE=where
  for width,filename in zip(['H4','L16'],files):
   key=which+'_'+width;results[key]=schedule_tools.schedule(filename)
   raw=load(where/filename);tensors=[];ops=[]
   for ident,instances in raw['timeline']['tensors'].items():
    prop=raw['description']['tensors'][ident]['properties']
    for instance in instances:tensors.append({'id':ident,'properties':prop,**instance})
   for ident,instances in raw['timeline']['operators'].items():
    desc=raw['description']['operators'][ident]
    if desc['kind'].startswith('dma')or desc['kind']in['rngd_cmd.tu_exec','rngd_cmd.sto_vrf']:
     for inst in instances:ops.append({'id':ident,'kind':desc['kind'],'description':desc['description'],'inputs':desc['inputs'],'outputs':desc['outputs'],**inst})
   timeline_details[key]={'full_tensor_allocations':tensors,'all_dma_and_tu':ops}
 assert [results[k]['complete_unfolded_schedule_cycles']for k in ['baseline_H4','baseline_L16','replicated_H4','replicated_L16']]==[29069,67271,29115,67239]
 assert all(not v['has_dram_stack']for v in results.values())
 images={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted((HERE/'resource-images').iterdir())}
 assert len(images)==6
 for idx,fn in [('a002','quantize_pack_input'),('a003','quantize_pack_hidden')]:
  for ext in ['bin','hash','edf']:
   name=f'furiosa_opt_gemma4::ops_int4_grouped_lut_replicas::{fn}.{ext}'
   assert sha(HERE/f'kernel-out-{idx}'/'furiosa-opt-gemma4'/name)==images[name]['sha256']
 attribution={}
 for width in ['H4','L16']:
  data=timeline_details['replicated_'+width];ops=data['all_dma_and_tu'];ts=data['full_tensor_allocations']
  find=lambda fragment,kind:[x for x in ops if fragment in x['description']and x['kind']==kind]
  base=find('replica_bases.to_dm','dma.dto_s');vrf=find('replica_dm.view()','rngd_cmd.sto_vrf');gather=find('.gather::<','dma_command_gather')
  assert len(base)==len(vrf)==len(gather)==1
  byid={x['id']:x for x in ts}
  # These remain whole-chip physical allocations, not per-PE inferred live sizes.
  selected=[]
  for op in base+vrf:
   for tid in op['inputs']+op['outputs']:
    selected.append(byid[tid])
  offset_store=find('offset_store.view().to_hbm_view','dma.sto_d');assert len(offset_store)==1
  assert offset_store[0]['lifetime']['end']<=gather[0]['lifetime']['begin']
  attribution[width]={'additional_base_dma':base[0],'additional_base_vrf':vrf[0],
   'base_path_allocations':selected,'lut_gather':gather[0],'offset_store_before_gather':True,
   'complete_delta_cycles':results['replicated_'+width]['complete_unfolded_schedule_cycles']-results['baseline_'+width]['complete_unfolded_schedule_cycles']}
 result={'source_commit':records[0]['commit'],'plan_sha256':sha(HERE/'plan.json'),'source_proof':proof,
 'sdk_contract_sha256':sha(HERE/'sdk-contract.json'),'records':records,'resource_images':images,
 'schedules':results,'full_allocation_and_source_timelines':timeline_details,'replica_cost_attribution':attribution,
 'memory':{'old_table_bytes':4096,'new_table_bytes':65536,'replica_base_owner_bytes':512,'additional_HBM_owned_bytes':61952,
 'per_group_record_payload_H_L_bytes':[65536,262144],'offset_owner_H_L_bytes':[8192,32768],'offset_write_plus_reread_H_L_bytes':[16384,65536]},
 'decision':'SOURCE_HOST_RESOURCE_PASS_ONLY; NO_ELF_NO_ARENA_NO_RUNTIME_OR_CAUSAL_GAIN_CLAIM',
 'limits':['Compile validates public mapping legality, not actual replica values. Runtime oracle must verify H P%16 split across slice/local axes and every new offset.',
 'New offset contents differ from old ABI:512*m+32*(p%16), same owner extents. Codes/scalebits remain the same intended math; independent quotient oracle and full old-source control are retained.',
 'The static gather spans remain7886/28521cycles. Data-dependent actual address-concentration benefit is not predicted by these resources; prior slow-case correlation remains only a hypothesis.',
 'Additional DMA/VRF work overlaps with other operations; summed spans are not wall-time savings. Complete endpoints differ+46/-32cycles and are not measured runtime.',
 'Table/base owner upload is explicit startup work; future ABBA includes every device base-load/index/store/read/pack/scale movement but excludes equal CPU upload/readback.',
 'No aggregate combination with seven-to-three FFN composition, no model precision change or promotion. Original retained-axis NO-GO/actual323173/current fullFFN HOLDs remain preserved.']}
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
