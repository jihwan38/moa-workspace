#!/usr/bin/env python3
"""Bounded offline E148 evidence gate; no compile, upload, execution or adoption."""
import argparse,hashlib,importlib.util,json,re,statistics,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;C=HERE.parent;ROOT=C.parents[3];BOUND=C/'boundary-suite';LABEL='e148-direct-v1';SELF=Path(__file__).resolve()
def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def sha(p):return {k:v for k,v in rec(p).items() if k!='path'}
def imp(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def run(p,*args):
 r=subprocess.run([sys.executable,str(p),*args],cwd=ROOT,text=True,capture_output=True);assert r.returncode==0,(p.name,r.stdout,r.stderr)
 return {'verifier':rec(p),'arguments':list(args),'exit_code':0}
def image(x):return {k:x[k] for k in ('bytes','sha256')}
def raw_graph_review():
 a=imp('e148_whole_audit',C/'audit.py');h=a.h;sig=a.a.signature
 pg=BOUND/'oracle-evidence-v1/probe-graph.json';ps=BOUND/'oracle-evidence-v1/probe-schedule.json';g=h.Graph(load(pg));w=h.Graph(load(C/'evidence'/LABEL/'graph.json'))
 select=lambda graph,rule:[o for o in graph.ops.values() if rule(o)]
 fused=lambda o:o['context']=='Main' and '2139095040i' in o['description'] and 'GReduceMaxF [ratio=16]' in o['description']
 rs,ws=select(g,fused),select(w,fused);assert len(rs)==len(ws)==1 and sig(g,rs[0])==sig(w,ws[0])
 switch=lambda o:o['context']=='Sub' and 'CustomWithBitmap' in o['description'] and 'ring_size: 16' in o['description'] and 'to Vrf' in o['description']
 ss,sw=select(g,switch),select(w,switch);assert len(ss)==len(sw)==2
 key=lambda o:len(h.offsets(h.sequence(o['description'],'fetch_sequencer')));ss.sort(key=key);sw.sort(key=key);transports=[]
 for ro,wo in zip(ss,sw):
  assert len(ro['inputs'])==len(wo['inputs'])==2
  rc=g.producer(ro['inputs'][1])[1];wc=w.producer(wo['inputs'][1])[1]
  assert rc['kind']==wc['kind']=='Dma.DtoS' and sig(g,rc)==sig(w,wc)
  assert sig(g,ro).replace('T'+str(ro['inputs'][1]),'T_BITMAP')==sig(w,wo).replace('T'+str(wo['inputs'][1]),'T_BITMAP')
  offsets=h.offsets(h.sequence(ro['description'],'fetch_sequencer'));assert offsets in (list(range(4)),list(range(8)))
  assert [int(v) for v in re.findall(r'SliceBitmap \[(\d+)\]',ro['description'])]==[16*(s//16) for s in range(256)]
  transports.append({'raw_operator':ro['name'],'whole_operator':wo['name'],'valid_source_indices':offsets,'actual_bitmap_producer_and_vrf_packet_match':True})
 assert ss[0]['inputs'][1:]==ss[1]['inputs'][1:]
 sums=lambda o:o['context']=='Main' and 'WriteVrf: WriteVrf' in o['description']
 sr,sp=select(g,sums),select(w,sums);assert len(sr)==len(sp)==2;sum_reports=[]
 for ro in sr:
  matches=[wo for wo in sp if sig(g,ro)==sig(w,wo)];assert len(matches)==1
  assert g.tensor(ro['outputs'][0])['buffer_type']=='Vrf' and g.tensor(ro['outputs'][0])['size']==524288
  sum_reports.append({'raw':ro['name'],'whole':matches[0]['name'],'actual_f32_reader_mul_add_and_vrf_match':True})
 native=lambda o:o['context']=='Main' and 'operation_unit:' in o['description'] and 'axes=OutputTerm' in o['description'] and 'GReduceAddF [ratio=16]' in o['description']
 rn,wn=select(g,native),select(w,native);assert len(rn)==len(wn)==2
 rn.sort(key=lambda o:g.view(o['inputs'][3])['start']);wn.sort(key=lambda o:w.view(o['inputs'][3])['start']);natives=[]
 for ro,wo,start in zip(rn,wn,[0,60]):
  assert sig(g,ro)==sig(w,wo)
  rv,wv=g.view(ro['inputs'][3]),w.view(wo['inputs'][3])
  for k in ('stride_elements','axis_extent','window','start'):assert rv[k]==wv[k]
  assert rv['start']==start and rv['window']==60 and rv['axis_extent']==120
  natives.append({'raw':ro['name'],'whole':wo['name'],'output_row_start':start,'rows':60,'actual_full_native_descriptor_match':True})
 s=load(ps);ts={int(t['index']):t for t in s['tensors']};regs=[];direct=[]
 for op in s['instructions']:
  if op['tpe']=='Main' and '.contract_outer' in op['description']:
   rr=[ts[t] for t in op['input_tensors'] if t in ts and ts[t]['buffer_type'] in ('Trf','Vrf')];assert sorted(t['buffer_type'] for t in rr)==['Trf','Vrf']
   for t in rr:assert t['lifetime']['begin']<=op['lifetime']['begin']<=op['lifetime']['end']<=t['lifetime']['end']
   regs.append({'instruction':op['index'],'lifetime':op['lifetime'],'registers':rr})
  if '.to_vrf(&mut device.sub)' in op['description']:
   assert set(op['contexts'])=={'MainContext','SubContext','VectorEngine'};direct.append({'instruction':op['index'],'contexts':op['contexts']})
 assert len(regs)==len(direct)==2 and len({tuple(t['index'] for t in rr['registers']) for rr in regs})==1
 vrfs=[t for t in ts.values() if t['buffer_type']=='Vrf']
 for i,x in enumerate(vrfs):
  assert 0<=x['address']<x['address']+x['size']<=8192*512
  for y in vrfs[i+1:]:
   temporal=max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end']);spatial=max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size']);assert not(temporal and spatial)
 return {'bindings':[rec(pg),rec(ps)],'one_main_scale':{'raw':rs[0]['name'],'whole':ws[0]['name'],'actual_f32_pipeline_match':True},'scalar_transports':transports,'direct_cumulative_sums':sum_reports,'native_stages':natives,'shared_registers':regs,'direct_sum_resources':direct,'all_vrf_allocations_nonoverlapping_in_8kib_per_slice':len(vrfs),'scope':'Separately compiled production-helper diagnostic graph, not inherited whole allocation. Hidden register timing and NPU output accuracy are separately scoped.'}
def review():
 checks=[run(C/'audit.py',LABEL,'--check'),run(C/'verify_graph.py',LABEL,'--check'),run(C/'numerical/scale-proof.py','--check'),run(C/'validation_performance.py','--check'),run(C/'validation_extra.py','--check'),run(C/'provenance.py','--check'),run(BOUND/'oracle.py','--check'),run(BOUND/'full_stress.py','--check')]
 b=load(C/f'build-{LABEL}-repro.json');p=load(C/f'package-{LABEL}.json');source=load(C/'measured-build-e148-adoption.json');manifest=load(C/'compiled-source-manifest.json')
 assert source['source_inputs']==b['source_inputs']==p['source_inputs'] and len(b['source_inputs'])==46
 assert manifest==b['source_inputs'] or manifest.get('source_inputs')==b['source_inputs']
 control=load(C.parent/'qkv-scalar-sub-vrf/measured-build-e147-adoption.json')
 assert {k for k in b['source_inputs'] if b['source_inputs'][k]!=control['source_inputs'][k]}=={'src/device/sliding/projection.rs'}
 for n in ('sliding_project_qkv','decoder_feedforward'):assert image(p['images'][n])==image(control['images'][n])==image(source['images'][n])
 assert image(p['images']['sliding_attention_output'])==b['exact_image']==image(source['images']['sliding_attention_output'])
 perf=load(HERE/'performance-summary.json');observations=perf['paired_observations'];assert perf['valid'] and len(observations)==3 and len({x['job_id'] for x in observations})==3
 recomputed=[];result_paths=[]
 for row in observations:
  path=C/f"{row['screen_label']}-job-{row['job_id']}-results.json";r=load(path);result_paths.append(path)
  assert r['valid'] and r['actual_upload_matches'] and r['screen_complete'] and r['correctness_passes']==45
  get=lambda n:r['stages'][n]['kernels']['sliding_attention_output']['median_cycles'];before,candidate,after=get('control_before'),get('candidate_whole'),get('control_after');bracket=(before+after)/2;delta=(candidate/bracket-1)*100
  assert abs(delta-row['delta_percent'])<1e-10 and delta<0
  recomputed.append({'job':row['job_id'],'before':before,'candidate':candidate,'after':after,'bracket':bracket,'paired_delta_percent':delta})
 median=statistics.median(x['paired_delta_percent'] for x in recomputed);assert abs(median-perf['median_paired_delta_percent'])<1e-10 and perf['all_paired_o_improved']
 assert perf['candidate_distinct_standard_inputs']==3 and perf['candidate_o_comparisons_passed']==9
 extra=load(HERE/'extra-results.json');er=load(HERE/'e148-extra-correctness.json');assert extra['valid'] and extra['actual_upload_matches'] and er['valid']
 assert extra['run_ids']==[3,4,5,6,7,8] and extra['output_comparisons_passed']==30
 raw=load(BOUND/'oracle-results-v1.json');rb=load(BOUND/'oracle-build-v1.json')
 assert raw['valid'] and raw['case_count']==9 and raw['output_elements']==34560 and raw['exact_bitwise'] and raw['signed_zero_equivalent'] and raw['actual_upload_matches']
 assert raw['production_source_full_file_byte_identical'] and rb['original_source_inputs']==b['source_inputs'] and rb['production_whole_exact_image']==b['exact_image']
 assert rb['base_build']==rec(C/f'build-{LABEL}-repro.json') and rb['fixture_identical_to_original_e137']
 full=load(BOUND/'stress-results-v1.json');fb=load(BOUND/'stress-build-v1.json')
 assert full['valid'] and full['actual_upload_matches'] and full['case_count']==1 and full['output_elements']==3840 and full['production_source_inputs_match']
 assert image(full['image'])==image(fb['image'])==b['exact_image'] and fb['base_build']==rec(C/f'build-{LABEL}-repro.json')
 assert {n:v for n,v in fb['source_inputs'].items() if n!='src/bin/probe_output.rs'}==b['source_inputs']
 raw_graph=raw_graph_review()
 bound=[C/'audit.py',C/f'audit-{LABEL}.json',C/f'graph-verification-{LABEL}.json',C/f'build-{LABEL}-repro.json',C/f'package-{LABEL}.json',C/'source-build-binding.json',C/'compiled-source-manifest.json',C/'measured-build-e148-adoption.json',HERE/'performance-summary.json',HERE/'e148-adoption-performance.json',HERE/'e148-e147-paired-control.json',HERE/'extra-results.json',HERE/'e148-extra-correctness.json',HERE/'extra-image-verification.json',BOUND/'preparation.json',BOUND/'oracle-build-v1.json',BOUND/'oracle-results-v1.json',BOUND/'oracle-truth-v1.json',BOUND/'stress-build-v1.json',BOUND/'stress-results-v1.json',C/'numerical/scale-proof.py',C/'numerical/scale-proof.json',C/'numerical/provenance.json',*result_paths]
 return {'valid':True,'adoption_ready':True,'candidate':'E148','label':LABEL,'verifier':rec(SELF),'inputdigests':{str(p.relative_to(ROOT)):rec(p) for p in bound},'checks':checks,'source':{'compiled_input_count':46,'source_manifest':rec(C/'compiled-source-manifest.json'),'image':b['exact_image'],'production_projection':b['source_inputs']['src/device/sliding/projection.rs'],'changed_from_e147':['src/device/sliding/projection.rs'],'e147_qkv_ffn_preserved':True,'unchanged_qkv_ffn_images':{n:image(p['images'][n]) for n in ('sliding_project_qkv','decoder_feedforward')},'whole_structural_gate':{'passed':True,'one_main_scale':True,'scalar_sub_vrf_broadcasts':2,'ring_size':16,'shared_bitmap_hbm_bytes':16384,'direct_f32_cumulative_sum_vrf':2,'native_stages':2,'rows_per_stage':60,'original_weight_hbm_bytes':15728640,'actual_weight_packet_bytes':256}},'performance':{'independent_allocations':3,'observations':recomputed,'median_paired_delta_percent':median,'all_paired_improved':True,'no_significance_or_rank_claim':True},'correctness':{'standard_distinct_inputs':9,'performance_repeated_inputs':3,'extra_distinct_inputs':6,'finite_bf16_scalar_cpu_proof':{'patterns':65280,'bitwise_f32_scale_equivalent':True,'nonfinite_not_claimed':True,'proof':rec(C/'numerical/scale-proof.json')},'raw_projection':{'cases':9,'values':34560,'bitwise_bf16':True,'signed_zero_equivalent':True,'actual_input_checksums':27,'original_oracle_and_host_unchanged':True,'diagnostic_image_distinct_from_whole':True,'graph_review':raw_graph},'full_o':{'cases':1,'values':3840,'actual_production_image':True,'original_atol':0.05,'original_rtol':0.01,'original_case7_fixture_and_reference':True}},'scope':'Finite original standard and selected O boundary inputs, exact source/image/upload bindings and three contemporary performance allocations support root adoption review. No arbitrary-input, nonfinite native arithmetic, end-to-end model, bottleneck or rank guarantee. Root owns adoption/live changes.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(review()));out=HERE/'independent-final-review.json'
 if args.check:assert load(out)==r,'Final review changed'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E148 independent final review')
if __name__=='__main__':main()
