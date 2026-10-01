#!/usr/bin/env python3
"""Bounded offline E151 final evidence review. No compile/NPU/live adoption."""
import argparse,hashlib,json,statistics,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;C=HERE.parent;ROOT=C.parents[3];RAW=C/'down-oracle';FULL=C/'full-ffn-stress';LABEL='e151-ingress-v1';SELF=Path(__file__).resolve()
def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def image(x):return {k:x[k]for k in ('bytes','sha256')}
def run(p,*args):
 r=subprocess.run([sys.executable,str(p),*args],cwd=ROOT,text=True,capture_output=True);assert r.returncode==0,(p.name,r.stdout,r.stderr)
 return {'verifier':rec(p),'arguments':list(args),'exit_code':0}
def review():
 # Success cannot be emitted before all actual results and their frozen independent parsers pass.
 checks=[run(C/'verify_graph.py',LABEL,'--check'),run(HERE/'raw-graph-review.py','--check'),run(C/'validation_performance.py','--check'),run(C/'validation_extra.py','--check'),run(C/'provenance.py','--check'),run(RAW/'probe.py','--check'),run(FULL/'verify_results.py','--check')]
 b=load(C/f'build-{LABEL}-repro.json');p=load(C/f'package-{LABEL}.json');source=load(C/'measured-build-e151-adoption.json');manifest=load(C/'compiled-source-manifest.json');control=load(C.parent/'ffn-down-weight-stages/measured-build-e150-adoption.json')
 assert source['source_inputs']==b['source_inputs']==p['source_inputs'] and len(b['source_inputs'])==46
 assert manifest==b['source_inputs'] or manifest.get('source_inputs')==b['source_inputs']
 assert b['base_commit']=='ad1ecb3d3868c7e0ca2e4d1428bbd66bf1111821'
 assert {k for k in b['source_inputs']if b['source_inputs'][k]!=control['source_inputs'][k]}=={'src/device/shared/mlp.rs'}
 for name in ('sliding_project_qkv','sliding_attention_output'):assert image(p['images'][name])==image(source['images'][name])==image(control['images'][name])
 assert image(p['images']['decoder_feedforward'])==image(source['images']['decoder_feedforward'])==b['exact_image']
 perf=load(HERE/'performance-summary.json');rows=perf['paired_observations'];assert perf['valid'] and perf['independent_allocations']==3 and len(rows)==3 and len({x['job_id']for x in rows})==3
 observations=[];result_paths=[]
 for row in rows:
  path=C/f"{row['screen_label']}-job-{row['job_id']}-results.json";r=load(path);result_paths.append(path)
  assert r['valid'] and r['actual_upload_matches'] and r['screen_complete'] and r['correctness_passes']==45
  get=lambda name:r['stages'][name]['kernels']['decoder_feedforward']['median_cycles']
  before,candidate,after=get('control_before'),get('candidate_whole'),get('control_after');bracket=(before+after)/2;delta=(candidate/bracket-1)*100
  assert abs(delta-row['delta_percent'])<1e-10 and delta<0
  observations.append({'job':row['job_id'],'before':before,'candidate':candidate,'after':after,'bracket':bracket,'paired_delta_percent':delta})
 median=statistics.median(x['paired_delta_percent']for x in observations);assert abs(median-perf['median_paired_delta_percent'])<1e-10 and perf['all_paired_ffn_improved']
 assert perf['candidate_distinct_standard_inputs']==3 and perf['candidate_ffn_comparisons_passed']==9
 extra=load(HERE/'extra-results.json');extra_report=load(HERE/'e151-extra-correctness.json');assert extra['valid'] and extra['actual_upload_matches'] and extra_report['valid']
 assert extra['run_ids']==list(range(3,9)) and extra['output_comparisons_passed']==30
 rb=load(RAW/'build.json');raw=load(RAW/'results.json');assert raw['valid'] and raw['actual_upload_matches'] and raw['case_count']==9 and raw['output_elements']==34560 and raw['mapping_cases_exact_bitwise'] and raw['signed_zero_equivalent']
 assert raw['production_source_full_file_byte_identical_prefix'] and rb['production_source_full_file_byte_identical_prefix']
 assert rb['production_whole_source_inputs']==b['source_inputs'] and rb['production_whole_exact_image']==b['exact_image'] and rb['production_build']==rec(C/f'build-{LABEL}-repro.json')
 assert rb['production_mlp_prefix']==b['source_inputs']['src/device/shared/mlp.rs'] and image(rb['image'])!=b['exact_image']
 fb=load(FULL/'build-e151.json');full=load(FULL/'e151-full-ffn-stress-results.json');fv=load(FULL/'verification.json')
 assert full['valid'] and full['actual_upload_matches'] and full['performance_excluded'] and fv['valid'] and fv['new_candidate_pass']
 assert full['stages']['e151']['passed'] and full['stages']['e151']['checked']==full['stages']['e151']['exact']==3840 and full['stages']['e151']['nonfinite']==0
 assert fb['valid'] and not fb['source_kernel_edits'] and fb['kernel_image_byte_identical']
 assert image(fb['image'])==b['exact_image'] and fb['production_manifest']==b['source_inputs'] and fb['production_build']==rec(C/f'build-{LABEL}-repro.json')
 gate=load(C/f'audit-{LABEL}.json');rg=load(HERE/'raw-graph-review.json');assert gate['passed'] and rg['passed']
 assert gate['both_native_half_writes_joined'] and gate['scale_ingress']['coverage']['bytes']==3686400 and gate['row_packets_bijection']==30720 and gate['five_original_consumers_unchanged'] and gate['all_vrf_lifetimes_nonoverlapping_within_8kib']
 bound=[C/'audit.py',C/f'audit-{LABEL}.json',C/f'graph-verification-{LABEL}.json',C/f'build-{LABEL}-repro.json',C/f'package-{LABEL}.json',C/'source-build-binding.json',C/'compiled-source-manifest.json',C/'measured-build-e151-adoption.json',HERE/'performance-summary.json',HERE/'e151-adoption-performance.json',HERE/'e151-e150-paired-control.json',HERE/'extra-results.json',HERE/'e151-extra-correctness.json',HERE/'extra-image-verification.json',HERE/'raw-graph-review.py',HERE/'raw-graph-review.json',RAW/'recipe.json',RAW/'build.json',RAW/'results.json',RAW/'probe.py',FULL/'build-e151.json',FULL/'e151-full-ffn-stress-results.json',FULL/'verification.json',FULL/'verify_results.py',*result_paths]
 return {'valid':True,'adoption_ready':True,'candidate':'E151','label':LABEL,'verifier':rec(SELF),'inputdigests':{str(x.relative_to(ROOT)):rec(x)for x in bound},'checks':checks,'source':{'compiled_input_count':46,'source_manifest':rec(C/'compiled-source-manifest.json'),'image':b['exact_image'],'production_mlp':b['source_inputs']['src/device/shared/mlp.rs'],'changed_from_e150':['src/device/shared/mlp.rs'],'e150_qkv_o_preserved':True,'unchanged_qkv_o_images':{n:image(p['images'][n])for n in ('sliding_project_qkv','sliding_attention_output')},'whole_structural_gate':{'passed':True,'record':rec(C/f'audit-{LABEL}.json'),'down_native_stages':2,'rows_per_stage':30,'native_halves_and_weight_ingress_unchanged':True,'scale_hbm_bytes':3686400,'actual_scale_hbm_segment_bytes':480,'scale_packets':7680,'scale_switches':5,'scale_vrf_bytes_per_slice':5760,'routed_row_packets':30720,'partial_slice_half_byte_ranges':[[0,14400],[14400,28800]],'partial_consumers':5,'consumer_rows':12,'cross_half_rows':[24,35],'shared_register_lifetimes_verified':True}},'performance':{'independent_allocations':3,'all_paired_improved':True,'median_paired_delta_percent':median,'observations':observations,'no_significance_or_rank_claim':True},'correctness':{'standard_distinct_inputs':9,'performance_repeated_inputs':3,'extra_distinct_inputs':6,'raw_down':{'job':raw['job']['id'],'cases':9,'values':34560,'bitwise_bf16':True,'signed_zero_equivalent':True,'production_mlp_prefix_preserved':True,'original_oracle_and_host_unchanged':True,'diagnostic_image_distinct_from_whole':True,'diagnostic_image':image(rb['image']),'diagnostic_source_input_count':len(rb['source_inputs']),'graph_review':rec(HERE/'raw-graph-review.json'),'scope':'Original synthetic BF16 down input through actual prepare/project/finish helper; not a general full-FFN or all-input equivalence proof.'},'full_ffn':{'job':full['job']['id'],'cases':1,'values':3840,'bitwise_bf16':True,'actual_production_image':True,'original_reference_and_host_unchanged':True,'original_input_tensor_count':13,'original_atol':0.01,'original_rtol':0.01}},'scope':'Finite original standard and changed-down-boundary cases, original full-FFN regression, exact source/image/upload binding and three contemporary paired allocations support root adoption review. No arbitrary-input, end-to-end model, physical bottleneck or rank guarantee. Root owns live adoption.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(review()));out=HERE/'independent-final-review.json'
 if args.check:assert load(out)==r,'Final review changed'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E151 final: source46,3positive allocations,9standard,raw9/34560,fullFFN3840')
