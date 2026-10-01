#!/usr/bin/env python3
"""Bounded offline E154 final evidence review. No compile/NPU/live adoption."""
import argparse,hashlib,json,statistics,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;C=HERE.parent;ROOT=C.parents[3];RAW=C/'validation';FULL=C/'full-ffn-stress';PREV=C.parent/'ffn-prerms-sharded-consumer';LABEL='e154-replicated-v1';SELF=Path(__file__).resolve()
def load(p):return json.loads(p.read_text())
def rec(p):
 b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def image(x):return {k:x[k]for k in ('bytes','sha256')}
def run(p,*args):
 r=subprocess.run([sys.executable,str(p),*args],cwd=ROOT,text=True,capture_output=True);assert r.returncode==0,(p.name,r.stdout,r.stderr)
 return {'verifier':rec(p),'arguments':list(args),'exit_code':0}
def review():
 # Success cannot be emitted before all actual results and their frozen independent parsers pass.
 checks=[run(C/'verify_graph.py',LABEL,'--check'),run(HERE/'raw-graph-review.py','--check'),run(C/'validation_performance.py','--check'),run(C/'validation_extra.py','--check'),run(C/'provenance.py','--check'),run(C/'probe.py','--check'),run(FULL/'verify_results.py','--check')]
 b=load(C/f'build-{LABEL}-repro.json');p=load(C/f'package-{LABEL}.json');source=load(C/'measured-build-e154-adoption.json');manifest=load(C/'compiled-source-manifest.json');control=load(C.parent/'ffn-down-scale-ingress-switch/measured-build-e151-adoption.json')
 assert source['source_inputs']==b['source_inputs']==p['source_inputs'] and len(b['source_inputs'])==46
 assert manifest==b['source_inputs'] or manifest.get('source_inputs')==b['source_inputs']
 assert b['base_commit']=='8adbbae1776aa8080a96348a9d9a64022a0cf6d0'
 assert {k for k in b['source_inputs']if b['source_inputs'][k]!=control['source_inputs'][k]}=={'src/device/shared/mlp.rs','src/device/shared/rmsnorm.rs','src/ops.rs'}
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
 extra=load(HERE/'extra-results.json');extra_report=load(HERE/'e154-extra-correctness.json');assert extra['valid'] and extra['actual_upload_matches'] and extra_report['valid']
 assert extra['run_ids']==list(range(3,9)) and extra['output_comparisons_passed']==30
 rb=load(RAW/'probe-build-v1.json');raw=load(RAW/'probe-results-v1.json');assert raw['valid'] and raw['actual_upload_matches'] and raw['case_count']==8 and raw['output_elements']==245760 and raw['exact_bitwise'] and raw['signed_zero_equivalent']
 assert raw['production_source_full_file_byte_identical_prefix'] and rb['production_source_full_file_byte_identical_prefix']
 assert rb['production_whole_source_inputs']==b['source_inputs'] and rb['production_whole_exact_image']==b['exact_image'] and rb['source_build_digest']==rec(C/f'build-{LABEL}-repro.json')
 assert rb['production_mlp_prefix']==b['source_inputs']['src/device/shared/mlp.rs'] and image(rb['image'])!=b['exact_image']
 fb=load(FULL/'build-e154.json');full=load(FULL/'e154-full-ffn-stress-results.json');fv=load(FULL/'verification.json')
 assert full['valid'] and full['actual_upload_matches'] and full['performance_excluded'] and fv['valid'] and fv['new_candidate_pass']
 assert full['stages']['e154']['passed'] and full['stages']['e154']['checked']==full['stages']['e154']['exact']==3840 and full['stages']['e154']['nonfinite']==0
 assert fb['valid'] and not fb['source_kernel_edits'] and fb['kernel_image_byte_identical']
 assert image(fb['image'])==b['exact_image'] and fb['production_manifest']==b['source_inputs'] and fb['production_build']==rec(C/f'build-{LABEL}-repro.json')
 gate=load(C/f'audit-{LABEL}.json');rg=load(HERE/'raw-graph-review.json');assert gate['passed'] and rg['passed']
 ga=gate['actual'];sr=load(PREV/'source-review/source_review.json')
 assert ga['rms_first_ten_emitted_operations_identical_to_e151'] and ga['valid_input_slices']==list(range(8)) and ga['input_cluster']==0
 assert ga['global_max']['actual_ratio']>0 and ga['global_max']['output_slices']==list(range(8)) and ga['quant_scalar_dma_removed'] and ga['global_max']['valid_source_shards']==8 and ga['late_packed_dma']['destination_bytes']==23040
 assert ga['all_vrf_lifetimes_nonoverlapping_within_8kib'] and ga['native_and_e151_scale_arithmetic_unchanged'] and ga['native_count']==4
 assert sr['valid'] and sr['finite_amax']['bf16_patterns']==65280 and sr['source']['same_rms_arithmetic_and_normalized_f32_dm']
 assert rb['historical_fixture_identity']['byte_identical_inputs_wrapper_reference'] and rg['actual_sharded_quantizer_called']

 bound=[C/'audit.py',C/'replicated-v1/prepare.py',C/'replicated-v1/source-contract.json',C/f'audit-{LABEL}.json',C/f'graph-verification-{LABEL}.json',C/f'build-{LABEL}-repro.json',C/f'package-{LABEL}.json',C/'source-build-binding.json',C/'compiled-source-manifest.json',C/'measured-build-e154-adoption.json',HERE/'performance-summary.json',HERE/'e154-adoption-performance.json',HERE/'e154-e151-paired-control.json',HERE/'extra-results.json',HERE/'e154-extra-correctness.json',HERE/'extra-image-verification.json',HERE/'raw-graph-review.py',HERE/'raw-graph-review.json',RAW/'probe-v1/recipe.json',RAW/'probe-build-v1.json',RAW/'probe-results-v1.json',C/'probe.py',PREV/'source-review/source_review.py',PREV/'source-review/source_review.json',FULL/'build-e154.json',FULL/'e154-full-ffn-stress-results.json',FULL/'verification.json',FULL/'verify_results.py',*result_paths]
 return {'valid':True,'adoption_ready':True,'candidate':'E154','label':LABEL,'verifier':rec(SELF),'inputdigests':{str(x.relative_to(ROOT)):rec(x)for x in bound},'checks':checks,'source':{'compiled_input_count':46,'source_manifest':rec(C/'compiled-source-manifest.json'),'image':b['exact_image'],'production_mlp':b['source_inputs']['src/device/shared/mlp.rs'],'changed_from_e151':['src/device/shared/mlp.rs','src/device/shared/rmsnorm.rs','src/ops.rs'],'production_rmsnorm':b['source_inputs']['src/device/shared/rmsnorm.rs'],'production_ops':b['source_inputs']['src/ops.rs'],'e151_qkv_o_preserved':True,'unchanged_qkv_o_images':{n:image(p['images'][n])for n in ('sliding_project_qkv','sliding_attention_output')},'whole_structural_gate':{'passed':True,'record':rec(C/f'audit-{LABEL}.json'), 'rms_f32_emitted_prefix_preserved':True,'rms_emitted_prefix_operation_count':10,'local_bf16_cast_h480':True,'valid_input_cluster':0,'valid_input_shards':8,'global_max_actual_ratio':ga['global_max']['actual_ratio'],'quant_scalar_dma_removed':True,'global_max_output_shards':8,'global_max_valid_shards':8,'finite_bf16_source_max_proof':rec(PREV/'source-review/source_review.json'),'finite_bf16_patterns':65280,'late_packed_destination_bytes':23040,'term_copy_bytes':480,'term_copy_offsets':[0,480,960],'ring32_bitmap_and_copy_bounds_verified':True,'native_and_e151_scale_arithmetic_unchanged':True,'shared_register_lifetimes_verified':True}},'performance':{'independent_allocations':3,'all_paired_improved':True,'median_paired_delta_percent':median,'observations':observations,'no_significance_or_rank_claim':True},'correctness':{'standard_distinct_inputs':9,'performance_repeated_inputs':3,'extra_distinct_inputs':6,'finite_bf16_scalar_cpu_proof':{'patterns':65280,'record':rec(PREV/'source-review/source_review.json'),'scope':'Finite absolute Max/floor and original scale formula; source and CPU only, not arbitrary RMS or NPU correctness.'},'raw_upgate':{'job':raw['job']['id'],'cases':8,'values':245760,'bitwise_bf16':True,'signed_zero_equivalent':True,'production_mlp_prefix_preserved':True,'original_oracle_and_host_unchanged':True,'diagnostic_image_distinct_from_whole':True,'diagnostic_image':image(rb['image']),'diagnostic_source_input_count':len(rb['source_inputs']),'graph_review':rec(HERE/'raw-graph-review.json'),'scope':'Original BF16 input through actual new sharded quantizer and original project_up_and_gate to first BF16. The compatibility DMA is diagnostic-only; preRMS is excluded from this oracle.'},'full_ffn':{'job':full['job']['id'],'cases':1,'values':3840,'bitwise_bf16':True,'actual_production_image':True,'original_reference_and_host_unchanged':True,'original_input_tensor_count':13,'original_atol':0.01,'original_rtol':0.01}},'scope':'Finite original standard and changed-quantizer-boundary cases, original full-FFN regression, exact source/image/upload binding and three contemporary paired allocations support root adoption review. No arbitrary-input, end-to-end model, physical bottleneck or rank guarantee. Root owns live adoption.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=json.loads(json.dumps(review()));out=HERE/'independent-final-review.json'
 if args.check:assert load(out)==r,'Final review changed'
 else:out.write_text(json.dumps(r,indent=2)+'\n')
 print('PASS E154 final: source46,3positive allocations,9standard,raw8/245760,fullFFN3840')
