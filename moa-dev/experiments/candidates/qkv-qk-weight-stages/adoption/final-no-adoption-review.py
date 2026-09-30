#!/usr/bin/env python3
"""Bounded final E146 no-adoption evidence review; no compile, upload, launch or adoption."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
C=HERE.parent
ROOT=C.parents[3]
BOUND=C/'boundary-suite'
LABEL='e146-qk-half-v4'
SELF=Path(__file__).resolve()

def load(p): return json.loads(p.read_text())
def rec(p):
    b=p.read_bytes()
    return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def sha(p): return {k:v for k,v in rec(p).items() if k!='path'}
def imp(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(p,*args):
    completed=subprocess.run([sys.executable,str(p),*args],cwd=ROOT,text=True,capture_output=True)
    assert completed.returncode==0, (p.name,completed.stdout,completed.stderr)
    return {'verifier':rec(p),'arguments':list(args),'exit_code':0}

def raw_graph_review():
    a=imp('e146_structural_audit',C/'audit.py');h=a.h;signature=a.a.signature
    pg=BOUND/'raw-evidence-v1/probe-graph.json';ps=BOUND/'raw-evidence-v1/probe-schedule.json'
    g=h.Graph(load(pg));whole=h.Graph(load(C/'evidence'/LABEL/'graph.json'))
    def ingress(graph,io):
        found=[]
        for op in graph.ops.values():
            if op['kind']!='Dma.DtoS':continue
            tid=op['inputs'][0];t=graph.tensor(tid)
            if t['element_type']!='Float8e4m3':continue
            start=0;view=None
            producer=graph.producers.get(tid)
            if producer and producer[1]['kind']=='IndexAccess':
                view=graph.view(tid);start=view['start'];tid=view['parent'];t=graph.tensor(tid)
            if t['optionals'].get('address')==f'IO({io})+0':found.append((start,op,view))
        return sorted(found,key=lambda x:x[0])
    reports=[]
    for io,name,rows,starts in [(1,'Q',16,[0,16]),(2,'K',8,[0,8]),(3,'V',16,[0])]:
        raw=ingress(g,io);prod=ingress(whole,io)
        assert [v[0] for v in raw]==[v[0] for v in prod]==starts
        for (offset,w,view),(woffset,ww,wview) in zip(raw,prod):
            assert signature(g,w)==signature(whole,ww)
            if view:
                for key in ('stride_elements','axis_extent','window','start'):assert view[key]==wview[key]
            ns=[o for o in g.ops.values() if o['context']=='Main' and w['outputs'][0] in o['inputs'] and 'InterTranspose' in o['description']]
            ws=[o for o in whole.ops.values() if o['context']=='Main' and ww['outputs'][0] in o['inputs'] and 'InterTranspose' in o['description']]
            assert len(ns)==len(ws)==1 and signature(g,ns[0])==signature(whole,ws[0])
            assert h.offsets(h.sequence(ns[0]['description'],'commit_projection'))==list(range(rows))
            if name!='V':
                rv,pv=g.view(ns[0]['inputs'][3]),whole.view(ws[0]['inputs'][3])
                for key in ('stride_elements','axis_extent','window','start'):assert rv[key]==pv[key]
                assert rv['start']==offset and rv['window']==rows
            for tid in ns[0]['inputs'][1:3]:assert g.producer(tid)[1]['kind'] in ('StoTrf','TuExec','StoVrf')
            reports.append({'projection':name,'stage_row_offset':offset,'public_weight_io':io,'raw_weight':w['name'],'raw_native':ns[0]['name'],'whole_weight':ww['name'],'whole_native':ws[0]['name'],'actual_description_io_shapes_and_index_offsets_equal':True})
    s=load(ps);ts={int(t['index']):t for t in s['tensors']};registers=[]
    for o in s['instructions']:
        if o['tpe']!='Main' or '.contract_outer' not in o['description']:continue
        rr=[ts[t] for t in o['input_tensors'] if t in ts and ts[t]['buffer_type'] in ('Trf','Vrf')]
        assert sorted(t['buffer_type'] for t in rr)==['Trf','Vrf']
        for t in rr:assert t['lifetime']['begin']<=o['lifetime']['begin']<=o['lifetime']['end']<=t['lifetime']['end']
        registers.append({'native':o['index'],'lifetime':o['lifetime'],'registers':rr})
    assert len(registers)==5 and len({tuple(t['index'] for t in x['registers']) for x in registers})==1
    vrfs=[t for t in ts.values() if t['buffer_type']=='Vrf']
    for i,x in enumerate(vrfs):
        assert 0<=x['address']<x['address']+x['size']<=512*8192
        for y in vrfs[i+1:]:
            concurrent=max(x['lifetime']['begin'],y['lifetime']['begin'])<min(x['lifetime']['end'],y['lifetime']['end'])
            overlap=max(x['address'],y['address'])<min(x['address']+x['size'],y['address']+y['size'])
            assert not(concurrent and overlap),(x['index'],y['index'])
    return {'bindings':[rec(pg),rec(ps)],'weights_and_native':reports,'scheduled_shared_registers':registers,'all_vrf_allocations_within_8kib_per_slice':len(vrfs),'simultaneously_live_vrf_ranges_nonoverlapping':True,'scope':'Actual separate diagnostic graph reviewed including both Q/K stages. Hidden SFR and runtime arbitration are not decoded; NPU exact outputs remain a separate check.'}

def review():
    checks=[run(C/'audit.py',LABEL,'--check'),run(C/'verify_graph.py',LABEL,'--check'),run(C/'validation_performance.py','--check'),run(C/'provenance.py','--check'),run(BOUND/'raw_oracle.py','--check'),run(BOUND/'check_full.py')]
    b=load(C/f'build-{LABEL}-repro.json');p=load(C/f'package-{LABEL}.json')
    source=load(C/'measured-build-e146-adoption.json');manifest=load(C/'compiled-source-manifest.json')
    assert source['source_inputs']==b['source_inputs']==p['source_inputs'] and len(b['source_inputs'])==46
    assert manifest==b['source_inputs'] or manifest.get('source_inputs')==b['source_inputs']
    control=load(ROOT/'moa-dev/experiments/candidates/qkv-fullh-weight-switch/measured-build-e144-adoption.json')
    assert {k for k in b['source_inputs'] if b['source_inputs'][k]!=control['source_inputs'][k]}=={'src/device/sliding/projection.rs'}
    image=lambda x:{k:x[k] for k in ('bytes','sha256')}
    for name in ('sliding_attention_output','decoder_feedforward'):
        assert image(p['images'][name])==image(control['images'][name])==image(source['images'][name])
    assert image(p['images']['sliding_project_qkv'])==b['exact_image']==image(source['images']['sliding_project_qkv'])
    perf=load(HERE/'performance-summary.json');observations=perf['paired_observations']
    assert perf['valid'] and len(observations)==3 and len({x['job_id'] for x in observations})==3
    recomputed=[];result_paths=[]
    for row in observations:
        path=C/f"{row['screen_label']}-job-{row['job_id']}-results.json";r=load(path);result_paths.append(path)
        assert r['valid'] and r['actual_upload_matches'] and r['screen_complete'] and r['correctness_passes']==45
        st=r['stages'];get=lambda n:st[n]['kernels']['sliding_project_qkv']['median_cycles']
        before,candidate,after=get('control_before'),get('candidate_whole'),get('control_after');bracket=(before+after)/2;delta=(candidate/bracket-1)*100
        assert abs(delta-row['delta_percent'])<1e-10
        recomputed.append({'job':row['job_id'],'before':before,'candidate':candidate,'after':after,'bracket':bracket,'paired_delta_percent':delta})
    median=statistics.median(x['paired_delta_percent'] for x in recomputed)
    assert abs(median-perf['median_paired_delta_percent'])<1e-10
    assert perf['all_paired_qkv_improved']==all(x['paired_delta_percent']<0 for x in recomputed)
    assert not perf['all_paired_qkv_improved'] and median>0, 'Root closure binds the actual mixed-sign, non-improving paired median.'
    assert not (HERE/'extra-results.json').exists(), 'Extra6 was prepared but not executed for this closure.'
    assert perf['candidate_distinct_standard_inputs']==3 and perf['candidate_qkv_comparisons_passed']==27
    raw=load(BOUND/'raw-results-v1.json');rb=load(BOUND/'raw-build-v1.json')
    assert raw['valid'] and raw['case_count']==9 and raw['output_elements']==73728 and raw['exact_bitwise'] and raw['actual_upload_matches']
    assert rb['original_source_inputs']==b['source_inputs'] and rb['production_whole_exact_image']==b['exact_image']
    selected=load(BOUND/'production-selection.json')
    assert selected['label']==LABEL and selected['source_inputs']==b['source_inputs'] and selected['exact_image']==b['exact_image']
    assert selected['build']==rec(C/f'build-{LABEL}-repro.json') and selected['gate']==rec(C/f'graph-verification-{LABEL}.json')
    original=ROOT/'moa-dev/experiments/candidates/qkv-native-consumer-local/boundary-suite-v2/raw-src'
    for name in ('probe.rs','probe_qkv_raw.rs','generate.py'):assert (BOUND/'raw-src'/name).read_bytes()==(original/name).read_bytes()
    assert (BOUND/'raw-truth-v1.json').read_bytes()==(ROOT/'moa-dev/experiments/candidates/qkv-native-consumer-local/boundary-suite/raw-truth-v1.json').read_bytes()
    assert '::project_qkv_native(device,&x,qw,kw,vw,qs,ks,vs)' in (BOUND/'raw-src/probe.rs').read_text()
    cfg=load(BOUND/'collector-config.json');full_path=ROOT/cfg['output'];full=load(full_path);full_build=load(ROOT/cfg['build_record'])
    assert full['verdict']=='PASS' and full['qkv_image_sha256']==b['exact_image']['sha256']
    assert sha(ROOT/full_build['selected_build'])==full_build['selected_build_digest']==sha(C/f'build-{LABEL}-repro.json')
    assert sum(x['count'] for x in full['results'])==16384 and sum(x['count'] for x in full['cache_checks'] if x['phase']=='after')==8380416
    actual_raw=raw_graph_review()
    bound=[C/'audit.py',C/f'audit-{LABEL}.json',C/f'graph-verification-{LABEL}.json',C/f'build-{LABEL}-repro.json',C/f'package-{LABEL}.json',C/'source-build-binding.json',C/'compiled-source-manifest.json',C/'measured-build-e146-adoption.json',HERE/'performance-summary.json',HERE/'e146-adoption-performance.json',HERE/'e146-e144-paired-control.json',BOUND/'preparation.json',BOUND/'production-selection.json',BOUND/'raw-build-v1.json',BOUND/'raw-results-v1.json',BOUND/'raw-truth-v1.json',BOUND/'collector-config.json',ROOT/cfg['build_record'],full_path,*result_paths]
    return {'valid':True,'adoption_ready':False,'candidate':'E146','label':LABEL,'verifier':rec(SELF),'inputdigests':{str(x.relative_to(ROOT)):rec(x) for x in bound},'checks':checks,'source':{'compiled_input_count':46,'source_manifest':rec(C/'compiled-source-manifest.json'),'image':b['exact_image'],'production_projection':b['source_inputs']['src/device/sliding/projection.rs'],'changed_from_e144':['src/device/sliding/projection.rs'],'e144_ffn_preserved':True,'unchanged_o_ffn_images':{n:image(p['images'][n]) for n in ('sliding_attention_output','decoder_feedforward')},'whole_structural_gate':{'passed':True,'row_quarter_packets':32768,'weight_hbm_bytes':31457280,'actual_packet_bytes':256,'qk_stage_native_count':4,'v_native_count':1,'direct_partial_writes_without_extra_copy':True}},'performance':{'independent_allocations':3,'observations':recomputed,'median_paired_delta_percent':median,'all_paired_improved':False,'no_significance_or_rank_claim':True},'correctness':{'standard_distinct_inputs':3,'performance_repeated_inputs':3,'candidate_standard_comparisons_across_allocations':45,'candidate_qkv_comparisons_across_allocations':27,'extra_distinct_inputs':0,'extra6_npu_status':'NOT_EXECUTED_AFTER_MIXED_PAIRED_PERFORMANCE','raw_before_head':{'cases':9,'values':73728,'bitwise_bf16':True,'signed_zero_equivalent':True,'actual_input_sha256_checks':63,'original_oracle_and_host_unchanged':True,'diagnostic_image_distinct_from_whole':True,'graph_review':actual_raw},'full_qkv':{'cases':2,'values':16384,'untouched_cache_values':8380416,'actual_input_sha256_checks':38,'actual_production_image':True,'original_tolerances':[.04,.01]}},'decision':{'adoption':'not_adopted','reason':'Three paired allocations mix signs and their median is a regression; the first positive does not reproduce.','further_tuning':'stop this staged prototype; no row sweep or extra6 NPU','major_research':'retain E144 and separately evaluate the selected scalar-supply readiness repair; no architecture-wide rejection'},'scope':'Valid finite accuracy/provenance evidence with no reproducible paired gain. Three original standard inputs repeated across three allocations plus selected raw/full boundary cases pass. Extra6 was not executed. No arbitrary-BF16, end-to-end, physical bottleneck or rank claim.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    result=json.loads(json.dumps(review()));out=HERE/'final-no-adoption-review.json'
    if args.check:assert load(out)==result,'Final independent review changed'
    else:out.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS E146 valid final evidence; adoption_ready=false; original E144 retained')
if __name__=='__main__':main()
