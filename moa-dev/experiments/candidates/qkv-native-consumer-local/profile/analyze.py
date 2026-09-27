#!/usr/bin/env python3
"""Source-bound complete QKV profiles. Never add overlapping spans as cycle savings."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import tarfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE))
from package_v2 import ARTIFACTS,digest,file_record,read,write,operation_kind,map_markers,bindings
from collect_v2 import parse_diagnostic
CONFIG={'control': {'input_hbm': '14', 'pre_rms_finish': '73', 'input_leader': '79', 'input_ring': '96', 'q_trf': '138', 'kv_trf': '134', 'q_weight': '125', 'k_weight': '145', 'v_weight': '204', 'q_native': '151', 'k_native': '163', 'v_native': '328', 'q_table': '144', 'k_table': '76', 'v_table': '220', 'q_gather': '166', 'k_gather': '198', 'v_gather': '344', 'q_head_rms': '197', 'k_head_rms': '232', 'v_head_rms': '354', 'q_rope_finish': '322', 'k_rope_finish': '311', 'q_output': '329', 'k_output': '331', 'v_output': '374'}, 'candidate': {'input_hbm': '22', 'pre_rms_finish': '94', 'consumer_input_dma': '100', 'local_amax': '107', 'global_k4_amax': '117', 'scale': '122', 'term0': '157', 'term1': '183', 'term2': '209', 'quant_done': '232', 'coefficient_pack': '187', 'coefficient_dma': '191', 'coefficient_vrf': '213', 'activation_trf': '239', 'v_weight': '64', 'q_weight': '200', 'k_weight': '314', 'v_native': '243', 'q_native': '258', 'k_native': '396', 'v_gather': '262', 'q_gather': '283', 'k_gather': '405', 'common_scale_dma': '272', 'common_scale_vrf': '277', 'v_scaled_bf16': '291', 'q_scaled_bf16': '324', 'k_scaled_bf16': '428', 'q_output': '417', 'v_output': '424', 'k_output': '494'}}
NATIVES={'control': ['151', '163', '328'], 'candidate': ['243', '258', '396']}

def stats(vals):return {'median':statistics.median(vals),'min':min(vals),'max':max(vals),'samples':vals}
def function_names(src):return [(i+1,re.search(r'fn (\w+)',t).group(1)) for i,t in enumerate(src.splitlines()) if re.match(r'^(?:pub\(crate\) )?fn \w+',t)]

def analyze(label,job):
    result_path=HERE/f'{label}-job-{job}-results.json';result=read(result_path)
    assert result['valid'] and result['actual_upload_matches'] and result['correctness_passes']==81
    build_path=ROOT/result['build_record']['path'];assert file_record(build_path)==result['build_record'];built=read(build_path)
    ppath=HERE/f'package-{built["build_label"]}.json';assert file_record(ppath)==built['candidate_package'];package=read(ppath)
    raw=(ROOT/result['log']).read_bytes();assert digest(raw)==result['log_digest']
    assert file_record(ROOT/result['archive']['path'])==result['archive']
    with tarfile.open(ROOT/result['archive']['path']) as tar:
        members=[m for m in tar if m.isfile()];assert len(members)==len({m.name for m in members})
        files={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert files==result['package_files'] and {k:v for k,v in files.items() if k!='job.yaml'}==built['package_files']
    sides,control,origin=bindings();assert control==package['control'] and origin==package['origin']
    samples={};maps={};summaries={}
    for side in ('control','candidate'):
        folder=HERE/'evidence'/built['build_label']
        mp=folder/f'{side}-marker-map.json';metadata=folder/f'{side}-qkv.image-profile.json'
        assert digest(mp.read_bytes())==package['diagnostic_files'][mp.name]
        assert digest(metadata.read_bytes())==package['diagnostic_files'][metadata.name]
        schedule_path=folder/side/'schedule.json';assert file_record(schedule_path)==package['sides'][side]['schedule']
        mapping=read(mp);schedule=read(schedule_path);assert mapping==map_markers(schedule,read(metadata))
        stage_name=side+'_trace';stage=result['stages'][stage_name]
        content=re.findall(r'^DIAG_STAGE_BEGIN '+stage_name+r'\n(.*?)^DIAG_STAGE_END '+stage_name+'$',raw.decode(),re.M|re.S);assert len(content)==1
        stage_spec=next(x for x in built['stages'] if x['name']==stage_name)
        assert parse_diagnostic(content[0],stage_spec,mapping)=={k:v for k,v in stage.items() if k!='exit_code'}
        by_index={r['schedule']['index']:r for r in mapping['operations']}
        assert set(CONFIG[side].values())<=by_index.keys()
        # Bind numerical/transfer names to actual operation bodies and producer edges.
        for n in NATIVES[side]:assert 'contract' in by_index[n]['schedule']['description'] and by_index[n]['schedule']['tpe']=='Main'
        if side=='candidate':
            assert 'InterSliceReduceOpF32::Max' in by_index[CONFIG[side]['global_k4_amax']]['schedule']['description']
            assert '.to_trf()' in by_index[CONFIG[side]['activation_trf']]['schedule']['description']
            assert 'x.to_dm' in by_index[CONFIG[side]['consumer_input_dma']]['schedule']['description']
        fs=function_names((folder/side/'production-projection.rs').read_text())
        observed=[]
        for obs in stage['trace_observations']:
            ops={r['schedule_index']:r for r in obs['operations']}
            assert set(ops)==set(by_index)-set(mapping['cluster_absent'][str(obs['cluster'])])
            selected={name:ops[idx] for name,idx in CONFIG[side].items() if idx in ops}
            metrics={}
            for projection in ('q','k','v'):
                native=selected[projection+'_native']
                metrics[projection+'_weight_ready_to_native']=native['begin_from_task']-selected[projection+'_weight']['end_from_task']
                metrics[projection+'_native_end_to_task_end']=obs['task_cycles']-native['end_from_task']
                if side=='candidate':
                    metrics[projection+'_trf_ready_to_native']=native['begin_from_task']-selected['activation_trf']['end_from_task']
            if side=='candidate':
                for name,start,stop in [('pre_rms_end_to_consumer_input_ready','pre_rms_finish','consumer_input_dma'),('input_ready_to_quant_done','consumer_input_dma','quant_done'),('quant_done_to_trf_ready','quant_done','activation_trf'),('pre_rms_end_to_trf_ready','pre_rms_finish','activation_trf')]:
                    metrics[name]=selected[stop]['end_from_task']-selected[start]['end_from_task']
            observed.append({'run':obs['run'],'cluster':obs['cluster'],'task_cycles':obs['task_cycles'],'selected':selected,'interval_metrics':metrics,'all_operations':ops})
        samples[side]=observed
        summaries[side]={'task_cycles':stage['cycles'],'selected':{},'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observed]) for k in observed[0]['interval_metrics']},'whole_median':result['stages']['candidate_whole' if side=='candidate' else 'control_before']['kernels'][FFN]['median_cycles']}
        for name,idx in CONFIG[side].items():
            summaries[side]['selected'][name]={'schedule_index':idx,**{key:stats([o['selected'][name][key] for o in observed if name in o['selected']]) for key in ['begin_from_task','end_from_task','duration']}}
        maps[side]={'mapping':file_record(mp),'schedule':file_record(schedule_path),'operation_count':len(mapping['operations']),'all_operations':{}}
        for idx,row in by_index.items():
            n=row['schedule'];match=re.search(r'src/device/sliding/projection.rs:(\d+):',n['description']);line=int(match[1]) if match else 0
            fn=next((name for start,name in reversed(fs) if start<=line),'external shared helper/table')
            maps[side]['all_operations'][idx]={'marker':row['marker'],'source_function':fn,'source_line':line,'schedule':n,'measurements':[{k:o['all_operations'][idx][k] for k in ['begin_from_task','end_from_task','duration']}|{'run':o['run'],'cluster':o['cluster']} for o in observed if idx in o['all_operations']]}
    comparison={name:{point: summaries['candidate']['selected'][name][point]['median']-summaries['control']['selected'][name][point]['median'] for point in ['begin_from_task','end_from_task','duration']} for name in CONFIG['control'] if name in CONFIG['candidate']}
    return {'id':'E138-NATIVE-V1-READINESS','valid':True,'job':job,'source_result':file_record(result_path),'package':file_record(ppath),'source_maps':maps,'samples':samples,'summaries':summaries,'candidate_minus_control_median_landmark_cycles':comparison,'profiling_effect':{side:{'standard_info':summaries[side]['whole_median'],'diagnostic_info':result['stages'][side+'_info']['cycles']['median'],'diagnostic_trace':result['stages'][side+'_trace']['cycles']['median']} for side in sides},'whole_comparison':result['whole_qkv_comparison'],'method':'Three original inputs, two clusters per trace, one allocation. Times relative to each cluster Task. All native markers checked against entire source-schedule sequence and issue order.','limitations':['Marker intervals include queue/dependency waits and are not isolated service times.','Interval metrics are elapsed windows, not sums of overlapping durations.','Trace timestamps and info/whole standard host timings have different profiling modes; trace is not a speedup/adoption sample.','Moving quantization changes input transfer, active DM, reduction ratio and schedule; an observed interval is not its predicted net saving.','A prefix-ready delta is observed; no causal bus stall fraction is inferred without a supported dependency edge.'],'analyzer':file_record(Path(__file__))}

FFN='sliding_project_qkv'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('job',type=int);p.add_argument('--check',action='store_true');a=p.parse_args();v=analyze(a.label,a.job);out=HERE/'profile-analysis.json'
    if a.check:assert read(out)==v
    else:write(out,v)
    print(json.dumps({'whole':v['whole_comparison'],'summaries':{side:{name:{k:r[k]['median'] for k in ['begin_from_task','end_from_task','duration']} for name,r in s['selected'].items()} for side,s in v['summaries'].items()}},indent=2))
