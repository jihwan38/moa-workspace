#!/usr/bin/env python3
"""Source-bound complete FFN profiles. Never add overlapping spans as cycle savings."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import tarfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from package import ARTIFACTS,digest,file_record,read,write,operation_kind,map_markers,bindings
from screen import parse_diagnostic
CONFIG={
'control':{'input_hbm':'24','pre_rms_finish':'136','initial_replication':'148','amax':'172','quantizer_scale':'188','terms_assembled':'275','terms_leader_gather':'284','term_broadcast':'294','input_scale_vrf':'303','input_trf':'314','up_packed_hbm':'305','gate_packed_hbm':'115','up_fp8_materialization':'375','gate_fp8_materialization':'145','up_scale_hbm':'377','gate_scale_hbm':'160','first_scale_vrf':'274','first_native':'318','last_native':'468','gate_local_gather':'376','up_local_gather':'474','geglu_output':'488','down_activation_compact':'500','down_activation_leader':'502','down_activation_broadcast':'515','down_activation_trf':'520','down_packed_hbm':'379','down_fp8_materialization':'499','down_bf16_dot':'525','down_scale_reduce_last':'610','down_result_gather':'618','down_cluster_reduce':'628','output_store':'672'},
'candidate':{'input_hbm':'132','pre_rms_finish':'175','initial_replication':'184','amax':'197','quantizer_scale':'213','terms_assembled':'279','terms_leader_gather':'288','term_broadcast':'305','input_scale_vrf':'313','input_trf':'366','up_packed_hbm':'16','gate_packed_hbm':'105','up_packed_copy':'103','gate_packed_copy':'115','up_scale_hbm':'116','gate_scale_hbm':'153','up_scale_copy':'131','gate_scale_copy':'180','first_scale_vrf':'278','first_native_table':'285','last_native_table':'353','first_native':'375','last_native':'464','gate_local_gather':'472','up_local_gather':'476','geglu_output':'512','down_activation_compact':'516','down_activation_leader':'519','down_activation_broadcast':'542','down_activation_trf':'547','down_packed_hbm':'348','down_fp8_materialization':'525','down_bf16_dot':'553','down_scale_reduce_last':'633','down_result_gather':'640','down_cluster_reduce':'661','output_store':'705'}}
NATIVES={'control':['318','331','343','355','366','397','428','445','457','468'],'candidate':['375','392','405','417','429','441','453','464']}
TABLES={'control':['23','182','281','492'],'candidate':['285','294','319','326','333','339','346','353','506','529']}

def stats(vals):return {'median':statistics.median(vals),'min':min(vals),'max':max(vals),'samples':vals}
def function_names(src):return [(i+1,re.search(r'fn (\w+)',t).group(1)) for i,t in enumerate(src.splitlines()) if re.match(r'^(?:pub\(crate\) )?fn \w+',t)]

def analyze(label,job):
    result_path=HERE/f'{label}-job-{job}-results.json';result=read(result_path)
    assert result['valid'] and result['actual_upload_matches'] and result['correctness_passes']==57
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
        mp=folder/f'{side}-marker-map.json';metadata=folder/f'{side}-ffn.image-profile.json'
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
        for name in ['input_trf','down_activation_trf']:assert '.to_trf()' in by_index[CONFIG[side][name]]['schedule']['description']
        for name in ['term_broadcast','down_activation_broadcast']:assert '.switch::' in by_index[CONFIG[side][name]]['schedule']['description']
        assert 'to_hbm_view' in by_index[CONFIG[side]['output_store']]['schedule']['description']
        down=by_index[CONFIG[side]['down_bf16_dot']]['schedule']
        assert 'fetch_table_lookup::<bf16>' in down['description']
        assert by_index[CONFIG[side]['down_fp8_materialization']]['schedule']['output_tensors'][0] in down['input_tensors']
        assert by_index[CONFIG[side]['down_activation_trf']]['schedule']['output_tensors'][0] in down['input_tensors']
        fs=function_names((folder/side/'production-mlp.rs').read_text())
        observed=[]
        for obs in stage['trace_observations']:
            ops={r['schedule_index']:r for r in obs['operations']}
            assert len(ops)==len(mapping['operations'])
            selected={name:ops[idx] for name,idx in CONFIG[side].items()}
            metrics={
                'upgate_native_envelope':ops[NATIVES[side][-1]]['end_from_task']-ops[NATIVES[side][0]]['begin_from_task'],
                'input_hbm_begin_to_first_native':selected['first_native']['begin_from_task']-selected['input_hbm']['begin_from_task'],
                'pre_rms_end_to_first_native':selected['first_native']['begin_from_task']-selected['pre_rms_finish']['end_from_task'],
                'geglu_ready_to_task_end':obs['task_cycles']-selected['geglu_output']['end_from_task'],
                'down_activation_trf_ready_to_task_end':obs['task_cycles']-selected['down_activation_trf']['end_from_task'],
                'first_down_dot_begin_to_task_end':obs['task_cycles']-selected['down_bf16_dot']['begin_from_task'],
                'last_native_end_to_geglu_end':selected['geglu_output']['end_from_task']-selected['last_native']['end_from_task']}
            if side=='candidate':
                metrics.update({'packed_gate_hbm_end_to_input_hbm_begin':selected['input_hbm']['begin_from_task']-selected['gate_packed_hbm']['end_from_task'],
                    'up_scale_hbm_end_to_input_hbm_begin':selected['input_hbm']['begin_from_task']-selected['up_scale_hbm']['end_from_task'],
                    'gate_packed_copy_end_to_pre_rms_begin':ops['137']['begin_from_task']-selected['gate_packed_copy']['end_from_task']})
            observed.append({'run':obs['run'],'cluster':obs['cluster'],'task_cycles':obs['task_cycles'],'selected':selected,'interval_metrics':metrics,'all_operations':ops})
        samples[side]=observed
        summaries[side]={'task_cycles':stage['cycles'],'selected':{},'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observed]) for k in observed[0]['interval_metrics']},'whole_median':result['stages']['candidate_whole' if side=='candidate' else 'control_before']['kernels'][FFN]['median_cycles']}
        for name,idx in CONFIG[side].items():
            summaries[side]['selected'][name]={'schedule_index':idx,**{key:stats([o['selected'][name][key] for o in observed]) for key in ['begin_from_task','end_from_task','duration']}}
        maps[side]={'mapping':file_record(mp),'schedule':file_record(schedule_path),'operation_count':len(mapping['operations']),'all_operations':{}}
        for idx,row in by_index.items():
            n=row['schedule'];match=re.search(r'src/device/shared/mlp.rs:(\d+):',n['description']);line=int(match[1]) if match else 0
            fn=next((name for start,name in reversed(fs) if start<=line),'external shared helper/table')
            maps[side]['all_operations'][idx]={'marker':row['marker'],'source_function':fn,'source_line':line,'schedule':n,'measurements':[{k:o['all_operations'][idx][k] for k in ['begin_from_task','end_from_task','duration']}|{'run':o['run'],'cluster':o['cluster']} for o in observed]}
    comparison={name:{point: summaries['candidate']['selected'][name][point]['median']-summaries['control']['selected'][name][point]['median'] for point in ['begin_from_task','end_from_task','duration']} for name in CONFIG['control'] if name in CONFIG['candidate']}
    return {'id':'E131-COST-RECOVERY-PROFILE','valid':True,'job':job,'source_result':file_record(result_path),'package':file_record(ppath),'source_maps':maps,'samples':samples,'summaries':summaries,'candidate_minus_control_median_landmark_cycles':comparison,'whole_comparison':result['whole_ffn_comparison'],'method':'Three original inputs, two clusters per trace, one allocation. Times relative to each cluster Task. All native markers checked against entire source-schedule sequence and issue order.','limitations':['Marker intervals include queue/dependency waits and are not isolated service times.','Interval metrics are elapsed windows, not sums of overlapping durations.','Trace timestamps and info/whole standard host timings have different profiling modes; trace is not a speedup/adoption sample.','Any suffix-removal headroom assumes the same prefix/schedule; a changed down path can reschedule earlier operations.','A prefix-ready delta is observed; no causal bus stall fraction is inferred without a supported dependency edge.'],'analyzer':file_record(Path(__file__))}

FFN='decoder_feedforward'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('job',type=int);p.add_argument('--check',action='store_true');a=p.parse_args();v=analyze(a.label,a.job);out=HERE/'profile-analysis.json'
    if a.check:assert read(out)==v
    else:write(out,v)
    print(json.dumps({'whole':v['whole_comparison'],'summaries':{side:{name:{k:r[k]['median'] for k in ['begin_from_task','end_from_task','duration']} for name,r in s['selected'].items()} for side,s in v['summaries'].items()}},indent=2))
