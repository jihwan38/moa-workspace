#!/usr/bin/env python3
"""Source-bound complete O profiles. Never add overlapping spans as cycle savings."""
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
from screen import parse_diagnostic
CONFIG={
'control':{'first_table_dma':'5','first_table_ready':'24','input_hbm':'25','weight0_hbm':'27','activation_trf':'32','native0':'37','second_table_dma':'38','second_table_ready':'41','weight1_hbm':'42','native1':'53','channel_hbm':'54','channel_vrf':'58','canonical_gather':'81','postscale':'89','rms_apply':'115','residual_add':'120','output_store':'123'},
'candidate':{'input_hbm':'7','local_amax':'46','weight0_hbm':'47','global_amax':'52','pow2_scale':'57','coefficient0':'62','coefficient1':'69','coefficient2':'76','coefficient3':'83','coefficient_pack':'89','quant_scale_dma':'90','quant_scale_vrf':'96','coefficient_rep_dma':'97','term0_quant':'101','hi_decode_vrf':'108','term1_quant':'114','sum01_decode':'121','sum01_vrf':'126','term2_quant':'131','sum012_decode':'138','sum012_vrf':'143','channel_hbm':'149','quant_done':'154','coefficient_vrf':'156','channel_vrf':'163','term_dma':'167','activation_trf':'175','weight1_hbm':'177','native0':'182','native1':'190','common_scale_dma':'191','common_scale_vrf':'195','canonical_gather':'209','postscale':'228','rms_apply':'254','residual_add':'259','output_store':'264'}}
NATIVES={'control':['37','53'],'candidate':['182','190']}

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
        mp=folder/f'{side}-marker-map.json';metadata=folder/f'{side}-o.image-profile.json'
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
        assert '.to_trf()' in by_index[CONFIG[side]['activation_trf']]['schedule']['description']
        assert 'to_hbm_view' in by_index[CONFIG[side]['output_store']]['schedule']['description']
        for name in ['weight0_hbm','weight1_hbm']:assert 'weight.tile' in by_index[CONFIG[side][name]]['schedule']['description']
        if side=='candidate':
            assert 'packed.to_dm' in by_index[CONFIG[side]['term_dma']]['schedule']['description']
            assert 'InterSliceReduceOpF32::Max' in by_index[CONFIG[side]['global_amax']]['schedule']['description']
            assert '134217728.0f32' in by_index[CONFIG[side]['quant_done']]['schedule']['description']
            assert 'fetch_table_lookup' not in by_index[NATIVES[side][0]]['schedule']['description']
        else:assert 'fetch_table_lookup::<bf16>' in by_index[NATIVES[side][0]]['schedule']['description']
        fs=function_names((folder/side/'production-projection.rs').read_text())
        observed=[]
        for obs in stage['trace_observations']:
            ops={r['schedule_index']:r for r in obs['operations']}
            assert len(ops)==len(mapping['operations'])
            selected={name:ops[idx] for name,idx in CONFIG[side].items()}
            metrics={
                'input_hbm_begin_to_native0':selected['native0']['begin_from_task']-selected['input_hbm']['begin_from_task'],
                'activation_trf_end_to_native0':selected['native0']['begin_from_task']-selected['activation_trf']['end_from_task'],
                'weight0_end_to_native0':selected['native0']['begin_from_task']-selected['weight0_hbm']['end_from_task'],
                'native0_end_to_native1_begin':selected['native1']['begin_from_task']-selected['native0']['end_from_task'],
                'weight1_end_to_native1':selected['native1']['begin_from_task']-selected['weight1_hbm']['end_from_task'],
                'native1_end_to_output_store_end':selected['output_store']['end_from_task']-selected['native1']['end_from_task']}
            if side=='candidate':
                metrics.update({'term_dma_end_to_weight1_begin':selected['weight1_hbm']['begin_from_task']-selected['term_dma']['end_from_task'],
                    'quant_done_end_to_term_dma_begin':selected['term_dma']['begin_from_task']-selected['quant_done']['end_from_task'],
                    'input_hbm_end_to_quant_done':selected['quant_done']['end_from_task']-selected['input_hbm']['end_from_task'],
                    'global_amax_end_to_quant_done':selected['quant_done']['end_from_task']-selected['global_amax']['end_from_task']})
            observed.append({'run':obs['run'],'cluster':obs['cluster'],'task_cycles':obs['task_cycles'],'selected':selected,'interval_metrics':metrics,'all_operations':ops})
        samples[side]=observed
        summaries[side]={'task_cycles':stage['cycles'],'selected':{},'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observed]) for k in observed[0]['interval_metrics']},'whole_median':result['stages']['candidate_whole' if side=='candidate' else 'control_before']['kernels'][FFN]['median_cycles']}
        for name,idx in CONFIG[side].items():
            summaries[side]['selected'][name]={'schedule_index':idx,**{key:stats([o['selected'][name][key] for o in observed]) for key in ['begin_from_task','end_from_task','duration']}}
        maps[side]={'mapping':file_record(mp),'schedule':file_record(schedule_path),'operation_count':len(mapping['operations']),'all_operations':{}}
        for idx,row in by_index.items():
            n=row['schedule'];match=re.search(r'src/device/sliding/projection.rs:(\d+):',n['description']);line=int(match[1]) if match else 0
            fn=next((name for start,name in reversed(fs) if start<=line),'external shared helper/table')
            maps[side]['all_operations'][idx]={'marker':row['marker'],'source_function':fn,'source_line':line,'schedule':n,'measurements':[{k:o['all_operations'][idx][k] for k in ['begin_from_task','end_from_task','duration']}|{'run':o['run'],'cluster':o['cluster']} for o in observed]}
    comparison={name:{point: summaries['candidate']['selected'][name][point]['median']-summaries['control']['selected'][name][point]['median'] for point in ['begin_from_task','end_from_task','duration']} for name in CONFIG['control'] if name in CONFIG['candidate']}
    return {'id':'E137-COST-RECOVERY-PROFILE','valid':True,'job':job,'source_result':file_record(result_path),'package':file_record(ppath),'source_maps':maps,'samples':samples,'summaries':summaries,'candidate_minus_control_median_landmark_cycles':comparison,'profiling_effect':{side:{'standard_info':summaries[side]['whole_median'],'diagnostic_info':result['stages'][side+'_info']['cycles']['median'],'diagnostic_trace':result['stages'][side+'_trace']['cycles']['median']} for side in sides},'whole_comparison':result['whole_o_comparison'],'method':'Three original inputs, two clusters per trace, one allocation. Times relative to each cluster Task. All native markers checked against entire source-schedule sequence and issue order.','limitations':['Marker intervals include queue/dependency waits and are not isolated service times.','Interval metrics are elapsed windows, not sums of overlapping durations.','Trace timestamps and info/whole standard host timings have different profiling modes; trace is not a speedup/adoption sample.','Moving quantization changes input transfer, active DM, reduction ratio and schedule; an observed interval is not its predicted net saving.','A prefix-ready delta is observed; no causal bus stall fraction is inferred without a supported dependency edge.'],'analyzer':file_record(Path(__file__))}

FFN='sliding_attention_output'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('job',type=int);p.add_argument('--check',action='store_true');a=p.parse_args();v=analyze(a.label,a.job);out=HERE/'profile-analysis.json'
    if a.check:assert read(out)==v
    else:write(out,v)
    print(json.dumps({'whole':v['whole_comparison'],'summaries':{side:{name:{k:r[k]['median'] for k in ['begin_from_task','end_from_task','duration']} for name,r in s['selected'].items()} for side,s in v['summaries'].items()}},indent=2))
