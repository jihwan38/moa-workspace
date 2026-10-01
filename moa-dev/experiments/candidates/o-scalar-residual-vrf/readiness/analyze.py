#!/usr/bin/env python3
"""Reparse actual source-bound O/current-FFN traces without summing overlap."""
import argparse,json,re,statistics,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE))
from package import read,write,file_record,digest,bindings,map_markers
from screen import parse_diagnostic

def stats(values):return {'median':statistics.median(values),'min':min(values),'max':max(values),'samples':values}
def analyze(label,job):
    rp=HERE/f'{label}-job-{job}-results.json';r=read(rp)
    assert r['valid'] and r['actual_upload_matches'] and r['correctness_passes']==63
    bp=ROOT/r['build_record']['path'];assert file_record(bp)==r['build_record'];b=read(bp)
    pp=ROOT/b['candidate_package']['path'];assert file_record(pp)==b['candidate_package'];p=read(pp)
    raw=(ROOT/r['log']).read_bytes();assert digest(raw)==r['log_digest'];assert file_record(ROOT/r['archive']['path'])==r['archive']
    with tarfile.open(ROOT/r['archive']['path']) as t:
        ms=[m for m in t if m.isfile()];assert len(ms)==len({m.name for m in ms});files={m.name:digest(t.extractfile(m).read()) for m in ms}
    assert files==r['package_files'] and {k:v for k,v in files.items() if k!='job.yaml'}==b['package_files']
    _,control,origin=bindings();assert control==p['control'] and origin==p['origin']
    cfg=read(HERE/'landmarks.json');summaries={};samples={};maps={}
    for side in ['control','candidate','ffn']:
        f=HERE/'evidence'/b['build_label'];mp=f/f'{side}-marker-map.json';md=f/('ffn.image-profile.json' if side=='ffn' else f'{side}-o.image-profile.json')
        for path in [mp,md]:assert digest(path.read_bytes())==p['diagnostic_files'][path.name]
        sp=f/side/'schedule.json';record=p['ffn'] if side=='ffn' else p['sides'][side];assert file_record(sp)==record['schedule']
        mapping=read(mp);assert mapping==map_markers(read(sp),read(md));bi={v['schedule']['index']:v for v in mapping['operations']};assert set(cfg[side].values())<=bi.keys()
        stage=side+'_trace';content=re.findall(r'^DIAG_STAGE_BEGIN '+stage+r'\n(.*?)^DIAG_STAGE_END '+stage+'$',raw.decode(),re.M|re.S);assert len(content)==1
        spec=next(s for s in b['stages'] if s['name']==stage);actual=parse_diagnostic(content[0],spec,mapping)
        assert actual=={k:v for k,v in r['stages'][stage].items() if k!='exit_code'}
        native_names=['gate_native','up_native','down_native'] if side=='ffn' else ['native0','native1']
        for name in native_names:assert 'contract' in bi[cfg[side][name]]['schedule']['description']
        observed=[]
        for obs in actual['trace_observations']:
            ops={x['schedule_index']:x for x in obs['operations']};sel={name:ops[i] for name,i in cfg[side].items()}
            def gap(a,z):return sel[z]['begin_from_task']-sel[a]['end_from_task']
            if side=='ffn':
                intervals={'gate_weight_end_to_native':gap('gate_weight_hbm','gate_native'),'shared_trf_end_to_gate_native':gap('shared_activation_trf','gate_native'),'up_weight_end_to_native':gap('up_weight_hbm','up_native'),'gate_native_end_to_up_native':gap('gate_native','up_native'),'geglu_end_to_down_native':gap('geglu_output','down_native'),'down_weight_end_to_native':gap('down_weight_hbm','down_native'),'down_trf_end_to_native':gap('down_activation_trf','down_native'),'geglu_end_to_down_trf_end':sel['down_activation_trf']['end_from_task']-sel['geglu_output']['end_from_task'],'down_native_end_to_output_end':sel['output_store']['end_from_task']-sel['down_native']['end_from_task']}
            else:
                intervals={'input_end_to_quant_done':sel['quant_done']['end_from_task']-sel['input_hbm']['end_from_task'],'quant_done_end_to_trf_begin':gap('quant_done','activation_trf'),'trf_end_to_native0':gap('activation_trf','native0'),'coefficient_end_to_native0':gap('coefficient_vrf','native0'),'weight0_end_to_native0':gap('weight0_hbm','native0'),'weight1_end_to_native1':gap('weight1_hbm','native1'),'native0_end_to_native1':gap('native0','native1'),'native1_end_to_output_end':sel['output_store']['end_from_task']-sel['native1']['end_from_task']}
            observed.append({'run':obs['run'],'cluster':obs['cluster'],'task_cycles':obs['task_cycles'],'selected':sel,'interval_metrics':intervals,'all_operations':ops})
        samples[side]=observed
        summaries[side]={'task_cycles':actual['cycles'],'selected':{name:{'schedule_index':i,**{k:stats([o['selected'][name][k] for o in observed]) for k in ['begin_from_task','end_from_task','duration']}} for name,i in cfg[side].items()},'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observed]) for k in observed[0]['interval_metrics']}}
        maps[side]={'operation_count':len(mapping['operations']),'mapping':file_record(mp),'schedule':file_record(sp),'selected_sources':{name:bi[i] for name,i in cfg[side].items()}}
    comparison={name:{k:summaries['candidate']['selected'][name][k]['median']-summaries['control']['selected'][name][k]['median'] for k in ['begin_from_task','end_from_task','duration']} for name in cfg['control'] if name in cfg['candidate']}
    return {'id':'E148-READINESS','valid':True,'job':job,'source_result':file_record(rp),'package':file_record(pp),'landmarks':file_record(HERE/'landmarks.json'),'analyzer':file_record(Path(__file__)),'source_maps':maps,'samples':samples,'summaries':summaries,'candidate_minus_control_median_landmark_cycles':comparison,'whole_comparison':r['whole_o_comparison'],'profiling_effect':{s:{'diagnostic_info':r['stages'][s+'_info']['cycles']['median'],'diagnostic_trace':r['stages'][s+'_trace']['cycles']['median'],'standard_info':r['stages']['candidate_whole' if s=='candidate' else 'control_before']['kernels']['decoder_feedforward' if s=='ffn' else 'sliding_attention_output']['median_cycles']} for s in ['control','candidate','ffn']},'limitations':['Six run/cluster observations per trace share one allocation, not six independent repetitions.','Endpoint summaries are medians; their differences are not medians of per-run differences. Interval metrics use paired endpoints in the same cluster clock.','Spans include waits/overlap and are not exclusive service or savings.','Current FFN is E142 exact source/image, only observed once; old E139 is not a paired control.','Names-only instrumentation and host process scope may affect timings; whole/info/trace are reported separately.']}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('label');a.add_argument('job',type=int);a.add_argument('--check',action='store_true');x=a.parse_args();v=analyze(x.label,x.job);out=HERE/'profile-analysis.json'
    if x.check:assert read(out)==v
    else:write(out,v)
    print(json.dumps({'valid':v['valid'],'whole':v['whole_comparison'],'profiles':v['profiling_effect'],'intervals':{s:{k:z['median'] for k,z in q['interval_metrics'].items()} for s,q in v['summaries'].items()}},indent=2))
