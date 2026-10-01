#!/usr/bin/env python3
"""Reparse actual source-bound E150 FFN traces without summing overlap."""
import argparse,json,re,statistics,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE))
from package import read,write,file_record,digest,bindings,map_markers
from screen import parse_diagnostic

def stats(values):return {'median':statistics.median(values),'min':min(values),'max':max(values),'samples':values}
def analyze(label,job):
    rp=HERE/f'{label}-job-{job}-results.json';r=read(rp)
    assert r['valid'] and r['actual_upload_matches'] and r['correctness_passes']==51
    bp=ROOT/r['build_record']['path'];assert file_record(bp)==r['build_record'];b=read(bp)
    pp=ROOT/b['candidate_package']['path'];assert file_record(pp)==b['candidate_package'];p=read(pp)
    raw=(ROOT/r['log']).read_bytes();assert digest(raw)==r['log_digest'];assert file_record(ROOT/r['archive']['path'])==r['archive']
    with tarfile.open(ROOT/r['archive']['path']) as t:
        ms=[m for m in t if m.isfile()];assert len(ms)==len({m.name for m in ms});files={m.name:digest(t.extractfile(m).read()) for m in ms}
    assert files==r['package_files'] and {k:v for k,v in files.items() if k!='job.yaml'}==b['package_files']
    _,control,origin=bindings();assert control==p['control'] and origin==p['origin']
    cfg=read(HERE/'landmarks.json');side='candidate'
    f=HERE/'evidence'/b['build_label'];mp=f/'candidate-marker-map.json';md=f/'candidate-ffn.image-profile.json'
    for path in [mp,md]:assert digest(path.read_bytes())==p['diagnostic_files'][path.name]
    sp=f/side/'schedule.json';assert file_record(sp)==p['sides'][side]['schedule']
    mapping=read(mp);assert mapping==map_markers(read(sp),read(md));bi={v['schedule']['index']:v for v in mapping['operations']};assert set(cfg.values())<=bi.keys()
    stage='candidate_trace';content=re.findall(r'^DIAG_STAGE_BEGIN '+stage+r'\n(.*?)^DIAG_STAGE_END '+stage+'$',raw.decode(),re.M|re.S);assert len(content)==1
    spec=next(s for s in b['stages'] if s['name']==stage);actual=parse_diagnostic(content[0],spec,mapping)
    assert actual=={k:v for k,v in r['stages'][stage].items() if k!='exit_code'}
    for name in ['gate_native','up_native','down_native0','down_native1']:assert 'contract' in bi[cfg[name]]['schedule']['description']
    observed=[]
    for obs in actual['trace_observations']:
        ops={x['schedule_index']:x for x in obs['operations']};sel={name:ops[i] for name,i in cfg.items()}
        def beg(n):return sel[n]['begin_from_task']
        def end(n):return sel[n]['end_from_task']
        def gap(a,z):return beg(z)-end(a)
        intervals={
            'down_weight0_end_to_up_gather_begin':gap('down_weight0_hbm','up_gather'),
            'up_gather_end_to_geglu_end':end('geglu_output')-end('up_gather'),
            'geglu_end_to_down_weight1_begin':gap('geglu_output','down_weight1_hbm'),
            'geglu_end_to_down_trf_end':end('down_activation_trf')-end('geglu_output'),
            'down_trf_end_to_native0_begin':gap('down_activation_trf','down_native0'),
            'down_weight0_end_to_native0_begin':gap('down_weight0_hbm','down_native0'),
            'down_table0_end_to_native0_begin':gap('down_table0','down_native0'),
            'down_coefficient_end_to_native0_begin':gap('down_coefficient_vrf','down_native0'),
            'down_native0_end_to_native1_begin':gap('down_native0','down_native1'),
            'down_weight1_end_to_native1_begin':gap('down_weight1_hbm','down_native1'),
            'down_table1_end_to_native1_begin':gap('down_table1','down_native1'),
            'down_native1_end_to_blockscale_dma_begin':gap('down_native1','down_blockscale_hbm'),
            'down_native1_end_to_first_reduce_begin':gap('down_native1','down_reduce0'),
            'down_blockscale_dma_end_to_first_vrf_begin':gap('down_blockscale_hbm','down_blockscale_vrf0'),
            'down_blockscale_dma_end_to_first_reduce_begin':gap('down_blockscale_hbm','down_reduce0'),
            'down_native1_end_to_output_end':end('output_store')-end('down_native1'),
        }
        for i in range(5):intervals[f'down_blockscale_vrf{i}_end_to_reduce{i}_begin']=gap(f'down_blockscale_vrf{i}',f'down_reduce{i}')
        predicates={
            'up_gather_geglu_between_down_weight_transfers':end('down_weight0_hbm')<=beg('up_gather')<=end('geglu_output')<=beg('down_weight1_hbm'),
            'blockscale_transfer_ends_after_last_native':end('down_blockscale_hbm')>end('down_native1'),
            'first_reduce_starts_after_blockscale_transfer':beg('down_reduce0')>=end('down_blockscale_hbm'),
            'first_reduce_starts_after_last_native':beg('down_reduce0')>=end('down_native1'),
        }
        observed.append({'run':obs['run'],'cluster':obs['cluster'],'task_cycles':obs['task_cycles'],'selected':sel,'interval_metrics':intervals,'predicates':predicates,'all_operations':ops})
    summary={'task_cycles':actual['cycles'],'selected':{name:{'schedule_index':i,**{k:stats([o['selected'][name][k] for o in observed]) for k in ['begin_from_task','end_from_task','duration']}} for name,i in cfg.items()},'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observed]) for k in observed[0]['interval_metrics']},'predicate_counts':{k:{'true':sum(o['predicates'][k] for o in observed),'observations':len(observed)} for k in observed[0]['predicates']}}
    source_map={'operation_count':len(mapping['operations']),'mapping':file_record(mp),'schedule':file_record(sp),'selected_sources':{name:bi[i] for name,i in cfg.items()}}
    return {'id':'E150-READINESS','valid':True,'job':job,'source_result':file_record(rp),'package':file_record(pp),'landmarks':file_record(HERE/'landmarks.json'),'analyzer':file_record(Path(__file__)),'source_map':source_map,'samples':observed,'summary':summary,'whole_comparison':r['whole_ffn_comparison'],'profiling_effect':{'diagnostic_info':r['stages']['candidate_info']['cycles']['median'],'diagnostic_trace':r['stages']['candidate_trace']['cycles']['median'],'standard_info':r['stages']['candidate_whole']['kernels']['decoder_feedforward']['median_cycles']},'limitations':['Six run/cluster observations share one allocation, not six independent repetitions.','Endpoint summaries are medians; their differences are not medians of per-run differences. Interval metrics use paired endpoints in the same cluster clock.','Spans include waits and overlap and are not exclusive service or savings. No elapsed spans are summed.','Only E150 has internal traces in this allocation. Historical E147 internal traces are not contemporaneous controls.','Whole/info/trace use different host process scopes. This fixed diagnostic is not an additional adoption performance allocation.','A readiness boundary does not by itself identify a pure bus stall or guarantee that changing one producer advances final completion.']}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('label');a.add_argument('job',type=int);a.add_argument('--check',action='store_true');x=a.parse_args();v=analyze(x.label,x.job);out=HERE/'profile-analysis.json'
    if x.check:assert read(out)==v
    else:write(out,v)
    print(json.dumps({'valid':v['valid'],'whole':v['whole_comparison'],'profiles':v['profiling_effect'],'intervals':{k:z['median'] for k,z in v['summary']['interval_metrics'].items()},'predicates':v['summary']['predicate_counts']},indent=2))
