#!/usr/bin/env python3
"""Reparse paired E151/E153 FFN readiness in each actual cluster clock."""
import argparse,json,re,statistics,tarfile,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE))
from package import read,write,file_record,digest,bindings,map_markers
from screen import parse_diagnostic

def stats(v):return dict(median=statistics.median(v),min=min(v),max=max(v),samples=v)
def analyze(label,job):
    rp=HERE/f'{label}-job-{job}-results.json';r=read(rp)
    assert r['valid'] and r['actual_upload_matches'] and r['correctness_passes']==57
    bp=ROOT/r['build_record']['path'];assert file_record(bp)==r['build_record'];b=read(bp)
    pp=ROOT/b['candidate_package']['path'];assert file_record(pp)==b['candidate_package'];p=read(pp)
    raw=(ROOT/r['log']).read_bytes();assert digest(raw)==r['log_digest'];assert file_record(ROOT/r['archive']['path'])==r['archive']
    with tarfile.open(ROOT/r['archive']['path']) as t:
        ms=[m for m in t if m.isfile()];assert len(ms)==len({m.name for m in ms});files={m.name:digest(t.extractfile(m).read()) for m in ms}
    assert files==r['package_files'] and {k:v for k,v in files.items() if k!='job.yaml'}==b['package_files']
    _,control,origin=bindings();assert control==p['control'] and origin==p['origin']
    cfg=read(HERE/'landmarks.json');sides={};source_maps={}
    for side,c in cfg.items():
        f=HERE/'evidence'/b['build_label'];mp=f/f'{side}-marker-map.json';md=f/f'{side}-ffn.image-profile.json'
        for path in [mp,md]:assert digest(path.read_bytes())==p['diagnostic_files'][path.name]
        sp=f/side/'schedule.json';assert file_record(sp)==p['sides'][side]['schedule']
        mapping=read(mp);assert mapping==map_markers(read(sp),read(md));bi={v['schedule']['index']:v for v in mapping['operations']};assert set(c.values())<=bi.keys()
        stage=side+'_trace';content=re.findall(r'^DIAG_STAGE_BEGIN '+stage+r'\n(.*?)^DIAG_STAGE_END '+stage+'$',raw.decode(),re.M|re.S);assert len(content)==1
        spec=next(s for s in b['stages'] if s['name']==stage);actual=parse_diagnostic(content[0],spec,mapping)
        assert actual=={k:v for k,v in r['stages'][stage].items() if k!='exit_code'}
        for name in ['gate_native','up_native','down_native0','down_native1']:assert 'contract' in bi[c[name]]['schedule']['description']
        observed=[]
        for obs in actual['trace_observations']:
            ops={x['schedule_index']:x for x in obs['operations']};sel={name:ops[i] for name,i in c.items()}
            def beg(n):return sel[n]['begin_from_task']
            def end(n):return sel[n]['end_from_task']
            def gap(a,z):return beg(z)-end(a)
            intervals={
                'prerms_bf16_end_to_amax_begin':gap('prerms_bf16','amax'),
                'prerms_bf16_end_to_scale_end':end('scale')-end('prerms_bf16'),
                'scale_end_to_quant_dma_begin':gap('scale','quant_scale_dma'),
                'scale_end_to_quant_vrf_end':end('quant_scale_vrf')-end('scale'),
                'quant_dma_end_to_quant_vrf_begin':gap('quant_scale_dma','quant_scale_vrf'),
                'quant_vrf_end_to_term0_begin':gap('quant_scale_vrf','term0'),
                'gate_weight_end_to_quant_dma_begin':gap('gate_weight','quant_scale_dma'),
                'term2_end_to_third_copy_end':end('copy_third')-end('term2'),
                'third_copy_end_to_ring_seed_begin':gap('copy_third','ring_seed_dma'),
                'ring_seed_end_to_ring_begin':gap('ring_seed_dma','activation_ring'),
                'ring_end_to_trf_begin':gap('activation_ring','activation_trf'),
                'trf_end_to_gate_native_begin':gap('activation_trf','gate_native'),
                'gate_weight_end_to_gate_native_begin':gap('gate_weight','gate_native'),
                'up_weight_end_to_up_native_begin':gap('up_weight','up_native'),
                'gate_native_end_to_up_native_begin':gap('gate_native','up_native'),
                'down_weight0_end_to_up_gather_begin':gap('down_weight0','up_gather'),
                'up_gather_end_to_geglu_end':end('geglu')-end('up_gather'),
                'geglu_end_to_down_trf_end':end('down_trf')-end('geglu'),
                'down_trf_end_to_native0_begin':gap('down_trf','down_native0'),
                'down_weight1_end_to_native1_begin':gap('down_weight1','down_native1'),
                'last_native_end_to_down_scale_begin':gap('down_native1','down_scale_dma'),
                'last_native_end_to_first_reduce_begin':gap('down_native1','down_reduce0'),
                'down_scale_end_to_first_reduce_begin':gap('down_scale_dma','down_reduce0'),
                'last_native_end_to_output_end':end('output_store')-end('down_native1'),
            }
            if side=='candidate':
                intervals.update(third_copy_end_to_late_copy_begin=gap('copy_third','late_packed_copy'),late_copy_end_to_ring_seed_begin=gap('late_packed_copy','ring_seed_dma'),global_max_end_to_scale_end=end('scale')-end('global_max'))
            observed.append(dict(run=obs['run'],cluster=obs['cluster'],task_cycles=obs['task_cycles'],selected=sel,interval_metrics=intervals,all_operations=ops))
        summary={'task_cycles':actual['cycles'],'selected':{name:{'schedule_index':i,**{k:stats([o['selected'][name][k] for o in observed]) for k in ['begin_from_task','end_from_task','duration']}} for name,i in c.items()},'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observed]) for k in observed[0]['interval_metrics']}}
        sides[side]={'samples':observed,'summary':summary}
        source_maps[side]={'operation_count':len(mapping['operations']),'mapping':file_record(mp),'schedule':file_record(sp),'selected_sources':{name:bi[i] for name,i in c.items()}}
    common=cfg['control'].keys()&cfg['candidate'].keys()
    differences={name:{k:sides['candidate']['summary']['selected'][name][k]['median']-sides['control']['summary']['selected'][name][k]['median'] for k in ['begin_from_task','end_from_task','duration']} for name in sorted(common)}
    profiles={side:{'info':r['stages'][side+'_info']['cycles']['median'],'trace':r['stages'][side+'_trace']['cycles']['median']} for side in cfg}
    return dict(id='E153-READINESS',valid=True,job=job,source_result=file_record(rp),package=file_record(pp),landmarks=file_record(HERE/'landmarks.json'),analyzer=file_record(Path(__file__)),source_maps=source_maps,sides=sides,endpoint_median_differences_candidate_minus_control=differences,whole_comparison=r['whole_ffn_comparison'],profiling_effect=profiles,limitations=['One allocation with three original inputs and two clusters per trace; six observations are not six independent allocations.','Endpoint median differences are not paired per-run deltas across the two serial stages. Within-side interval metrics use endpoints from the same run/cluster clock.','Both current E151 and E153 internal traces are observed; historical E150 is not used as the control.','Spans include waits and overlap; no durations are summed as service, bus stalls, or obtainable speedup.','Advancing one producer need not advance whole completion because resource scheduling and overlap can change.','This names-only diagnostic is not an additional adoption performance allocation.'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('label');a.add_argument('job',type=int);a.add_argument('--check',action='store_true');x=a.parse_args();v=analyze(x.label,x.job);out=HERE/'profile-analysis.json'
    if x.check:assert read(out)==v
    else:write(out,v)
    print(json.dumps({'valid':v['valid'],'whole':v['whole_comparison'],'profiles':v['profiling_effect'],'intervals':{side:{k:z['median'] for k,z in d['summary']['interval_metrics'].items()} for side,d in v['sides'].items()}},indent=2))
