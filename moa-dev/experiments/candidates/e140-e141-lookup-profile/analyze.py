#!/usr/bin/env python3
"""Compare frozen E140/E141 whole-FFN readiness on each run/cluster clock."""
import argparse,json,re,statistics,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from package import bindings,digest,file_record,read,write,map_markers,operation_kind,FFN
from screen import parse_diagnostic
CONFIG={
'control':{'input_hbm':'26','pre_rms_finish':'141','initial_replication':'160','amax':'181','quantizer_scale':'197','quantizer_third':'267','terms_assembled':'280','terms_leader_gather':'287','term_broadcast':'309','input_scale_vrf':'295','input_trf':'324','gate_packed_hbm':'120','gate_fp8_materialization':'154','gate_scale_hbm':'157','gate_native_first':'332','gate_native_last':'379','gate_local_gather':'385','gate_first_bf16':'388','up_packed_hbm':'311','up_fp8_materialization':'401','up_scale_hbm':'389','up_native_first':'430','up_native_last':'481','up_local_gather':'487','up_first_bf16':'490','geglu_output':'501','down_packed_hbm':'410','down_table_ready':'505','down_activation_canonical':'509','down_amax':'512','down_quant_third':'607','down_term_broadcast':'632','down_coeff_vrf':'640','down_activation_trf':'648','down_native':'652','down_reduce_first':'670','down_reduce_last':'722','down_result_gather':'730','down_finish':'740','post_rms_finish':'771','output_store':'784'},
'candidate':{'input_hbm':'15','pre_rms_finish':'135','initial_replication':'140','amax':'161','quantizer_scale':'170','quantizer_third':'215','terms_assembled':'243','terms_leader_gather':'265','term_broadcast':'282','input_scale_vrf':'238','input_trf':'322','gate_packed_hbm':'221','gate_scale_hbm':'114','gate_native_first':'331','gate_native_last':'384','gate_local_gather':'392','gate_first_bf16':'403','up_packed_hbm':'303','up_scale_hbm':'296','up_native_first':'431','up_native_last':'517','up_local_gather':'523','up_first_bf16':'527','geglu_output':'538','down_packed_hbm':'465','down_table_ready':'541','down_activation_canonical':'545','down_amax':'548','down_quant_third':'630','down_term_broadcast':'656','down_coeff_vrf':'665','down_activation_trf':'673','down_native':'677','down_reduce_first':'699','down_reduce_last':'747','down_result_gather':'754','down_finish':'775','post_rms_finish':'806','output_store':'819'}}
NATIVES={'control':['332','344','356','368','379','430','446','458','470','481'],
'candidate':['331','348','360','372','384','431','460','483','500','517']}

def stats(vals):return {'median':statistics.median(vals),'min':min(vals),'max':max(vals),'samples':vals}
def overlap(a,b):return max(0,min(a['end_from_task'],b['end_from_task'])-max(a['begin_from_task'],b['begin_from_task']))
def union(spans):
    total=0;end=-1
    for row in sorted(spans,key=lambda r:r['begin_from_task']):
        a,b=row['begin_from_task'],row['end_from_task'];total+=max(0,b-max(a,end));end=max(end,b)
    return total

def source_plan(side,schedule,mapping,source):
    marked={x['schedule']['index']:x for x in mapping['operations']}
    allops={x['index']:x for x in schedule['instructions']}
    producers={str(t):n['index'] for n in schedule['instructions'] for t in n['output_tensors']}
    def ancestors(t,seen=()):
        idx=producers.get(str(t))
        if idx is None:return []
        assert idx not in seen
        if idx in marked:return [idx]
        return sorted({a for p in allops[idx]['input_tensors'] for a in ancestors(p,seen+(idx,))},key=int)
    funcs=[(i+1,re.search(r'fn (\w+)',t).group(1)) for i,t in enumerate(source.splitlines()) if re.match(r'^(?:pub\(crate\) )?fn \w+',t)]
    mapped={}
    for idx,row in marked.items():
        op=row['schedule'];d=op['description'];match=re.search(r'src/device/shared/mlp.rs:(\d+):',d);line=int(match[1]) if match else 0
        fn=next((n for start,n in reversed(funcs) if start<=line),'external shared helper/table')
        mapped[idx]={'marker':row['marker'],'source_function':fn,'source_line':line,'schedule':op}
    assert set(CONFIG[side].values())<=marked.keys()
    for name in ('input_trf','down_activation_trf'):assert '.to_trf()' in marked[CONFIG[side][name]]['schedule']['description']
    for name in ('term_broadcast','down_term_broadcast'):assert '.switch::' in marked[CONFIG[side][name]]['schedule']['description']
    assert 'to_hbm_view' in marked[CONFIG[side]['output_store']]['schedule']['description']
    natives=[]
    for order,idx in enumerate(NATIVES[side]):
        op=allops[idx];d=op['description'];assert op['tpe']=='Main' and 'contract' in d
        assert mapped[idx]['source_function']=='project_rows_with_scale_contraction'
        assert ('fetch_table_lookup::<f8e5m2>' in d)==(side=='candidate')
        roles=['weight','activation_trf','input_scale_vrf','block_scale_vrf']+(['lookup_table'] if side=='candidate' else [])
        dependencies={name:ancestors(t) for name,t in zip(roles,op['input_tensors'])}
        assert all(len(v)==1 for v in dependencies.values()),dependencies
        dependencies={k:v[0] for k,v in dependencies.items()}
        projection='gate' if order<5 else 'up'
        assert dependencies['activation_trf']==CONFIG[side]['input_trf']
        assert dependencies['input_scale_vrf']==CONFIG[side]['input_scale_vrf']
        weight=CONFIG[side][projection+('_packed_hbm' if side=='candidate' else '_fp8_materialization')]
        assert dependencies['weight']==weight,(side,idx,dependencies,weight)
        assert 'scales.view().tile' in marked[dependencies['block_scale_vrf']]['schedule']['description']
        if side=='candidate':
            assert operation_kind(allops[dependencies['lookup_table']])=='Renegade::StoTab'
            tab=allops[dependencies['lookup_table']];ds=ancestors(tab['input_tensors'][0]);assert len(ds)==1
            dependencies['lookup_table_dma']=ds[0];assert operation_kind(allops[ds[0]])=='DMA'
        natives.append({'index':idx,'projection':projection,'six_row_tile':order%5,'inputs':op['input_tensors'],'marked_producers':dependencies})
    table_ids=[]
    if side=='candidate':table_ids=[r['marked_producers']['lookup_table'] for r in natives]
    else:
        for projection in ('gate','up'):
            decoded=allops[CONFIG[side][projection+'_fp8_materialization']]
            tables=[a for t in decoded['input_tensors'] for a in ancestors(t) if operation_kind(allops[a])=='Renegade::StoTab']
            assert len(tables)==1;table_ids+=tables
    assert len(set(table_ids))==(10 if side=='candidate' else 2)
    return {'operations':mapped,'natives':natives,'upgate_table_ids':table_ids,'landmarks':CONFIG[side],
            'scope':'Tensor producer traversal uses only the schedule namespace. Unmarked core/view operations remain unmeasured and are not erased as execution cost.'}

def analyze(label,job):
    rp=HERE/f'{label}-job-{job}-results.json';r=read(rp)
    assert r['valid'] and r['actual_upload_matches'] and r['correctness_passes']==57
    bp=ROOT/r['build_record']['path'];assert file_record(bp)==r['build_record'];batch=read(bp)
    pp=HERE/f'package-{batch["build_label"]}.json';assert file_record(pp)==batch['candidate_package'];pkg=read(pp)
    raw=(ROOT/r['log']).read_bytes();assert digest(raw)==r['log_digest'] and file_record(ROOT/r['archive']['path'])==r['archive']
    with tarfile.open(ROOT/r['archive']['path']) as tar:
        members=[m for m in tar if m.isfile()];assert len(members)==len({m.name for m in members})
        files={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert files==r['package_files'] and {k:v for k,v in files.items() if k!='job.yaml'}==batch['package_files']
    sides,cr,origin=bindings();assert cr==pkg['control'] and origin==pkg['origin']
    samples={};summaries={};plans={}
    for side in sides:
        folder=HERE/'evidence'/batch['build_label'];mp=folder/f'{side}-marker-map.json';ip=folder/f'{side}-ffn.image-profile.json'
        for path in (mp,ip):assert digest(path.read_bytes())==pkg['diagnostic_files'][path.name]
        sp=folder/side/'schedule.json';assert file_record(sp)==pkg['sides'][side]['schedule']
        mapping=read(mp);schedule=read(sp);assert mapping==map_markers(schedule,read(ip))
        src=folder/side/'production-mlp.rs';plan=source_plan(side,schedule,mapping,src.read_text())
        stage_name=side+'_trace';stage=r['stages'][stage_name]
        content=re.findall(r'^DIAG_STAGE_BEGIN '+stage_name+r'\n(.*?)^DIAG_STAGE_END '+stage_name+'$',raw.decode(),re.M|re.S);assert len(content)==1
        stage_spec=next(s for s in batch['stages'] if s['name']==stage_name)
        assert parse_diagnostic(content[0],stage_spec,mapping)=={k:v for k,v in stage.items() if k!='exit_code'}
        observations=[]
        for obs in stage['trace_observations']:
            ops={o['schedule_index']:o for o in obs['operations']};assert len(ops)==len(mapping['operations'])
            selected={n:ops[idx] for n,idx in CONFIG[side].items()}
            def begin(n):return selected[n]['begin_from_task']
            def end(n):return selected[n]['end_from_task']
            metrics={'pre_rms_end_to_first_native':begin('gate_native_first')-end('pre_rms_finish'),
                'input_trf_end_to_first_native':begin('gate_native_first')-end('input_trf'),
                'gate_weight_hbm_end_to_first_native':begin('gate_native_first')-end('gate_packed_hbm'),
                'up_weight_hbm_end_to_first_native':begin('up_native_first')-end('up_packed_hbm'),
                'upgate_native_envelope':end('up_native_last')-begin('gate_native_first'),
                'gate_native_envelope':end('gate_native_last')-begin('gate_native_first'),
                'up_native_envelope':end('up_native_last')-begin('up_native_first'),
                'between_projection_native_gap':begin('up_native_first')-end('gate_native_last'),
                'last_native_to_geglu_ready':end('geglu_output')-end('up_native_last'),
                'geglu_ready_to_down_native':begin('down_native')-end('geglu_output'),
                'down_trf_end_to_native':begin('down_native')-end('down_activation_trf'),
                'down_weight_end_to_native':begin('down_native')-end('down_packed_hbm'),
                'down_table_end_to_native':begin('down_native')-end('down_table_ready'),
                'down_native_to_task_end':obs['task_cycles']-begin('down_native'),
                'geglu_ready_to_task_end':obs['task_cycles']-end('geglu_output'),
                'down_native_end_to_first_reduce':begin('down_reduce_first')-end('down_native'),
                'last_down_reduce_to_output':end('output_store')-end('down_reduce_last'),
                'upgate_native_interval_union':union([ops[n] for n in NATIVES[side]]),
                'upgate_table_interval_union':union([ops[n] for n in plan['upgate_table_ids']])}
            details=[]
            for i,row in enumerate(plan['natives']):
                native=ops[row['index']];prod={k:ops[v] for k,v in row['marked_producers'].items()}
                latest=max(prod,key=lambda k:prod[k]['end_from_task'])
                item={'index':row['index'],'projection':row['projection'],'six_row_tile':row['six_row_tile'],
                    'native':native,'producers':prod,'latest_marked_producer':latest,
                    'native_start_minus_latest_marked_input_end':native['begin_from_task']-prod[latest]['end_from_task'],
                    'native_start_minus_producer_end':{k:native['begin_from_task']-v['end_from_task'] for k,v in prod.items()}}
                if i:
                    prev=ops[NATIVES[side][i-1]]
                    item.update(previous_native_end_to_start=native['begin_from_task']-prev['end_from_task'],
                        table_overlap_previous_native=overlap(prod['lookup_table'],prev) if side=='candidate' else None,
                        blockscale_overlap_previous_native=overlap(prod['block_scale_vrf'],prev))
                details.append(item)
            observations.append({'run':obs['run'],'cluster':obs['cluster'],'task_cycles':obs['task_cycles'],
                'selected':selected,'interval_metrics':metrics,'native_readiness':details,'all_operations':ops})
        samples[side]=observations
        summary={'task_cycles':stage['cycles'],'operation_count':len(mapping['operations']),
                 'upgate_lookup_table_count':len(plan['upgate_table_ids']),
                 'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observations]) for k in observations[0]['interval_metrics']},
                 'selected':{},'native_readiness':[]}
        for name,idx in CONFIG[side].items():summary['selected'][name]={'schedule_index':idx,**{k:stats([o['selected'][name][k] for o in observations]) for k in ('begin_from_task','end_from_task','duration')}}
        for i,row in enumerate(plan['natives']):
            rows=[o['native_readiness'][i] for o in observations]
            summary['native_readiness'].append({'index':row['index'],'projection':row['projection'],'six_row_tile':row['six_row_tile'],
                'native_duration':stats([d['native']['duration'] for d in rows]),
                'start_minus_latest_input_end':stats([d['native_start_minus_latest_marked_input_end'] for d in rows]),
                'latest_marked_producer_per_sample':[d['latest_marked_producer'] for d in rows],
                'start_minus_producer_end':{k:stats([d['native_start_minus_producer_end'][k] for d in rows]) for k in rows[0]['producers']},
                'previous_native_end_to_start':stats([d['previous_native_end_to_start'] for d in rows]) if i else None,
                'producer_intervals':{n:{k:stats([d['producers'][n][k] for d in rows]) for k in ('begin_from_task','end_from_task','duration')} for n in rows[0]['producers']}})
        summaries[side]=summary
        plans[side]={'mapping':file_record(mp),'schedule':file_record(sp),'source':file_record(src),
                     'graph':pkg['sides'][side]['graph'],'operation_count':len(mapping['operations']),**plan}
    common=CONFIG['control'].keys()&CONFIG['candidate'].keys()
    compare={n:{k:summaries['candidate']['selected'][n][k]['median']-summaries['control']['selected'][n][k]['median'] for k in ('begin_from_task','end_from_task','duration')} for n in sorted(common)}
    return {'id':'D-E141','valid':True,'job':job,'source_result':file_record(rp),'package':file_record(pp),
        'source_maps':plans,'samples':samples,'summaries':summaries,'candidate_minus_control_median_landmark_cycles':compare,
        'whole_comparison':r['whole_ffn_comparison'],'analyzer':file_record(Path(__file__)),
        'instrumentation_cycles':{n:r['stages'][n]['cycles'] for n in ('control_info','control_trace','candidate_info','candidate_trace')},
        'limitations':['Marker intervals include waits; no isolated lookup/engine service costs are inferred.',
            'Direct Main combines lookup/native/scale/reduce; this diagnostic cannot split its internal service time.',
            'Latest completed marked producer is observed readiness, not proof that it is the only cause of scheduling delay.',
            'Interval unions describe occupied marker windows; they are not summed into reclaimable latency or a speedup forecast.',
            'Time differences use each run/cluster Task origin; clocks from different clusters are not directly subtracted.',
            'Three inputs and two clusters are not six independent hardware allocations. One diagnostic allocation only.',
            'Two large direct streams would add full F32 block-partial stores/reads and separate scale/reduce, requiring fresh bounds/numerical/whole gates.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('job',type=int);p.add_argument('--check',action='store_true');a=p.parse_args()
    value=analyze(a.label,a.job);out=HERE/'profile-analysis.json'
    if a.check:assert read(out)==value
    else:write(out,value)
    print(json.dumps({'whole':value['whole_comparison'],'landmark_changes':value['candidate_minus_control_median_landmark_cycles']},indent=2))
