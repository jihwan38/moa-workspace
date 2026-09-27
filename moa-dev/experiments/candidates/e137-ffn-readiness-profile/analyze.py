#!/usr/bin/env python3
"""E139 source-bound current FFN readiness profile. Never add overlapping spans as cycle savings."""
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
CONFIG={'control': {'input_hbm': '26', 'pre_rms_canonical': '28', 'pre_rms_reduced': '114', 'pre_rms_finish': '136', 'pre_rms_gather': '141', 'up_packed_hbm': '120', 'up_fp8_materialization': '153', 'up_scale_hbm': '156', 'up_input_gather': '159', 'up_amax': '180', 'up_scale': '196', 'up_quant_shard_dma': '175', 'up_quant_hi': '231', 'up_quant_lo': '246', 'up_quant_third': '272', 'up_terms_assembled': '292', 'up_leader_dma': '299', 'up_term_broadcast': '305', 'up_activation_trf': '324', 'up_native_first': '328', 'up_native_last': '376', 'up_local_gather': '386', 'up_bf16_cast': '407', 'gate_packed_hbm': '307', 'gate_fp8_materialization': '385', 'gate_scale_hbm': '388', 'gate_native_first': '422', 'gate_native_last': '469', 'gate_local_gather': '475', 'gate_bf16_cast': '478', 'geglu_output': '499', 'down_activation_canonical_gather': '506', 'down_amax': '509', 'down_quant_shard_dma': '510', 'down_scale': '519', 'down_coefficient0': '524', 'down_coefficient1': '532', 'down_coefficient2': '551', 'down_quant_scale_dma': '525', 'down_quant_scale_vrf': '535', 'down_quant_hi': '546', 'down_hi_copy': '553', 'down_hi_decode_vrf': '562', 'down_quant_lo': '568', 'down_sum01': '574', 'down_lo_copy': '576', 'down_sum01_vrf': '585', 'down_quant_third': '589', 'down_third_copy': '594', 'down_term_compact_gather': '603', 'down_leader_dma': '609', 'down_term_broadcast': '617', 'down_coefficient_dma': '618', 'down_coefficient_vrf': '625', 'down_activation_trf': '633', 'down_packed_hbm': '390', 'down_table_dma': '479', 'down_table_ready': '502', 'down_scale_hbm': '626', 'down_native_partial_write': '637', 'down_blockscale_vrf_first': '642', 'down_partial_scale_reduce_first': '655', 'down_partial_scale_reduce_last': '707', 'down_result_gather': '715', 'down_cluster_reduce_bf16': '730', 'down_global_scale': '735', 'post_rms_apply': '761', 'residual_add': '766', 'layer_scale': '771', 'output_store': '774'}}
NATIVES={'control':['328','341','353','365','376','422','434','446','458','469','637']}

def stats(vals):return {'median':statistics.median(vals),'min':min(vals),'max':max(vals),'samples':vals}
def function_names(src):return [(i+1,re.search(r'fn (\w+)',t).group(1)) for i,t in enumerate(src.splitlines()) if re.match(r'^(?:pub\(crate\) )?fn \w+',t)]

def analyze(label,job):
    result_path=HERE/f'{label}-job-{job}-results.json';result=read(result_path)
    assert result['valid'] and result['actual_upload_matches'] and result['correctness_passes']==36
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
    for side in ('control',):
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
        for name in ['up_activation_trf','down_activation_trf']:assert '.to_trf()' in by_index[CONFIG[side][name]]['schedule']['description']
        for name in ['up_term_broadcast','down_term_broadcast']:assert '.switch::' in by_index[CONFIG[side][name]]['schedule']['description']
        assert 'to_hbm_view' in by_index[CONFIG[side]['output_store']]['schedule']['description']
        down=by_index[CONFIG[side]['down_native_partial_write']]['schedule']
        assert 'fetch_table_lookup::<f8e5m2>' in down['description']
        assert down['input_tensors']==[213,297,294,257]
        for name,tensor in [('down_packed_hbm',213),('down_activation_trf',297),('down_coefficient_vrf',294),('down_table_ready',257)]:
            assert tensor in by_index[CONFIG[side][name]]['schedule']['output_tensors']
        assert 'vector_intra_slice_reduce::<TileL' in by_index[CONFIG[side]['down_amax']]['schedule']['description']
        assert 'TileL % 480' in by_index[CONFIG[side]['down_quant_hi']]['schedule']['description'] or 'TileL / 16 % 30' in by_index[CONFIG[side]['down_quant_hi']]['schedule']['description']
        fs=function_names((folder/side/'production-mlp.rs').read_text())
        observed=[]
        for obs in stage['trace_observations']:
            ops={r['schedule_index']:r for r in obs['operations']}
            assert len(ops)==len(mapping['operations'])
            selected={name:ops[idx] for name,idx in CONFIG[side].items()}
            def begin(name):return selected[name]['begin_from_task']
            def end(name):return selected[name]['end_from_task']
            metrics={
                'pre_rms_ready_to_up_native':begin('up_native_first')-end('pre_rms_gather'),
                'up_materialization_ready_to_up_native':begin('up_native_first')-end('up_fp8_materialization'),
                'up_trf_ready_to_up_native':begin('up_native_first')-end('up_activation_trf'),
                'up_last_native_to_up_gather':begin('up_local_gather')-end('up_native_last'),
                'gate_materialization_ready_to_gate_native':begin('gate_native_first')-end('gate_fp8_materialization'),
                'gate_last_native_to_gate_gather':begin('gate_local_gather')-end('gate_native_last'),
                'down_weight_end_to_gate_gather':begin('gate_local_gather')-end('down_packed_hbm'),
                'gate_gather_end_to_geglu_ready':end('geglu_output')-end('gate_local_gather'),
                'geglu_ready_to_down_trf_ready':end('down_activation_trf')-end('geglu_output'),
                'down_amax_end_to_quant_third_ready':end('down_quant_third')-end('down_amax'),
                'down_third_copy_end_to_native':begin('down_native_partial_write')-end('down_third_copy'),
                'down_weight_ready_to_native':begin('down_native_partial_write')-end('down_packed_hbm'),
                'down_trf_ready_to_native':begin('down_native_partial_write')-end('down_activation_trf'),
                'down_coeff_ready_to_native':begin('down_native_partial_write')-end('down_coefficient_vrf'),
                'down_table_ready_to_native':begin('down_native_partial_write')-end('down_table_ready'),
                'down_native_end_to_first_partial_reduce':begin('down_partial_scale_reduce_first')-end('down_native_partial_write'),
                'down_scale_hbm_end_to_first_partial_reduce':begin('down_partial_scale_reduce_first')-end('down_scale_hbm'),
                'last_partial_reduce_end_to_cluster_finish':end('down_cluster_reduce_bf16')-end('down_partial_scale_reduce_last'),
                'down_native_begin_to_task_end':obs['task_cycles']-begin('down_native_partial_write'),
                'geglu_ready_to_task_end':obs['task_cycles']-end('geglu_output')}
            observed.append({'run':obs['run'],'cluster':obs['cluster'],'task_cycles':obs['task_cycles'],'selected':selected,'interval_metrics':metrics,'all_operations':ops})
        samples[side]=observed
        summaries[side]={'task_cycles':stage['cycles'],'selected':{},'interval_metrics':{k:stats([o['interval_metrics'][k] for o in observed]) for k in observed[0]['interval_metrics']},'whole_median':result['stages']['control_before']['kernels'][FFN]['median_cycles']}
        for name,idx in CONFIG[side].items():
            summaries[side]['selected'][name]={'schedule_index':idx,**{key:stats([o['selected'][name][key] for o in observed]) for key in ['begin_from_task','end_from_task','duration']}}
        maps[side]={'mapping':file_record(mp),'schedule':file_record(schedule_path),'graph':package['sides'][side]['graph'],'production_source':file_record(folder/side/'production-mlp.rs'),'schedule_tensors':{str(t['index']):t for t in schedule['tensors']},'operation_count':len(mapping['operations']),'all_operations':{}}
        for idx,row in by_index.items():
            n=row['schedule'];match=re.search(r'src/device/shared/mlp.rs:(\d+):',n['description']);line=int(match[1]) if match else 0
            fn=next((name for start,name in reversed(fs) if start<=line),'external shared helper/table')
            maps[side]['all_operations'][idx]={'marker':row['marker'],'source_function':fn,'source_line':line,'schedule':n,'measurements':[{k:o['all_operations'][idx][k] for k in ['begin_from_task','end_from_task','duration']}|{'run':o['run'],'cluster':o['cluster']} for o in observed]}
    # Schedule tensors and final graph use different identifier namespaces.
    # Preserve both; join measured markers only to the schedule namespace.
    return {'id':'E139-CURRENT-FFN-READINESS','valid':True,'job':job,'source_result':file_record(result_path),'package':file_record(ppath),'source_maps':maps,'samples':samples,'summaries':summaries,'current_ffn_comparison':result['current_ffn_comparison'],'method':'Three original inputs, two clusters, one trace/allocation. Times relative to each cluster Task. Complete137 operation ordering/kinds checked; all current source/images preserved.','limitations':['Marker intervals include waits and are not isolated service times.','Elapsed readiness windows are not sums of overlapping intervals or predicted net savings.','Standard info and diagnostic info/trace are different observation modes, not candidate speedup measurements.','A replacement may change earlier scheduling, conversion and transfer work.','Schedule tensor IDs are not joined numerically to final graph IDs.'],'analyzer':file_record(Path(__file__))}

FFN='decoder_feedforward'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('job',type=int);p.add_argument('--check',action='store_true');a=p.parse_args();v=analyze(a.label,a.job);out=HERE/'profile-analysis.json'
    if a.check:assert read(out)==v
    else:write(out,v)
    print(json.dumps({'whole':v['current_ffn_comparison'],'summaries':{side:{name:{k:r[k]['median'] for k in ['begin_from_task','end_from_task','duration']} for name,r in s['selected'].items()} for side,s in v['summaries'].items()}},indent=2))
