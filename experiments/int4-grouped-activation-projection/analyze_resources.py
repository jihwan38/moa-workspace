#!/usr/bin/env python3
"""Replay compiler-only grouped projection gates; never run a compiler/device."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, hashlib, json
HERE=Path(__file__).resolve().parent
TREE=HERE.parents[1]
ROUND2=TREE.parent
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(path):
    d=json.loads(path.read_text()); desc=d['description'];time=d['timeline']
    flat=[x for v in time['operators'].values() for x in v]
    spills=[{'tensor':k,**x} for k,v in time['tensors'].items() for x in v
            if x['allocation']['buffer_type'].startswith('DramStack')]
    tu=defaultdict(list); ordered=[]
    for k,o in desc['operators'].items():
        if o['kind']!='rngd_cmd.tu_exec':continue
        body=o['description']
        kind=('dot' if 'vector_fxp_to_fp' in body else 'post_reduce_scale' if 'begin(reduced.view())' in body
              else 'global' if '&global' in body else 'transpose' if '.transpose' in body else 'ingress')
        for x in time['operators'][k]:
            tu[kind].append(x['lifetime']['end']-x['lifetime']['begin'])
        if kind=='dot':
            needles=['FpMulAlu::Mul0),&block','FpMulAlu::Mul1),&xs','vector_inter_slice_reduce','vector_final().cast::<bf16']
            offsets=[body.find(s) for s in needles]
            ordered.append(all(x>=0 for x in offsets) and offsets==sorted(offsets))
    reduced=[];bf16columns=[]
    for k,t in desc['tensors'].items():
        p=t['properties']
        if p['buffer_type']=='sram' and ('Element: (L % 240, P512' in p['shape'] or 'Element: (H % 240, P512' in p['shape']):
            item={'tensor':k,'dtype':p['element_type'],'padded_shape':p['shape'],
                  'declared_size':int(p['size']),'allocations':time['tensors'][k]}
            if p['element_type']=='f32':reduced.append(item)
            elif p['element_type']=='bf16':bf16columns.append(item)
    return {'path':str(path.relative_to(ROUND2)),'sha256':SHA(path),
            'timeline_begin':min(x['lifetime']['begin'] for x in flat),
            'timeline_endpoint':max(x['lifetime']['end'] for x in flat),
            'operator_instances':len(flat),'operator_kinds':dict(sorted(Counter(o['kind'] for o in desc['operators'].values()).items())),
            'dram_stack':spills,'tu_duration_cycles_by_source':dict(sorted(tu.items())),
            'all_dot_programs_show_weight_then_activation_then_reduce_then_bf16':all(ordered),
            'f32_reduced_column_sram':reduced,'bf16_column_sram':bf16columns}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    result={'scope':'Compiler schedules only; no standalone ELF, device run, production speed or model-quality result.',
            'source_commit':'71cdf450','source_review':'e2e_evidence independent read-only PASS; no host/resource claim by reviewer',
            'old_complete_speed_gate':'HOLD (actual323067 slower)','old_down_resource_gate':'HOLD (245684 > 240000); historical branch not changed',
            'original_dependency_hashes':{p:SHA(TREE/p) for p in ['Cargo.toml','Cargo.lock']},'runs':{},'projections':{}}
    assert result['original_dependency_hashes']['Cargo.lock']=='a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
    for name in ['a001-int4-grouped-activation-projection-check','a002-int4-grouped-activation-upgate-resource','a003-int4-grouped-activation-down-resource']:
        path=ROUND2/'runs'/name;record=json.loads((path/'run.json').read_text());hashes=json.loads((path/'source-hashes.json').read_text())
        assert record['exit_code']==0 and record['commit'].startswith('71cdf450')
        for p,h in hashes.items():assert SHA(TREE/p)==h,(name,p)
        result['runs'][name]={'run_sha256':SHA(path/'run.json'),'log_sha256':SHA(path/'output.log'),'input_hashes_sha256':SHA(path/'source-hashes.json'),
            'commit':record['commit'],'started_at':record['started_at'],'finished_at':record['finished_at'],'exit_code':record['exit_code'],'source_input_count':len(hashes),'command':record['command']}
    for name,oldend in [('upgate',356610),('down',245684)]:
        old=analyze(ROUND2/'int4-group960-consumer/experiments/int4-group960-consumer'/f'{name}-schedule.json')
        new=analyze(HERE/f'{name}-schedule.json')
        assert old['timeline_endpoint']==oldend and not new['dram_stack']
        assert new['all_dot_programs_show_weight_then_activation_then_reduce_then_bf16']
        assert not new['f32_reduced_column_sram']
        assert len(old['f32_reduced_column_sram'])==(4 if name=='upgate' else 2)
        assert new['tu_duration_cycles_by_source']['dot']==old['tu_duration_cycles_by_source']['dot']
        if name=='down':assert new['tu_duration_cycles_by_source']['global']==old['tu_duration_cycles_by_source']['global']
        result['projections'][name]={'old':old,'grouped':new,'endpoint_cycles_removed':oldend-new['timeline_endpoint'],
            'endpoint_reduction_percent':100*(oldend-new['timeline_endpoint'])/oldend}
    result['sum_of_separate_endpoint_reductions_cycles']=sum(v['endpoint_cycles_removed'] for v in result['projections'].values())
    result['removed_postreduce_TU_duration_sum_cycles']=sum(sum(v['old']['tu_duration_cycles_by_source']['post_reduce_scale']) for v in result['projections'].values())
    result['storage']={'new_scale_owner_bytes':{'H':8192,'L':32768},'old_scale_owner_bytes_each':2048,
        'scale_live_delivery_bytes':{'upgate_old':262144,'upgate_grouped':1048576,'down_old':32768,'down_grouped':524288},
        'packed_input_output_weight_owners':'unchanged; group scales alone change the raw input extent',
        'removed_f32_column_note':'Each scheduled reduced-column tensor occupied62914560B padded SRAM. Four UP and two DOWN instances are different lifetimes, not simultaneously freed memory.',
        'retained':'BF16 column result/transpose/output, input/weight/scales/TRF, and separate DOWN BF16-global-BF16 pass. No hidden DramStack.',
        'scale_vrf_per_chunk_padded_allocation_bytes':262144}
    result['kernel_images']={str(p.relative_to(TREE)):{'sha256':SHA(p),'bytes':p.stat().st_size} for p in sorted(HERE.glob('kernel-out-*/*/*.bin'))}
    result['limitations']=['No speed prediction from compiler cycle endpoints or sum of overlapping TU spans.',
        'Dense-E4 reference host-typechecked only; reference device compilation/actual parity pending.',
        'Group quantization/packing and complete FFN are not in these projection schedules.',
        'No fullpath10percent gate or model-quality gate has passed.']
    target=HERE/'resource-result.json';text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.verify:assert target.read_text()==text,'resource report differs';print('PASS resource report, source hashes, ordering and preserved controls')
    else:target.write_text(text);print(target)
if __name__=='__main__':main()
