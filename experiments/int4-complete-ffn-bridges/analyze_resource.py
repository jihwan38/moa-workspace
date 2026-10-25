#!/usr/bin/env python3
"""Replay frozen source/resource evidence, without invoking the SDK or device."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = '381fc0239c212e53009294e19084e5cbf850e51c'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def union_size(intervals):
    merged = []
    for start,end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end,merged[-1][1])
        else:
            merged.append([start,end])
    return sum(end-start for start,end in merged)

def schedule(filename):
    path = HERE/filename
    data = json.loads(path.read_text())
    expanded = []
    for ident,instances in data['timeline']['operators'].items():
        op = data['description']['operators'][ident]
        for instance in instances:
            begin,end = (instance['lifetime'][k] for k in ['begin','end'])
            expanded.append(dict(id=ident,kind=op['kind'],begin=begin,end=end,
                                 cycles=end-begin,iteration=instance['iteration_index'],
                                 source=op['description'].splitlines()[0] if op['description'] else '',
                                 inputs=op['inputs'],outputs=op['outputs']))
    counts = Counter(op['kind'] for op in expanded)
    bykind = {}
    for kind in sorted(counts):
        if kind.startswith('dma.') or kind in ['rngd_cmd.tu_exec','rngd_cmd.sto_vrf']:
            selected = [op for op in expanded if op['kind']==kind]
            bykind[kind] = dict(instances=len(selected),
                               summed_cycles=sum(op['cycles'] for op in selected),
                               union_cycles=union_size((op['begin'],op['end']) for op in selected))
    allocations = []
    for ident,instances in data['timeline']['tensors'].items():
        desc = data['description']['tensors'][ident]['properties']
        for instance in instances:
            a = instance['allocation']
            if a['buffer_type']=='Sram':
                allocations.append(dict(id=ident,name=desc['name'],shape=desc['shape'],
                                        lifetime=instance['lifetime'],iteration=instance['iteration_index'],**a))
    placed_peak = live_sum_peak = 0
    for time in {a['lifetime']['begin'] for a in allocations}:
        active = [a for a in allocations if a['lifetime']['begin']<=time<a['lifetime']['end']]
        placed_peak = max(placed_peak,union_size((a['address'],a['address']+a['size'])
                           for a in active if a['address'] is not None))
        live_sum_peak = max(live_sum_peak,sum(a['size'] for a in active))
    minimum = min(op['begin'] for op in expanded)
    maximum = max(op['end'] for op in expanded)
    lifetimes = [op['properties']['lir_lifetime'] for op in data['description']['operators'].values()
                 if 'lir_lifetime' in op['properties']]
    return dict(schedule_sha256=sha(path),
                timeline_begin=minimum,timeline_end=maximum,
                complete_unfolded_schedule_cycles=maximum-minimum,
                max_local_lir_endpoint=max(int(re.findall(r'\d+',x)[1]) for x in lifetimes),
                lir_caveat='Loop-local endpoints are not a whole-task duration. Unfolded span includes SDK scheduling/start offsets and is not measured device time.',
                has_dram_stack='DramStack' in json.dumps(data),
                resource_kinds=bykind,
                bridge_dma_instances=[x for x in expanded if x['kind'].startswith('dma.')
                                      and 'ops_int4_ffn_bridges.rs' in x['source']],
                max_physical_sram_tensor_bytes=max(a['size'] for a in allocations),
                placed_live_address_union_peak_bytes=placed_peak,
                live_sram_record_sum_upper_bound_bytes=live_sum_peak,
                unplaced_sram_records=sum(a['address'] is None for a in allocations),
                memory_caveat='The SDK leaves addresses null for loop-local/aliased records. Placed union is not a certified whole-graph peak; the live-record sum is a conservative alias-counting upper bound. Preserve the exact allocation records.',
                compiler_sram_allocations=allocations)

def main():
    proof=json.loads(subprocess.check_output(['python3',str(HERE/'verify_source.py')],cwd=ROOT))
    assert proof==json.loads((HERE/'source-proof.json').read_text())
    records=[]
    for directory in sorted((HERE/'runs').iterdir()):
        record=json.loads((directory/'run.json').read_text())
        hashes=json.loads((directory/'source-hashes.json').read_text())
        assert record['commit']==SOURCE and record['exit_code']==0
        assert record['result']=='command_succeeded'
        assert (directory/'source.diff').read_bytes()==b''
        for relative,expected in hashes.items():
            assert sha(ROOT/relative)==expected,relative
        records.append(dict(name=record['name'],commit=record['commit'],exit_code=record['exit_code'],
                            started_at=record['started_at'],finished_at=record['finished_at'],
                            verified_source_inputs=len(hashes),command=record['command'],
                            recorded_files={p.name:sha(p) for p in sorted(directory.iterdir())}))
    results={name:schedule(filename) for name,filename in [
        ('prefix_sliding','prefix-sliding-schedule.json'),
        ('prefix_full','prefix-full-schedule.json'),('geglu','geglu-schedule.json')]}
    assert [x['complete_unfolded_schedule_cycles'] for x in results.values()]==[42180,33155,139190]
    assert all(not x['has_dram_stack'] for x in results.values())
    assert results['geglu']['max_physical_sram_tensor_bytes']==15728640
    images={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))
            for directory in sorted(HERE.glob('kernel-out-*')) for p in sorted(directory.rglob('*'))
            if p.suffix in ['.bin','.edf','.hash']}
    assert len(images)==9
    deps={p:sha(ROOT/p) for p in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']}
    assert deps=={
        'Cargo.toml':'a4154d908d25a512d031dea859b1b013794f821a47deb739334332bd0e4fdf85',
        'Cargo.lock':'a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e',
        'rust-toolchain.toml':'1e6c9fed0427862faf47f5993fca93848e1b5d62127b38a058fd20ef6a03cc6d'}
    out=dict(source_commit=SOURCE,plan_sha256=sha(HERE/'plan.json'),source_proof=proof,
             recorded_runs=records,dependency_hashes=deps,resource_images=images,results=results,
             source_review=dict(agent='/root/e2e_evidence',source=SOURCE,result='PASS',
                                scope='Caller/callee arithmetic, rounding, scale and row-map source review only.'),
             new_boundary_traffic=dict(prefix_write_bytes=3932160,hidden_write_bytes=15728640,
                                      combined_write_bytes=19660800,future_equal_quantizer_read_bytes=19660800,
                                      total_new_BF16_boundary_write_read_bytes=39321600,
                                      original_joined_read_bytes=31457280),
             decision='SOURCE_HOST_RESOURCE_PASS; actual bridge parity and complete-path latency/quality remain unmeasured.',
             scope_limits=['No standalone ELF built','No Arena or device execution',
                           'No full FFN assembly or production change',
                           'No relative speed claim from standalone resources',
                           'Operation span sums are not a critical-path speedup prediction'])
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=='__main__':
    main()
