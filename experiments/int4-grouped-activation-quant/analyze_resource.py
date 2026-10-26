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
SOURCE = '55a55aff'
BASE = 'a4152dba16b6b3de27b3cc474f1a890cfdb5d0a9'

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
        if kind.startswith('dma') or kind in ['rngd_cmd.tu_exec','rngd_cmd.sto_vrf']:
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
                all_movement_and_TU_instances=[x for x in expanded if x['kind'].startswith('dma') or x['kind'] in ['rngd_cmd.tu_exec','rngd_cmd.sto_vrf']],
                max_physical_sram_tensor_bytes=max(a['size'] for a in allocations),
                placed_live_address_union_peak_bytes=placed_peak,
                live_sram_record_sum_upper_bound_bytes=live_sum_peak,
                unplaced_sram_records=sum(a['address'] is None for a in allocations),
                memory_caveat='The SDK leaves addresses null for loop-local/aliased records. Placed union is not a certified whole-graph peak; the live-record sum is a conservative alias-counting upper bound. Preserve the exact allocation records.',
                compiler_sram_allocations=allocations)


def main():
    records=[]
    for directory in sorted((HERE/'runs').iterdir()):
        record=json.loads((directory/'run.json').read_text())
        hashes=json.loads((directory/'source-hashes.json').read_text())
        assert record['commit'].startswith(SOURCE) and record['exit_code']==0
        assert record['result']=='command_succeeded'
        assert (directory/'source.diff').read_bytes()==b''
        for relative,expected in hashes.items(): assert sha(ROOT/relative)==expected,relative
        records.append(dict(name=record['name'],commit=record['commit'],exit_code=record['exit_code'],
                            started_at=record['started_at'],finished_at=record['finished_at'],
                            verified_source_inputs=len(hashes),command=record['command'],
                            recorded_files={p.name:sha(p) for p in sorted(directory.iterdir())}))
    results={name:schedule(filename) for name,filename in [
        ('input_H4','input-schedule.json'),('hidden_L16','hidden-schedule.json'),
        ('prior_wholerow_H','prior-wholerow-input-schedule.json'),
        ('prior_wholerow_L','prior-wholerow-hidden-schedule.json')]}
    assert [x['complete_unfolded_schedule_cycles'] for x in results.values()]==[60292,74636,23918,45745]
    assert all(not x['has_dram_stack'] for x in results.values())
    original_paths=['src/ops_int4_rational_fused_pack.rs','src/ops_int4_pack_p512.rs',
                    'src/ops_int4_group960_projection.rs','src/bin/support/rational_a4_oracle.rs']
    unchanged={}
    for rel in original_paths:
        original=subprocess.check_output(['git','show',f'{BASE}:{rel}'],cwd=ROOT)
        assert original==(ROOT/rel).read_bytes()
        unchanged[rel]=sha(ROOT/rel)
    deps={p:sha(ROOT/p) for p in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']}
    assert deps=={'Cargo.toml':'a4154d908d25a512d031dea859b1b013794f821a47deb739334332bd0e4fdf85',
                 'Cargo.lock':'a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e',
                 'rust-toolchain.toml':'1e6c9fed0427862faf47f5993fca93848e1b5d62127b38a058fd20ef6a03cc6d'}
    images={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((HERE/'resource-images').iterdir())}
    assert len(images)==6
    out=dict(source_commit=records[0]['commit'],base=BASE,plan_sha256=sha(HERE/'plan.json'),
             recorded_runs=records,dependency_hashes=deps,unchanged_original_source_hashes=unchanged,
             resource_images=images,results=results,
             attribution={'H4':'Two16B-per-slice compact metadata stores cost17823 estimated cycles each; explicit LUT gather7886; input4783; real alignedH16 packed move1826/store1240.',
                          'L16':'Two64B-per-slice compact metadata stores cost8457 each; full Record8 gather28521; input15950; natural16row packed store3679 and no packed DM join.',
                          'memory':'HBM offsets are explicit DramIO scratch8192B H/32768B L. No DramStack or equivalent large SPM list; full compiler allocation records retained.',
                          'comparison_limit':'The prior whole-row quantizer computes different code/scales. Its static schedule is a cost context, not a numerical reference or calibrated runtime baseline.'},
             source_review=dict(agent='/root/e2e_evidence',commit=records[0]['commit'],result='PASS',scope='Source/fixture/probe only; resource values not independently reviewed; runtime untested.'),
             decision='CHECKPOINT_ONLY; source and host/resource compile pass; resource cost increases. No repair, standalone ELF, device execution or serving activation authorized at checkpoint.',
             scope_limits=['Probe source typechecked but was never executed.',
                           'No actual grouped code/scale/offset/pack equality proof.',
                           'No speed, quantization-error, complete-FFN or model-quality claim.',
                           'Original complete FFN323067 correctness PASS/performance HOLD and large synthetic matched-input drift remain unchanged.',
                           'Whole timeline, operation unions and source descriptions are static compiler evidence; overlapping durations are not additive wall savings.'])
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
