#!/usr/bin/env python3
"""Read-only replay of the preregistered H-only staging resource gate.

Run from any directory; JSON goes to stdout. No compiler/device invocation.
The new staging gate uses the SDK LIR task estimate. The older original-pack
report used the schedule-view endpoint; preserve both metrics explicitly.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent
BASE = '11e8118be33de43ef0b1cdd8ef7f2763b3c54312'
SOURCE = '15a335b6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(path):
    d = json.loads(path.read_text())
    ops = []
    maximum = 0
    for key, op in d['description']['operators'].items():
        value = op.get('properties', {}).get('lir_lifetime')
        if value is None:
            continue
        begin, end = map(int, re.findall(r'\d+', value))
        maximum = max(maximum, end)
        if op['kind'].startswith('dma.') or op['kind'] in ('rngd_cmd.tu_exec', 'cluster_sync'):
            ops.append(dict(id=key, kind=op['kind'], begin=begin, end=end,
                            cycles=end-begin, source=op['description'].splitlines()[0]))
    allocations = []
    for key, items in d['timeline']['tensors'].items():
        for item in items:
            alloc = item.get('allocation', {})
            if alloc.get('buffer_type') == 'Sram':
                allocations.append(dict(id=key, lifetime=item['lifetime'], **alloc))
    # Union live address intervals, rather than counting reshape aliases twice.
    peak = 0
    for time in {a['lifetime']['begin'] for a in allocations}:
        intervals = sorted((a['address'], a['address']+a['size']) for a in allocations
                           if a['lifetime']['begin'] <= time < a['lifetime']['end'])
        merged = []
        for begin, end in intervals:
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([begin, end])
        peak = max(peak, sum(end-begin for begin, end in merged))
    return dict(sha256=sha(path), lir_task_cycles=maximum,
                timeline_last_end=max(i['lifetime']['end'] for v in d['timeline']['operators'].values() for i in v),
                dram_stack='DramStack' in json.dumps(d),
                operations=sorted(ops, key=lambda x: (x['begin'], x['id'])),
                sram_peak_live_address_union_bytes=peak,
                sram_allocations=allocations)


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    results = {}
    for label, path in [
        ('original_H', HERE/'controls/original-H-schedule.json'),
        ('four_row_H', HERE/'controls/four-row-H-schedule.json'),
        ('sixteen_row_L', HERE/'controls/sixteen-row-L-schedule.json'),
        ('sixteen_row_H_padded', HERE/'input-schedule.json'),
    ]:
        results[label] = schedule(path)
    assert [results[k]['lir_task_cycles'] for k in results] == [51982, 54722, 49254, 19994]
    assert [results[k]['timeline_last_end'] for k in results] == [52085, 54825, 49363, 20097]
    assert all(not v['dram_stack'] for v in results.values())
    records = []
    for name in ['a001-int4-pack-input-aligned-hostcheck', 'a002-int4-pack-input-aligned-resource']:
        directory = HERE/'runs'/name
        record = json.loads((directory/'run.json').read_text())
        assert record['commit'].startswith(SOURCE) and record['exit_code'] == 0
        hashes = json.loads((directory/'source-hashes.json').read_text())
        for relative, expected in hashes.items():
            assert sha(TREE/relative) == expected, relative
        records.append(dict(name=name, commit=record['commit'], result=record['result'],
                            exit_code=record['exit_code'], checked_input_files=len(hashes),
                            command=record['command'], files={p.name: sha(p) for p in directory.iterdir()}))
    unchanged = {}
    for relative in ['src/ops_int4_pack_p512.rs', 'src/ops_int4_pack_staged_store.rs',
                     'src/i4_p512_raw_adapter.rs', 'src/bin/test_int4_pack_p512.rs',
                     'src/bin/test_int4_pack_staged_store.rs', 'Cargo.toml', 'rust-toolchain.toml']:
        original = subprocess.check_output(['git', '-C', str(TREE), 'show', f'{BASE}:{relative}'])
        assert (TREE/relative).read_bytes() == original, relative
        unchanged[relative] = sha(TREE/relative)
    assert sha(TREE/'Cargo.lock') == 'a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
    unchanged['Cargo.lock'] = sha(TREE/'Cargo.lock')
    images = {str(p.relative_to(TREE)): dict(sha256=sha(p), bytes=p.stat().st_size)
              for p in sorted((HERE/'kernel-out-a002').rglob('*')) if p.suffix in ('.bin', '.edf', '.hash')}
    assert len(images) == 3
    candidate = results['sixteen_row_H_padded']
    gate = plan['gates']['H_complete_resource_max_cycles']
    passed = candidate['lir_task_cycles'] <= gate and not candidate['dram_stack']
    assert passed and gate == 41668
    result = dict(
        source_commit=records[0]['commit'], base_commit=BASE,
        plan_sha256=sha(HERE/'plan.json'), records=records,
        unchanged_base_files=unchanged, resource_images=images, results=results,
        gate=dict(max_lir_task_cycles=gate, pass_resource=passed,
                  candidate_timeline_also_below_same_cap=candidate['timeline_last_end'] <= gate),
        preserved_holds={'original_H': '52085 > 50000', 'four_row_H': '54722 > 41668'},
        attribution={
            'historical_metric_note': 'Original H52085 is the timeline endpoint (LIR51982); staged H54722 and candidate19994 are LIR task estimates (timeline54825/20097). All original HOLD decisions remain; the candidate passes41668 under either metric.',
            'dm_staging_cycles_H4_and_H16': 3276,
            'store_cycles_H4_to_H16': [35968, 1240],
            'only_estimated_critical_delta_cycles': 34728,
            'physical_H16_staging_bytes': 3932160,
            'live_H16_staging_bytes': 983040,
            'dense_plus_staged_physical_bytes_during_copy': 4915200,
            'meaning': 'Controlled H-only mapping change removes the modeled final-store penalty; no assertion about an undocumented TDMA threshold or actual bandwidth.'},
        review={'agent': '/root/e2e_evidence', 'source': SOURCE, 'result': 'PASS',
                'scope': 'Source-only leader/padding/row coverage; actual padding writes and raw bytes remain ungated.'},
        no_standalone_elf_build=True, no_arena_execution=True,
        next_gate='Root decision: strict complete H pack/raw-copy byte oracle and matched completed-call timing. Resource PASS is neither device correctness nor full-FFN speed.')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
