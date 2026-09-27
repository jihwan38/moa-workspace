#!/usr/bin/env python3
"""Check the frozen E128 six-row emitted graph; --check never writes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def record(path):
    b = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def analyze(label):
    require(label == 'e128-six-row-v1', 'This checker is for frozen E128 six-row-v1 only')
    build_path = HERE / f'build-{label}-repro.json'
    source = HERE / 'six-row-v1/mlp.rs'
    evidence = HERE / 'evidence' / label
    artifact = ROOT / 'moa-dev/experiments/artifacts/ffn-native-fused-scale' / label
    paths = {'build_record': build_path, 'source': source, 'recipe': HERE / 'six-row-v1/recipe.json',
             'graph': evidence / 'graph.json', 'schedule': evidence / 'schedule.json',
             'exact_image': artifact / 'exact-kernel.bin'}
    result = {'label': label, 'passed': False, 'inputdigests': {k: record(p) for k, p in paths.items()}}
    build = json.loads(build_path.read_text())
    require(build['exact_compile_exit_code'] == 0 and build['source_inputs_verified_in_container'], 'Exact build did not pass')
    require(build['label'] == label and build['variant'] == 'six-row-v1', 'Build recipe mismatch')
    for k, buildkey in [('source', 'src/device/shared/mlp.rs')]:
        require({x: result['inputdigests'][k][x] for x in ('bytes', 'sha256')} == build['source_inputs'][buildkey], 'Source differs from compiled manifest')
    for k in ('graph', 'schedule', 'exact_image'):
        require({x: result['inputdigests'][k][x] for x in ('bytes', 'sha256')} == build[k], f'{k} digest mismatch')
    for k in ('graph', 'schedule'):
        require(paths[k].read_bytes() == (artifact / f'{k}.json').read_bytes(), f'{k} artifact/evidence mismatch')
    require(source.read_bytes() == (evidence / 'src/device/shared/mlp.rs').read_bytes(), 'Evidence source mismatch')
    require(source.read_bytes() == (artifact / 'crate/src/device/shared/mlp.rs').read_bytes(), 'Built crate source mismatch')
    origin = build['source_plan']['source_origin_inputs']
    unchanged = [k for k in origin if k != 'src/device/shared/mlp.rs']
    require(all(build['source_inputs'][k] == origin[k] for k in unchanged), 'Unrelated compile input changed')
    g = json.loads(paths['graph'].read_text())
    schedule = json.loads(paths['schedule'].read_text())
    tensors, ops = g['tensors'], g['operators']
    producers = {tid: (key, op) for key, op in ops.items() for tid in op['outputs']}

    def tensor(tid):
        return tensors[str(tid)]

    def mappings(tid):
        return tensor(tid)['shape']['MappingShape']['inner']['mappings']

    def producer(tid, kind=None):
        pair = producers[tid]
        require(kind is None or pair[1]['kind'] == kind, f'T{tid} producer is not {kind}')
        return pair

    def scalar(tid):
        _, op = producer(tid, 'CoreImpCmd')
        m = re.search(r'__scalar_out\[\] = (\d+)i32;', op['description'])
        require(m is not None, f'T{tid}: nonconstant tile index')
        return int(m.group(1))

    def view(tid, stride):
        key, op = producer(tid, 'IndexAccess')
        require(f'axis: stride {stride} x 30, window_size: 6\n' in op['description'], f'T{tid}: wrong tile extent/stride')
        require(len(op['inputs']) == 2, 'Malformed IndexAccess')
        row = scalar(op['inputs'][1])
        require(row in (0, 6, 12, 18, 24), 'Bad tile row offset')
        return op['inputs'][0], row, key

    def root_output(tid):
        _, op = producer(tid)
        if op['kind'] == 'Scratchpad':
            return tid
        require(op['kind'] == 'IndexWrite', 'Unexpected output alias chain')
        return root_output(op['inputs'][0])

    def seq(description, field):
        m = re.search(rf'^\s*{field}: (\d+)\+\[([^\]]*)\]', description, re.M)
        require(m is not None, f'Missing emitted {field}')
        dims = re.findall(r'[^,]+=(\d+):(\d+)', m.group(2))
        require(bool(dims), f'Empty emitted {field}')
        return int(m.group(1)), [(int(n), int(s)) for n, s in dims]

    def offsets(sequence):
        base, dims = sequence
        return [base + sum(i * stride for i, (_, stride) in zip(index, dims))
                for index in itertools.product(*(range(n) for n, _ in dims))]

    native = [(k, op) for k, op in ops.items() if op['kind'] == 'TuExec'
              and 'broadcast_kind: Direct, f8_e5' in op['description'] and 'operation_unit:' in op['description']]
    require(len(native) == 10, f'Expected ten native streams, got {len(native)}')
    groups = {}
    native_records = []
    xids, trfids, scaleids = set(), set(), set()
    for key, op in native:
        d = op['description']
        markers = ['operation_unit:', 'FpMul0: %l * %r [true] %r=LoadVrf { id: VrfId(0) }',
                   'FpMul1: %l * %r [true] %r=LoadVrf { id: VrfId(1) }', 'ReduceAccTree: LReduce(+) [axes=H]',
                   'transpose_engine:', 'commit_unit:']
        positions = [d.index(m) for m in markers]
        require(positions == sorted(positions), f'{key}: fused numerical order changed')
        require(d.count('ReduceAccTree:') == 1 and d.count('FpMul0:') == 1 and d.count('FpMul1:') == 1, 'Unexpected extra VE stage')
        require('FfnRow_1=6, H_64=60, <ve_fp_way>H_16=4' in d, 'Wrong native block/VE packet ordering')
        require('data_type: Float32' in d and 'commit_in_size: 8B (=2EA)' in d, 'Non-F32 or padded output store')
        require(len(op['inputs']) == 5 and len(op['outputs']) == 1, 'Unexpected native IO/partial materialization')
        weight, trf, xs, bs, dst = op['inputs']
        require([tensor(x)['buffer_type'] for x in op['inputs']] == ['Sram', 'Trf', 'Vrf', 'Vrf', 'Sram'], 'Unexpected native memory tiers')
        require(tensor(weight)['element_type'] == 'Float8e5m2', 'Weight dtype mismatch')
        require(tensor(xs)['element_type'] == tensor(bs)['element_type'] == tensor(dst)['element_type'] == 'Float32', 'Scale/output dtype mismatch')
        base, row, access = view(weight, 3840)
        _, decode = producer(base, 'TuExec')
        require('operation_unit:' not in decode['description'] and 'commit_in_size: 32B (=32EA)' in decode['description'], 'Missing materialized E5 weight producer')
        require(tensor(base)['size'] == 512 * 30 * 3840, 'Weight backing extent mismatch')
        weight_offsets = offsets(seq(d, 'fetch_sequencer'))
        require(weight_offsets == list(range(6 * 3840)), 'Weight fetch does not exactly cover six rows')
        require('fetch_size: 32\n' in d, 'Unexpected weight fetch width')
        skey, scale_load = producer(bs, 'TuExec')
        require(scale_load['context'] == 'Sub' and 'to Vrf\n' in scale_load['description'] and len(scale_load['inputs']) == 1, 'Block scale spilled/materialized')
        scale_base, scale_row, scale_access = view(scale_load['inputs'][0], 240)
        require(scale_row == row and tensor(scale_base)['element_type'] == 'Float8e4m3', 'Weight/scale row or dtype mismatch')
        require(tensor(scale_base)['size'] == 512 * 30 * 240, 'Scale backing extent mismatch')
        require(offsets(seq(scale_load['description'], 'fetch_sequencer')) == list(range(6 * 240)), 'Scale fetch coverage mismatch')
        require(tensor(bs)['size'] == 512 * 6 * 240 * 4, 'Scale VRF compact extent mismatch')
        m = mappings(bs)
        require(m['InSlice'] == {'Pair': {'left': {'Resize': {'inner': {'Symbol': {'symbol': 'FfnRow', 'size': 30}}, 'resize': 6}}, 'right': {'Stride': {'inner': {'Symbol': {'symbol': 'H', 'size': 3840}}, 'stride': 16}}}}, 'Block VRF row/block order changed')
        require(mappings(xs)['InSlice'] == {'Padding': {'inner': {'Broadcast': {'size': 1}}, 'padding': 8, 'kind': 'Top'}}, 'x scale is not scalar broadcast')
        require(all(mappings(t)['Cluster'] == m['Cluster'] and mappings(t)['Partitioning'] == m['Partitioning'] for t in [weight, trf, xs, dst]), 'Spatial ownership mismatch')
        xaddr, baddr = int(tensor(xs)['optionals']['address']), int(tensor(bs)['optionals']['address'])
        require(xaddr + 32 <= baddr or baddr + 5760 <= xaddr, 'x/block scale VRF physical overlap')
        dparent, drow, daccess = view(dst, 1)
        require(drow == row and tensor(dst)['size'] == 512 * 30 * 4, 'Destination tile does not match source rows')
        require(offsets(seq(d, 'commit_projection')) == list(range(6)), 'Commit does not write six contiguous F32 rows')
        out = op['outputs'][0]
        require(tensor(out)['element_type'] == 'Float32' and tensor(out)['optionals']['address'] == tensor(dst)['optionals']['address'], 'Output store alias mismatch')
        writers = [(wk, w) for wk, w in ops.items() if w['kind'] == 'IndexWrite' and w['inputs'][-1] == out]
        require(len(writers) == 1, 'Missing output IndexWrite')
        wk, writer = writers[0]
        require(writer['inputs'][0] == dparent and scalar(writer['inputs'][1]) == row, 'IndexWrite offset/parent mismatch')
        dest_root = root_output(dparent)
        group = groups.setdefault(base, {'rows': [], 'scale_base': scale_base, 'output_root': dest_root})
        require(group['scale_base'] == scale_base and group['output_root'] == dest_root, 'Mixed projection weights/scales/output')
        group['rows'].append(row)
        xids.add(xs); trfids.add(trf); scaleids.add(bs)
        native_records.append({'graph_operator': key, 'name': op['name'], 'row_start': row, 'weight_base_tensor': base,
                               'weight_view_operator': access, 'weight_relative_bytes': [row * 3840, (row + 6) * 3840],
                               'scale_base_tensor': scale_base, 'scale_view_operator': scale_access, 'scale_relative_bytes': [row * 240, (row + 6) * 240],
                               'scale_load_operator': skey, 'block_scale_vrf': bs, 'output_root': dest_root,
                               'output_view_operator': daccess, 'index_write_operator': wk, 'output_relative_bytes': [row * 4, (row + 6) * 4],
                               'weight_fetch_elements': len(weight_offsets), 'scale_fetch_elements': 1440, 'f32_commit_packets': 3, 'f32_commit_bytes': 24})
    require(len(groups) == 2 and all(sorted(x['rows']) == [0, 6, 12, 18, 24] for x in groups.values()), 'Projection row coverage is not disjoint/exhaustive')
    require(len({x['scale_base'] for x in groups.values()}) == 2 and len({x['output_root'] for x in groups.values()}) == 2, 'Up/gate source/output alias')
    require(len(xids) == len(trfids) == 1 and len(scaleids) == 10, 'Unexpected activation/scale register reuse')
    for tid in xids: producer(tid, 'StoVrf')
    for tid in trfids: producer(tid, 'StoTrf')
    def contains_axes(t):
        text = json.dumps(t['shape'])
        return 'FfnRow' in text and '"H"' in text
    partials = [tid for tid, t in tensors.items() if t['element_type'] == 'Float32' and t['buffer_type'] == 'Sram' and contains_axes(t)]
    require(not partials, f'F32 native block partial DM remains: {partials}')
    vrfs = {int(tid) for tid, t in tensors.items() if t['element_type'] == 'Float32' and t['buffer_type'] == 'Vrf' and contains_axes(t)}
    require(vrfs == scaleids, 'Unexpected native partial VRF or scale spill')
    separate_reduce = [k for k, o in ops.items() if 'FfnRow' in o['description'] and 'ReduceAccTree:' in o['description'] and 'operation_unit:' not in o['description']]
    require(not separate_reduce, 'Separate up/gate scale/reduce stream remains')

    st = {int(t['index']): t for t in schedule['tensors']}
    instructions = schedule['instructions']
    sp = {tid: ins for ins in instructions for tid in ins['output_tensors']}
    fused = [x for x in instructions if x['tpe'] == 'Main' and '.contract_outer::' in x['description'] and 'FpMulAlu::Mul1' in x['description'] and 'FfnRow = 6' in x['description']]
    require(len(fused) == 10, 'Final schedule does not contain ten fused native commands')
    life_records = []
    shared_x, shared_trf = set(), set()
    for ins in fused:
        _, trf, xs, bs, _ = ins['input_tensors']
        shared_x.add(xs); shared_trf.add(trf)
        for tid in (trf, xs, bs):
            t = st[tid]; load = sp[tid]
            require(load['lifetime']['end'] <= ins['lifetime']['begin'], 'Register consumed before production')
            require(t['lifetime']['begin'] <= ins['lifetime']['begin'] and t['lifetime']['end'] >= ins['lifetime']['end'], 'Register lifetime does not cover native use')
            require(load['tpe'] == 'Sub', 'Register production unexpectedly changed engine')
        life_records.append({'instruction': ins['index'], 'lifetime': ins['lifetime'], 'x_scale': xs, 'x_trf': trf, 'block_scale': bs})
    require(len(shared_x) == len(shared_trf) == 1, 'Final schedule activation reuse differs from graph')
    focus = {int(t['index']) for t in st.values() if int(t['index']) in shared_x or (t['buffer_type'] == 'Vrf' and 'FfnRow = 6' in t['shape'])}
    for tid in focus:
        t = st[tid]
        for uid, u in st.items():
            if uid == tid or u['buffer_type'] != 'Vrf': continue
            live_overlap = max(t['lifetime']['begin'], u['lifetime']['begin']) < min(t['lifetime']['end'], u['lifetime']['end'])
            address_overlap = max(t['address'], u['address']) < min(t['address'] + t['size'], u['address'] + u['size'])
            require(not (live_overlap and address_overlap), f'VRF live address overlap: {tid}/{uid}')
    result.update({'passed': True, 'unchanged_compile_inputs': len(unchanged), 'gates': {
        'ten_fused_native_commands': True, 'emitted_fetch_and_commit_coverage': True,
        'input_scale_then_block_scale_then_h_reduce': True, 'two_vrf_mapping_and_lifetimes': True,
        'upgate_f32_block_partial_dm_absent': True, 'native_partial_vrf_absent': True,
        'separate_upgate_scale_reduce_commands': 0, 'disjoint_five_sixrow_tiles_per_projection': True},
        'native_streams': sorted(native_records, key=lambda x: (x['weight_base_tensor'], x['row_start'])),
        'final_schedule': life_records, 'compiler_shared_registers': {'graph_x_scale': sorted(xids), 'graph_x_trf': sorted(trfids),
            'schedule_x_scale': sorted(shared_x), 'schedule_x_trf': sorted(shared_trf),
            'note': 'Both source reloads lowered to one shared register producer; emitted lifetimes and non-overlapping VRF reuse verified.'},
        'logical_removed_partial_write_bytes': 29491200, 'logical_removed_partial_read_bytes': 29491200,
        'limits': ['Source and emitted graph/schedule audit; no NPU numerical correctness or measured speed claim.',
                   'Fetch/commit sequencers plus IndexAccess constants prove six-row DM views. The unchanged whole HBM ingress is source/manifest-bound, not independently enumerated here.',
                   'Graph exposes separate VrfId inputs and logical mapping but not encoded vISA VRF read-indexer bitfields; their numerical behavior requires the prepared NPU probe.']})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    out = HERE / 'graph-verification.json'
    try:
        result = analyze(args.label)
    except Exception as exc:
        result = {'label': args.label, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        if not args.check:
            out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), file=sys.stderr)
        return 1
    if args.check:
        require(out.exists() and json.loads(out.read_text()) == result, 'Frozen graph verification is stale')
    else:
        out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'label': args.label, 'native_streams': len(result['native_streams']), 'check': args.check}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
