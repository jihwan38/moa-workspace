#!/usr/bin/env python3
"""Bounded, frozen-v2 source/graph gate; this is not a numerical oracle."""
import argparse
import itertools
import json
import re
import subprocess
from pathlib import Path
from package import HERE, ROOT, ARTIFACTS, checked_compile, file_record, read, write


def source_ranges(description, expected_bytes):
    intervals = []
    for dims, base in re.findall(r'^src\d+: \[(.*?)\], Dram\(chip=0, in_chip=0x([0-9A-Fa-f]+)\)', description, re.M):
        dims = [(int(n), int(stride)) for n, stride in re.findall(r'[^, ]+=(\d+):(\d+)', dims)]
        assert dims[-1][0] == 1
        width = dims[-1][1]
        for indexes in itertools.product(*(range(n) for n, _ in dims[:-1])):
            begin = int(base, 16) + sum(i * d[1] for i, d in zip(indexes, dims[:-1]))
            intervals.append((begin, begin + width))
    intervals.sort()
    assert intervals and intervals[0][0] == 0 and intervals[-1][1] == expected_bytes
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    return {'packets': len(intervals), 'packet_bytes': width, 'first': 0,
            'last_exclusive': expected_bytes, 'exact_once_contiguous_source_coverage': True}


def verify(label):
    build_path, built, control, _, _ = checked_compile(label)
    assert built['variant'] == 'sub-consumer-v2'
    artifact = ARTIFACTS / label
    graph = read(artifact / 'graph.json')
    schedule = read(artifact / 'schedule.json')
    ops, ts = graph['operators'], graph['tensors']
    producer = {t: (oid, op) for oid, op in ops.items() for t in op['outputs']}
    consumers = {}
    for oid, op in ops.items():
        for t in op['inputs']:
            consumers.setdefault(t, []).append((oid, op))
    pin = {'build_record': file_record(build_path), 'graph': file_record(artifact / 'graph.json'),
           'schedule': file_record(artifact / 'schedule.json'),
           'exact_image': file_record(artifact / 'exact-kernel.bin')}
    for path, frozen in built['source_plan']['source_overrides'].items():
        pin[path] = file_record(ROOT / frozen['path'])
        assert pin[path] == frozen
    pin['recipe'] = file_record(ROOT / built['source_plan']['recipe']['path'])
    assert pin['recipe'] == built['source_plan']['recipe']
    base = built['base_commit']
    def original(name):
        return subprocess.check_output(['git', 'show', base + ':furiosa-opt-gemma4-12B/src/device/sliding/' + name], cwd=ROOT).decode()
    def normalize(text):
        return re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', text))
    for name in ('rmsnorm.rs', 'rope.rs'):
        candidate = (artifact / 'crate/src/device/sliding' / name).read_text()
        assert normalize(candidate) == normalize(original(name).replace('.main', '.sub'))
    projection = (artifact / 'crate/src/device/sliding/projection.rs').read_text()
    assert projection.split('pub(crate) fn project_output(')[1] == original('projection.rs').split('pub(crate) fn project_output(')[1]
    assert built['source_inputs']['src/ops.rs'] == control['source_inputs']['src/ops.rs']
    rows = []
    for name, dot_id, scale_id, switch_id, weight_io, scale_io, count in [
            ('Q', '56', '57', '58', 1, 4, 4096),
            ('K', '101', '111', '112', 2, 5, 2048),
            ('V', '59', '80', '81', 3, 6, 2048)]:
        dot, scale, switch = ops[dot_id], ops[scale_id], ops[switch_id]
        weight, activation, table = dot['inputs']
        raw, = dot['outputs']
        raw_t = ts[str(raw)]
        assert dot['kind'] == 'TuExec' and dot['context'] == 'Main'
        assert 'operation_unit:' in dot['description'] and 've_pass:' not in dot['description']
        assert 'cast_compaction:' not in dot['description']
        assert 'data_type: Float32' in dot['description'] and 'in_rows: InRows(2u5)' in dot['description']
        assert 'commit_in_size: 8B (=2EA)' in dot['description']
        local = count // 512
        assert f'commit_projection: 0+[None={local//2}:2, None=2:1]' in dot['description']
        assert raw_t['element_type'] == 'Float32' and raw_t['size'] == count * 4
        assert raw_t['buffer_type'] == 'Sram'
        assert ts[str(weight)]['element_type'] == 'Float8e4m3'
        assert ts[str(activation)]['element_type'] == 'Bfloat16' and ts[str(activation)]['buffer_type'] == 'Trf'
        weight_dma = producer[weight][1]
        assert weight_dma['inputs'] == [weight_io]
        bounds = source_ranges(weight_dma['description'], count * 3840)
        assert consumers[raw] == [(scale_id, scale)]
        assert scale['context'] == 'Sub' and scale['inputs'][0] == raw
        scale_vrf = scale['inputs'][1]
        assert ts[str(scale_vrf)]['element_type'] == 'Float32' and ts[str(scale_vrf)]['buffer_type'] == 'Vrf'
        scale_loader = producer[scale_vrf][1]
        assert scale_loader['context'] == 'Sub'
        assert producer[scale_loader['inputs'][0]][1]['inputs'] == [scale_io]
        desc = scale['description']
        assert desc.index('FpMul0:') < desc.index('cast_compaction:')
        assert 'output_element_type: bf16' in desc and 'fetch_size: 8' in desc
        assert f'fetch_sequencer: 0+[None={local}:1, None=2:1]' in desc
        assert '[[]+1]=2' in desc and '[[]+7]=8' in desc
        assert switch['context'] == 'Sub' and switch['inputs'] == scale['outputs']
        assert 'InterTranspose' in switch['description']
        assert ts[str(scale['outputs'][0])]['element_type'] == 'Bfloat16'
        # The second F32 word is an undefined padding lane, not a second output.
        # Last packet intentionally extends four bytes past the logical tensor.
        # Check it does not intersect another live SRAM payload in this schedule.
        base_address = int(raw_t['optionals']['address'])
        st = [t for t in schedule['tensors'] if t['buffer_type'] == 'Sram'
              and t['address'] == base_address * 512 and t['size'] == count * 4]
        assert len(st) == 1
        st = st[0]
        tail_begin = st['address'] + st['size']
        tail_end = tail_begin + 4 * 512
        conflicts = [t['index'] for t in schedule['tensors'] if t['index'] != st['index']
            and t['buffer_type'] == 'Sram'
            and max(t['lifetime']['begin'], st['lifetime']['begin']) < min(t['lifetime']['end'], st['lifetime']['end'])
            and max(t['address'], tail_begin) < min(t['address'] + t['size'], tail_end)]
        assert not conflicts, conflicts
        rows.append({'projection': name, 'dot': dot_id, 'scale': scale_id, 'row_switch': switch_id,
            'weight_input': weight_io, 'scale_input': scale_io, 'weight_dma': bounds,
            'raw_tensor': raw, 'raw_schedule_tensor': st['index'], 'raw_bytes': count * 4,
            'raw_rows_per_slice': local, 'commit_packet_bytes': 8,
            'commit_offsets_per_slice': list(range(local)),
            'sub_fetch_packet_bytes': 8, 'sub_fetch_packet_sum_bytes': count * 8,
            'valid_scalar_read_bytes': count * 4,
            'last_packet_read_per_slice': [base_address + 4*(local-1), base_address + 4*(local+1)],
            'intentional_trailing_padding_read_bytes_per_slice': 4,
            'padding_semantics': 'Only first scalar is live; #2 then #8 padding is never reduced into it; first scalar alone survives BF16 transpose/commit.',
            'trailing_padding_overlaps_other_live_schedule_payload': False})
    instructions = schedule['instructions']
    dpes = [i for i in instructions if 'contract_outer::' in i['description'] and 'weight_f8.view()' in i['description']]
    assert len(dpes) == 3 and all(i['contexts'] == ['MainContext'] for i in dpes)
    heads = [i for i in instructions if re.search(r'src/device/sliding/(rmsnorm|rope)\.rs:', i['description']) and i['tpe'] in ('Main', 'Sub') and i['lifetime']['end'] > i['lifetime']['begin'] and i['contexts'] != ['Noop']]
    assert heads and all(i['tpe'] == 'Sub' for i in heads)
    overlap = []
    for dot in dpes:
        for sub in instructions:
            if sub['tpe'] != 'Sub' or 'VectorEngine' not in sub['contexts']:
                continue
            begin = max(dot['lifetime']['begin'], sub['lifetime']['begin'])
            end = min(dot['lifetime']['end'], sub['lifetime']['end'])
            if begin < end:
                overlap.append({'main': dot['index'], 'sub_ve': sub['index'], 'begin': begin, 'end': end, 'cycles': end-begin})
    assert overlap
    return {'schema_version': 1, 'experiment': 'E130', 'label': label, 'passed': True,
        'inputdigests': pin, 'verifiers': [file_record(Path(__file__))],
        'unchanged_public_ops_and_O_suffix': True, 'head_arithmetic_unchanged_except_context': True,
        'projections': rows, 'added_raw_f32_write_bytes': sum(r['raw_bytes'] for r in rows),
        'added_raw_f32_valid_read_bytes': sum(r['valid_scalar_read_bytes'] for r in rows),
        'added_sub_fetch_packet_sum_bytes': sum(r['sub_fetch_packet_sum_bytes'] for r in rows),
        'main_dpe_schedule': [{k:i[k] for k in ('index','contexts','lifetime')} for i in dpes],
        'main_dpe_sub_ve_overlap': overlap, 'static_cycles': built['static_cycles'],
        'static_comparison': {'control_cycles': 51293, 'candidate_change_percent': 100*(built['static_cycles']/51293-1)},
        'limits': ['Static packet sums are not measured bus traffic.', 'Logical bounds deliberately exclude ignored trailing padding reads; schedule allocation/lifetime check is separate.', 'No numerical or NPU performance claim: original host is the next gate.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    value = verify(args.label)
    target = HERE / f'graph-verification-{args.label}.json'
    if args.check:
        assert read(target) == value
    else:
        write(target, value)
    print('PASS: frozen source/graph/image, 3 pure Main F32 dots, matching Sub scale-before-BF16, full weight source bounds and overlap')
