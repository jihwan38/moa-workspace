#!/usr/bin/env python3
"""Check source coverage and emitted transfer extents, separately from NPU timing."""
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILDS = {'v1': 'm07-dma-only-v1', 'v3': 'm07-stream-assembly-v3'}


def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def coordinates():
    up, down = [], []
    for c in range(2):
        for s in range(256):
            local_up, local_down = [], []
            r, k = divmod(s, 4)
            for chunk in range(5):
                local_up.append((6 * chunk * 1920, 6 * (chunk + 1) * 1920))
                local_down.append((12 * chunk * 960, 12 * (chunk + 1) * 960))
                for t in range(6 * chunk, 6 * (chunk + 1)):
                    l = 7680 * c + 30 * s + t
                    up.append((l * 1920, (l + 1) * 1920))
                for t in range(12 * chunk, 12 * (chunk + 1)):
                    h, l = 60 * r + t, 7680 * c + 1920 * k
                    start = (h * 15360 + l) // 2
                    down.append((start, start + 960))
            for spans in [local_up, local_down]:
                assert spans[0][0] == 0 and spans[-1][1] == 57600
                assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
                assert all(a % 8 == b % 8 == 0 and b - a == 11520 for a, b in spans)
    for spans in [up, down]:
        spans.sort()
        assert spans[0][0] == 0 and spans[-1][1] == 29491200
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    return {'upgate_row_segments': len(up), 'down_row_segments': len(down),
            'per_matrix_logical_bytes': 29491200, 'intended_per_transfer_bytes': 5898240,
            'per_slice_chunk_bytes': 11520, 'per_slice_full_packed_bytes': 57600,
            'all_logical_elements_covered_once': True, 'destination_tiles_disjoint': True,
            'scope': 'Source coordinates only; emitted DMA may cover padded extents too.'}


def descriptors(path):
    graph = json.loads(path.read_text())
    result = []
    for index, op in graph['operators'].items():
        if op['kind'] != 'Dma.DtoS':
            continue
        src = graph['tensors'][str(op['inputs'][0])]
        if src['element_type'] != 'Float4e2m1':
            continue
        commands = re.findall(r'^src\d+: \[([^\]]+)\], Dram', op['description'], re.M)
        assert commands
        sizes = []
        for command in commands:
            loops = re.findall(r'=([0-9]+):([0-9]+)', command)
            assert loops and len(loops) == len(command.split(','))
            # Last loop is the contiguous byte packet; outer strides are pitches.
            sizes.append(math.prod(int(count) for count, _ in loops) * int(loops[-1][1]))
        result.append({'graph_operator': index, 'name': op['name'],
                       'projection_layout': 'upgate' if 'FfnSlice' in json.dumps(src['shape']) else 'down',
                       'source_type': src['element_type'], 'source_shape': src['shape'],
                       'source_descriptors': len(commands), 'descriptor_byte_volume': sum(sizes),
                       'description': op['description']})
    assert len(result) == 15
    return {'graph': digest(path.read_bytes()), 'weight_dma': result,
            'total_descriptor_byte_volume': sum(n['descriptor_byte_volume'] for n in result),
            'scope': 'Emitted descriptor loop volume, not actual bus transactions/bandwidth or hardware time.'}


def schedule(path, chunked=False):
    s = json.loads(path.read_text())
    ins = s['instructions']
    main = [n for n in ins if n['tpe'] == 'Main']
    dma = [n for n in ins if n['tpe'].startswith('Dma')]
    expansion = [n for n in main if 'fetch_table_lookup::<f8e' in n['description']]
    dots = [n for n in main if 'contract_outer' in n['description']]
    norm = [n for n in main if 'src/device/shared/rmsnorm.rs:95:' in n['description']]
    amax = [n for n in main if '0x7fff_ffff' in n['description']]
    broadcast = [n for n in main if 'FfnTerm' in n['description'] and 'CustomBroadcast { ring_size: 32 }' in n['description']]
    weights = [n for n in dma if n['tpe'] == 'DmaLoad' and 'src/device/shared/mlp.rs:' in n['description'] and
               (('weight' in n['description'] and '.tile' in n['description']) if chunked else
                any(x in n['description'] for x in ['weight_packed.to_dm', 'weight.to_dm']))]
    assembly = [n for n in ins if n['tpe'] == 'Sub' and 'begin(chunk.view())' in n['description']]
    assert len(expansion) == len(dots) == 3
    assert len(norm) == len(amax) == len(broadcast) == 1
    assert len(weights) == (15 if chunked else 3)
    def detail(nodes):
        return [{'index': n['index'], 'type': n['tpe'], **n['lifetime'],
                 'description': n['description']} for n in nodes]
    return {'schedule': digest(path.read_bytes()),
            'makespan': max(n['lifetime']['end'] for n in ins),
            'dma_count': len(dma), 'main_count': len(main), 'instruction_count': len(ins),
            'fp8_expansion': detail(expansion), 'dots': detail(dots), 'weight_loads': detail(weights),
            'sub_assembly': detail(assembly), 'pre_rms': detail(norm), 'amax': detail(amax),
            'pre_rms_to_amax_gap': amax[0]['lifetime']['begin'] - norm[0]['lifetime']['end'],
            'input_broadcast': detail(broadcast),
            'fp8_logical_write_bytes': 3 * 3840 * 15360,
            'scope': 'Compiler schedule, not NPU performance/stall measurement; intervals overlap.'}


def result():
    paths = {'E104': HERE.parent / 'e104-weight-path/evidence/profile-v1/schedule.json',
             'M03': HERE.parent / 'm03-producer-local-ffn/evidence/m03-native-local-v1/schedule.json',
             **{k: HERE / 'evidence' / b / 'schedule.json' for k, b in BUILDS.items()}}
    schedules = {k: schedule(p, k in BUILDS) for k, p in paths.items()}
    emitted = {k: descriptors(HERE / 'evidence' / b / 'ingress-graph.json') for k, b in BUILDS.items()}
    assert sorted(n['descriptor_byte_volume'] for n in emitted['v1']['weight_dma']) == [5898240] * 5 + [29491200] * 10
    assert [n['descriptor_byte_volume'] for n in emitted['v3']['weight_dma']] == [5898240] * 15
    assert len(schedules['v3']['sub_assembly']) == 10
    return {'coordinates': coordinates(), 'schedules': schedules, 'descriptors': emitted}


if __name__ == '__main__':
    value = result()
    (HERE / 'static-analysis.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: {f: v[f] for f in ['makespan', 'dma_count', 'pre_rms_to_amax_gap']}
                      for k, v in value['schedules'].items()}, indent=2))
