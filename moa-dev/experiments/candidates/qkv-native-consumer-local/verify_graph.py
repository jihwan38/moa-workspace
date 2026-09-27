#!/usr/bin/env python3
"""Bounded E138 emitted native/order/register gate and independent review binding."""
import argparse
import json
from pathlib import Path
import package as p

HERE = Path(__file__).resolve().parent


def analyze(label):
    assert label == 'e138-native-v1'
    path, built, _, _, _ = p.checked_compile(label)
    artifact = p.ARTIFACTS / label
    graph, schedule = [p.read(artifact / (name + '.json')) for name in ('graph', 'schedule')]
    review_path = HERE / 'independent-native-v1-review.json'
    review = p.read(review_path)
    assert review['passed'] is True
    for name, expected in review['inputdigests'].items():
        assert p.digest((p.ROOT / name).read_bytes()) == expected, name
    natives = [o for o in graph['operators'].values()
               if 'ReduceAccTree: LReduce(+) [axes=QkvTerm]' in o.get('description', '')]
    assert len(natives) == 3
    for op in natives:
        d = op['description']
        assert op['context'] == 'Main' and 'broadcast_kind: Direct, f8_e4' in d
        assert '<ve_fp_way>QkvTerm_1=4' in d
        assert d.index('FpMul0:') < d.index('ReduceAccTree:') < d.index('GReduceAdd:')
        assert 'GReduceAddF [ratio=4]' in d and 'commit_in_size: 8B (=2EA)' in d
        assert 'cast_compaction' not in d
        assert any(f'fetch_sequencer: 0+[None={n}:32, None=32:1]' in d for n in (480,960))
        regs = [graph['tensors'][str(t)] for t in op['inputs']
                if graph['tensors'][str(t)]['buffer_type'] in ('Vrf', 'Trf')]
        assert len(regs) == 2
        for reg in regs:
            if reg['buffer_type'] == 'Trf':
                assert reg['element_type'] == 'Float8e4m3' and reg['size'] == 512 * 3840
                capacity = 65536
            else:
                assert reg['element_type'] == 'Float32' and reg['size'] == 512 * 32
                capacity = 8192
            assert int(reg['optionals']['address']) + reg['size'] // 512 <= capacity
    snative = [i for i in schedule['instructions'] if i['tpe'] == 'Main'
               and '.contract_lane::' in i['description']]
    assert len(snative) == 3
    shared_registers = set(snative[0]['input_tensors']) & set(snative[1]['input_tensors']) & set(snative[2]['input_tensors'])
    tensors = {int(t['index']): t for t in schedule['tensors']}
    assert len([i for i in shared_registers if i in tensors and tensors[i]['buffer_type'] in ('Trf','Vrf')]) == 2
    checks = []
    for instruction in snative:
        regs = [tensors[t] for t in instruction['input_tensors'] if t in tensors and tensors[t]['buffer_type'] in ('Vrf', 'Trf')]
        assert len(regs) == 2
        for reg in regs:
            assert reg['lifetime']['begin'] <= instruction['lifetime']['begin']
            assert reg['lifetime']['end'] >= instruction['lifetime']['end']
            capacity = 65536 if reg['buffer_type'] == 'Trf' else 8192
            assert reg['address'] + reg['size'] <= 512 * capacity
        checks.append({'index': instruction['index'], 'lifetime': instruction['lifetime'], 'registers': regs})
    return {'schema_version': 1, 'label': label, 'passed': True,
            'inputdigests': {'build_record': p.file_record(path),
                            'graph': p.file_record(artifact / 'graph.json'),
                            'schedule': p.file_record(artifact / 'schedule.json')},
            'verifiers': [p.file_record(Path(__file__).resolve()), p.file_record(HERE / 'package.py'), p.file_record(review_path)],
            'independent_descriptor_and_source_review': p.file_record(review_path),
            'native_names': [o['name'] for o in natives], 'register_capacity_and_lifetimes': checks,
            'weight_fetch_per_slice_bytes': {'q':30720,'k':15360,'v':15360},
            'shared_trf_and_coefficient_ids': sorted(shared_registers),
            'scope': 'Source/input/image provenance plus independent descriptor review, spatial Term4 F32 coefficient/reduce then K4/8B commit, native register bounds/lifetimes. Hidden SFR execution, numerical accuracy and whole performance require NPU.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', nargs='?', default='e138-native-v1')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = analyze(args.label)
    path = HERE / f'graph-verification-{args.label}.json'
    if args.check:
        assert p.read(path) == result
    else:
        p.write(path, result)
    print(json.dumps({'passed': True, 'check': args.check, 'path': str(path)}))
