#!/usr/bin/env python3
"""Recheck the frozen E136 descriptor review and native register/order gate.

This bounded gate is specific to native-v1; it is not an ISA simulator, hardware
liveness proof, or numerical/performance result. --check performs no writes.
"""
import argparse
import json
from pathlib import Path
import package as p

HERE = Path(__file__).resolve().parent


def analyze(label):
    assert label == 'e136-native-v1'
    build_path, built, _, _, _ = p.checked_compile(label)
    artifact = p.ARTIFACTS / label
    graph = p.read(artifact / 'graph.json')
    schedule = p.read(artifact / 'schedule.json')
    review_path = HERE / 'independent-native-review.json'
    review = p.read(review_path)
    assert review['passed']
    for path, expected in review['inputdigests'].items():
        assert p.digest((p.ROOT / path).read_bytes()) == expected, path
    ops = {o['name']: o for o in graph['operators'].values()}
    tensors = graph['tensors']
    native_rows = []
    for item in review['native_streams']:
        native = ops[item['native']]
        trf = ops[item['weight_trf']]
        ingress = ops[item['weight_hbm']]
        post = ops[item['postscale']]
        assert trf['kind'] == 'StoTrf' and trf['context'] == 'Sub'
        assert ingress['outputs'][0] in trf['inputs']
        assert trf['outputs'][0] in native['inputs']
        assert 'lookup' not in trf['description'].lower()
        weight = tensors[str(trf['outputs'][0])]
        assert weight['element_type'] == 'Float8e4m3'
        assert weight['size'] == item['weight_bytes']
        assert int(weight['optionals']['address']) + weight['size'] // 512 <= 65536
        d = native['description']
        assert 'QkvTerm_1=4' in d and 'broadcast_kind: Direct, f8_e4' in d
        assert d.index('FpMul0:') < d.index('ReduceAccTree: LReduce(+) [axes=QkvTerm]')
        assert 'cast_compaction' not in d
        raw = tensors[str(native['outputs'][0])]
        assert raw['element_type'] == 'Float32' and raw['size'] == item['f32_raw_bytes']
        assert native['outputs'][0] in post['inputs']
        pd = post['description']
        assert pd.index('FpMul0:') < pd.index('FpMul1:') < pd.index('cast_compaction:')
        coeff = [tensors[str(t)] for t in native['inputs'] if tensors[str(t)]['buffer_type'] == 'Vrf']
        assert len(coeff) == 1 and coeff[0]['size'] == 512 * 128
        assert int(coeff[0]['optionals']['address']) + 128 <= 8192
        native_rows.append(item)
    instructions = {i['index']: i for i in schedule['instructions']}
    stensors = {t['index']: t for t in schedule['tensors']}
    lifetime_checks = []
    for row in review['schedule_separate_namespace']:
        instruction = instructions[row['schedule_index']]
        assert instruction['lifetime'] == row['lifetime']
        for recorded in row['registers']:
            actual = stensors[recorded['index']]
            assert actual == recorded
            assert int(actual['index']) in instruction['input_tensors']
            begin, end = instruction['lifetime']['begin'], instruction['lifetime']['end']
            assert actual['lifetime']['begin'] <= begin <= end <= actual['lifetime']['end']
            capacity = 65536 if actual['buffer_type'] == 'Trf' else 8192
            assert actual['address'] + actual['size'] <= 512 * capacity
        lifetime_checks.append(row)
    # Q occupies TRF upper half; K/V share lower half at disjoint live intervals.
    weights = [r['registers'][0] for r in lifetime_checks]
    for i, a in enumerate(weights):
        for b in weights[i+1:]:
            address_overlap = max(a['address'], b['address']) < min(a['address']+a['size'], b['address']+b['size'])
            time_overlap = max(a['lifetime']['begin'], b['lifetime']['begin']) < min(a['lifetime']['end'], b['lifetime']['end'])
            assert not (address_overlap and time_overlap)
    return {
        'schema_version': 1, 'label': label, 'passed': True,
        'inputdigests': {'build_record': p.file_record(build_path),
                        'graph': p.file_record(artifact / 'graph.json'),
                        'schedule': p.file_record(artifact / 'schedule.json')},
        'verifiers': [p.file_record(Path(__file__).resolve()), p.file_record(HERE / 'package.py'),
                      p.file_record(review_path)],
        'source_manifest_and_exact_image_verified': True,
        'independent_review': p.file_record(review_path),
        'native_weight_and_numerical_order': native_rows,
        'native_register_capacity_and_lifetime': lifetime_checks,
        'shared_weight_trf_live_intervals_do_not_overlap': True,
        'scope': 'Digest-bound independent exhaustive weight DMA mapping plus emitted native term/scale order, register capacity and lifetime checks. Does not establish hidden SFR iterator correctness, NPU liveness, arithmetic accuracy or speed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', nargs='?', default='e136-native-v1')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = analyze(args.label)
    path = HERE / f'graph-verification-{args.label}.json'
    if args.check:
        assert p.read(path) == result
    else:
        p.write(path, result)
    print(json.dumps({'passed': True, 'check': args.check, 'record': str(path)}))
