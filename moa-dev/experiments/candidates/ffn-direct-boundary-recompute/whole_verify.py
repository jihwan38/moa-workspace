#!/usr/bin/env python3
"""Verify whole E123 provenance and all emitted weight-ingress source ranges."""
import json
import math
import re
import sys
from build import HERE, ROOT, ARTIFACTS, digest

def verify(label):
    b = json.loads((HERE / f'build-{label}-repro.json').read_text())
    p = ARTIFACTS / label
    assert b['exact_compile_exit_code'] == b['host_build_exit_code'] == 0
    assert b['source_inputs_verified_in_container']
    for name, expected in b['source_inputs'].items():
        assert digest((p / 'crate' / name).read_bytes()) == expected, name
    for name, path in b['frozen_files'].items():
        expected = b['source_inputs'].get(name) or b['logs'].get(name) or b.get(name.removesuffix('.json'))
        assert digest((ROOT / path).read_bytes()) == expected, name
    plan = b['source_plan']
    assert digest((ROOT / plan['recipe']['path']).read_bytes()) == {k: plan['recipe'][k] for k in ['bytes', 'sha256']}
    inputs = dict(plan['source_origin_inputs'])
    for target, source in plan['source_overrides'].items():
        expected = {k: source[k] for k in ['bytes', 'sha256']}
        assert digest((ROOT / source['path']).read_bytes()) == expected
        inputs[target] = expected
    assert inputs == b['source_inputs']
    assert {f.name: digest(f.read_bytes()) for f in (p / 'package').iterdir()} == b['package_files']
    host = (p / 'package/test_runtime').read_bytes()
    for name, expected in b['images'].items():
        data = (p / 'kernels' / f'furiosa_opt_gemma4::ops::{name}.bin').read_bytes()
        assert {**digest(data), 'host_occurrences': host.count(data)} == expected
        assert host.count(data) == 1
    assert digest((p / 'exact-kernel.bin').read_bytes()) == b['exact_image']
    assert b['exact_image'] == {k: b['images']['decoder_feedforward'][k] for k in ['bytes', 'sha256']}
    for name, expected in plan['expected_unchanged_images'].items():
        assert b['images'][name]['sha256'] == expected
    assert digest((p / 'graph.json').read_bytes()) == b['graph']
    assert digest((p / 'schedule.json').read_bytes()) == b['schedule']
    # No quantizer or consumer changes hidden in the composed source file.
    current = (p / 'crate/src/device/shared/mlp.rs').read_text()
    control = (HERE / 'v4/mlp.rs').read_text()
    boundary = 'fn cast_gathered_rows('
    assert current[current.index(boundary):] == control[control.index(boundary):]
    g = json.loads((p / 'graph.json').read_text())
    ingress, copies = [], []
    for op in g['operators'].values():
        ts = [g['tensors'][str(i)] for i in op['inputs']]
        weights = [t for t in ts if t['element_type'] == 'Float4e2m1' and t['buffer_type'] == 'Dram']
        if op['kind'] == 'Dma.DtoS' and weights:
            assert len(weights) == 1
            descriptors = []
            for pe, shape, base in re.findall(r'src(\d+): \[(.*?)\], Dram\(chip=0, in_chip=(0x[0-9A-Fa-f]+)\)', op['description']):
                dims = [[int(n), int(s)] for n, s in re.findall(r'=(\d+):(\d+)', shape)]
                size = math.prod(n for n, s in dims) * dims[-1][1]
                end = int(base, 16) + sum((n - 1) * s for n, s in dims) + dims[-1][1]
                assert end <= weights[0]['size'], (op['name'], end, weights[0]['size'])
                descriptors.append({'pe': int(pe), 'shape': dims, 'base': int(base, 16), 'bytes': size, 'end_exclusive': end})
            assert len(descriptors) == 8
            assert sum(d['bytes'] for d in descriptors) == 29491200
            ingress.append({'operator': op['name'], 'input_allocation_bytes': weights[0]['size'], 'source': weights[0]['optionals']['address'], 'descriptors': descriptors})
        if op['kind'] == 'TuExec' and op['context'] == 'Main' and op['outputs'] and all(g['tensors'][str(i)]['element_type'] == 'Float4e2m1' for i in op['outputs']):
            assert 'fetch_sequencer: 0+[None=1800:64, None=64:1]' in op['description']
            assert 'commit_projection: 0+[None=1800:64, None=64:1]' in op['description']
            assert 've_pass:' not in op['description']
            copies.append({'operator': op['name'], 'logical_bytes': 29491200, 'description': op['description']})
    assert len(ingress) == 3
    assert len(copies) == 2
    return {'label': label, 'status': 'PASS', 'scope': 'Provenance and logical weight descriptors only; not hardware performance', 'inputs': b['source_inputs'], 'images': b['images'], 'unchanged_e104_quantizer_consumers': True, 'fp4_ingress': ingress, 'raw_joint_copies': copies}

if __name__ == '__main__':
    label = sys.argv[1]
    result = verify(label)
    if '--write' in sys.argv:
        with (HERE / f'verification-{label}.json').open('x') as f:
            json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ['label', 'status', 'scope', 'unchanged_e104_quantizer_consumers']}, indent=2))
