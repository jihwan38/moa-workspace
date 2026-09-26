#!/usr/bin/env python3
"""Audit emitted FP4 ingress and explicit raw Main copies; not hardware traffic."""
import json
import math
import re
import sys
from build import HERE, ARTIFACTS, digest

def audit(label):
    build = json.loads((HERE / f'build-{label}.json').read_text())
    out = {'label': label, 'scope': 'Logical emitted descriptor volumes/ranges; no transaction or performance estimate', 'kernels': {}}
    for mode in ['materialized', 'direct_fixed', 'hybrid']:
        path = ARTIFACTS / label / f'{mode}-graph.json'
        assert digest(path.read_bytes()) == build['dumps'][path.name]
        graph = json.loads(path.read_text())
        tensors = graph['tensors']
        ingress, copies, merges = [], [], []
        for op in graph['operators'].values():
            inputs = [tensors[str(i)] for i in op['inputs']]
            outputs = [tensors[str(i)] for i in op['outputs']]
            weights = [t for t in inputs if t['element_type'] == 'Float4e2m1' and t['buffer_type'] == 'Dram']
            if op['kind'] == 'Dma.DtoS' and weights:
                assert len(weights) == 1
                tensor = weights[0]
                descriptors = []
                for pe, shape, base in re.findall(r'src(\d+): \[(.*?)\], Dram\(chip=0, in_chip=(0x[0-9A-Fa-f]+)\)', op['description']):
                    dims = [[int(n), int(s)] for n, s in re.findall(r'=(\d+):(\d+)', shape)]
                    volume = math.prod(n for n, s in dims) * dims[-1][1]
                    end = int(base, 16) + sum((n - 1) * s for n, s in dims) + dims[-1][1]
                    assert end <= tensor['size'], (mode, op['name'], end, tensor['size'])
                    descriptors.append({'pe': int(pe), 'dimensions': dims, 'base': int(base, 16), 'bytes': volume, 'end_exclusive': end})
                assert len(descriptors) == 8
                volume = sum(d['bytes'] for d in descriptors)
                assert volume in [29491200, 983040]
                ingress.append({'operator': op['name'], 'source_address': tensor['optionals']['address'], 'input_allocation_bytes': tensor['size'], 'descriptor_bytes': volume, 'maximum_end_exclusive': max(d['end_exclusive'] for d in descriptors), 'descriptors': descriptors})
            if op['kind'] == 'TuExec' and op['context'] == 'Main':
                if outputs and all(t['element_type'] == 'Float4e2m1' for t in outputs):
                    assert 'fetch_sequencer: 0+[None=1800:64, None=64:1]' in op['description']
                    assert 'commit_projection: 0+[None=1800:64, None=64:1]' in op['description']
                    assert 've_pass:' not in op['description']
                    copies.append({'operator': op['name'], 'bytes_per_slice': 57600, 'total_bytes': 29491200, 'inputs': op['inputs'], 'outputs': op['outputs'], 'description': op['description']})
                if 'fetch_sequencer: 0+[None=30:8, None=8:1]' in op['description'] and 've_pass:' not in op['description'] and outputs and all(t['element_type'] == 'Float32' for t in outputs):
                    assert 'commit_projection: 0+[None=30:8, None=8:1]' in op['description']
                    merges.append({'operator': op['name'], 'bytes_per_slice': 960, 'total_bytes': 491520, 'inputs': op['inputs'], 'outputs': op['outputs'], 'description': op['description']})
        assert sorted(i['descriptor_bytes'] for i in ingress) == ([983040] if mode == 'hybrid' else []) + [29491200, 29491200]
        assert len(copies) == (0 if mode == 'materialized' else 2)
        assert len(merges) == (1 if mode == 'hybrid' else 0)
        out['kernels'][mode] = {'graph': digest(path.read_bytes()), 'ingress': ingress, 'joint_raw_copies': copies, 'partial_merge': merges}
    return out

if __name__ == '__main__':
    label = sys.argv[1]
    result = audit(label)
    path = HERE / f'descriptor-audit-{label}.json'
    if '--write' in sys.argv:
        with path.open('x') as f:
            json.dump(result, f, indent=2); f.write('\n')
    elif path.exists():
        assert json.loads(path.read_text()) == result
    print(json.dumps({mode: {'ingress_bytes': [i['descriptor_bytes'] for i in data['ingress']], 'copy_bytes': sum(i['total_bytes'] for i in data['joint_raw_copies']), 'partial_merge_bytes': sum(i['total_bytes'] for i in data['partial_merge'])} for mode, data in result['kernels'].items()}, indent=2))
