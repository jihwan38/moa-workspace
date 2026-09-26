#!/usr/bin/env python3
"""Validate one frozen M08 graph and bind the result to its build/image/source inputs.

    python3 verify_descriptors.py m08-pair4-v1
    python3 verify_descriptors.py m08-pair4-v1 --check

The first command preserves descriptor-analysis-LABEL.json (success or failure).
--check recomputes and compares without writing. No compilation or NPU execution.
Descriptor PASS is not NPU numerical correctness or a performance claim.
"""
import argparse
import copy
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts' / HERE.name
DEFAULT_BASELINE = ROOT / 'moa-dev/experiments/artifacts/output-bf16-activation-e47/candidate-graph.json'
BASELINE_GRAPH_SHA = '24d45fc706964d6dbfb7eb3a52a29381dd59347cd714d681ed2ab603f506cee0'
SLICE_STRIDE = 4194304  # virtual descriptor address encoding, not slice capacity
H, K, NCLUSTER, NSLICE = 3840, 4096, 2, 256
TRANSPARENT = {'LowLevelReshape', 'ClusterSync'}

class AuditError(Exception):
    pass

def require(ok, message):
    if not ok:
        raise AuditError(message)

def only(values, message):
    require(len(values) == 1, f'{message}: expected one, got {len(values)}')
    return values[0]

def parse_descriptors(op):
    parsed = {'src': {}, 'dst': {}}
    for line in op['description'].splitlines():
        if not re.match(r'^(src|dst)\d+:', line):
            continue
        m = re.fullmatch(r'(src|dst)(\d+): \[(.*?)\], (Dram|Sram)\((.*)\)', line.strip())
        require(m is not None, f'unrecognized descriptor: {line}')
        dims = []
        for item in m[3].split(', '):
            dim = re.fullmatch(r'(.*?)=(\d+):(-?\d+)', item)
            require(dim is not None, f'unrecognized dimension: {item}')
            label, count, stride = dim.groups()
            require(int(count) > 0 and int(stride) >= 0, 'negative stride/empty axis requires separate review')
            dims.append({'label': label, 'count': int(count), 'stride': int(stride)})
        require(dims and dims[-1]['count'] == 1 and dims[-1]['stride'] > 0,
                'unsupported packet representation; last dimension must be count1:packet_bytes')
        base = {k: int(v, 0) for k, v in re.findall(r'(\w+)=(0x[0-9A-Fa-f]+|\d+)', m[5])}
        ident = int(m[2])
        require(ident not in parsed[m[1]], 'duplicate descriptor ID')
        parsed[m[1]][ident] = {'memory': m[4], 'dims': dims, 'base': base}
    require(parsed['src'] and parsed['src'].keys() == parsed['dst'].keys(), 'unmatched source/destination descriptor IDs')
    return parsed

def packets(desc):
    dims = desc['dims']
    base = desc['base'].get('in_chip', desc['base'].get('in_slice', 0))
    for indices in itertools.product(*(range(d['count']) for d in dims[:-1])):
        yield base + sum(i*d['stride'] for i, d in zip(indices, dims[:-1])), dims[-1]['stride']

def paired_packets(op):
    ds = parse_descriptors(op)
    sentinel = object()
    for ident in sorted(ds['src']):
        src, dst = ds['src'][ident], ds['dst'][ident]
        for a, b in itertools.zip_longest(packets(src), packets(dst), fillvalue=sentinel):
            require(a is not sentinel and b is not sentinel, 'source/destination packet count mismatch')
            require(a[1] == b[1], 'different packet sizes require byte-stream pairing review')
            yield ident, src, dst, a[0], b[0], a[1]

def physical(desc, address):
    require(desc['memory'] == 'Sram', 'physical address expected SRAM')
    base = desc['base']
    require(base.get('chip', 0) == 0, 'unexpected chip')
    virtual_slice, local = divmod(address, SLICE_STRIDE)
    within_pair = base.get('dmn', 0)*32 + base.get('slice', 0) + virtual_slice
    require(0 <= within_pair < 64, 'descriptor crosses its PE DMN pair')
    cluster, sl = base.get('cluster', 0), base.get('pe', 0)*64 + within_pair
    require(0 <= cluster < 2 and 0 <= sl < 256, 'physical destination outside chip topology')
    return cluster, sl, local

def row_of(layout, cluster, sl, local_row):
    require(0 <= local_row < 120, f'local row {local_row} outside120')
    if layout == 'baseline':
        return 1920*cluster + 120*(sl//16) + local_row
    require(layout == 'pair4', 'unknown layout')
    g, r = sl//64, (sl%64)//16
    t, j = divmod(local_row, 2)
    return 1920*cluster + 480*g + 8*t + 2*r + j

class Graph:
    def __init__(self, raw):
        self.raw = raw
        self.tensors = {int(k): v for k, v in raw['tensors'].items()}
        self.ops = list(raw['operators'].values())
        self.producer = {}
        for op in self.ops:
            for t in op['outputs']:
                require(t not in self.producer, 'tensor has multiple producers')
                self.producer[t] = op

    def ancestry(self, t, transparent_only=False, seen=None):
        seen = set() if seen is None else seen
        if t in seen:
            return seen
        seen.add(t)
        op = self.producer.get(t)
        if op and (not transparent_only or op['kind'] in TRANSPARENT):
            for inp in op['inputs']:
                self.ancestry(inp, transparent_only, seen)
        return seen

    def from_io(self, t, io):
        return any(re.fullmatch(fr'IO\({io}\)\+0', self.tensors[n].get('optionals', {}).get('address', ''))
                   for n in self.ancestry(t, True))

    def dma_from_io(self, io):
        return only([op for op in self.ops if op['kind'] == 'Dma.DtoS' and
                     any(self.from_io(t, io) for t in op['inputs'])], f'HBM IO{io} load')

    def dependent(self, op, ancestor):
        outs = set(ancestor['outputs'])
        return any(outs & self.ancestry(t) for t in op['inputs'])

    def identify(self):
        weight, activation, scale_dma = self.dma_from_io(2), self.dma_from_io(0), self.dma_from_io(3)
        dot = only([op for op in self.ops if op['kind'] == 'TuExec' and op['context'] == 'Main' and
                    'GReduceAdd:' in op['description'] and self.dependent(op, weight)], 'weight dot/reduce')
        gather = only([op for op in self.ops if op['kind'] == 'Dma.StoS' and
                       any(set(dot['outputs']) & self.ancestry(t, True) for t in op['inputs'])], 'first F32 gather')
        scale = only([op for op in self.ops if op['kind'] == 'TuExec' and op['context'] == 'Main' and
                      'FpMul0:' in op['description'] and
                      any(set(gather['outputs']) & self.ancestry(t, True) for t in op['inputs']) and
                      self.dependent(op, scale_dma)], 'channel scale immediately after gather')
        trf = only([op for op in self.ops if op['kind'] == 'StoTrf' and self.dependent(op, activation) and
                    set(op['outputs']) & set(dot['inputs'])], 'activation StoTrf consumed by dot')
        return {'weight': weight, 'activation': activation, 'scale_dma': scale_dma,
                'dot': dot, 'gather': gather, 'scale': scale, 'trf': trf,
                'scale_vrf': self.producer[only([t for t in scale['inputs'] if self.tensors[t]['buffer_type'] == 'Vrf'], 'scale VRF')]}

def check_weight(op, layout):
    source, dest, by_slice, sizes = Counter(), set(), Counter(), Counter()
    for _, sd, dd, sa, da, n in paired_packets(op):
        require(sd['memory'] == 'Dram' and dd['memory'] == 'Sram', 'weight must be HBM→DM')
        require(sd['base'].get('chip', 0) == 0, 'weight chip mismatch')
        require(n == 256 and sa%256 == 0 and 0 <= sa <= H*K-n, 'weight source bounds/alignment/packet')
        c, sl, local = physical(dd, da)
        require(local%256 == 0 and 0 <= local <= 120*256-n, 'weight destination bounds/alignment')
        local_row = local//256
        expected = row_of(layout, c, sl, local_row)*K + (sl%16)*256
        require(sa == expected, f'weight owner mismatch c={c} slice={sl} row={local_row}: {sa}!={expected}')
        key = (c, sl, local_row)
        require(key not in dest, 'duplicate weight destination packet')
        dest.add(key); source[sa] += 1; by_slice[c, sl] += n; sizes[n] += 1
    require(set(source) == set(range(0, H*K, 256)) and set(source.values()) == {1}, 'weight source holes/duplicates')
    require(len(dest) == 512*120 and len(by_slice) == 512 and set(by_slice.values()) == {30720}, 'weight destination coverage')
    return {'status': 'PASS', 'source_bytes_exact_once': H*K, 'physical_slices': 512,
            'bytes_per_slice': 30720, 'descriptor_packet_histogram': dict(sizes),
            'layout': layout, 'element_mapping_verified': 'Each256 B contiguous K chunk is checked against physical row owner and K shard; all chunks cover source and destination exactly.'}

def check_activation(op):
    source, dest, by_slice, sizes = Counter(), set(), Counter(), Counter()
    ds = parse_descriptors(op)
    zeros = sum(d['count'] > 1 and d['stride'] == 0 for s in ds['src'].values() for d in s['dims'])
    for _, sd, dd, sa, da, n in paired_packets(op):
        require(sd['memory'] == 'Dram' and dd['memory'] == 'Sram', 'activation must be HBM→DM')
        require(n%2 == 0 and sa%2 == 0 and 0 <= sa <= K*2-n, 'activation source BF16 bounds')
        c, sl, local = physical(dd, da)
        require(local%2 == 0 and 0 <= local <= 512-n, 'activation local K256 bounds')
        require(sa == (sl%16)*512 + local, f'activation K shard mismatch c={c} slice={sl}')
        for b in range(0, n, 2):
            key = (c, sl, (local+b)//2)
            require(key not in dest, 'activation duplicate destination BF16')
            dest.add(key); source[(sa+b)//2] += 1
        by_slice[c, sl] += n; sizes[n] += 1
    require(len(dest) == 512*256 and len(by_slice) == 512 and set(by_slice.values()) == {512}, 'activation destination coverage')
    require(set(source) == set(range(K)) and set(source.values()) == {32}, 'activation source replication mismatch')
    return {'status': 'PASS', 'unique_hbm_bytes': K*2, 'logical_replicated_payload_bytes': 512*512,
            'source_copies_per_BF16': 32, 'zero_stride_axes': zeros, 'destination_k_matches_s_mod_16': True,
            'descriptor_packet_histogram': dict(sizes), 'note': 'Repeated descriptor payload is not measured HBM traffic or multicast proof.'}

def check_gather(op, layout):
    source_rows, destination_rows, sizes = set(), set(), Counter()
    for _, sd, dd, sa, da, n in paired_packets(op):
        require(sd['memory'] == dd['memory'] == 'Sram', 'gather must be direct DM→DM')
        sc, ss, slocal = physical(sd, sa)
        dc, ds, dlocal = physical(dd, da)
        require(ss%16 == 0, 'gather source is not reduced K-column0 owner')
        require(dc == 0 and 0 <= ds < 8, 'gather destination not canonical cluster0 slices0..7')
        require(n%8 == sa%8 == da%8 == 0, 'F32 gather8 B packet/alignment contract')
        require(0 <= slocal <= 480-n and 0 <= dlocal <= 1920-n, 'gather source/destination local bounds')
        for byte in range(0, n, 4):
            h = row_of(layout, sc, ss, (slocal+byte)//4)
            dh = ds*480 + (dlocal+byte)//4
            require(h == dh, f'gather row mismatch {h}!={dh}')
            require(h not in source_rows and dh not in destination_rows, 'duplicate F32 row in gather')
            source_rows.add(h); destination_rows.add(dh)
        sizes[n] += 1
    require(source_rows == destination_rows == set(range(H)), 'F32 gather source/destination coverage')
    return {'status': 'PASS', 'payload_bytes': H*4, 'descriptor_packet_histogram': dict(sizes),
            'observed_packets': sum(sizes.values()), 'design_expected_packets': 32 if layout == 'baseline' else 1920,
            'expected_packet_count_matches': sum(sizes.values()) == (32 if layout == 'baseline' else 1920),
            'note': 'Observed count is reported, not forced to a theoretical lowering. Byte correspondence is mandatory. No latency/bus-transaction claim.'}

def check_scale_dma(op):
    source, dest = set(), set()
    for _, sd, dd, sa, da, n in paired_packets(op):
        require(sd['memory'] == 'Dram' and dd['memory'] == 'Sram', 'scale must be HBM→DM')
        c, sl, local = physical(dd, da)
        require(c == 0 and sl < 8 and n%2 == sa%2 == local%2 == 0, 'scale destination/dtype')
        require(0 <= sa <= H*2-n and 0 <= local <= 960-n, 'scale source/destination bounds')
        for b in range(0, n, 2):
            sh, dh = (sa+b)//2, sl*480+(local+b)//2
            require(sh == dh and sh not in source and dh not in dest, 'scale channel correspondence/duplicate')
            source.add(sh); dest.add(dh)
    require(source == dest == set(range(H)), 'scale channel coverage')
    return {'status': 'PASS', 'channel_count': H, 'scale_matches_canonical_gather_row': True}

def normalize_description(op):
    return re.sub(r'name: .*?#O\d+', 'name: OP', op['description']).strip()

def check_arithmetic(g, selected, baseline_selected):
    dot, gather, scale, trf = (selected[k] for k in ('dot','gather','scale','trf'))
    desc = dot['description']
    require(g.tensors[selected['weight']['outputs'][0]]['element_type'] == 'Float8e4m3', 'weight dtype changed')
    require(g.tensors[trf['outputs'][0]]['element_type'] == 'Bfloat16', 'x TRF dtype changed')
    require('operation_unit:' in desc and 'lookup_table:' in desc, 'dot or fixed lookup evidence absent')
    require('broadcast_kind: Direct, bf16' in desc, 'dot fetch conversion not BF16')
    require(desc.count('GReduceAdd:') == 1 and 'GReduceAddF [ratio=16]' in desc,
            'expected exactly one16-way F32 additive reduction')
    require('<global_reduce>Qs_256=16<global_reduce>' in desc, 'reduction axis not16 K256 columns')
    require('cast_compaction:' not in desc and 'FpMul0:' not in desc, 'early cast/scale inserted in dot')
    require(all(g.tensors[t]['element_type'] == 'Float32' for t in dot['outputs'] + gather['outputs']), 'dot/gather output not F32')
    # All data on the direct dot→gather→scale route remains F32; dependencies are
    # checked structurally, not by operator IDs or dictionary insertion order.
    for t in gather['inputs'] + scale['inputs']:
        for a in g.ancestry(t, True):
            if set(dot['outputs']) & g.ancestry(a, True) or set(gather['outputs']) & g.ancestry(a, True):
                require(g.tensors[a]['element_type'] == 'Float32', 'pre-scale path has non-F32 intermediate')
    sdesc = scale['description']
    require('FpMul0:' in sdesc and 'cast_compaction:' in sdesc and
            sdesc.index('FpMul0:') < sdesc.index('cast_compaction:') and
            'output_element_type: bf16' in sdesc, 'scale must precede first BF16 conversion')
    require(all(g.tensors[t]['element_type'] == 'Bfloat16' for t in scale['outputs']), 'scaled output dtype changed')
    require(normalize_description(scale) == normalize_description(baseline_selected['scale']),
            'emitted scale/first BF16 operator differs from baseline; requires independent review')
    # The scale VRF is an independent input. Require identical emitted channel
    # fetch/cast description as baseline, plus provenance from IO3 audited above.
    vrf_inputs = [t for t in scale['inputs'] if g.tensors[t]['buffer_type'] == 'Vrf']
    scale_vrf = only(vrf_inputs, 'scale VRF input')
    vrf_op = g.producer[scale_vrf]
    require(g.dependent(vrf_op, selected['scale_dma']), 'scale VRF disconnected from IO3')
    require(g.tensors[scale_vrf]['element_type'] == 'Float32' and 'to Vrf' in vrf_op['description'], 'scale VRF conversion')
    require(normalize_description(vrf_op) == normalize_description(baseline_selected['scale_vrf']),
            'scale VRF channel fetch differs from baseline; independent review required')
    return {'status': 'PASS', 'reduction': 'one GReduceAddF ratio16 over Qs_256',
            'path': 'FP8 fixed lookup→BF16 dot/TRF→F32 K sum→F32 gather→channel MulF→BF16',
            'scale_command_identical_to_baseline_ignoring_name': True,
            'limitation': 'Graph dtype, emitted operation text and dependencies verified; exact DPE arithmetic/register wiring and numerical correctness still require source/vISA review and NPU validation.'}

def audit(raw, layout, baseline_selected):
    g = Graph(raw); selected = g.identify()
    result = {'layout': layout, 'operators': {k:v['name'] for k,v in selected.items()}, 'checks': {}}
    for name, fn in [('weight',lambda:check_weight(selected['weight'],layout)),
                     ('activation',lambda:check_activation(selected['activation'])),
                     ('gather',lambda:check_gather(selected['gather'],layout)),
                     ('scale_channels',lambda:check_scale_dma(selected['scale_dma'])),
                     ('arithmetic_order',lambda:check_arithmetic(g,selected,baseline_selected))]:
        result['checks'][name] = fn()
    result['status'] = 'PASS_GRAPH_CONTRACT_ONLY'
    return result

def negative_self_tests(raw, selected):
    mutations = [
        ('weight_source_pitch', selected['weight']['name'], 'H_1=120:4096', 'H_1=120:8192'),
        ('activation_broadcast_offset', selected['activation']['name'], 'anonymbroadcast(2)_1=2:0', 'anonymbroadcast(2)_1=2:512'),
        ('gather_destination_pitch', selected['gather']['name'], 'H_120=2:480', 'H_120=2:488'),
        ('K_reduce_ratio', selected['dot']['name'], 'GReduceAddF [ratio=16]', 'GReduceAddF [ratio=8]'),
        ('early_BF16_cast', selected['dot']['name'], 'operation_unit:', 'cast_compaction:\noutput_element_type: bf16\noperation_unit:'),
    ]
    outcomes = []
    for name, op_name, old, new in mutations:
        mutant = copy.deepcopy(raw)
        op = only([o for o in mutant['operators'].values() if o['name'] == op_name], name)
        require(old in op['description'], 'negative-test mutation not applied')
        op['description'] = op['description'].replace(old,new,1)
        try:
            audit(mutant, 'baseline', selected)
        except AuditError as exc:
            outcomes.append({'case':name,'status':'EXPECTED_REJECTION','reason':str(exc)})
        else:
            raise AuditError('checker accepted injected error: '+name)
    return outcomes

def file_record(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def evaluate(label):
    out = {'label': label, 'passed': False,
           'scope': 'Independent descriptor/dataflow validation; no NPU numerical or performance claim.',
           'inputdigests': {},
           'address_model': '4MiB virtual slice stride;32 slices/DMN,2DMNs/PE,4PEs/cluster. Base cluster/pe/dmn/slice included. Virtual strides are not physical capacities.'}
    try:
        require(re.fullmatch(r'[a-z0-9-]{1,70}', label) is not None, 'invalid label')
        build_path = HERE / f'build-{label}-repro.json'
        out['inputdigests']['build'] = file_record(build_path)
        build = json.loads(build_path.read_text())
        require(build['label'] == label and build['exact_compile_exit_code'] == 0, 'compile label/status mismatch')
        require(build['target_kernel'] == 'sliding_attention_output', 'wrong kernel')
        require(build['source_inputs_verified_in_container'] is True, 'container source hashes not verified')
        artifact = ARTIFACTS / label
        evidence = HERE / 'evidence' / label
        files = {'graph': evidence/'graph.json', 'schedule': evidence/'schedule.json',
                 'exact_image': artifact/'exact-kernel.bin'}
        for name, path in files.items():
            rec = file_record(path)
            out['inputdigests'][name] = rec
            require({k:rec[k] for k in ('bytes','sha256')} == build[name], name+' digest differs from build')
        # The preserved evidence, compile artifact, frozen changed source and source
        # recipe must identify the same exact compilation, not similarly named code.
        for name in ('graph','schedule'):
            rec = file_record(artifact/f'{name}.json')
            out['inputdigests']['artifact_'+name] = rec
            require({k:rec[k] for k in ('bytes','sha256')} == build[name], 'artifact '+name+' differs')
        source_records = {}
        for path, expected in build['source_inputs'].items():
            rec = file_record(artifact/'crate'/path)
            require({k:rec[k] for k in ('bytes','sha256')} == expected, 'compile source changed: '+path)
            source_records[path] = rec
        out['inputdigests']['compiled_source_inputs'] = source_records
        out['source_inputs_verified'] = True
        overrides = build['source_plan']['source_overrides']
        out['inputdigests']['source_overrides'] = {}
        out['inputdigests']['frozen_sources'] = {}
        for relative, expected in overrides.items():
            rec = file_record(ROOT/expected['path'])
            require({k:rec[k] for k in ('bytes','sha256')} == build['source_inputs'][relative], 'variant source changed: '+relative)
            out['inputdigests']['source_overrides'][relative] = rec
            frozen = file_record(ROOT/build['frozen_files'][relative])
            require({k:frozen[k] for k in ('bytes','sha256')} == build['source_inputs'][relative], 'frozen source changed: '+relative)
            out['inputdigests']['frozen_sources'][relative] = frozen
        recipe = build['source_plan']['recipe']
        rec = file_record(ROOT/recipe['path'])
        require(rec == recipe, 'recipe changed since compilation')
        out['inputdigests']['recipe'] = rec
        out['inputdigests']['baseline_graph'] = file_record(DEFAULT_BASELINE)
        require(out['inputdigests']['baseline_graph']['sha256'] == BASELINE_GRAPH_SHA, 'baseline graph changed')
        out['inputdigests']['checker'] = file_record(Path(__file__).resolve())
        baseline_raw = json.loads(DEFAULT_BASELINE.read_text())
        baseline_selected = Graph(baseline_raw).identify()
        out['baseline'] = audit(baseline_raw, 'baseline', baseline_selected)
        out['negative_self_tests'] = negative_self_tests(baseline_raw, baseline_selected)
        out['candidate'] = audit(json.loads(files['graph'].read_text()), 'pair4', baseline_selected)
        out['static_cycles'] = build['static_cycles']
        out['status'] = 'PASS_GRAPH_CONTRACT_ONLY'
        out['passed'] = True
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        out['status'] = 'FAIL_OR_UNSUPPORTED_REQUIRES_REVIEW'
        out['error'] = str(exc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label', help='exact compilation label, e.g. m08-pair4-v1')
    ap.add_argument('--check', action='store_true', help='recompute and compare; never write any file')
    args = ap.parse_args()
    if re.fullmatch(r'[a-z0-9-]{1,70}', args.label) is None:
        ap.error('label must contain only lowercase letters, digits and hyphens')
    path = HERE/f'descriptor-analysis-{args.label}.json'
    result = json.loads(json.dumps(evaluate(args.label)))
    if args.check:
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print('FAIL: preserved descriptor analysis unavailable:', exc)
            return 1
        if existing != result:
            print('FAIL: descriptor analysis differs from current inputs; no file written')
            if result.get('error'):
                print(result['error'])
            return 1
    else:
        if path.exists():
            if json.loads(path.read_text()) != result:
                print('FAIL: refusing to overwrite frozen analysis; use a new label or inspect with --check')
                return 1
        else:
            with path.open('x') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                f.write('\n')
    if result['passed']:
        print('PASS: exact provenance, weight/input/F32-gather coordinates,16K reduction and scale→BF16;', args.label)
    else:
        print('FAIL:', result.get('error',result['status']))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
