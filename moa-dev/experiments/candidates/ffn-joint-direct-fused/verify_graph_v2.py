#!/usr/bin/env python3
"""Verify the frozen E131 emitted graph. --check compares without writing."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts' / HERE.name


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def record(path):
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def digest(rec):
    return {k: rec[k] for k in ('bytes', 'sha256')}


def sequence(description, field):
    match = re.search(rf'^\s*{field}: (\d+)\+\[([^\]]*)\]', description, re.M)
    require(match is not None, f'Missing emitted {field}')
    dims = [(int(n), int(s)) for n, s in re.findall(r'[^,]+=(\d+):(\d+)', match[2])]
    require(dims, f'Empty {field}')
    return int(match[1]), dims


def offsets(seq):
    base, dims = seq
    return [base + sum(i * stride for i, (_, stride) in zip(index, dims))
            for index in itertools.product(*(range(n) for n, _ in dims))]


def intervals(base, dims):
    # Graph DMA descriptions make the final dimension the contiguous packet.
    packet = dims[-1][0] * dims[-1][1]
    require(packet > 0, 'Empty DMA packet')
    for index in itertools.product(*(range(n) for n, _ in dims[:-1])):
        lo = base + sum(i * stride for i, (_, stride) in zip(index, dims[:-1]))
        yield lo, lo + packet


def full_coverage(packets, size, why):
    ordered = sorted(packets)
    end = 0
    for lo, hi in ordered:
        require(lo == end and hi > lo and hi <= size,
                f'{why}: gap/overlap/out-of-bounds packet [{lo},{hi}) after {end}, extent {size}')
        end = hi
    require(end == size, f'{why}: coverage ends at {end}, expected {size}')
    return {'packet_count': len(ordered), 'bytes': size, 'bounds': [0, size],
            'no_gaps_or_overlaps': True}


def dma_descriptors(description):
    result = {}
    pattern = r'^(src|dst)(\d+): \[(.*?)\], (Dram|Sram)\((.*?)\)$'
    for side, pe, shape, media, address in re.findall(pattern, description, re.M):
        dims = [(int(n), int(s)) for n, s in re.findall(r'=(\d+):(\d+)', shape)]
        require(dims, 'Empty DMA dimensions')
        fields = {k: int(v, 0) for k, v in re.findall(r'(\w+)=(0x[0-9A-Fa-f]+|\d+)', address)}
        result.setdefault(int(pe), {})[side] = {'dims': dims, 'media': media, 'address': fields}
    require(result and all(set(pair) == {'src', 'dst'} for pair in result.values()), 'Unpaired DMA descriptors')
    return result


class Graph:
    def __init__(self, data):
        self.data = data
        self.ts = data['tensors']
        self.ops = data['operators']
        self.producers = {tid: (k, op) for k, op in self.ops.items() for tid in op['outputs']}

    def tensor(self, tid):
        return self.ts[str(tid)]

    def mappings(self, tid):
        return self.tensor(tid)['shape']['MappingShape']['inner']['mappings']

    def producer(self, tid, kind=None):
        pair = self.producers[tid]
        require(kind is None or pair[1]['kind'] == kind, f'T{tid}: expected {kind}, got {pair[1]["kind"]}')
        return pair

    def scalar(self, tid):
        _, op = self.producer(tid, 'CoreImpCmd')
        match = re.search(r'__scalar_out\[\] = (\d+)i32;', op['description'])
        require(match is not None, f'T{tid}: tile index is not a literal')
        return int(match[1])

    def view(self, tid):
        key, op = self.producer(tid, 'IndexAccess')
        match = re.search(r'axis: stride (\d+) x (\d+), window_size: (\d+)', op['description'])
        require(match is not None and len(op['inputs']) == 2, f'T{tid}: malformed indexed view')
        return {'tensor': tid, 'operator': key, 'parent': op['inputs'][0],
                'stride_elements': int(match[1]), 'axis_extent': int(match[2]),
                'window': int(match[3]), 'start': self.scalar(op['inputs'][1])}

    def root_output(self, tid):
        key, op = self.producer(tid)
        if op['kind'] == 'Scratchpad':
            return tid
        require(op['kind'] == 'IndexWrite', f'T{tid}: unexpected mutable output chain {op["kind"]}')
        return self.root_output(op['inputs'][0])


def provenance(label):
    require(re.fullmatch(r'[a-z0-9-]{1,70}', label), 'Invalid label')
    build_path = HERE / f'build-{label}-repro.json'
    build = json.loads(build_path.read_text())
    variant = build['variant']
    paths = {'build_record': build_path, 'source': HERE / variant / 'mlp.rs',
             'recipe': HERE / variant / 'recipe.json',
             'graph': ARTIFACTS / label / 'graph.json',
             'schedule': ARTIFACTS / label / 'schedule.json',
             'exact_image': ARTIFACTS / label / 'exact-kernel.bin'}
    recs = {k: record(p) for k, p in paths.items()}
    require(build['exact_compile_exit_code'] == 0 and build['source_inputs_verified_in_container'], 'Exact compile did not pass')
    require(build['label'] == label, 'Build label mismatch')
    require(digest(recs['source']) == build['source_inputs']['src/device/shared/mlp.rs'], 'Compiled source digest mismatch')
    require(recs['recipe'] == build['source_plan']['recipe'], 'Recipe digest mismatch')
    for key in ('graph', 'schedule', 'exact_image'):
        require(digest(recs[key]) == build[key], f'{key} build digest mismatch')
    for key in ('graph', 'schedule'):
        require(paths[key].read_bytes() == (HERE / 'evidence' / label / f'{key}.json').read_bytes(), f'{key} artifact differs')
    require(paths['source'].read_bytes() == (HERE / 'evidence' / label / 'src/device/shared/mlp.rs').read_bytes(), 'Frozen source differs')
    plan = build['source_plan']
    planned = dict(plan['source_origin_inputs'])
    for name, override in plan['source_overrides'].items():
        require(record(ROOT / override['path']) == override, f'Override changed: {name}')
        planned[name] = digest(override)
    require(planned == build['source_inputs'], 'Recipe source manifest mismatch')
    for name, expected in build['source_inputs'].items():
        require(digest(record(ARTIFACTS / label / 'crate' / name)) == expected, f'Compiled crate changed: {name}')
    return build, paths, recs



def hbm_ingress(g):
    records = []
    for key, op in g.ops.items():
        if op['kind'] != 'Dma.DtoS':
            continue
        ids = [tid for tid in op['inputs'] if g.tensor(tid)['buffer_type'] == 'Dram'
               and g.tensor(tid)['element_type'] in ('Float4e2m1', 'Float8e4m3')]
        if not ids:
            continue
        require(len(ids) == 1 and len(op['outputs']) == 1, 'Unexpected matrix ingress IO')
        tid, out = ids[0], op['outputs'][0]
        t = g.tensor(tid)
        ds = dma_descriptors(op['description'])
        require(set(ds) == set(range(8)), f'{key}: expected all eight DMA PE descriptors')
        packets = []
        details = []
        for pe, sides in sorted(ds.items()):
            src, dst = sides['src'], sides['dst']
            require(src['media'] == 'Dram' and dst['media'] == 'Sram', 'Wrong ingress memory tier')
            require(src['address']['chip'] == dst['address']['chip'] == 0, 'Unexpected chip')
            require(math.prod(n for n, _ in src['dims']) * src['dims'][-1][1]
                    == math.prod(n for n, _ in dst['dims']) * dst['dims'][-1][1], 'DMA read/write volume mismatch')
            spans = list(intervals(src['address']['in_chip'], src['dims']))
            packets.extend(spans)
            details.append({'descriptor': pe, 'source': src, 'destination': dst,
                            'bytes': sum(hi-lo for lo, hi in spans),
                            'source_bounds': [min(lo for lo, _ in spans), max(hi for _, hi in spans)]})
        expected = 29491200 if t['element_type'] == 'Float4e2m1' else 3686400
        require(t['size'] == expected, f'{key}: wrong public matrix allocation extent')
        coverage = full_coverage(packets, expected, f'{key} {t["element_type"]} HBM')
        records.append({'operator': key, 'name': op['name'], 'input': tid, 'output': out,
                        'type': t['element_type'], 'coverage': coverage, 'descriptors': details})
    require(sum(r['type'] == 'Float4e2m1' for r in records) == 3, 'Expected up/gate/down FP4 ingress')
    require(sum(r['type'] == 'Float8e4m3' for r in records) == 3, 'Expected up/gate/down FP8 scale ingress')
    return records


def compact_destination(g, ingress, bits, rows, columns):
    # Exact packet correspondence: public l-major source rows -> M03 L-half/slice rows.
    element_bytes = rows * columns * bits // 8
    row_bytes = columns * bits // 8
    for item in ingress['descriptors']:
        src, dst = item['source'], item['destination']
        ss = list(intervals(src['address']['in_chip'], src['dims']))
        dd = list(intervals(0, dst['dims']))
        require(len(ss) == len(dd), 'DMA packet walks differ')
        addr = dst['address']
        require(addr['cluster'] == item['descriptor'] // 4 and addr['pe'] == item['descriptor'] % 4,
                'Unexpected compact ingress ownership')
        for (lo, hi), (dlo, dhi) in zip(ss, dd):
            require(hi-lo == dhi-dlo, 'Compact DMA packet sizes differ')
            physical_slice = addr['pe'] * 64 + addr['dmn'] * 32 + addr['slice'] + dlo // 4194304
            in_slice = dlo % 4194304
            require(in_slice + hi-lo <= element_bytes, 'DMA crosses compact local element bound')
            expect = (addr['cluster'] * 7680 + physical_slice * rows) * row_bytes + in_slice
            require(lo == expect, 'HBM row-to-cluster/slice mapping mismatch')
    return {'cluster_owner': 'L/7680', 'slice_owner': 'local_L/30',
            'local_element_bytes': element_bytes, 'packet_correspondence': True}


def analyze(label):
    require(label == 'e131-joint-four-row-v2', 'Checker frozen for E131 joint-four-row-v2')
    build, paths, recs = provenance(label)
    require(build['variant'] == 'joint-four-row-v2', 'Unexpected source variant')
    source = paths['source'].read_text()
    prior_path = HERE.parent / 'ffn-direct-boundary-recompute/whole-v1/mlp.rs'
    control_path = HERE.parent / 'ffn-native-fused-scale/six-row-v1/mlp.rs'
    prior, control = prior_path.read_text(), control_path.read_text()
    def function(text, name):
        start = text.index('fn ' + name + '(')
        i = text.index('{', start) + 1
        depth = 1
        while depth:
            depth += (text[i] == '{') - (text[i] == '}')
            i += 1
        return text[start:i]
    require(function(source, 'load_joint_compact') == function(prior, 'load_joint_compact'), 'E123 compact weight assembly changed')
    boundary = 'fn broadcast_geglu_scale('
    require(source[source.index(boundary):] == control[control.index(boundary):], 'E128 quantizer/GeGLU/down/public entry suffix changed')
    recs['prior_assembly_source'] = record(prior_path)
    recs['control_suffix_source'] = record(control_path)
    g = Graph(json.loads(paths['graph'].read_text()))
    schedule = json.loads(paths['schedule'].read_text())
    ingress = hbm_ingress(g)
    by_output = {r['output']: r for r in ingress}
    copies, joint_roots = [], {}
    for key, op in g.ops.items():
        if op['kind'] != 'TuExec' or len(op['inputs']) != 2 or len(op['outputs']) != 1:
            continue
        src, dst = op['inputs']
        if src not in by_output or 'FfnRow' not in op['description']:
            continue
        item = by_output[src]
        bits, cols, context = (4, 3840, 'Main') if item['type'] == 'Float4e2m1' else (8, 240, 'Sub')
        count = 30 * cols
        require(op['context'] == context and 've_pass:' not in op['description'] and 'lookup_table:' not in op['description'], 'Assembly copy contains unexpected compute')
        require(offsets(sequence(op['description'], 'fetch_sequencer')) == list(range(count)), 'Raw copy fetch coverage')
        require(offsets(sequence(op['description'], 'commit_projection')) == list(range(count)), 'Raw copy commit coverage')
        require(('commit_in_size: 32B (=64EA)' if bits == 4 else 'commit_in_size: 16B (=16EA)') in op['description'], 'Wrong raw packet width')
        view = g.view(dst)
        require((view['stride_elements'], view['axis_extent'], view['window']) == (count, 2, 1), 'Raw copy projection pitch/extent')
        require(view['start'] in (0, 1), 'Invalid projection')
        require(g.tensor(dst)['size'] == 512 * 2 * count * bits // 8, 'Wrong joint allocation extent')
        root = g.root_output(view['parent'])
        writers = [(k,w) for k,w in g.ops.items() if w['kind'] == 'IndexWrite' and w['inputs'][-1] == op['outputs'][0]]
        require(len(writers) == 1 and writers[0][1]['inputs'][0] == view['parent'] and g.scalar(writers[0][1]['inputs'][1]) == view['start'], 'Assembly writeback mismatch')
        item['compact_destination'] = compact_destination(g, item, bits, 30, cols)
        joint_roots.setdefault(item['type'], []).append((view['start'], root))
        copies.append({'operator': key, 'type': item['type'], 'projection': view['start'], 'source': src,
                       'joint_root': root, 'source_bytes_per_slice': count * bits // 8,
                       'destination_bytes_per_slice': [view['start'] * count * bits // 8, (view['start'] + 1) * count * bits // 8],
                       'logical_read_bytes': item['coverage']['bytes'], 'logical_write_bytes': item['coverage']['bytes']})
    require(len(copies) == 4, 'Expected two FP4 Main and two FP8 Sub assembly copies')
    for dtype, entries in joint_roots.items():
        require(sorted(p for p,_ in entries) == [0,1] and len({r for _,r in entries}) == 1, f'{dtype}: projection assembly gap/alias')
    weight_root = joint_roots['Float4e2m1'][0][1]
    scale_root = joint_roots['Float8e4m3'][0][1]
    native = [(k,o) for k,o in g.ops.items() if o['kind'] == 'TuExec' and 'operation_unit:' in o['description'] and 'FfnProjection' in o['description']]
    require(len(native) == 8, 'Expected eight fused native streams')
    native_records, row_starts, xids, trfids, scaleids, out_roots = [], [], set(), set(), set(), set()
    sym = lambda name,size: {'Symbol': {'symbol': name, 'size': size}}
    pair = lambda a,b: {'Pair': {'left':a,'right':b}}
    for key, op in native:
        d = op['description']
        markers = ['operation_unit:', 'FpMul0: %l * %r [true] %r=LoadVrf { id: VrfId(0) }',
                   'FpMul1: %l * %r [true] %r=LoadVrf { id: VrfId(1) }', 'ReduceAccTree: LReduce(+) [axes=H]', 'transpose_engine:', 'commit_unit:']
        pos = [d.index(m) for m in markers]
        require(pos == sorted(pos) and all(d.count(m) == 1 for m in markers[:4]), 'Fused numerical order/stages changed')
        require('data_type: Float32' in d and 'commit_in_size: 8B (=2EA)' in d, 'Native result must commit two F32 rows')
        require(len(op['inputs']) == 6 and len(op['outputs']) == 1, 'Unexpected native IO')
        weight, trf, xs, bs, table, dst = op['inputs']
        require([g.tensor(t)['buffer_type'] for t in op['inputs']] == ['Sram','Trf','Vrf','Vrf','Sram','Sram'], 'Native operand tiers changed')
        require(g.tensor(weight)['element_type'] == 'Float4e2m1', 'Native input is not compact FP4')
        require(g.tensor(trf)['element_type'] == 'Float8e5m2', 'Activation TRF dtype')
        require(all(g.tensor(t)['element_type'] == 'Float32' for t in (xs,bs,dst)), 'Scale/output type')
        require(g.tensor(table)['element_type'] == 'RawInt16' and f'T{table},' in d, 'Lookup table not bound')
        g.producer(table, 'StoTab')
        wv, ov = g.view(weight), g.view(dst)
        row = wv['start']; tile = wv['window']
        require((row in range(0,28,4) and tile == 4) or (row == 28 and tile == 2), 'Native fixed four-row/tail grouping')
        require((wv['stride_elements'],wv['axis_extent']) == (3840,30), 'Weight tile pitch')
        require((ov['stride_elements'],ov['axis_extent'],ov['window'],ov['start']) == (1,30,tile,row), 'Output tile alignment')
        require(f'FfnProjection_1=2, FfnRow_1={tile}, H_64=60, <ve_fp_way>H_16=4' in d, 'Native block order changed')
        expected_scale_mapping = pair(sym('FfnProjection',2), pair({'Resize': {'inner':sym('FfnRow',30),'resize':tile}}, {'Stride': {'inner':sym('H',3840),'stride':16}}))
        require(g.root_output(wv['parent']) == weight_root, 'Wrong joint weight source')
        require(offsets(sequence(d,'fetch_sequencer')) == [p*115200+r*3840+k for p in range(2) for r in range(tile) for k in range(3840)], 'FP4 projection pitch/fetch coverage')
        require(offsets(sequence(d,'commit_projection')) == [p*30+r for p in range(2) for r in range(tile)], 'P-major output commit coordinates')
        skey, load = g.producer(bs,'TuExec')
        require(load['context'] == 'Sub' and 'to Vrf\n' in load['description'] and len(load['inputs']) == 1, 'Scale VRF load changed')
        sv = g.view(load['inputs'][0])
        require((sv['stride_elements'],sv['axis_extent'],sv['window'],sv['start']) == (240,30,tile,row), 'Scale/weight tile misalignment')
        require(g.root_output(sv['parent']) == scale_root, 'Wrong joint scale source')
        require(offsets(sequence(load['description'],'fetch_sequencer')) == [p*7200+r*240+k for p in range(2) for r in range(tile) for k in range(240)], 'Scale fetch projection pitch')
        require(g.mappings(bs)['InSlice'] == expected_scale_mapping and g.tensor(bs)['size'] == 512*2*tile*240*4, 'Block scale VRF element ordering')
        require(g.mappings(xs)['InSlice'] == {'Padding': {'inner': {'Broadcast': {'size':1}},'padding':8,'kind':'Top'}}, 'Activation scale is not scalar')
        spatial = g.mappings(bs)
        require(all(g.mappings(t)['Cluster'] == spatial['Cluster'] and g.mappings(t)['Partitioning'] == spatial['Partitioning'] for t in (weight,trf,xs,dst)), 'Operand spatial mismatch')
        xa, ba = (int(g.tensor(t)['optionals']['address']) for t in (xs,bs))
        vrf_bytes = 2*tile*240*4
        require(xa+32 <= ba or ba+vrf_bytes <= xa, 'Activation/block scale VRF address overlap')
        require(xa+32 <= 8192 and ba+vrf_bytes <= 8192, 'VRF allocation exceeds physical 8KiB')
        writers = [(k,w) for k,w in g.ops.items() if w['kind']=='IndexWrite' and w['inputs'][-1] == op['outputs'][0]]
        require(len(writers)==1 and writers[0][1]['inputs'][0] == ov['parent'] and g.scalar(writers[0][1]['inputs'][1])==row, 'Native output writeback mismatch')
        out_roots.add(g.root_output(ov['parent'])); row_starts.append(row)
        xids.add(xs);trfids.add(trf);scaleids.add(bs)
        native_records.append({'operator':key,'row_start':row,'row_count':tile,'block_scale_vrf_bytes':vrf_bytes,'block_scale_vrf_address':ba,'x_scale_vrf_address':xa,'block_scale':bs,'scale_load':skey,
                               'weight_source_elements':[[row*3840,(row+tile)*3840],[115200+row*3840,115200+(row+tile)*3840]],
                               'output_bytes_per_slice':[[row*4,(row+tile)*4],[120+row*4,120+(row+tile)*4]],'lookup_table':table})
    require(sorted(row_starts)==list(range(0,28,4))+[28] and len(out_roots)==1, 'Native output coverage incomplete/aliased')
    require(len(xids)==len(trfids)==1 and len(scaleids)==8, 'Register reuse differs from intended graph')
    for tid in xids: g.producer(tid,'StoVrf')
    for tid in trfids: g.producer(tid,'StoTrf')
    partials = [tid for tid,t in g.ts.items() if t['buffer_type']=='Sram' and t['element_type']=='Float32' and 'FfnRow' in json.dumps(t['shape']) and '"H"' in json.dumps(t['shape'])]
    materialized = [tid for tid,t in g.ts.items() if t['buffer_type']=='Sram' and t['element_type']=='Float8e5m2' and 'FfnRow' in json.dumps(t['shape'])]
    require(not partials and not materialized, 'Up/gate F32 block partial or materialized E5 weight remains')
    separate_reduce = [k for k,o in g.ops.items() if 'FfnRow' in o['description'] and 'ReduceAccTree:' in o['description'] and 'operation_unit:' not in o['description']]
    require(not separate_reduce, 'Separate up/gate reduction remains')
    gathers=[]
    for key,op in g.ops.items():
        if op['kind']!='Dma.StoS' or len(op['inputs'])!=2: continue
        src,dst=op['inputs']
        if g.tensor(src)['element_type']!='Float32' or 'TileL' not in json.dumps(g.tensor(src)['shape']): continue
        try: v=g.view(src)
        except ValueError: continue
        require((v['stride_elements'],v['axis_extent'],v['window'])==(30,2,1) and v['start'] in (0,1), 'Gather projection selection')
        _,reshape=g.producer(v['parent'],'LowLevelReshape')
        require(g.root_output(reshape['inputs'][0]) in out_roots, 'Gather reads wrong output')
        require(g.mappings(v['parent'])['InSlice']==pair(sym('FfnProjection',2),{'Modulo': {'inner':sym('TileL',7680),'modulo':30}}), 'Gather parent projection pitch')
        require(g.mappings(dst)['InSlice']=={'Modulo':{'inner':sym('TileL',7680),'modulo':120}}, 'Gather target local row extent')
        require(g.mappings(dst)['Partitioning']==pair({'Stride':{'inner':sym('TileL',7680),'stride':120}}, {'Padding':{'inner':{'Broadcast':{'size':1}},'padding':4,'kind':'Top'}}), 'M03 destination slice owners changed')
        ds=dma_descriptors(op['description']);require(set(ds)==set(range(8)), 'Gather must cover eight PEs')
        src_cells=set(); dst_cells=set(); packets=0
        for pe,sides in ds.items():
            a,b=sides['src'],sides['dst']; aa,ba=a['address'],b['address']
            require(a['media']==b['media']=='Sram' and aa['cluster']==ba['cluster']==pe//4 and aa['pe']==ba['pe']==pe%4, 'Gather crosses L-half owner')
            ss=list(intervals(aa['in_slice'],a['dims']));dd=list(intervals(ba['in_slice'],b['dims']))
            require(len(ss)==len(dd), 'Gather packet walk mismatch')
            for (lo,hi),(dlo,dhi) in zip(ss,dd):
                require(hi-lo==dhi-dlo==120, 'Gather must transfer compact thirty F32 rows')
                s=aa['pe']*64+aa['dmn']*32+aa['slice']+lo//4194304
                t=ba['pe']*64+ba['dmn']*32+ba['slice']+dlo//4194304
                si=lo%4194304+v['start']*120; di=dlo%4194304
                require(si==v['start']*120 and si+120<=(v['start']+1)*120, 'Gather projection source bounds')
                require(t%4==0 and t*30+di//4==s*30, 'Gather source/destination L coordinate mismatch')
                require(di%8==0 and di+120<=480, 'Gather target alignment/bounds')
                for r in range(30):
                    sc=(aa['cluster'],v['start'],s,r);dc=(ba['cluster'],t,di//4+r)
                    require(sc not in src_cells and dc not in dst_cells, 'Duplicate gather element')
                    src_cells.add(sc);dst_cells.add(dc)
                packets+=1
        require(len(src_cells)==len(dst_cells)==15360 and packets==512, 'Incomplete projection gather')
        casts=[(ck,c) for ck,c in g.ops.items() if c['kind']=='TuExec' and c['inputs']==op['outputs'] and 'cast_compaction:' in c['description']]
        require(len(casts)==1 and 'output_element_type: bf16' in casts[0][1]['description'] and 'commit_in_size: 8B (=4EA)' in casts[0][1]['description'], 'Missing final F32-to-BF16 cast')
        require(offsets(sequence(casts[0][1]['description'],'commit_projection'))==list(range(120)), 'Final BF16 output coverage')
        gathers.append({'operator':key,'projection':v['start'],'source_view_operator':v['operator'],
                        'source_projection_offset_bytes':v['start']*120,'source_packet_bytes':120,
                        'packet_count':packets,'logical_bytes':61440,'output_rows':15360,'cast_operator':casts[0][0],
                        'mapping':'source l=7680*c+30*s+r -> target c, slice=4*floor(s/4), r=(s%4)*30+r'})
    require(sorted(x['projection'] for x in gathers)==[0,1], 'Missing up/gate output gather')
    st={int(t['index']):t for t in schedule['tensors']}
    sp={tid:i for i in schedule['instructions'] for tid in i['output_tensors']}
    fused=[i for i in schedule['instructions'] if i['tpe']=='Main' and '.contract_outer::' in i['description'] and 'FpMulAlu::Mul1' in i['description'] and 'FfnProjection' in i['description']]
    require(len(fused)==8,'Final schedule missing fused streams')
    life=[];focus=set()
    for ins in fused:
        _,trf,xs,bs,*_=ins['input_tensors']
        for tid in (trf,xs,bs):
            require(sp[tid]['lifetime']['end']<=ins['lifetime']['begin'],'Register production after consumer')
            require(st[tid]['lifetime']['begin']<=ins['lifetime']['begin'] and st[tid]['lifetime']['end']>=ins['lifetime']['end'],'Register lifetime does not cover use')
        focus.update((xs,bs));life.append({'instruction':ins['index'],'lifetime':ins['lifetime'],'trf':trf,'x_scale':xs,'block_scale':bs})
    for tid in focus:
        t=st[tid]
        for uid,u in st.items():
            if tid==uid or u['buffer_type']!='Vrf':continue
            overlap=max(t['lifetime']['begin'],u['lifetime']['begin'])<min(t['lifetime']['end'],u['lifetime']['end'])
            addresses=max(t['address'],u['address'])<min(t['address']+t['size'],u['address']+u['size'])
            require(not(overlap and addresses),f'Live VRF collision {tid}/{uid}')
    return {'label':label,'passed':True,'inputdigests':recs,'verifiers':[record(Path(__file__).resolve())],
            'source_gates':{'e123_compact_assembly_identical':True,'e128_consumer_quantizer_down_suffix_identical':True},
            'hbm_ingress':ingress,'joint_assembly_copies':copies,'native_streams':sorted(native_records,key=lambda x:x['row_start']),
            'output_gathers':sorted(gathers,key=lambda x:x['projection']),'register_lifetimes':life,
            'gates':{'all_public_weight_and_scale_source_coverage':True,'four_compact_ingress_destination_coordinate_checks':True,
                     'two_fp4_main_and_two_fp8_sub_assembly_copies_retained':True,'seven_four_row_plus_one_two_row_native_streams':True,
                     'dot_then_xscale_then_blockscale_then_h_reduce':True,'block_scale_vrf_order_and_no_live_overlap':True,
                     'upgate_materialized_e5_weights_absent':True,'upgate_f32_block_partials_absent':True,
                     'projection_offsets_0_120_and_m03_Lhalf_gather':True},
            'limits':['Emitted logical descriptors and source/schedule audit; NPU numerical validation remains required.',
                      'IndexAccess/IndexWrite constants are combined with emitted relative sequencers; graph addresses alone do not encode the view offset.',
                      'Down arithmetic is source-identical to E128; down HBM source coverage is checked, while its unchanged destination mapping is outside the new-path coordinate audit.',
                      'Lookup tables remain; extra packed/scale assembly copies are included, and no static duration is interpreted as NPU speedup.']}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    output = HERE / f'graph-verification-{args.label}.json'
    try:
        result = json.loads(json.dumps(analyze(args.label)))
    except Exception as exc:
        result = {'label': args.label, 'passed': False,
                  'verifiers': [record(Path(__file__).resolve())],
                  'error': f'{type(exc).__name__}: {exc}'}
        if not args.check:
            output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), file=sys.stderr)
        return 1
    if args.check:
        require(output.exists() and json.loads(output.read_text()) == result, 'Frozen graph gate is stale')
    else:
        output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'label': args.label, 'passed': result['passed'], 'check': args.check}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
