#!/usr/bin/env python3
"""Verify the frozen E133 down v2 emitted graph. --check compares without writing."""
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


def analyze(label):
    require(label == 'e133-down-e5-v2', 'This checker binds the frozen repaired v2 only')
    build, paths, recs = provenance(label)
    require(build['variant'] == 'down-e5-v2', 'Unexpected variant')
    source = paths['source'].read_text()
    original_path = HERE.parent / 'ffn-native-fused-scale/six-row-v1/mlp.rs'
    original = original_path.read_text()
    require(source[:source.index('// E133: quantize')] == original[:original.index('fn project_down_tile(')], 'Up/gate/GeGLU changed')
    require(source[source.index('\nfn finish_down('):] == original[original.index('\nfn finish_down('):], 'Epilogue/public entries/up quantizer changed')
    def reduction(s):
        return s[s.index('    let mut down:', s.index('fn project_down_tile(')):s.index('\nfn finish_down(')]
    require(reduction(source) == reduction(original), 'Existing blockscale/reduction changed')
    g = Graph(json.loads(paths['graph'].read_text()))
    ops = {int(re.search(r'#O(\d+)', o['name'])[1]): o for o in g.ops.values()}
    def op(n, kind=None):
        o = ops[n]
        require(kind is None or o['kind'] == kind, f'O{n}: expected {kind}')
        return o
    def desc(n): return op(n)['description']
    def owned(tid):
        require(g.mappings(tid)['Cluster'] == {'Symbol': {'symbol':'Dummy2','size':2}}, f'T{tid}: cluster ownership lost')
    def same_cluster_dma(n):
        ds = dma_descriptors(desc(n))
        for pair in ds.values():
            require(pair['src']['address']['cluster'] == pair['dst']['address']['cluster'], f'O{n}: cross-cluster source replacement')
        return ds
    ingress = hbm_ingress(g)
    down_loads = []
    for n, scalar_bytes, expected_packet in [(228, .5, 960), (230, 1, 120)]:
        o = op(n, 'Dma.DtoS'); owned(o['outputs'][0])
        item = next(x for x in ingress if x['name'] == o['name'])
        for d in item['descriptors']:
            ss, dd = d['source'], d['destination']; addr = dd['address']
            require(addr['cluster'] == d['descriptor']//4 and addr['pe'] == d['descriptor']%4, 'Down ingress PE placement')
            reads = list(intervals(ss['address']['in_chip'], ss['dims']))
            writes = list(intervals(addr['in_slice'], dd['dims']))
            require(len(reads) == len(writes), 'Down packet correspondence count')
            for (a,b),(x,y) in zip(reads,writes):
                sl = addr['pe']*64 + addr['dmn']*32 + addr['slice'] + x//4194304
                off = x%4194304
                require(b-a == y-x == expected_packet, 'Down ingress packet extent')
                row, local_k = divmod(off, expected_packet)
                require(row < 60 and local_k == 0, 'Down local row boundary')
                h = 60*(sl//4)+row
                if scalar_bytes == .5:
                    expected = h*7680 + addr['cluster']*3840 + (sl%4)*960
                else:
                    expected = h*960 + addr['cluster']*480 + (sl%4)*120
                require(a == expected, 'Down HBM-to-cluster/quarter/row mapping')
        down_loads.append({'operator':n,'public_coverage':item['coverage'],'packet_bytes':expected_packet,
                           'coordinate':'h=60*(slice//4)+row; l=7680*cluster+1920*(slice%4)+k'})

    # Cluster-local amax/scale and the exact E128 residual arithmetic, adapted to TileL.
    for n in (192,193,197,198,199,200,201,202):
        for t in op(n)['inputs'] + op(n)['outputs']: owned(t)
    require(offsets(sequence(desc(192),'fetch_sequencer')) == list(range(7680)), 'Amax does not cover compact L half')
    require('LReduce(max_f) [axes=TileL]' in desc(192) and '1.92593e-34f' in desc(192), 'Amax/floor changed')
    require('2139095040i' in desc(193) and '6.1035156e-5f' in desc(193), 'Exponent mask/2^-14 scale changed')
    for n in (197,199,202):
        require(offsets(sequence(desc(n),'fetch_sequencer')) == list(range(480)), 'Quantizer chunk coverage')
        require('FpFpu: %l / %r' in desc(n) and 'output_element_type: f8_e5' in desc(n), 'E5 division/cast changed')
        require('FpMul0:' not in desc(n) and 'FpMul1:' not in desc(n), 'Unexpected per-term scale')
    require(op(197)['inputs'][1] == op(199)['inputs'][1] == op(202)['inputs'][1], 'Terms use distinct scale operands')
    require('FpFma: %l - %r' in desc(199) and 'FpFma: %l - %r' in desc(202), 'Residual subtraction missing')
    require('FpFma: %l + %r' in desc(200), 'hi+lo F32 sum missing')
    for n in (194,196,219,220,224): same_cluster_dma(n)
    ds = dma_descriptors(desc(196))
    for pair in ds.values():
        full_coverage(intervals(0,pair['src']['dims']), 15360, 'BF16 quantizer partition')
        require(pair['dst']['dims'] == [(16,4194304),(1,960)], 'Quantizer chunk placement changed')
    copies=[]
    for term,n in enumerate((207,212,217)):
        o=op(n,'TuExec');require(o['context']=='Sub','Term copy is not Sub')
        require(offsets(sequence(desc(n),'fetch_sequencer')) == list(range(480)), 'Term source overread')
        require(offsets(sequence(desc(n),'commit_projection')) == list(range(480)), 'Term commit extent')
        require('commit_in_size: 32B (=32EA)' in desc(n) and 'operation_unit:' not in desc(n) and 've_pass:' not in desc(n), 'Term copy changed values')
        v=g.view(o['inputs'][1]);require(v['start']==term and v['stride_elements']==480 and v['axis_extent']==3 and v['window']==1,'Term destination offset')
        require(term*480+480<=1440,'Term destination overrun')
        owned(o['inputs'][0]);owned(o['outputs'][0])
        writers=[w for w in g.ops.values() if w['kind']=='IndexWrite' and w['inputs'][-1]==o['outputs'][0]]
        require(len(writers)==1 and g.scalar(writers[0]['inputs'][1])==term,'Term dependency/index write mismatch')
        copies.append({'operator':n,'term':term,'source_bytes':480,'destination_bounds':[term*480,(term+1)*480]})
    for pair in dma_descriptors(desc(219)).values():
        require(pair['src']['dims']==[(3,480),(16,4194304),(1,480)],'Term gather source coordinates')
        require(pair['dst']['dims']==[(3,7680),(16,480),(1,480)],'Term gather destination coordinates')
        full_coverage(intervals(0,pair['dst']['dims']),23040,'Compact three-term cluster payload')
    for pair in dma_descriptors(desc(220)).values():
        require(pair['src']['address']['pe']==0,'Broadcast source not compact owner')
        reads=list(intervals(0,pair['src']['dims']));writes=list(intervals(0,pair['dst']['dims']))
        require(len(reads)==len(writes),'Quarter seed packet count')
        for (a,b),(x,y) in zip(reads,writes):
            sl=x//4194304; off=x%4194304;term,k=divmod(off,1920)
            require(term<3 and sl%32<4 and a==term*7680+(sl%32)*1920+k and b-a==y-x==960,'Quarter seed coordinate mismatch')
    require('CustomBroadcast' in source and 'ring_size: 32' in source,'Ring32 source missing')
    require(offsets(sequence(desc(222),'fetch_sequencer'))==list(range(5760)),'Term broadcast fetch order')
    require(offsets(sequence(desc(222),'commit_projection'))==list(range(5760)),'Term broadcast commit order')
    require('FfnTerm_1=3, TileL_1=1920' in desc(226) and 'f8_e5' in desc(226),'TRF term/quarter layout')
    for pair in dma_descriptors(desc(224)).values():
        require(all(s==0 for _,s in pair['src']['dims'][:-1]),'Input scale is not scalar broadcast')
        require(pair['src']['dims'][-1]==(1,32),'Input scale packet width')

    native=op(233,'TuExec');w,trf,xs,table=native['inputs']
    require([g.tensor(t)['buffer_type'] for t in native['inputs']]==['Sram','Trf','Vrf','Sram'],'Native operand tiers')
    require(g.tensor(w)['element_type']=='Float4e2m1' and g.tensor(trf)['element_type']=='Float8e5m2','Direct native operand types')
    require(g.tensor(native['outputs'][0])['size']==14745600,'F32 partial extent')
    require(g.tensor(native['outputs'][0])['element_type']=='Float32','Partial narrowed')
    require(offsets(sequence(desc(233),'fetch_sequencer'))==list(range(60*1920)),'Packed fetch element coverage')
    require(offsets(sequence(desc(233),'commit_projection'))==list(range(60*120)),'F32 partial commit coverage')
    require('commit_in_size: 16B (=4EA)' in desc(233),'Native partial packet')
    require(desc(233).index('operation_unit:')<desc(233).index('FpMul0:')<desc(233).index('commit_unit:'),'Native/restore/write order')
    require('lookup_table:' in desc(233) and 'broadcast_kind: Direct, f8_e5' in desc(233),'Direct FP4 lookup missing')
    for tid in native['inputs'][:3]+native['outputs']:owned(tid)
    # No new full E5/E4 down weight materialization; only activation E5 contains TileL.
    decoded=[int(tid) for tid,t in g.ts.items() if t['buffer_type']=='Sram' and t['element_type'] in ('Float8e4m3','Float8e5m2') and t['size']==58982400 and 'TileL' in json.dumps(t['shape'])]
    require(not decoded,'Full decoded down weight DM remains')
    reductions=[]
    for row,n in zip((0,12,24,36,48),(243,253,263,273,283)):
        o=op(n,'TuExec');pv=g.view(o['inputs'][0]);dv=g.view(o['inputs'][2])
        require(pv['start']==dv['start']==row and pv['stride_elements']==120 and dv['stride_elements']==1 and pv['window']==dv['window']==12,'Reduction tile coordinates')
        scale=g.producer(o['inputs'][1],'TuExec')[1];sv=g.view(scale['inputs'][0]);require(sv['start']==row and sv['stride_elements']==120,'Blockscale row alignment')
        require(offsets(sequence(scale['description'],'fetch_sequencer'))==list(range(1440)),'Blockscale read coverage')
        require(offsets(sequence(desc(n),'fetch_sequencer'))==list(range(1440)),'Partial read coverage')
        require(offsets(sequence(desc(n),'commit_projection'))==list(range(12)),'Reduced output coverage')
        markers=['FpMul0:','LReduce(+) [axes=TileL]','GReduceAdd:','commit_unit:'];require([desc(n).index(m) for m in markers]==sorted(desc(n).index(m) for m in markers),'Scale/localK/four-column/store order')
        require('[ratio=4]' in desc(n) and 'commit_in_size: 8B (=2EA)' in desc(n),'Four-column sum/F32pair store changed')
        reductions.append({'operator':n,'rows':[row,row+12],'blockscale_bytes_per_slice':5760})

    schedule=json.loads(paths['schedule'].read_text());st={int(t['index']):t for t in schedule['tensors']}
    ins=schedule['instructions'];sp={t:o for o in ins for t in o['output_tensors']}
    dn=[o for o in ins if '.contract_outer::' in o['description'] and 'TileL' in o['description']]
    require(len(dn)==1,'Expected one native down task')
    require('FfnTerm], m![TileL % 64]' in dn[0]['description'],'Three same-scale terms absent from actual native schedule')
    registers=[t for t in st.values() if t['buffer_type'] in ('Vrf','Trf') and 'TileL' in t['shape']]
    for t in registers:
        require(t['address']+t['size']<=512*(8192 if t['buffer_type']=='Vrf' else 65536),'Register capacity exceeded')
        for u in st.values():
            if u['index']==t['index'] or u['buffer_type']!=t['buffer_type']:continue
            live=max(t['lifetime']['begin'],u['lifetime']['begin'])<min(t['lifetime']['end'],u['lifetime']['end'])
            overlap=max(t['address'],u['address'])<min(t['address']+t['size'],u['address']+u['size'])
            require(not(live and overlap),'Live register address collision')
    for tid in dn[0]['input_tensors'][1:3]:
        t=st[tid];p=sp[tid]
        require(p['lifetime']['end']<=dn[0]['lifetime']['begin'],'Register not ready')
        require(t['lifetime']['begin']<=dn[0]['lifetime']['begin'] and t['lifetime']['end']>=dn[0]['lifetime']['end'],'Register lifetime ends early')
    return {'passed':True,'label':label,'inputdigests':recs,'verifiers':[record(Path(__file__))],
            'original_e128_source':record(original_path),'source_scope':'Up/gate/GeGLU and finish_down/public entries/up quantizer byte-preserved; existing five12row down reduction byte-preserved.',
            'public_matrix_ingress':ingress,'down_coordinate_checks':down_loads,'term_copies':copies,'reductions':reductions,
            'down_native':{'operator':233,'fp4_input':w,'e5_trf':trf,'cluster_scale_vrf':xs,'partial':native['outputs'][0],'f32_partial_bytes':14745600,'schedule_instruction':dn[0]['index'],'lifetime':dn[0]['lifetime']},
            'registers':[{'index':t['index'],'kind':t['buffer_type'],'shape':t['shape'],'address':t['address'],'size':t['size'],'lifetime':t['lifetime']} for t in sorted(registers,key=lambda t:int(t['index']))],
            'gates':{'actual_compact_hbm_full_coverage':True,'down_h_row_k_quarter_coordinate_mapping':True,'independent_cluster_amax_scale':True,'three_exact480B_term_copies':True,'term_compaction_quarter_seed_mapping':True,'native_inputscale_then_f32partial_then_original_reduction':True,'register_lifetimes_no_live_collision':True,'decoded_down_weight_dm_absent':True},
            'limits':['Static source/descriptor/register gate, not NPU accuracy or speed.','No all-BF16 exactness claim; underflow/cancellation requires actual down boundary oracle.','TRF axis/reduction order is bound to source and emitted mapping; binary DPE arithmetic is not independently emulated here.','Inherited E128 up quantizer and untouched boundaries retain prior evidence; this does not independently certify every inherited DMA.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('label');p.add_argument('--check',action='store_true');a=p.parse_args()
    path=HERE/f'graph-verification-{a.label}.json'
    try:result=json.loads(json.dumps(analyze(a.label)))
    except Exception as e:
        result={'passed':False,'label':a.label,'error':repr(e)}
        if not a.check:path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return 1
    if a.check:require(json.loads(path.read_text())==result,'Frozen gate is stale')
    else:
        if path.exists() and json.loads(path.read_text()).get('passed'):raise ValueError('Do not overwrite a successful frozen gate')
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'label':a.label,'check':a.check}));return 0


if __name__=='__main__':raise SystemExit(main())
