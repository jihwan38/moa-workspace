#!/usr/bin/env python3
"""Frozen E135 direct-six-row descriptor gate; --check never writes.

Graph tensor/operator IDs and schedule IDs are checked in their own namespaces.
This gate proves emitted layout/provenance contracts, not NPU arithmetic or speed.
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HELPER = HERE.parent / 'ffn-native-down-weighted-residual/verify_graph.py'
spec = importlib.util.spec_from_file_location('e134_descriptor_helpers', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.HERE, h.ROOT = HERE, ROOT
h.ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts' / HERE.name
require, record = h.require, h.record


def analyze(label):
    require(label == 'e135-direct-six-row-v1', 'Only frozen E135 v1 is supported')
    build, paths, recs = h.provenance(label)
    require(build['variant'] == 'direct-six-row-v1', 'Wrong frozen source variant')
    source = paths['source'].read_text()
    control_path = HERE.parent / 'ffn-native-down-weighted-residual/weighted-v3/mlp.rs'
    control = control_path.read_text()
    suffix = '\nfn cast_gathered_rows('
    require(source[source.index(suffix):] == control[control.index(suffix):],
            'GeGLU, weighted down, quantizer or public entry suffix changed')
    for fragment in (
        '.fetch_table_lookup::<f8e5m2>()',
        '.contract_packet::<m![H / 16 % 4]>()',
        '.contract_time::<m![FfnRow = 6, H / 64]>()',
        'FpMulAlu::Mul0), x_scale_vrf', 'FpMulAlu::Mul1), &scale_vrf',
        '.vector_intra_slice_reduce::<H, m![FfnRow = 6]',
    ):
        require(fragment in source[:source.index(suffix)], f'Missing source arithmetic contract: {fragment}')
    g = h.Graph(json.loads(paths['graph'].read_text()))
    ingress = h.hbm_ingress(g)
    owned_ingress = {}
    for item in ingress:
        mapping = g.mappings(item['output'])
        if mapping['Partitioning'] != {'Symbol': {'symbol': 'FfnSlice', 'size': 256}}:
            continue
        require(mapping['Cluster'] == {'Symbol': {'symbol': 'Dummy2', 'size': 2}}, 'Up/gate cluster ownership')
        rowbytes = 1920 if item['type'] == 'Float4e2m1' else 240
        for d in item['descriptors']:
            src, dst = d['source'], d['destination']
            a = dst['address']
            require(a['cluster'] == d['descriptor']//4 and a['pe'] == d['descriptor']%4,
                    'Up/gate ingress PE placement')
            reads = list(h.intervals(src['address']['in_chip'], src['dims']))
            writes = list(h.intervals(a['in_slice'], dst['dims']))
            require(len(reads) == len(writes), 'Ingress packet correspondence')
            for (lo, hi), (outlo, outhi) in zip(reads, writes):
                sl = a['pe']*64 + a['dmn']*32 + a['slice'] + outlo//4194304
                off = outlo % 4194304
                require(0 <= sl < 256 and hi-lo == outhi-outlo, 'Ingress packet extent')
                require(0 <= off and outhi-outlo+off <= 30*rowbytes, 'Compact slice bounds')
                require(lo == (a['cluster']*7680+sl*30)*rowbytes+off,
                        'HBM to compact cluster/slice row mapping')
        owned_ingress[item['output']] = item
    require(len(owned_ingress) == 4, 'Expected independent up/gate weight and scale ingress')
    streams = []
    outputs, weights, scales = {}, {}, {}
    native_ops = [(k,o) for k,o in g.ops.items() if o['kind']=='TuExec'
                  and o['context']=='Main' and 'lookup_table:' in o['description']
                  and 'FfnRow_1=6' in o['description'] and 'LReduce(+) [axes=H]' in o['description']]
    require(len(native_ops) == 10, 'Expected five direct fused streams per projection')
    for key, op in native_ops:
        ins, desc = op['inputs'], op['description']
        require(len(ins) == 6, 'Expected packed weight/TRF/two VRF/table/output inputs')
        w, trf, xs, bs, table, dest = ins
        require(g.tensor(w)['element_type']=='Float4e2m1', 'Weight is no longer direct FP4')
        require(g.tensor(trf)['buffer_type']=='Trf' and g.tensor(trf)['element_type']=='Float8e5m2', 'E5 activation TRF')
        require(g.tensor(xs)['buffer_type']==g.tensor(bs)['buffer_type']=='Vrf', 'Scale VRF bindings')
        require(g.tensor(xs)['size']==512*32 and g.tensor(bs)['size']==512*5760, 'Scale payload sizes')
        require(g.tensor(trf)['size']==512*3*3840, 'Three-term activation TRF payload')
        require('FfnTerm_1=3, H_1=3840' in g.producer(trf, 'StoTrf')[1]['description'], 'TRF term/H mapping')
        require(g.producer(table, 'StoTab')[1]['outputs']==[table], 'Lookup table producer')
        require(f'T{table},' in desc, 'Lookup table dependency')
        wv, ov = g.view(w), g.view(dest)
        bprod = g.producer(bs, 'TuExec')[1]
        require(bprod['context']=='Sub' and 'to Vrf' in bprod['description'], 'Block scale conversion stream')
        sv = g.view(bprod['inputs'][0])
        for view, stride in ((wv,3840),(sv,240),(ov,1)):
            require((view['stride_elements'],view['axis_extent'],view['window'])==(stride,30,6), 'Six-row view contract')
        require(wv['start']==sv['start']==ov['start'] and wv['start'] in (0,6,12,18,24), 'Weight/scale/output tile alignment')
        require(wv['parent'] in owned_ingress and sv['parent'] in owned_ingress, 'Direct ingress to consumer')
        require(h.sequence(desc,'fetch_sequencer')==(0,[(6,3840),(3840,1)]), 'Weight fetch must be six compact rows')
        require(h.sequence(bprod['description'],'fetch_sequencer')==(0,[(6,240),(240,1)]), 'Scale fetch must match six rows')
        require('commit_in_size: 8B (=2EA)' in desc and h.offsets(h.sequence(desc,'commit_projection'))==list(range(6)), 'Six F32 rows via 8B pairs')
        require(desc.index('FpMul0:') < desc.index('FpMul1:') < desc.index('LReduce(+) [axes=H]'), 'Native -> input scale -> block scale -> H reduction order')
        require('cast_compaction:' not in desc and g.tensor(op['outputs'][0])['element_type']=='Float32', 'Keep F32 until gather')
        root = g.root_output(ov['parent'])
        writers=[o for o in g.ops.values() if o['kind']=='IndexWrite' and op['outputs'][0] in o['inputs']]
        require(len(writers)==1 and g.scalar(writers[0]['inputs'][1])==ov['start'], 'Output tile write-back index')
        outputs.setdefault(root,[]).append(ov['start'])
        weights.setdefault(wv['parent'],[]).append(wv['start'])
        scales.setdefault(sv['parent'],[]).append(sv['start'])
        streams.append({'graph_key':key,'name':op['name'],'weight_view':wv,'scale_view':sv,'output_view':ov,
                        'output_root':root,'trf':trf,'input_scale_vrf':xs,'block_scale_vrf':bs,'lookup_table':table,
                        'weight_byte_range':[wv['start']*1920,(wv['start']+6)*1920],
                        'scale_byte_range':[sv['start']*240,(sv['start']+6)*240],
                        'output_byte_range':[ov['start']*4,(ov['start']+6)*4]})
    for groups in (outputs,weights,scales):
        require(len(groups)==2 and all(sorted(v)==[0,6,12,18,24] for v in groups.values()), 'Complete disjoint five-tile coverage for both projections')
    # Gather: follow actual producer lineage, then enumerate physical packet coordinates.
    def root(tid):
        _,o=g.producer(tid)
        if o['kind'] in ('LowLevelReshape','ClusterSync'): return root(o['inputs'][0])
        return g.root_output(tid)
    gathers=[]
    for key,op in g.ops.items():
        if op['kind']!='Dma.StoS':continue
        try:r=root(op['inputs'][0])
        except (ValueError,KeyError):continue
        if r not in outputs:continue
        pairs=h.dma_descriptors(op['description']); count=0
        for sides in pairs.values():
            a,b=sides['src'],sides['dst'];sa,da=a['address'],b['address']
            require(sa['cluster']==da['cluster'] and sa['pe']==da['pe'], 'Gather changed ownership')
            aa=list(h.intervals(sa['in_slice'],a['dims']));bb=list(h.intervals(da['in_slice'],b['dims']))
            require(len(aa)==len(bb), 'Gather packet count')
            for (lo,hi),(dl,dh) in zip(aa,bb):
                ss=sa['pe']*64+sa['dmn']*32+sa['slice']+lo//4194304
                ds=da['pe']*64+da['dmn']*32+da['slice']+dl//4194304
                require(hi-lo==dh-dl==120 and lo%4194304==0, 'Gather exact 30-F32 source packet')
                require(ds==ss//4*4 and dl%4194304==(ss%4)*120, 'Gather four adjacent slices to canonical 120 rows')
                count+=1
        require(count==512, 'Gather all 512 slice packets')
        consumers=[o for o in g.ops.values() if op['outputs'][0] in o['inputs'] and o['kind']=='TuExec']
        require(len(consumers)==1 and 'output_element_type: bf16' in consumers[0]['description'], 'BF16 cast follows F32 gather')
        gathers.append({'operator':key,'name':op['name'],'root':r,'packets':count,'packet_bytes':120})
    require(len(gathers)==2, 'Both projection gathers found')
    schedule=json.loads(paths['schedule'].read_text())
    ts={int(t['index']):t for t in schedule['tensors']}
    instructions=[i for i in schedule['instructions'] if i['tpe']=='Main'
                  and '.fetch_table_lookup::<f8e5m2>()' in i['description'] and 'FfnRow = 6' in i['description']]
    require(len(instructions)==10, 'Schedule contains ten fused direct streams')
    resource_checks=[]
    for ins in instructions:
        regs=[ts[x] for x in ins['input_tensors'] if x in ts and ts[x]['buffer_type'] in ('Vrf','Trf')]
        require(sorted(t['buffer_type'] for t in regs)==['Trf','Vrf','Vrf'], 'Schedule register bindings')
        for t in regs:
            require(t['lifetime']['begin']<=ins['lifetime']['begin'] and t['lifetime']['end']>=ins['lifetime']['end'], 'Register live throughout native consumer')
        vrfs=sorted((t for t in regs if t['buffer_type']=='Vrf'),key=lambda t:t['address'])
        require(vrfs[0]['address']+vrfs[0]['size']<=vrfs[1]['address'], 'Input/block scale VRF overlap')
        require(max(t['address']+t['size'] for t in vrfs)<=8192*512, 'VRF address extent')
        resource_checks.append({'schedule_index':ins['index'],'lifetime':ins['lifetime'],
                                'registers':regs,'namespace':'schedule-only; no join on graph IDs'})
    require(sum(o['kind']=='StoTab' for o in g.ops.values())==11, 'Ten up/gate lookup tables plus inherited down')
    return {'passed':True,'label':label,'inputdigests':recs,'verifiers':[record(Path(__file__)),record(HELPER)],
            'source_control':record(control_path),'unchanged_suffix':'cast_gathered_rows through EOF, including E134 weighted down',
            'public_hbm_ingress':ingress,'upgate_streams':streams,'f32_gathers':gathers,'schedule_register_checks':resource_checks,
            'counts':{'upgate_native_lookup_streams':10,'all_lookup_tables':11,'six_row_tiles_per_projection':5},
            'scope':'Actual emitted bounds/index/coverage/scale order and register bindings verified. No NPU arithmetic, hardware bank-conflict, hidden SFR read-index or speed claim; whole/probe validation remains required.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('label');ap.add_argument('--check',action='store_true');a=ap.parse_args()
    path=HERE/f'graph-verification-{a.label}.json'
    try:result=json.loads(json.dumps(analyze(a.label)))
    except Exception as exc:
        result={'passed':False,'label':a.label,'error':str(exc)}
        if not a.check:path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return 1
    if a.check:require(json.loads(path.read_text())==result,'Frozen graph gate is stale')
    else:
        require(not path.exists() or not json.loads(path.read_text()).get('passed'),'Do not overwrite a successful frozen gate')
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'label':a.label,'check':a.check}));return 0


if __name__=='__main__':raise SystemExit(main())
