#!/usr/bin/env python3
"""Check frozen C09 graph contracts and exact original-image identity.

Usage: python3 verify_descriptors.py c09-first-npu-v1 [--check]
Normal mode preserves a new analysis without replacing an existing result.
--check recomputes and compares without writing. No build or NPU execution.
"""
import argparse
import copy
import json
import re
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts' / HERE.name
COMMON_PATH = HERE.parent/'m08-output-pair4-ingress/verify_descriptors.py'
# Execute the preserved helper without creating __pycache__, including --check.
common = types.ModuleType('frozen_m08_descriptor_helpers')
common.__file__ = str(COMMON_PATH)
exec(compile(COMMON_PATH.read_text(), str(COMMON_PATH), 'exec'), common.__dict__)
AuditError, require, only = common.AuditError, common.require, common.only
file_record = common.file_record
ORIGINAL_IMAGE_SHA = 'fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495'
ORIGINAL_SOURCE_SHA = '5692369dd5b8f6ddbeecdfaa9289a9683fae18798b920a71775362f306560fe6'
H, K = 3840, 4096


def constant(g, t):
    op = g.producer[t]
    require(op['kind'] == 'CoreImpCmd' and not op['inputs'] and
            g.tensors[t]['element_type'] == 'RawInt32', 'index must be a constant i32 producer')
    matches = re.findall(r'__scalar_out\[\] = (-?\d+)i32;', op['description'])
    return int(only(matches, 'constant scalar assignment'))


def index_contract(g, op, kind, stride):
    require(op['kind'] == kind and len(op['inputs']) == (2 if kind == 'IndexAccess' else 3),
            'unsupported indexed view/write operation')
    axis = re.search(r'^axis: stride (\d+) x (\d+), window_size: (\d+)$', op['description'], re.M)
    require(axis is not None and tuple(map(int, axis.groups())) == (stride, 120, 60),
            'indexed axis must be a60-row window of120 with expected stride')
    start = constant(g, op['inputs'][1])
    require(start in (0, 60), 'stage index must be0 or60')
    return start


def identify(g):
    stages = []
    for op in g.ops:
        if op['kind'] != 'Dma.DtoS' or len(op['inputs']) != 1:
            continue
        view = g.producer.get(op['inputs'][0])
        if not view or view['kind'] != 'IndexAccess' or not g.from_io(view['inputs'][0], 2):
            continue
        start = index_contract(g, view, 'IndexAccess', K)
        dot = only([d for d in g.ops if d['kind'] == 'TuExec' and d['context'] == 'Main' and
                    'GReduceAdd:' in d['description'] and set(op['outputs']) & set(d['inputs'])], 'stage dot')
        output_view = only([g.producer[t] for t in dot['inputs'] if t in g.producer and
                            g.producer[t]['kind'] == 'IndexAccess'], 'dot mutable output view')
        require(index_contract(g, output_view, 'IndexAccess', 1) == start, 'weight/output stage offsets differ')
        write = only([w for w in g.ops if w['kind'] == 'IndexWrite' and len(w['inputs']) == 3 and
                      w['inputs'][2] in dot['outputs']], 'stage IndexWrite')
        require(index_contract(g, write, 'IndexWrite', 1) == start, 'IndexWrite offset differs from dot view')
        require(write['inputs'][0] == output_view['inputs'][0], 'IndexWrite and dot use different result bases')
        stages.append({'start': start, 'weight': op, 'weight_view': view,
                       'dot': dot, 'output_view': output_view, 'write': write})
    require(len(stages) == 2 and sorted(s['start'] for s in stages) == [0, 60], 'expected exactly two disjoint stages')
    stages.sort(key=lambda s:s['start'])
    first, second = stages
    result_base = first['output_view']['inputs'][0]
    require(g.producer[result_base]['kind'] == 'Scratchpad', 'first stage must initialize common scratchpad')
    require(second['output_view']['inputs'][0] == only(first['write']['outputs'], 'first written buffer'),
            'second output view must follow first IndexWrite')
    gather = only([o for o in g.ops if o['kind'] == 'Dma.StoS' and
                   any(set(second['write']['outputs']) & g.ancestry(t, True) for t in o['inputs'])], 'complete F32 gather')
    activation, scale_dma = g.dma_from_io(0), g.dma_from_io(3)
    trf = only([o for o in g.ops if o['kind'] == 'StoTrf' and g.dependent(o, activation) and
                all(set(o['outputs']) & set(s['dot']['inputs']) for s in stages)], 'shared activation TRF')
    scale = only([o for o in g.ops if o['kind'] == 'TuExec' and o['context'] == 'Main' and
                  'FpMul0:' in o['description'] and g.dependent(o, scale_dma) and
                  any(set(gather['outputs']) & g.ancestry(t, True) for t in o['inputs'])], 'post-gather channel scale')
    return {'stages': stages, 'result_base': result_base, 'activation': activation,
            'trf': trf, 'gather': gather, 'scale_dma': scale_dma, 'scale': scale}


def check_weights(g, selected):
    all_source, allocation_ranges, results = Counter(), [], []
    for stage in selected['stages']:
        op, start = stage['weight'], stage['start']
        tensor = g.tensors[only(op['outputs'], 'weight allocation')]
        require(tensor['element_type'] == 'Float8e4m3' and tensor['buffer_type'] == 'Sram' and
                tensor['size'] == 512*60*256, 'weight stage is not compact60x256 FP8')
        local_shape = tensor['shape']['MappingShape']['inner']['mappings']['InSlice']
        require('Padding' not in json.dumps(local_shape), 'weight DM tile must be compact, not padded')
        allocation = int(tensor['optionals']['address'])
        allocation_ranges.append((allocation, allocation+15360))
        src_tensor = g.tensors[only(op['inputs'], 'weight source view')]
        require(src_tensor['element_type'] == 'Float8e4m3' and src_tensor['buffer_type'] == 'Dram', 'weight source dtype/region')
        base_tensor = g.tensors[stage['weight_view']['inputs'][0]]
        require(base_tensor['size'] == H*K and base_tensor['element_type'] == 'Float8e4m3', 'weight HBM allocation size/dtype')
        offset = start*K  # IndexAccess stride is in elements, FP8 is one byte.
        source, dest, per_slice, sizes = Counter(), set(), Counter(), Counter()
        for _, sd, dd, raw_sa, da, n in common.paired_packets(op):
            sa = raw_sa + offset
            require(sd['memory'] == 'Dram' and dd['memory'] == 'Sram' and sd['base'].get('chip',0) == 0,
                    'weight must be local-chip HBM→DM')
            require(n == 256 and sa%256 == 0 and 0 <= sa <= H*K-n, 'weight source bounds/packet')
            c, sl, local = common.physical(dd, da)
            require(local%256 == 0 and 0 <= local <= 60*256-n, 'weight destination exceeds compact60x256')
            row = local//256
            expected = (1920*c + 120*(sl//16) + start + row)*K + 256*(sl%16)
            require(sa == expected, f'weight stage{start}: source/destination ownership differs')
            key = c, sl, row
            require(key not in dest, 'duplicate stage weight destination')
            dest.add(key); source[sa] += 1; all_source[sa] += 1
            per_slice[c,sl] += n; sizes[n] += 1
        require(len(source) == 512*60 and set(source.values()) == {1} and
                len(dest) == 512*60 and len(per_slice) == 512 and set(per_slice.values()) == {15360},
                'stage source/destination coverage')
        results.append({'status':'PASS','operator':op['name'],'row_start':start,
                        'index_access_source_offset_bytes':offset, 'payload_bytes':512*60*256,
                        'bytes_per_slice':15360,'DM_allocation_local_range':[allocation,allocation+15360],
                        'HBM_min_byte':min(source), 'HBM_max_exclusive':max(source)+256,
                        'descriptor_packet_histogram':dict(sizes)})
    require(set(all_source) == set(range(0,H*K,256)) and set(all_source.values()) == {1},
            'combined stages must cover original HBM exactly once')
    (a,b),(c,d) = allocation_ranges
    require(b <= c or d <= a, 'live stage weight allocations overlap')
    return {'status':'PASS','source_bytes_exact_once':H*K,'stages':results,
            'source_offset_model':'Effective HBM address = descriptor-relative address + constant IndexAccess row*4096. Both stages retain the original120-row group pitch.',
            'allocation_limit':'Reported allocation ranges are graph addresses, not a hardware capacity/bank-contention claim.'}


def check_outputs_and_arithmetic(g, selected, baseline):
    written, stages = set(), []
    base = g.tensors[selected['result_base']]
    require(base['element_type'] == 'Float32' and base['optionals']['address'] == '0', 'common F32 result base')
    for s in selected['stages']:
        dot, start = s['dot'], s['start']
        desc = dot['description']
        require('operation_unit:' in desc and 'lookup_table:' in desc and 'broadcast_kind: Direct, bf16' in desc,
                'FP8 lookup/BF16 dot evidence absent')
        require(desc.count('GReduceAdd:') == 1 and 'GReduceAddF [ratio=16]' in desc and
                '<global_reduce>Qs_256=16<global_reduce>' in desc, 'stage must reduce16 K256 columns in F32')
        require('cast_compaction:' not in desc and 'FpMul0:' not in desc, 'stage has early cast/scale')
        require('commit_in_size: 8B (=2EA)' in desc and 'data_type: Float32' in desc,
                'output commit must trim to two F32 rows')
        projection = re.search(r'commit_projection: (\d+)\+\[None=(\d+):(\d+), None=(\d+):(\d+)\]', desc)
        require(projection is not None, 'unsupported commit projection')
        origin,n0,p0,n1,p1 = map(int,projection.groups())
        local_rows = [origin+i*p0+j*p1 for i in range(n0) for j in range(n1)]
        require(n1 == 2 and p1 == 1 and len(local_rows) == 60 and set(local_rows) == set(range(60)),
                'commit must write60 contiguous rows exactly once')
        for t in s['output_view']['outputs'] + dot['outputs'] + s['write']['outputs']:
            require(g.tensors[t]['element_type'] == 'Float32' and g.tensors[t]['optionals']['address'] == base['optionals']['address'],
                    'indexed output must stay in common F32 allocation')
        for c in range(2):
            for group in range(16):
                for row in local_rows:
                    key = c, group*16, start+row
                    require(key not in written, 'stage output writes overlap')
                    written.add(key)
        stages.append({'operator':dot['name'],'index_access':s['output_view']['name'],
                       'index_write':s['write']['name'],'local_F32_byte_range':[4*start,4*(start+60)],
                       'valid_rows':1920,'commit_packet_bytes':8,'K_reduce_ratio':16})
    require(len(written) == H and {1920*c+120*(sl//16)+row for c,sl,row in written} == set(range(H)),
            'combined valid F32 output rows must cover3840')
    trf = g.tensors[only(selected['trf']['outputs'], 'activation TRF')]
    require(trf['element_type'] == 'Bfloat16', 'TRF activation dtype')
    require(common.normalize_description(selected['trf']) == common.normalize_description(baseline['trf']),
            'activation TRF loading changed from baseline')
    # Reuse the baseline scale/dtype/order checks with each stage. IndexWrite is
    # deliberately validated above rather than treated as transparent arithmetic.
    for s in selected['stages']:
        selected_one = {k:selected[k] for k in ('gather','scale','trf','scale_dma')}
        selected_one.update(weight=s['weight'],dot=s['dot'])
        common.check_arithmetic(g,selected_one,baseline)
    gather = selected['gather']
    path_tensors = set()
    for t in gather['inputs']:
        path_tensors |= g.ancestry(t,True)
    require(all(g.tensors[t]['element_type'] == 'Float32' for t in path_tensors), 'gather handoff has non-F32 data')
    return {'status':'PASS','stages':stages,'disjoint_valid_rows':H,
            'shared_activation_TRF_identical_to_baseline':True,
            'scale_and_first_BF16_identical_to_baseline':True,
            'order':'each FP8 lookup→BF16 dot→F32 K16 reduction→disjoint F32 IndexWrite→canonical F32 gather→F32 channel scale→first BF16',
            'handoff_operators':[o['name'] for o in sorted(g.ops,key=lambda x:x['name'])
                                 if set(o['outputs']) & path_tensors],
            'limit':'Graph coordinates, dtypes, command text and dependencies only; exact DPE register wiring, NPU numerics and actual stall/overlap durations are not established.'}


def audit(raw, baseline):
    g = common.Graph(raw); selected = identify(g)
    checks = {'weights':check_weights(g,selected), 'activation':common.check_activation(selected['activation']),
              'gather':common.check_gather(selected['gather'],'baseline'),
              'scale_channels':common.check_scale_dma(selected['scale_dma']),
              'outputs_and_arithmetic':check_outputs_and_arithmetic(g,selected,baseline)}
    require(checks['gather']['descriptor_packet_histogram'] == {480:32}, 'canonical gather must remain32x480B')
    return {'status':'PASS_GRAPH_CONTRACT_ONLY', 'checks':checks,
            'operators':{k:v['name'] for k,v in selected.items() if isinstance(v,dict)}}


def negative_tests(raw, baseline):
    g = common.Graph(raw); sel = identify(g); first,second = sel['stages']
    mutations = [
        ('second_weight_source_offset',g.producer[second['weight_view']['inputs'][1]],'= 60i32','= 0i32'),
        ('padded_weight_extent',first['weight'],'H_1=60:4096','H_1=120:4096'),
        ('second_output_write_offset',g.producer[second['write']['inputs'][1]],'= 60i32','= 0i32'),
        ('activation_wrong_K',sel['activation'],'anonymbroadcast(2)_1=2:0','anonymbroadcast(2)_1=2:512'),
        ('gather_wrong_pitch',sel['gather'],'H_120=2:480','H_120=2:488'),
        ('K_reduce_ratio',second['dot'],'GReduceAddF [ratio=16]','GReduceAddF [ratio=8]'),
        ('early_BF16_cast',first['dot'],'operation_unit:','cast_compaction:\noutput_element_type: bf16\noperation_unit:'),
    ]
    outcomes = []
    for name, op, old, new in mutations:
        mutant = copy.deepcopy(raw)
        target = only([o for o in mutant['operators'].values() if o['name'] == op['name']], 'negative mutation target')
        require(old in target['description'], 'negative mutation was not applied')
        target['description'] = target['description'].replace(old,new,1)
        try:
            audit(mutant,baseline)
        except AuditError as exc:
            outcomes.append({'case':name,'status':'EXPECTED_REJECTION','reason':str(exc)})
        else:
            raise AuditError('checker accepted injected error: '+name)
    return outcomes


def evaluate(label):
    out = {'label':label,'passed':False,'inputdigests':{},
           'scope':'Original frozen C09 image identity and independent graph descriptor/dataflow contracts; no NPU numerical/performance claim.',
           'address_model':'4MiB virtual slice stride; include cluster/PE/DMN/slice, tensor allocation base and constant IndexAccess provenance. Virtual strides are not physical capacities.'}
    try:
        require(re.fullmatch(r'[a-z0-9-]{1,70}',label) is not None,'invalid label')
        inputs = out['inputdigests']
        build_path = HERE/f'build-{label}-repro.json'
        inputs['build'] = file_record(build_path)
        build = json.loads(build_path.read_text())
        require(build['label'] == label and build['exact_compile_exit_code'] == 0,'compile label/status mismatch')
        require(build['target_kernel'] == 'sliding_attention_output' and build['source_inputs_verified_in_container'] is True,
                'kernel/container-source provenance mismatch')
        artifact, evidence = ARTIFACTS/label, HERE/'evidence'/label
        files = {'graph':evidence/'graph.json','schedule':evidence/'schedule.json','exact_image':artifact/'exact-kernel.bin'}
        for name,path in files.items():
            rec = file_record(path); inputs[name] = rec
            require({k:rec[k] for k in ('bytes','sha256')} == build[name],name+' digest differs from build')
        require(inputs['exact_image']['sha256'] == ORIGINAL_IMAGE_SHA,'exact image differs from original frozen C09')
        for name in ('graph','schedule'):
            rec = file_record(artifact/f'{name}.json'); inputs['artifact_'+name] = rec
            require({k:rec[k] for k in ('bytes','sha256')} == build[name],'artifact '+name+' differs')
        inputs['compiled_source_inputs'] = {}
        for path,expected in build['source_inputs'].items():
            rec = file_record(artifact/'crate'/path)
            require({k:rec[k] for k in ('bytes','sha256')} == expected,'compiled source changed: '+path)
            inputs['compiled_source_inputs'][path] = rec
        inputs['source_overrides'], inputs['frozen_sources'] = {}, {}
        for relative,expected in build['source_plan']['source_overrides'].items():
            for group,path in [('source_overrides',ROOT/expected['path']),('frozen_sources',ROOT/build['frozen_files'][relative])]:
                rec = file_record(path)
                require({k:rec[k] for k in ('bytes','sha256')} == build['source_inputs'][relative],'changed/frozen source differs: '+relative)
                inputs[group][relative] = rec
        recipe_rec = build['source_plan']['recipe']
        inputs['recipe'] = file_record(ROOT/recipe_rec['path'])
        require(inputs['recipe'] == recipe_rec,'recipe changed')
        recipe = json.loads((ROOT/recipe_rec['path']).read_text())
        inputs['original_source'] = file_record(ROOT/recipe['source_snapshot'])
        inputs['original_build'] = file_record(ROOT/recipe['original_build'])
        require(inputs['original_source']['sha256'] == ORIGINAL_SOURCE_SHA ==
                build['source_inputs']['src/device/sliding/projection.rs']['sha256'],'source differs from original frozen C09')
        original = json.loads((ROOT/recipe['original_build']).read_text())
        require(original['exact_image']['sha256'] == ORIGINAL_IMAGE_SHA,'original C09 build image record differs')
        inputs['baseline_graph'] = file_record(common.DEFAULT_BASELINE)
        require(inputs['baseline_graph']['sha256'] == common.BASELINE_GRAPH_SHA,'baseline graph changed')
        inputs['checker'],inputs['shared_checker'] = file_record(Path(__file__).resolve()),file_record(COMMON_PATH)
        baseline_raw = json.loads(common.DEFAULT_BASELINE.read_text())
        baseline = common.Graph(baseline_raw).identify()
        out['baseline'] = common.audit(baseline_raw,'baseline',baseline)
        raw = json.loads(files['graph'].read_text())
        out['candidate'] = audit(raw,baseline)
        out['negative_self_tests'] = negative_tests(raw,baseline)
        out['static_cycles'] = build['static_cycles']
        out['source_inputs_verified'] = out['exact_original_C09_image_identity'] = True
        out['status'],out['passed'] = 'PASS_GRAPH_CONTRACT_ONLY',True
    except (AuditError,OSError,ValueError,KeyError,TypeError) as exc:
        out['status'],out['error'] = 'FAIL_OR_UNSUPPORTED_REQUIRES_REVIEW',str(exc)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--check',action='store_true',help='recompute and compare; never write')
    args = parser.parse_args()
    if re.fullmatch(r'[a-z0-9-]{1,70}',args.label) is None:
        parser.error('label must contain lowercase letters, digits and hyphens only')
    result = json.loads(json.dumps(evaluate(args.label)))
    path = HERE/f'descriptor-analysis-{args.label}.json'
    if args.check or path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError,ValueError) as exc:
            print('FAIL: preserved descriptor analysis unavailable:',exc); return 1
        if existing != result:
            print('FAIL: frozen analysis differs; no file written.',result.get('error',''))
            return 1
    else:
        with path.open('x') as f:
            json.dump(result,f,indent=2,ensure_ascii=False); f.write('\n')
    print('PASS: original C09 image; two compact weight stages, indexed F32 writes, canonical gather and scale order; '+args.label
          if result['passed'] else 'FAIL: '+result.get('error',result['status']))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
