#!/usr/bin/env python3
"""Read preserved O images/descriptors; enumerate bytes. No compile/NPU/mutations.

Run: python3 docs/research/o-weight-ingress-20260927/kernel-audit.py --check
Candidate addresses are a mathematical proposed mapping, NOT emitted descriptors.
"""
import hashlib
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'moa-dev/experiments'
EXACT = EXP / 'artifacts/output-bf16-activation-e47'
PROFILE = EXP / 'candidates/e104-qkv-output-profile'
SCHEDULE = PROFILE / 'evidence/profile-v1/sliding_attention_output.json'
IMAGE_SHA = '359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b'
SLICE_ADDRESS_STRIDE = 4194304  # descriptor address encoding, NOT physical slice capacity
WEIGHT_BYTES = 3840 * 4096

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def descriptors(op):
    result = {'src': [], 'dst': []}
    for line in op['description'].splitlines():
        m = re.match(r'(src|dst)(\d+): \[(.*?)\], (Dram|Sram)\((.*)\)', line)
        if not m:
            continue
        dims = []
        for item in m[3].split(', '):
            label, size, stride = re.match(r'(.*?)=(\d+):(\d+)$', item).groups()
            dims.append({'axis': label, 'count': int(size), 'byte_stride': int(stride)})
        coords = {k: int(v, 0) for k, v in re.findall(r'(\w+)=(0x[0-9A-Fa-f]+|\d+)', m[5])}
        result[m[1]].append({'id': int(m[2]), 'memory': m[4], 'dims': dims, 'base': coords})
    return result

def enumerate_packets(desc):
    dims = desc['dims']
    assert dims[-1]['count'] == 1
    packet = dims[-1]['byte_stride']
    base = desc['base'].get('in_chip', desc['base'].get('in_slice', 0))
    for indices in itertools.product(*(range(d['count']) for d in dims[:-1])):
        yield base + sum(i*d['byte_stride'] for i, d in zip(indices, dims[:-1])), packet

def packet_summary(ds):
    packets = [x for d in ds for x in enumerate_packets(d)]
    return {'descriptors': len(ds), 'packets': len(packets),
            'logical_payload_bytes': sum(n for _, n in packets),
            'packet_size_histogram': dict(Counter(n for _, n in packets)),
            'first_descriptor_first_32_packet_offsets': [a for a, _ in list(enumerate_packets(ds[0]))[:32]]}

graph = json.loads((EXACT / 'candidate-graph.json').read_text())
ops = {v['name']: v for v in graph['operators'].values()}
weight = descriptors(ops['DmaCommand#O4'])
gather = descriptors(ops['DmaCommand#O11'])
activation = descriptors(ops['DmaCommand#O1'])
assert sha(EXACT / 'candidate.bin') == IMAGE_SHA
profile_build = json.loads((PROFILE/'build-profile-v1.json').read_text())
assert profile_build['images']['sliding_attention_output']['sha256'] == IMAGE_SHA
old_schedule = json.loads((EXACT / 'candidate.json').read_text())
new_schedule = json.loads(SCHEDULE.read_text())
without_source = lambda x: {k: v for k, v in x.items() if k != 'description'}
assert list(map(without_source, old_schedule['instructions'])) == list(map(without_source, new_schedule['instructions']))
assert old_schedule['tensors'] == new_schedule['tensors']
source_packets = [x for d in weight['src'] for x in enumerate_packets(d)]
assert all(n == 256 and a % 256 == 0 for a, n in source_packets)
assert sorted(a for a, _ in source_packets) == list(range(0, WEIGHT_BYTES, 256))
assert packet_summary(weight['dst'])['logical_payload_bytes'] == WEIGHT_BYTES
assert packet_summary(gather['src'])['logical_payload_bytes'] == 3840*4

# Decode physical destinations from the shape and virtual slice address stride.
# 1 PE = one DMN pair. Descriptor dst bases enumerate cluster 0/1, PE 0..3.
per_slice = Counter()
first = []
for s, d in zip(weight['src'], weight['dst']):
    for (sa, n), (da, dn) in zip(enumerate_packets(s), enumerate_packets(d)):
        assert n == dn == 256
        virtual_slice, local = divmod(da, SLICE_ADDRESS_STRIDE)
        assert local < 120*256
        dmn_in_pair, slice_in_dmn = divmod(virtual_slice, 32)
        key = (d['base']['cluster'], 2*d['base']['pe']+dmn_in_pair, slice_in_dmn)
        per_slice[key] += n
        if s['id'] == 0 and len(first) < 32:
            first.append({'hbm_relative_offset': sa, 'hbm_relative_bit8': (sa >> 8)&1,
                          'dmn_in_pair': dmn_in_pair, 'slice_in_dmn': slice_in_dmn,
                          'local_byte_offset': local})
assert len(per_slice) == 512 and set(per_slice.values()) == {30720}

# Proposed four-way row-pair interleave. Identical Q-column placement and 120 rows/slice.
# h=1920*c+480*g+8*t+2*r+j, slice=64*g+16*r+q.
candidate = []
candidate_gather_dest = []
candidate_gather_first = []
for c, g, t, j, b, q, d in itertools.product(range(2), range(4), range(60), range(2), range(2), range(16), range(2)):
    r = 2*d+b
    h = 1920*c + 480*g + 8*t + 2*r+j
    a = 4096*h+256*q
    candidate.append(a)
assert sorted(candidate) == list(range(0, WEIGHT_BYTES, 256))
for c, g, t, b, d in itertools.product(range(2), range(4), range(60), range(2), range(2)):
    r=2*d+b
    src=8*t
    dst=32*t+8*r
    assert src%8 == dst%8 == 0
    candidate_gather_dest.extend((1920*c+480*g)*4+dst+u for u in range(8))
    if c==g==0 and len(candidate_gather_first)<16:
        candidate_gather_first.append({'row_group_r': r, 'time_pair_t': t, 'src_local_byte': src, 'dst_local_byte': dst, 'packet_bytes': 8})
assert sorted(candidate_gather_dest) == list(range(3840*4))

analysis = json.loads((PROFILE/'profile-analysis.json').read_text())['kernels']['sliding_attention_output']
sdk_files = ['src/context.rs', 'src/tensor/memory.rs', 'src/tensor/memory/dma_layout.rs', 'src/engine/fetch_adapter.rs', 'src/engine/collect.rs']
out = {
  'scope': 'Read-only descriptor enumeration; matching E47 image equals E104/E122 O image. No new compile/NPU execution.',
  'source_manifest': [record(p) for p in [EXACT/'candidate.bin', EXACT/'candidate-graph.json', EXACT/'candidate.json', SCHEDULE, PROFILE/'build-profile-v1.json', PROFILE/'profile-analysis.json', ROOT/'furiosa-opt-gemma4-12B/src/device/sliding/projection.rs']],
  'sdk_manifest': [dict(record(Path('/tmp/moa-o-sdk-std')/p), original_container_path='/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/'+p) for p in sdk_files],
  'identity': {'current_o_image_sha256': IMAGE_SHA, 'schedule_non_source_fields_equal': True, 'schedule_tensors_equal': True, 'static_makespan': 27239},
  'weight': {'operator': ops['DmaCommand#O4'], 'descriptors': weight,
      'source_summary': packet_summary(weight['src']), 'destination_summary': packet_summary(weight['dst']),
      'hbm_byte_coverage_exact': True, 'relative_256B_alignment': True, 'source_max_end': max(a+n for a,n in source_packets),
      'physical_slices': len(per_slice), 'bytes_per_physical_slice': 30720,
      'first_pe_packet_coordinates': first,
      'hbm_relative_stack_bit_counts': dict(Counter((a>>8)&1 for a,n in source_packets)),
      'dm_allocation_address': graph['tensors']['9']['optionals']['address'],
      'dm_bank_bytes_per_256B_packet': {str(b): 16 for b in range(16)},
      'scope_note': 'Bank formula is documented 8B bank rows, 16 banks. Uniform address coverage does not measure queue conflicts. HBM allocation absolute base/channel mapping not established.'},
  'activation': {'descriptors': activation, 'source_summary': packet_summary(activation['src']), 'logical_unique_input_bytes': 8192,
      'note': 'Zero source strides express repeats; 262144B is descriptor logical payload, not measured HBM bus bytes nor proof of hardware multicast.'},
  'gather': {'operator': ops['DmaCommand#O11'], 'descriptors': gather, 'source_summary': packet_summary(gather['src']), 'dtype': 'Float32'},
  'npu_trace': {k:v['all_run_clusters'] for k,v in analysis['window_summary'].items()},
  'static_important_nodes': [v for v in new_schedule['instructions'] if v['index'] in ['14','16','26','54']],
  'candidate_row_pair_fourway': {
      'classification': 'Reopened row-interleave family; mathematical layout only, not compiler output',
      'formula': 'h=1920*c+480*g+8*t+2*r+j; slice=64*g+16*r+q; K=256*q+u',
      'ranges': {'c':2,'g':4,'t':60,'r':4,'j':2,'q':16,'u':256},
      'weight_bytes_unchanged': WEIGHT_BYTES, 'source_coverage_exact': True, 'local_weight_bytes_per_slice':30720,
      'same_order_assumption_expected_source_strides': {'t':32768,'j':4096,'b':8192,'q':256,'d':16384},
      'first_pe_first_32_packet_offsets_under_same_loop_assumption': candidate[:32],
      'pair_dmn_hbm_jump_baseline':983040, 'pair_dmn_hbm_jump_candidate':16384,
      'gather_source_byte': '8*t+4*j', 'gather_destination_byte':'32*t+8*r+4*j',
      'gather_math_min_packet_bytes':8, 'gather_math_packets':1920, 'gather_payload_bytes':15360,
      'gather_coverage_exact':True, 'gather_first_pairs': candidate_gather_first,
      'cost_warning':'32x480B baseline packets become a mathematical 1920x8B grid. One to_dm stage does not imply equal service cost; actual lowering and full gather must be measured.'},
  'conditional_two_stage_model': {'W':26736,'D':12014,'serial_W_plus_D':38750,'ideal_two_stage_W_plus_D_over_2':32743,'ideal_saved_D_over_2':6007,
      'note':'Assumes equal halves, independent resources, no extra lookup/setup/gather, unchanged epilogue. Not a measured speedup or upper bound on all possible architectures.'}
}
if '--check' in sys.argv:
    expected = json.loads(Path(__file__).with_suffix('.json').read_text())
    assert json.loads(json.dumps(out)) == expected, 'kernel audit differs from preserved evidence'
    print('PASS: image/schedule identity, weight descriptors, proposed coordinates and gather geometry')
else:
    print(json.dumps(out, indent=2, ensure_ascii=False))
