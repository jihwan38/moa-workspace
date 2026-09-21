#!/usr/bin/env python3
"""Verify the frozen one-call experiment and row mapping; no build or NPU work."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function(source, name):
    start = source.index('fn ' + name + '(')
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    manifest = json.loads((HERE / 'build-manifest.json').read_text())
    snapshot = ROOT / manifest['baseline_snapshot']
    original = (HERE / 'projection.original.rs').read_text()
    candidate = (HERE / 'projection.rs').read_text()
    old_call = 'project_value_two_dot(device, unsafe { x.view().reshape() }, &scale_vrf, v_weight, v_weight_scale);'
    new_call = 'project_one_kv_matrix(device, &x_trf, &scale_vrf, v_weight, v_weight_scale);'
    assert original.count(old_call) == 1
    assert candidate == original.replace(old_call, new_call)
    for path, digest in manifest['baseline_files'].items():
        assert sha(snapshot / path) == digest, path
    assert sha(HERE / 'projection.original.rs') == manifest['baseline_files']['src/device/sliding/projection.rs']
    assert sha(HERE / 'projection.rs') == manifest['candidate_files']['src/device/sliding/projection.rs']
    for path, digest in manifest['candidate_files'].items():
        if path != 'src/device/sliding/projection.rs':
            assert digest == manifest['baseline_files'][path], path
    e30 = (ROOT / 'moa-dev/experiments/artifacts/e30-pair-skip-control/src/device/sliding/projection.rs').read_text()
    helper = function(candidate, 'project_one_kv_matrix')
    assert helper == function(original, 'project_one_kv_matrix') == function(e30, 'project_one_kv_matrix')
    assert function(candidate, 'prepare_fp8_split') == function(original, 'prepare_fp8_split')
    seen = set()
    for cluster in range(2):
        for group in range(32):
            for row in range(4):
                for lane in range(8):
                    p = 1024 * cluster + 32 * group + 8 * row + lane
                    before_slice = 8 * group + lane
                    before_local = row
                    after_slice = 8 * group + row
                    after_local = lane
                    n, d = divmod(p, 256)
                    head_slice = ((n % 4) * 8 + d // 32) * 8 + (d // 8) % 4
                    assert n // 4 == cluster and head_slice == after_slice and d % 8 == after_local
                    assert p == 1024 * cluster + 32 * (before_slice // 8) + 8 * before_local + before_slice % 8
                    assert after_slice % 8 < 4  # padded local rows never become output rows
                    assert p not in seen
                    seen.add(p)
    assert seen == set(range(2048))
    baseline = json.loads((HERE / 'baseline.json').read_text())
    candidate_schedule = json.loads((HERE / 'candidate.json').read_text())
    makespan = lambda d: max(i['lifetime']['end'] for i in d['instructions'])
    assert makespan(baseline) == 50364 and makespan(candidate_schedule) == 50728
    tensors = {t['index']: t for t in baseline['tensors']}
    assert tensors['92']['buffer_type'] == 'Sram' and tensors['92']['size'] == 65536
    assert tensors['94']['name'] == 'low_vrf' and tensors['94']['size'] == 65536
    assert all('project_value_two_dot' not in i['description'] for i in candidate_schedule['instructions'])
    assert all(t['name'] != 'low_vrf' for t in candidate_schedule['tensors'])
    result = {
        'status': 'PASS', 'checked_source_files': manifest['source_file_count'],
        'checked_source_and_manifest_files': manifest['checked_files'],
        'only_change': 'project_key_value V call uses existing paired helper',
        'helper_byte_identical_to_e30_and_e37': True,
        'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
        'input_quantizer_unchanged': True, 'verified_row_coordinates': len(seen),
        'baseline_cycles': makespan(baseline), 'candidate_cycles': makespan(candidate_schedule),
        'named_low_dot_sram_and_low_vrf_removed': True,
        'hardware_correctness_checked': False,
        'limits': 'Source/shape/static checks do not prove runtime numerical tolerance or correct resource lifetimes.'
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
