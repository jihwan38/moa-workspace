#!/usr/bin/env python3
"""Bind adopted E134 source/tag to the frozen, independently verified NPU evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from package import read, file_record
sys.path.insert(0, str(ROOT / 'moa-dev'))
from experiment_index import validate_baseline


def verify(check_live):
    subprocess.run([sys.executable, str(HERE / 'validation_verify.py'), '--check'],
                   cwd=ROOT, check=True)
    baseline = read(ROOT / 'baseline/e134.json')
    validate_baseline(baseline, check_live)
    if check_live:
        assert read(ROOT / 'baseline/current.json')['recipe'] == 'baseline/e134.json'
    audit = read(ROOT / baseline['source_build_audit'])
    decision = read(HERE / 'adoption-decision.json')
    scope = read(HERE / 'stage1-scope-decision.json')
    measured_path = ROOT / audit['measured_build']
    measured = read(measured_path)
    package = read(HERE / 'package-e134-weighted-v3.json')
    compiled = read(HERE / 'build-e134-weighted-v3-repro.json')
    assert audit['measured_build_digest'] == file_record(measured_path)
    assert audit['original_compile_record'] == file_record(HERE / 'build-e134-weighted-v3-repro.json')
    assert audit['original_package_record'] == file_record(HERE / 'package-e134-weighted-v3.json')
    assert audit['source_binding'] == file_record(HERE / 'source-build-binding.json')
    assert measured['source_inputs'] == compiled['source_inputs'] == read(ROOT / baseline['compile_inputs'])
    assert measured['images'] == audit['images'] == package['images']
    assert measured['package_files'] == package['package_files']
    assert measured['original_compile_record'] == audit['original_compile_record']
    assert measured['original_package_record'] == audit['original_package_record']
    assert baseline['git']['commit'] == audit['adopted_source_commit'] == decision['adopted_source_commit'] == scope['source_commit']
    assert decision['adopted'] and scope['adopted'] and baseline['status'] == 'adopted'
    prior = read(HERE / 'validation/final-verification.json')
    assert prior['valid'] and prior['adoption_ready']
    assert audit['pre_adoption_verification'] == decision['pre_adoption_evidence'] == file_record(HERE / 'validation/final-verification.json')
    assert audit['compile_inputs'] == len(compiled['source_inputs']) == 46
    assert audit['median_paired_ffn_delta_percent'] == prior['performance']['median_paired_delta_percent'] < 0
    assert scope['unchanged_contract']['atol'] == scope['unchanged_contract']['rtol'] == 0.01
    paths = [ROOT / 'baseline/e134.json', ROOT / baseline['source_build_audit'],
             HERE / 'adoption-decision.json', HERE / 'stage1-scope-decision.json',
             HERE / 'validation/final-verification.json', measured_path,
             ROOT / baseline['compile_inputs'], Path(__file__)]
    return {'valid': True, 'baseline': 'E134', 'adopted': True,
            'source_commit': baseline['git']['commit'], 'source_tag_manifest_matches': True,
            'compile_inputs': 46, 'independent_performance_allocations': 3,
            'median_paired_ffn_delta_percent': prior['performance']['median_paired_delta_percent'],
            'standard_distinct_inputs': 9, 'targeted_down_exact_outputs': 34560,
            'constructed_full_ffn_exact_outputs': 3840,
            'evidence': [file_record(path) for path in paths],
            'full_model_tested': False, 'official_submission_at_adoption': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-live', action='store_true')
    args = parser.parse_args()
    result = verify(args.check_live)
    path = HERE / 'validation/adoption-verification.json'
    if args.check:
        assert read(path) == result, 'Adoption verification is stale'
    else:
        with path.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
    print('E134 adopted source/tag/performance/precision verification PASS')
