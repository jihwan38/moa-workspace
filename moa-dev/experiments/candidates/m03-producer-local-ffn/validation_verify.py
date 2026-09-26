#!/usr/bin/env python3
"""Verify the M03 adoption evidence without recompiling, submitting or editing runtime."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tarfile
from validation_results import records

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'moa-dev'))
from extra_seeds import parse_log as parse_extra


def digest(content):
    return {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def read(path):
    return json.loads((ROOT / path).read_text())


def verify():
    reports = records()
    for name, data in reports.items():
        assert read(name) == data
    candidate = reports['docs/experiments/e122-m03-adoption-validation.json']
    spec = importlib.util.spec_from_file_location('validation_index', ROOT / 'moa-dev/experiment_index.py')
    index = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(index)
    for name in reports:
        for kernel in index.KERNELS:
            index.measurement(name, kernel)
    index.check_comparable('docs/experiments/e122-m03-adoption-validation.json', 'docs/experiments/e122-m03-e104-paired-control.json')
    base = json.loads((HERE / 'build-m03-native-local-v1-repro.json').read_text())
    extra = json.loads((HERE / 'build-m03-adoption-extra6-20260927-repro.json').read_text())
    assert base['images'] == extra['images']
    assert {k for k in base['source_inputs'] if base['source_inputs'][k] != extra['source_inputs'][k]} == {'src/bin/test_kernels.rs'}
    for name, built in [('m03-native-local-v1', base), ('m03-adoption-extra6-20260927', extra)]:
        artifacts = ROOT / 'moa-dev/experiments/artifacts/m03-producer-local-ffn' / name
        for path, expected in built['source_inputs'].items():
            assert digest((artifacts / 'crate' / path).read_bytes()) == expected
        for path, frozen in built['frozen_files'].items():
            expected = built['source_inputs'].get(path, built['logs'].get(path))
            assert digest((ROOT / frozen).read_bytes()) == expected
        host = (artifacts / 'package/test_runtime').read_bytes()
        for kernel, expected in built['images'].items():
            image = (artifacts / 'kernels' / f'furiosa_opt_gemma4::ops::{kernel}.bin').read_bytes()
            assert digest(image) == {k: expected[k] for k in ('bytes', 'sha256')}
            assert host.count(image) == 1
    additional = read('docs/experiments/e122-m03-extra-correctness.json')
    assert additional['valid'] and additional['results']['output_comparisons_passed'] == 30
    raw = (ROOT / additional['log']).read_bytes()
    assert digest(raw)['sha256'] == additional['log_sha256']
    parts = dict(re.findall(r'^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$', raw.decode(), re.M | re.S))
    parsed = dict(parse_extra(parts['candidate_0']), valid=True)
    assert parsed == additional['results']
    tar_path = ROOT / f"moa-dev/experiments/artifacts/m03-producer-local-ffn/job-{additional['job']['id']}.tar"
    with tarfile.open(tar_path) as archive:
        actual = {m.name: digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
    assert actual == additional['package_files']
    for name in ('test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh'):
        assert actual['candidate_0_' + name] == extra['package_files'][name]
    probe = read('docs/experiments/e122-m03-probe-correctness.json')
    assert probe['valid'] and probe['complete']
    built = read(probe['build'])
    original = (HERE / 'native-local-v1/mlp.rs').read_bytes()
    assert digest(original) == base['source_inputs']['src/device/shared/mlp.rs'] == built['production_mlp_prefix']
    assert built['production_body_unchanged']
    label = Path(probe['build']).name.removeprefix('build-').removesuffix('.json')
    frozen = HERE / 'evidence' / label
    assert (frozen / 'src/device/mlp.rs').read_bytes() == original + (HERE / 'validation/probe_module.rs').read_bytes()
    for path, expected in built['source_inputs'].items():
        assert digest((frozen / path).read_bytes()) == expected
    for path, expected in built['logs'].items():
        assert digest((frozen / path).read_bytes()) == expected
    raw = (ROOT / probe['log']).read_bytes()
    assert digest(raw)['sha256'] == probe['log_sha256']
    details = re.findall(r'^M03_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)$', raw.decode(), re.M)
    assert details == [tuple(d) for d in probe['details']]
    expected = {(str(c), p, str(n), '0') for c in range(7) for p, n in [('up',15360),('gate',15360),('down',3840)]}
    assert len(details) == 21 and set(details) == expected
    assert raw.count(b'M03_PROBE_COMPLETE oracle_only=false cases=7 checked=241920 mismatches=0') == 1
    tar_path = ROOT / f"moa-dev/experiments/artifacts/m03-producer-local-ffn/validation-job-{probe['job']['id']}.tar"
    with tarfile.open(tar_path) as archive:
        actual = {m.name: digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
    assert actual == probe['package_files']
    assert {k: v for k, v in actual.items() if k != 'job.yaml'} == built['package_files']
    return {'valid': True, 'adopted': False, 'performance_report': 'docs/experiments/e122-m03-adoption-validation.json',
            'control_report': 'docs/experiments/e122-m03-e104-paired-control.json',
            'extra_report': 'docs/experiments/e122-m03-extra-correctness.json',
            'probe_report': 'docs/experiments/e122-m03-probe-correctness.json',
            'basic_arena_allocations': 6, 'same_m03_stage_payload_across_all_jobs': True,
            'raw_logs_and_original_uploaded_archives_verified': True, 'index_measurement_schema_check': 'PASS',
            'basic_m03_comparisons_passed': 90, 'basic_ffn_comparisons_passed': 18, 'distinct_standard_ffn_inputs': 9,
            'additional_standard_comparisons_passed': 30, 'additional_ffn_output_elements': 23040,
            'extra_device_images_identical_to_basic': True, 'projection_cases': 7, 'projection_output_elements': 241920,
            'cross_cluster_cancellation_case_passed': True, 'production_mlp_body_unchanged_in_probe': True,
            'source_origin': base['base_commit'], 'mlp': base['source_inputs']['src/device/shared/mlp.rs'],
            'ffn_image': base['exact_image'], 'median_paired_ffn_delta_percent': candidate['median_paired_delta_percent'],
            'aggregate_median_ffn_delta_percent': candidate['aggregate_median_delta_percent'],
            'scope': 'Limited adoption evidence; same basic3 inputs in six allocations, extra6 new standard inputs, seven separate projection diagnostics. No full-model/all-input equivalence or official submission claim. Root review required before live adoption.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args=parser.parse_args()
    result=verify()
    path=HERE / 'validation/verification.json'
    if args.check:
        assert json.loads(path.read_text()) == result
    else:
        with path.open('x') as out:
            json.dump(result,out,indent=2)
            out.write('\n')
    print(json.dumps(result,indent=2))
