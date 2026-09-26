#!/usr/bin/env python3
"""Verify frozen source scope, complete packages and the actual screened upload."""
import argparse
import json
import re
import subprocess
import sys
import tarfile

from analyze import HERE, ROOT, digest, result as analysis, coordinates
from compose import compose

BUILDS = {'dma-only-v1': 'm07-dma-only-v1', 'compact-upgate-v2': 'm07-compact-upgate-v2',
          'stream-assembly-v3': 'm07-stream-assembly-v3'}
SCREEN = 'm07-stream-assembly-screen'
ART = ROOT / 'moa-dev/experiments/artifacts' / HERE.name


def source_check():
    sys.path.insert(0, str(ROOT / 'moa-dev/experiments'))
    from structural_build import source_manifest
    reused = json.loads((HERE / 'build-m03-reference-repro.json').read_text())
    info = reused.pop('reused_build_record')
    assert digest((ROOT / info['path']).read_bytes())['sha256'] == info['sha256']
    assert reused == json.loads((ROOT / info['path']).read_text())
    assert {p.name: digest(p.read_bytes()) for p in (ART / 'm03-reference/package').iterdir()} == reused['package_files']
    for variant, label in BUILDS.items():
        recipe = json.loads((HERE / variant / 'recipe.json').read_text())
        component = recipe['source_components'][0]
        raw = subprocess.check_output(['git', 'show', component['commit'] + ':' + component['path']], cwd=ROOT)
        assert digest(raw)['sha256'] == component['sha256']
        assert (HERE / variant / 'mlp.rs').read_text() == compose(variant)
        build = json.loads((HERE / f'build-{label}-repro.json').read_text())
        plan = build['source_plan']
        inputs = source_manifest(recipe['base_commit'])
        assert inputs == plan['source_origin_inputs']
        assert digest((ROOT / plan['recipe']['path']).read_bytes()) == {k: plan['recipe'][k] for k in ['bytes', 'sha256']}
        for dest, value in plan['source_overrides'].items():
            inputs[dest] = digest((ROOT / value['path']).read_bytes())
            assert inputs[dest] == {k: value[k] for k in ['bytes', 'sha256']}
        assert inputs == build['source_inputs']
        assert {k: v for k, v in inputs.items() if k != 'src/device/shared/mlp.rs'} == {
            k: v for k, v in reused['source_inputs'].items() if k != 'src/device/shared/mlp.rs'}
        for name, path in build['frozen_files'].items():
            assert digest((ROOT / path).read_bytes()) == build['source_inputs'].get(name, build['logs'].get(name))
        if variant == 'compact-upgate-v2':
            assert build['failed_exit_code'] == 1
            assert 'source_accessing_range_local.end as usize <= input_tensor_range' in (HERE / 'evidence' / label / 'compile.log').read_text()
            continue
        assert build['exact_compile_exit_code'] == build['host_build_exit_code'] == 0
        out = ART / label
        assert digest((out / 'exact-kernel.bin').read_bytes()) == build['exact_image']
        assert digest((out / 'schedule.json').read_bytes()) == build['schedule'] == digest((HERE / 'evidence' / label / 'schedule.json').read_bytes())
        assert {p.name: digest(p.read_bytes()) for p in (out / 'package').iterdir()} == build['package_files']
        host = (out / 'package/test_runtime').read_bytes()
        for name, value in build['images'].items():
            raw = (out / 'kernels' / f'furiosa_opt_gemma4::ops::{name}.bin').read_bytes()
            assert digest(raw) == {k: value[k] for k in ['bytes', 'sha256']}
            assert host.count(raw) == value['host_occurrences'] == 1
            if name == build['target_kernel']:
                assert digest(raw) == build['exact_image']
            else:
                assert value['sha256'] == reused['images'][name]['sha256']
        suffix = 'v1' if variant == 'dma-only-v1' else 'v3'
        inspection = json.loads((HERE / f'inspection-{suffix}.json').read_text())
        assert inspection['exit_code'] == 0 and inspection['image'] == build['exact_image']
        for name, value in inspection['files'].items():
            assert digest((HERE / 'evidence' / label / name).read_bytes()) == value
        assert inspection['files']['inspection-schedule.json'] == build['schedule']
        assert digest((HERE / 'evidence' / label / 'inspection-compile.log').read_bytes()) == inspection['log']
    coordinates()
    assert json.loads((HERE / 'static-analysis.json').read_text()) == analysis()
    return {'frozen_variants': 3, 'executable_variants': 2, 'source_api_contract_and_scope_verified': True,
            'm03_companion_original_package_reused': True, 'qkv_o_images_unchanged': True,
            'source_coverage_and_emitted_descriptors_checked_separately': True,
            'v1_v3_inspection_recompiles_match_original_images': True}


def check():
    value = source_check()
    manifest = json.loads((HERE / f'{SCREEN}-build.json').read_text())
    measured = json.loads((HERE / f'{SCREEN}-results.json').read_text())
    job = measured['job']['id']
    raw = (HERE / f'evidence/arena-{job}.log').read_bytes()
    assert digest(raw)['sha256'] == measured['log_sha256']
    with tarfile.open(ART / f'job-{job}.tar') as archive:
        members = [m for m in archive if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: digest(archive.extractfile(m).read()) for m in members}
    assert files == measured['package_files']
    assert {k: v for k, v in files.items() if k != 'job.yaml'} == manifest['package_files']
    sys.path.insert(0, str(ROOT / 'moa-dev'))
    from measure import parse_log
    sections = re.findall(r'^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$', raw.decode(), re.M | re.S)
    assert [name for name, _ in sections] == [s['name'] for s in manifest['stages']]
    assert raw.count(b'STRUCTURAL_COMPLETE') == 1
    for name, log in sections:
        assert re.findall(r'^STRUCTURAL_EXIT ' + name + r' (\d+)$', log, re.M) == ['0']
        assert parse_log(log) == measured['stages'][name]['kernels']
    assert measured['valid'] and measured['job']['status'] == 'SUCCEEDED'
    value.update({'job': job, 'actual_upload_and_raw_results_verified': True,
                  'basic_comparisons_passed': sum(s['comparisons_passed'] for s in measured['stages'].values()),
                  'candidate_ffn_inputs': 3, 'candidate_ffn_elements': 11520,
                  'scope': 'Single Arena screen, no independent repetitions, expanded accuracy or measured internal engine timings.'})
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    value = source_check() if args.source_only else check()
    if not args.source_only:
        path = HERE / 'verification.json'
        if args.check:
            assert json.loads(path.read_text()) == value
        else:
            path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))
