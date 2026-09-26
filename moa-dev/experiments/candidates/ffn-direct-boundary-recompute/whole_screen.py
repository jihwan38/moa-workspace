#!/usr/bin/env python3
"""E104 -> E123 -> original M03 -> E104, with checked source/image provenance."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from build import HERE, ROOT, ARTIFACTS, digest
from whole_verify import verify

spec = importlib.util.spec_from_file_location('screen_common', ROOT / 'moa-dev/experiments/structural_screen.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
common.HERE, common.ARTIFACTS = HERE, ARTIFACTS

def checked_package(candidate, label):
    directory = ROOT / 'moa-dev/experiments/candidates' / candidate
    b = json.loads((directory / f'build-{label}-repro.json').read_text())
    p = ROOT / 'moa-dev/experiments/artifacts' / candidate / label
    assert b['exact_compile_exit_code'] == b['host_build_exit_code'] == 0
    files = {f.name: f.read_bytes() for f in (p / 'package').iterdir()}
    assert {name: digest(data) for name, data in files.items()} == b['package_files']
    assert set(files) == {'test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh'}
    for name, expected in b['source_inputs'].items():
        assert digest((p / 'crate' / name).read_bytes()) == expected, (label, name)
    for name, expected in b['images'].items():
        data = (p / 'kernels' / f'furiosa_opt_gemma4::ops::{name}.bin').read_bytes()
        assert {**digest(data), 'host_occurrences': files['test_runtime'].count(data)} == expected
        assert files['test_runtime'].count(data) == 1
    return b, files

def prepare(label, candidate):
    verify(candidate)
    e104, control_files = checked_package('c07-native-scale-dpe', 'c07-e104-reviewed-v1')
    e104_recipe_path = ROOT / 'baseline/e104.json'
    e104_recipe = json.loads(e104_recipe_path.read_text())
    assert e104['source_inputs'] == json.loads((ROOT / e104_recipe['compile_inputs']).read_text())
    m03, m03_files = checked_package('m03-producer-local-ffn', 'm03-native-local-v1')
    mp = m03['source_plan']
    m03_expected = dict(mp['source_origin_inputs'])
    for target, source in mp['source_overrides'].items():
        expected = {k: source[k] for k in ['bytes', 'sha256']}
        assert digest((ROOT / source['path']).read_bytes()) == expected
        m03_expected[target] = expected
    assert m03_expected == m03['source_inputs']
    cand, candidate_files = checked_package(HERE.name, candidate)
    for b in [m03, cand]:
        declared = b['source_plan']['performance_comparison']
        assert declared['id'] == 'E104'
        assert declared['sha256'] == digest(e104_recipe_path.read_bytes())['sha256']
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    stages = []
    for stage, build_label, files in [('control_before', 'e104', control_files), ('candidate', candidate, candidate_files), ('m03_control', 'm03-native-local-v1', m03_files), ('control_after', 'e104', control_files)]:
        for name, data in files.items():
            (out / f'{stage}_{name}').write_bytes(data)
        stages.append({'name': stage, 'build': build_label, 'extra': False})
    runner = ['#!/bin/sh', 'set -eu', 'cd "$(dirname "$0")"', 'any_failed=0']
    for stage in stages:
        name = stage['name']
        runner += [f'mkdir {name}', *[f'cp {name}_{f} {name}/{f}' for f in ['test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh']], f'chmod +x {name}/test_runtime', f'echo STRUCTURAL_BEGIN {name}', f'if sh {name}/remote_entrypoint.sh; then code=0; else code=$?; any_failed=1; fi', f'echo STRUCTURAL_EXIT {name} "$code"', f'echo STRUCTURAL_END {name}']
    runner += ['echo STRUCTURAL_COMPLETE', 'exit "$any_failed"']
    (out / 'screen.sh').write_text('\n'.join(runner) + '\n')
    data = {'label': label, 'stages': stages, 'timeout_seconds': 70, 'performance_control': {'recipe': 'baseline/e104.json', **digest(e104_recipe_path.read_bytes()), 'source_manifest_matches_actual_control_build': True, 'build_record': 'moa-dev/experiments/candidates/c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json'}, 'secondary_control': {'build_record': 'moa-dev/experiments/candidates/m03-producer-local-ffn/build-m03-native-local-v1-repro.json', 'actual_original_source_and_images_verified': True, 'images': m03['images']}, 'candidate_source_images_descriptors_verified': True, 'method': 'One Arena allocation; separate unchanged benchmark processes E104before -> E123 -> original M03 -> E104after. Not independent repetitions.', 'package_files': {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())}}
    common.record(HERE / f'{label}-build.json', data)
    print(f'Prepared {label}: four verified stages')

action, label = sys.argv[1:3]
if action == 'prepare':
    prepare(label, sys.argv[3])
elif action == 'submit':
    common.submit(label)
elif action == 'collect':
    try:
        common.collect(label)
    finally:
        path = HERE / f'{label}-results.json'
        if path.exists():
            result = json.loads(path.read_text())
            dest = HERE / 'evidence' / label
            dest.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / result['log'], dest / 'arena.log')
else:
    raise ValueError(action)
