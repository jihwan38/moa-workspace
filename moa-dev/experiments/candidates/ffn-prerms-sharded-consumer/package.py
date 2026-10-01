#!/usr/bin/env python3
"""Build the unchanged standard host around a frozen exact FFN compilation.

Compilation evidence is immutable. This separate package record binds E151's
actual source manifest, three images and benchmark inputs to the candidate.
The caller owns the shared compiler target while package() runs. No Arena calls.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts' / HERE.name
KERNELS = ('sliding_project_qkv', 'sliding_attention_output', 'decoder_feedforward')
TARGET = KERNELS[2]
FILES = ('test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh')


def digest(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def file_record(path):
    return {'path': str(path.relative_to(ROOT)), **digest(path.read_bytes())}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def clean_label(label):
    assert re.fullmatch(r'[a-z0-9-]{1,70}', label), label
    return label


def package_files(folder):
    actual = {p.name: digest(p.read_bytes()) for p in folder.iterdir() if p.is_file()}
    assert set(actual) == set(FILES), folder
    return actual


def source_inputs(crate):
    return {str(p.relative_to(crate)): digest(p.read_bytes()) for p in crate.rglob('*')
            if p.is_file() and (p.is_relative_to(crate / 'src') or p.name in
                               {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'})}


def checked_control():
    recipe_path = ROOT / 'baseline/e151.json'
    recipe = read(recipe_path)
    audit_path = ROOT / recipe['source_build_audit']
    audit = read(audit_path)
    build_path = ROOT / audit['measured_build']
    build = read(build_path)
    manifest_path = ROOT / recipe['compile_inputs']
    planner = module('qkv_native_weight_resident_source_planner', ROOT / 'moa-dev/experiments/structural_build.py')
    assert recipe['id'] == 'E151'
    assert file_record(build_path) == audit['measured_build_digest']
    assert build['source_inputs'] == read(manifest_path) == planner.source_manifest(recipe['git']['commit'])
    package = ROOT / build['package_path']
    assert package_files(package) == build['package_files']
    host = (package / 'test_runtime').read_bytes()
    for kernel in KERNELS:
        image_file = build['standard_image_files'][kernel]
        image_path = ROOT / image_file['path']
        raw = image_path.read_bytes()
        assert file_record(image_path) == image_file
        assert digest(raw) == {k: build['images'][kernel][k] for k in ('bytes', 'sha256')}
        assert build['images'][kernel] == audit['images'][kernel]
        assert host.count(raw) == build['images'][kernel]['host_occurrences'] == 1
    runner = subprocess.check_output(['git', 'show', recipe['git']['commit'] +
        ':furiosa-opt-gemma4-12B/scripts/rngd/remote_entrypoint.sh'], cwd=ROOT)
    assert runner == (package / 'remote_entrypoint.sh').read_bytes()
    assert b'here=$(dirname "$0")' in runner and b'cd "$here" || exit 1' in runner
    assert b'binary=./test_runtime' in runner and b'chmod +x "$binary"' in runner
    assert digest((package / 'fixtures.safetensors').read_bytes())['sha256'] == \
        '338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359'
    return recipe, build, package, {
        'recipe': file_record(recipe_path), 'source_build_audit': file_record(audit_path),
        'measured_build': file_record(build_path), 'compile_inputs': file_record(manifest_path),
        'source_inputs_match_adopted_manifest': True, 'source_inputs': build['source_inputs'],
        'package_files': build['package_files'], 'images': build['images'],
        'original_runner_and_fixture_verified': True}


def checked_compile(label):
    clean_label(label)
    path = HERE / f'build-{label}-repro.json'
    built = read(path)
    recipe, control, control_dir, control_record = checked_control()
    assert built['label'] == label and built['target_kernel'] == TARGET
    assert built['exact_compile_exit_code'] == 0 and built['compile_only'] is True
    assert not built['extra_inputs'] and not built['edge_inputs']
    declared = built['source_plan']['performance_comparison']
    assert declared['id'] == recipe['id'] and declared['commit'] == recipe['git']['commit']
    assert declared['sha256'] == control_record['recipe']['sha256']
    artifact = ARTIFACTS / label
    assert source_inputs(artifact / 'crate') == built['source_inputs']
    changed = {key for key in built['source_inputs'] if built['source_inputs'][key] != control['source_inputs'].get(key)}
    assert set(built['source_inputs']) == set(control['source_inputs'])
    assert changed == {'src/ops.rs', 'src/device/shared/rmsnorm.rs', 'src/device/shared/mlp.rs'}, changed
    assert digest((artifact / 'exact-kernel.bin').read_bytes()) == built['exact_image']
    for name in ('schedule', 'graph'):
        assert digest((artifact / f'{name}.json').read_bytes()) == built[name]
    for name, frozen in built['frozen_files'].items():
        if name in built['source_inputs']:
            assert digest((ROOT / frozen).read_bytes()) == built['source_inputs'][name]
    return path, built, control, control_dir, control_record


def checked_gate(label):
    path = HERE / f'graph-verification-{clean_label(label)}.json'
    value = read(path)
    assert value['passed'] is True
    assert value['inputdigests']['build_record'] == file_record(HERE / f'build-{label}-repro.json')
    artifact = ARTIFACTS / label
    for name in ('graph', 'schedule'):
        assert value['inputdigests'][name] == file_record(artifact / f'{name}.json')
    for item in value.get('verifiers', []):
        assert file_record(ROOT / item['path']) == item
    return {'record': file_record(path), 'rechecked_input_digests': True}


def checked_package(label):
    compile_path, built, control, control_dir, control_record = checked_compile(label)
    path = HERE / f'package-{label}.json'
    value = read(path)
    assert value['status'] == 'ready' and value['host_build_exit_code'] == 0
    assert value['compile_record'] == file_record(compile_path)
    assert value['graph_gate'] == checked_gate(label)
    assert value['control'] == control_record and value['source_inputs'] == built['source_inputs']
    folder = ARTIFACTS / label / 'package'
    assert package_files(folder) == value['package_files']
    host = (folder / 'test_runtime').read_bytes()
    for kernel in KERNELS:
        image_path = ARTIFACTS / label / 'standard-kernels' / f'furiosa_opt_gemma4::ops::{kernel}.bin'
        raw = image_path.read_bytes()
        assert file_record(image_path) == value['image_files'][kernel]
        expected = built['exact_image'] if kernel == TARGET else {k: control['images'][kernel][k] for k in ('bytes', 'sha256')}
        assert digest(raw) == expected
        assert value['images'][kernel] == {**digest(raw), 'host_occurrences': 1}
        assert host.count(raw) == 1
    for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
        assert (folder / name).read_bytes() == (control_dir / name).read_bytes()
    return path, value, folder, control_dir


def package(label, target_dir):
    assert target_dir.startswith('/tmp/') and '..' not in Path(target_dir).parts
    path = HERE / f'package-{clean_label(label)}.json'
    assert not path.exists(), path
    compile_path, built, control, control_dir, control_record = checked_compile(label)
    graph_gate = None  # Host compilation may overlap independent descriptor review.
    artifact = ARTIFACTS / label
    folder = artifact / 'package'
    folder.mkdir(exist_ok=False)
    images_dir = artifact / 'standard-kernels'
    images_dir.mkdir(exist_ok=False)
    evidence = HERE / 'evidence' / label
    evidence.mkdir(parents=True, exist_ok=True)
    remote = '/tmp/moa-c02-c03/' + label
    value = {'schema_version': 1, 'label': label, 'status': 'building',
             'compile_record': file_record(compile_path), 'source_inputs': built['source_inputs'],
             'control': control_record, 'target_dir': target_dir, 'commands': [], 'logs': {},
             'graph_gate': graph_gate,
             'builder': file_record(Path(__file__).resolve()), 'adopted': False,
             'official_submission': False,
             'scope': 'Original standard host and inputs; source/image integrity is not an arithmetic or performance claim.'}

    def run(command, log_path=None):
        value['commands'].append(command)
        if log_path:
            with log_path.open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    def verify_container():
        raw = subprocess.check_output(['docker', 'exec', '--workdir', remote, 'moa2026-dev',
            'tar', '-cf', '-', 'src', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            actual = {m.name: digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
        assert actual == built['source_inputs']

    try:
        print(f'{label}: restoring and checking exact compiled sources', flush=True)
        run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', remote])
        run(['docker', 'cp', str(artifact / 'crate') + '/.', 'moa2026-dev:' + remote])
        verify_container()
        value['source_inputs_verified_in_container'] = True
        env = ['docker', 'exec', '--workdir', remote, '-e', 'CARGO_HOME=/home/moa/.cargo',
               '-e', 'CARGO_NET_OFFLINE=true', '-e', 'CARGO_TARGET_DIR=' + target_dir,
               '-e', 'FURIOSA_OPT_OUT_DIR=' + remote + '/kernels', 'moa2026-dev']
        run(env + ['cargo', 'clean', '--release', '-p', 'furiosa-opt-gemma4'])
        print(f'{label}: building original standard host; log={evidence / "standard-build.log"}', flush=True)
        run(env + ['cargo', 'furiosa-opt', 'build', '--release', '--locked', '--bin', 'test_kernels'],
            evidence / 'standard-build.log')
        value['host_build_exit_code'] = 0
        print(f'{label}: host build passed; checking all three embedded images', flush=True)
        verify_container()
        run(['docker', 'cp', 'moa2026-dev:' + target_dir + '/release/test_kernels', str(folder / 'test_runtime')])
        run(['docker', 'cp', 'moa2026-dev:' + remote + '/kernels/furiosa-opt-gemma4/.', str(images_dir)])
        host = (folder / 'test_runtime').read_bytes()
        value['images'], value['image_files'] = {}, {}
        for kernel in KERNELS:
            image_path = images_dir / f'furiosa_opt_gemma4::ops::{kernel}.bin'
            raw = image_path.read_bytes()
            expected = built['exact_image'] if kernel == TARGET else {k: control['images'][kernel][k] for k in ('bytes', 'sha256')}
            assert digest(raw) == expected and host.count(raw) == 1, kernel
            value['images'][kernel] = {**digest(raw), 'host_occurrences': 1}
            value['image_files'][kernel] = file_record(image_path)
        for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
            shutil.copyfile(control_dir / name, folder / name)
        for name in ('test_runtime', 'remote_entrypoint.sh'):
            (folder / name).chmod(0o755)
        value['package_files'] = package_files(folder)
        value['package_path'] = str(folder.relative_to(ROOT))
        deadline = time.monotonic() + 300
        while not (HERE / f'graph-verification-{label}.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Independent graph gate not ready after host build')
            time.sleep(1)
        value['graph_gate'] = checked_gate(label)
        value['status'] = 'ready'
    except Exception as error:
        value.update(status='failed', error=repr(error))
        if isinstance(error, subprocess.CalledProcessError):
            value.update(failed_command=error.cmd, failed_exit_code=error.returncode)
        raise
    finally:
        log = evidence / 'standard-build.log'
        if log.exists():
            value['logs']['standard-build.log'] = file_record(log)
        write(path, value)
    checked_package(label)
    print(f'Packaged {label}: exact FFN plus unchanged E151 QKV/O; no Arena submission')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--target-dir', default='/tmp/moa-c09-validation-probe/target')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        checked_package(args.label)
        print('Candidate package and actual E151 control verified')
    else:
        package(args.label, args.target_dir)
