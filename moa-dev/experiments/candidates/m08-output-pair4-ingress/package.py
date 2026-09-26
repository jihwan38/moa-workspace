#!/usr/bin/env python3
"""Package a descriptor-approved M08 exact image and diagnostic-only profiles.

The caller owns the shared target while this script runs. It never submits Arena
jobs. Existing compile evidence is read-only; package-<label>.json is separate.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts' / HERE.name
PROFILE = HERE.parent / 'e104-weight-path'
QO = HERE.parent / 'e104-qkv-output-profile'
KERNELS = ('sliding_project_qkv', 'sliding_attention_output', 'decoder_feedforward')
O = KERNELS[1]
sys.path.insert(0, str(PROFILE))
from image_profile import digest, inspect


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def file_record(path):
    return {'path': str(path.relative_to(ROOT)), **digest(path.read_bytes())}


def checked_control():
    recipe_path = ROOT / 'baseline/e122.json'
    recipe = json.loads(recipe_path.read_text())
    audit_path = ROOT / recipe['source_build_audit']
    audit = json.loads(audit_path.read_text())
    build_path = ROOT / audit['measured_build']
    build = json.loads(build_path.read_text())
    manifest = json.loads((ROOT / recipe['compile_inputs']).read_text())
    planner = module('m08_source_planner', ROOT / 'moa-dev/experiments/structural_build.py')
    assert recipe['id'] == 'E122'
    assert build['source_inputs'] == manifest == planner.source_manifest(recipe['git']['commit'])
    family = build_path.parent.name
    artifact = ROOT / 'moa-dev/experiments/artifacts' / family / build['label']
    package = artifact / 'package'
    assert {p.name: digest(p.read_bytes()) for p in package.iterdir()} == build['package_files']
    host = (package / 'test_runtime').read_bytes()
    for kernel in KERNELS:
        raw = (artifact / 'kernels' / f'furiosa_opt_gemma4::ops::{kernel}.bin').read_bytes()
        assert digest(raw) == {k: build['images'][kernel][k] for k in ('bytes', 'sha256')}
        assert build['images'][kernel] == audit['images'][kernel]
        assert host.count(raw) == build['images'][kernel]['host_occurrences'] == 1
    return recipe, build, package, {
        'recipe': file_record(recipe_path), 'source_build_audit': file_record(audit_path),
        'measured_build': file_record(build_path), 'source_inputs_match_adopted_manifest': True,
        'package_files': build['package_files'], 'images': build['images']}


def label_image(raw):
    labelled, metadata = inspect(raw, True)
    _, checked = inspect(labelled)
    for key in ('version', 'chips', 'pes', 'header_bytes', 'tasks', 'weight', 'stacks',
                'slots', 'inputs', 'outputs', 'profile_depth'):
        assert metadata[key] == checked[key], key
    allowed = {i for row in metadata['spans'] for s in row
               for i in range(s['name_offset'], s['name_offset'] + s['name_bytes'])}
    assert len(raw) == len(labelled)
    assert all(a == b or i in allowed for i, (a, b) in enumerate(zip(raw, labelled)))
    assert [[(s['begin_marker'], s['end_marker']) for s in row] for row in metadata['spans']] == [
        [(s['begin_marker'], s['end_marker']) for s in row] for row in checked['spans']]
    return labelled, metadata


def map_markers(schedule, metadata):
    analyzer = module('m08_qo_marker_classifier', QO / 'analyze.py')
    assert len(metadata['spans']) == 1
    spans = sorted(metadata['spans'][0], key=lambda s: s['begin_marker'])
    markers = [s for s in spans if s['name'] not in analyzer.WRAPPERS]
    operations = [n for n in schedule['instructions'] if analyzer.operation_kind(n)]
    assert len(markers) == len(operations) > 0
    assert [s['name'] for s in markers] == [analyzer.operation_kind(n) for n in operations]
    assert len({s['label'] for s in spans}) == len(spans)
    task, = [s for s in spans if s['name'] == 'Task']
    return {'task_label': task['label'], 'spans': spans,
            'operations': [{'marker': s, 'schedule': n} for s, n in zip(markers, operations)],
            'scope': 'Complete native marker-kind / schedule-kind ordered correspondence; no duration-based source inference.'}


def package(label, target_dir):
    assert re.fullmatch(r'[a-z0-9-]{1,70}', label)
    assert target_dir.startswith('/tmp/') and '..' not in Path(target_dir).parts
    output_path = HERE / f'package-{label}.json'
    assert not output_path.exists(), output_path
    # A successful earlier check is insufficient if its source/graph has changed.
    subprocess.run([sys.executable, str(HERE / 'verify_descriptors.py'), label, '--check'],
                   cwd=ROOT, check=True)
    gate_path = HERE / f'descriptor-analysis-{label}.json'
    gate = json.loads(gate_path.read_text())
    assert gate['passed'] is True
    assert gate.get('inputdigests'), 'Descriptor gate must bind its checked inputs'
    compile_path = HERE / f'build-{label}-repro.json'
    compiled = json.loads(compile_path.read_text())
    assert compiled['exact_compile_exit_code'] == 0 and compiled['compile_only'] is True
    assert compiled['target_kernel'] == O
    recipe, control_build, control_package, control_record = checked_control()
    declared = compiled['source_plan']['performance_comparison']
    assert declared['id'] == 'E122' and declared['commit'] == recipe['git']['commit']
    assert declared['sha256'] == control_record['recipe']['sha256']
    out = ARTIFACTS / label
    crate = out / 'crate'
    inputs = {str(p.relative_to(crate)): digest(p.read_bytes()) for p in crate.rglob('*')
              if p.is_file() and (p.is_relative_to(crate / 'src') or p.name in
                                 {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'})}
    assert inputs == compiled['source_inputs']
    assert {k: v for k, v in inputs.items() if k != 'src/device/sliding/projection.rs'} == {
        k: v for k, v in control_build['source_inputs'].items() if k != 'src/device/sliding/projection.rs'}
    exact = (out / 'exact-kernel.bin').read_bytes()
    assert digest(exact) == compiled['exact_image']
    assert digest((out / 'schedule.json').read_bytes()) == compiled['schedule']
    standard = out / 'package'
    diagnostic = out / 'diagnostic'
    standard.mkdir(exist_ok=False)
    diagnostic.mkdir(exist_ok=False)
    remote = '/tmp/moa-c02-c03/' + label
    record = {'label': label, 'status': 'building', 'compile_record': file_record(compile_path),
              'descriptor_gate': file_record(gate_path), 'descriptor_gate_rechecked': True,
              'source_inputs': inputs, 'control': control_record, 'target_dir': target_dir,
              'commands': [], 'logs': {}, 'adopted': False, 'official_submission': False}

    def run(command, log_name=None):
        record['commands'].append(command)
        if log_name:
            path = diagnostic / log_name
            with path.open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    def capture_images(host, destination):
        destination.mkdir(exist_ok=False)
        image_dir = remote + '/kernels/furiosa-opt-gemma4'
        run(['docker', 'cp', 'moa2026-dev:' + image_dir + '/.', str(destination)])
        result = {}
        for kernel in KERNELS:
            raw = (destination / f'furiosa_opt_gemma4::ops::{kernel}.bin').read_bytes()
            expected = compiled['exact_image'] if kernel == O else {
                k: control_build['images'][kernel][k] for k in ('bytes', 'sha256')}
            assert digest(raw) == expected, kernel
            assert host.count(raw) == 1, kernel
            result[kernel] = {**digest(raw), 'host_occurrences': 1}
        return result

    try:
        # Restore exact compiled inputs before building, even if scratch was reused.
        run(['docker', 'exec', 'moa2026-dev', 'mkdir', '-p', remote])
        run(['docker', 'cp', str(crate) + '/.', 'moa2026-dev:' + remote])
        archived = subprocess.check_output(['docker', 'exec', '--workdir', remote,
                                           'moa2026-dev', 'tar', '-cf', '-', *inputs])
        with tarfile.open(fileobj=io.BytesIO(archived)) as tar:
            assert {m.name: digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == inputs
        record['source_inputs_verified_in_container'] = True
        env = ['docker', 'exec', '--workdir', remote, '-e', 'CARGO_HOME=/home/moa/.cargo',
               '-e', 'CARGO_NET_OFFLINE=true', '-e', 'CARGO_TARGET_DIR=' + target_dir,
               '-e', 'FURIOSA_OPT_OUT_DIR=' + remote + '/kernels', 'moa2026-dev']
        run(env + ['cargo', 'clean', '--release', '-p', 'furiosa-opt-gemma4'])
        run(env + ['cargo', 'furiosa-opt', 'build', '--release', '--locked', '--bin', 'test_kernels'], 'standard-build.log')
        run(['docker', 'cp', 'moa2026-dev:' + target_dir + '/release/test_kernels', str(standard / 'test_runtime')])
        host = (standard / 'test_runtime').read_bytes()
        record['images'] = capture_images(host, out / 'standard-kernels')
        for name in ('fixtures.safetensors', 'remote_entrypoint.sh'):
            shutil.copyfile(control_package / name, standard / name)
        for name in ('test_runtime', 'remote_entrypoint.sh'):
            (standard / name).chmod(0o755)
        record['package_files'] = {p.name: digest(p.read_bytes()) for p in standard.iterdir()}

        profiler = module('m08_profile_host_transform', PROFILE / 'build.py')
        transformed = profiler.host_source((crate / 'src/bin/test_kernels.rs').read_text()).encode()
        old_build = json.loads((PROFILE / 'build-profile-v1.json').read_text())
        assert digest(transformed) == old_build['source_inputs']['src/bin/test_kernels.rs']
        host_path = diagnostic / 'test_kernels.rs'
        host_path.write_bytes(transformed)
        record['diagnostic_source_inputs'] = {**inputs, 'src/bin/test_kernels.rs': digest(transformed)}
        run(['docker', 'cp', str(host_path), 'moa2026-dev:' + remote + '/src/bin/test_kernels.rs'])
        copied_host = subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'cat', remote + '/src/bin/test_kernels.rs'])
        assert copied_host == transformed
        diagnostic_archive = subprocess.check_output(['docker', 'exec', '--workdir', remote,
            'moa2026-dev', 'tar', '-cf', '-', *inputs])
        with tarfile.open(fileobj=io.BytesIO(diagnostic_archive)) as tar:
            assert {m.name: digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == record['diagnostic_source_inputs']
        record['diagnostic_source_inputs_verified_in_container'] = True
        # Same crate path, new host bytes; no device source is changed.
        run(env + ['cargo', 'furiosa-opt', 'build', '--release', '--locked', '--bin', 'test_kernels'], 'diagnostic-build.log')
        run(['docker', 'cp', 'moa2026-dev:' + target_dir + '/release/test_kernels', str(diagnostic / 'unlabelled-host')])
        unlabelled = (diagnostic / 'unlabelled-host').read_bytes()
        assert capture_images(unlabelled, out / 'diagnostic-kernels') == record['images']
        labelled_o, metadata = label_image(exact)
        assert unlabelled.count(exact) == 1
        probe = unlabelled.replace(exact, labelled_o)
        assert len(probe) == len(unlabelled) and probe.count(labelled_o) == 1
        (diagnostic / 'candidate-probe').write_bytes(probe)
        write(diagnostic / 'candidate-o.image-profile.json', metadata)
        write(diagnostic / 'candidate-marker-map.json', map_markers(json.loads((out / 'schedule.json').read_text()), metadata))

        # Reuse only the O launch of the old generic host: its FFN is E104 and is
        # never executed here. Whole benchmark controls use the actual E122 host.
        old_art = ROOT / 'moa-dev/experiments/artifacts/e104-weight-path/profile-v1'
        original = (old_art / 'unlabelled-host').read_bytes()
        assert digest(original) == old_build['unlabelled_host']
        control_o = None
        for kernel in KERNELS:
            raw = (old_art / 'kernels/furiosa-opt-gemma4' / f'furiosa_opt_gemma4::ops::{kernel}.bin').read_bytes()
            assert digest(raw) == old_build['images'][kernel] and original.count(raw) == 1
            if kernel == O:
                assert digest(raw) == {k: control_build['images'][O][k] for k in ('bytes', 'sha256')}
                control_o = raw
        assert control_o is not None
        labelled_control, control_metadata = label_image(control_o)
        control_probe = original.replace(control_o, labelled_control)
        (diagnostic / 'control-probe').write_bytes(control_probe)
        write(diagnostic / 'control-o.image-profile.json', control_metadata)
        control_schedule = QO / 'evidence/profile-v1/sliding_attention_output.json'
        qo_build = json.loads((QO / 'build-profile-v1.json').read_text())
        assert digest(control_schedule.read_bytes()) == qo_build['dumps'][control_schedule.name]
        assert qo_build['images'][O] == digest(control_o)
        write(diagnostic / 'control-marker-map.json', map_markers(json.loads(control_schedule.read_text()), control_metadata))
        for name in ('candidate-probe', 'control-probe'):
            (diagnostic / name).chmod(0o755)
        record.update(status='ready', standard_host_build_exit_code=0, diagnostic_host_build_exit_code=0,
                      unlabelled_diagnostic_host=digest(unlabelled),
                      diagnostic_host_source=file_record(host_path),
                      profile_name_bytes_only=True, task_bytes_unchanged=True,
                      control_diagnostic_reuse={
                          'build': file_record(PROFILE / 'build-profile-v1.json'),
                          'unlabelled_host': digest(original), 'same_host_transform': True,
                          'exact_o_matches_e122': True, 'other_kernels_executed': False,
                          'scope': 'E104 generic host O-only launch, exact adopted E122 O image. Not a full E122 diagnostic host.'},
                      diagnostic_files={p.name: digest(p.read_bytes()) for p in diagnostic.iterdir()
                                        if p.name in {'candidate-probe', 'control-probe', 'candidate-marker-map.json',
                                                      'control-marker-map.json', 'candidate-o.image-profile.json',
                                                      'control-o.image-profile.json'}},
                      fixture_and_runner_match_e122=True)
        print('M08 standard and O-only diagnostic packages verified; no Arena submission')
    except Exception as error:
        record.update(status='failed', error=repr(error))
        if isinstance(error, subprocess.CalledProcessError):
            record.update(failed_command=error.cmd, failed_exit_code=error.returncode)
        raise
    finally:
        record['logs'] = {p.name: digest(p.read_bytes()) for p in diagnostic.glob('*-build.log')}
        frozen = HERE / 'evidence' / label / 'profile-package'
        frozen.mkdir(parents=True, exist_ok=False)
        record['frozen_files'] = {}
        for name in ('test_kernels.rs', 'standard-build.log', 'diagnostic-build.log',
                     'control-marker-map.json', 'candidate-marker-map.json',
                     'control-o.image-profile.json', 'candidate-o.image-profile.json'):
            path = diagnostic / name
            if path.exists():
                shutil.copyfile(path, frozen / name)
                assert (frozen / name).read_bytes() == path.read_bytes()
                record['frozen_files'][name] = file_record(frozen / name)
        record['builder'] = file_record(Path(__file__).resolve())
        write(output_path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--target-dir', default='/tmp/moa-c01-upgate/target')
    args = parser.parse_args()
    package(args.label, args.target_dir)
