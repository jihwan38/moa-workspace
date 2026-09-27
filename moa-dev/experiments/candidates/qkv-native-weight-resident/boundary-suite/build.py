#!/usr/bin/env python3
"""Bind the existing two-case QKV diagnostic to an exact production image.

Only a host binary is added. Every original compilation input and the emitted
QKV image must match the selected candidate build. No device wrapper is used.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CANDIDATE = HERE.parent
LEGACY = ROOT / 'moa-dev/experiments/candidates/qkv-boundary-probe-e53'


def digest(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def run(cmd, log=None):
    if log:
        with log.open('w') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--target', default='/tmp/moa-m03-validation-probe/target')
    ap.add_argument('--fixture', type=Path, required=True)
    args = ap.parse_args()
    assert args.label.replace('-', '').isalnum()
    selected = json.loads(args.build.read_text())
    assert selected['exact_compile_exit_code'] == 0
    assert selected['source_inputs_verified_in_container']
    out = ROOT / 'moa-dev/experiments/artifacts/qkv-native-weight-resident' / args.label
    out.mkdir(parents=True, exist_ok=False)
    crate = out / 'crate'
    source = ROOT / 'moa-dev/experiments/artifacts/qkv-native-weight-resident' / selected['label'] / 'crate'
    shutil.copytree(source, crate)
    original = {str(p.relative_to(crate)): digest(p.read_bytes()) for p in sorted(crate.rglob('*'))
                if p.is_file() and (p.is_relative_to(crate/'src') or p.name in
                                   {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'})}
    assert original == selected['source_inputs']
    host = (LEGACY/'probe_qkv.rs').read_bytes()
    assert digest(host)['sha256'] == '893e8d17e21387962f5ff97ac8fb25420272c6c4469fd277ac30bcbad4b25292'
    (crate/'src/bin/probe_qkv.rs').write_bytes(host)
    expected = dict(original, **{'src/bin/probe_qkv.rs': digest(host)})
    cpu = json.loads((LEGACY/'cpu-references.json').read_text())
    fixture = args.fixture.read_bytes()
    assert digest(fixture) == cpu['payload']
    scratch = '/tmp/moa-e136-boundary/' + args.label
    run(['docker','exec','moa2026-dev','mkdir','-p',scratch])
    run(['docker','cp',str(crate)+'/.','moa2026-dev:'+scratch])
    packed = subprocess.check_output(['docker','exec','--workdir',scratch,'moa2026-dev',
              'tar','-cf','-','src','Cargo.toml','Cargo.lock','rust-toolchain.toml'])
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        actual = {m.name:digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
    assert actual == expected
    env = ['docker','exec','--workdir',scratch,'-e','CARGO_HOME=/home/moa/.cargo',
           '-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+args.target,
           '-e','FURIOSA_OPT_OUT_DIR='+scratch+'/kernels','moa2026-dev']
    record = {'selected_build':str(args.build.resolve().relative_to(ROOT)),
              'selected_build_digest':digest(args.build.read_bytes()),
              'source_inputs':actual, 'production_inputs_match':True,
              'only_added_source':'src/bin/probe_qkv.rs', 'target':args.target,
              'scratch':scratch,'host_source':digest(host),
              'cpu_reference_source':str((LEGACY/'cpu-references.json').relative_to(ROOT)),
              'cpu_reference_digest':digest((LEGACY/'cpu-references.json').read_bytes()),
              'scope':'Two full-production-QKV cancellation cases outside official generator support; no performance or all-input claim.'}
    try:
        command = env + ['cargo','furiosa-opt','build','--release','--locked','--bin','probe_qkv']
        record['build_command'] = command
        run(command, out/'build.log')
        record['host_build_exit_code'] = 0
        package = out/'package'
        package.mkdir()
        run(['docker','cp','moa2026-dev:'+args.target+'/release/probe_qkv',str(package/'test_runtime')])
        run(['docker','cp','moa2026-dev:'+scratch+'/kernels',str(out/'kernels')])
        images = list((out/'kernels').rglob('furiosa_opt_gemma4::ops::sliding_project_qkv.bin'))
        assert len(images) == 1
        image = images[0].read_bytes()
        assert digest(image) == selected['exact_image']
        assert (package/'test_runtime').read_bytes().count(image) == 1
        (out/'qkv.bin').write_bytes(image)
        record['image'] = dict(digest(image), host_occurrences=1,
                               path=str((out/'qkv.bin').relative_to(ROOT)))
        (package/'boundary.safetensors').write_bytes(fixture)
        shutil.copyfile(LEGACY/'remote_entrypoint.sh',package/'remote_entrypoint.sh')
        for name in ['test_runtime','remote_entrypoint.sh']:
            (package/name).chmod(0o755)
        run(['docker','cp',str(package),'moa2026-dev:'+scratch+'/package'])
        run(['docker','exec','--workdir',scratch+'/package','moa2026-dev','./test_runtime',
             '--fixture','boundary.safetensors','--host-check-only','--dump-generated',scratch+'/dump'],out/'host-check.log')
        run(['docker','cp','moa2026-dev:'+scratch+'/dump',str(out/'dump')])
        run([sys.executable,str(LEGACY/'verify_host.py'),'--cpu',str(LEGACY/'cpu-references.json'),
             '--dump',str(out/'dump'),'--log',str(out/'host-check.log')],out/'host-check-summary.json')
        verification = json.loads((out/'host-check-summary.json').read_text())
        assert verification['full_input_sha256_passed'] == 38
        record['host_check'] = {'checksums_passed':38,'full_input_sha256_passed':38,'npu_launches':0}
        record['package'] = {p.name:digest(p.read_bytes()) for p in sorted(package.iterdir())}
        record['container_package'] = scratch+'/package'
        record['status'] = 'READY_NOT_NPU_TESTED'
    except subprocess.CalledProcessError as exc:
        record.update(status='BUILD_OR_HOST_CHECK_FAILED',failed_command=exc.cmd,exit_code=exc.returncode)
        raise
    finally:
        record['logs'] = {p.name:digest(p.read_bytes()) for p in [out/'build.log',out/'host-check.log',out/'host-check-summary.json'] if p.exists()}
        (HERE/('build-'+args.label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'package':record['container_package'],'image':record['image']}),flush=True)


if __name__ == '__main__':
    main()
