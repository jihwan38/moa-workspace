#!/usr/bin/env python3
"""Validate six new standard inputs using the frozen E150 FFN down two-stage weight/native device images."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABEL = 'e150-staged-v1-extra6'
ART = ROOT/'moa-dev/experiments/artifacts'/HERE.name/LABEL
VALID = HERE/'adoption'
sys.path.insert(0, str(ROOT/'moa-dev'))
from measure import cli
from extra_seeds import extended_harness, verify_fixture, parse_log


def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def file(path):
    return {'path': str(path.relative_to(ROOT)), **digest(path.read_bytes())}


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify():
    source_path = HERE/'build-e150-staged-v1-repro.json'
    package_path = HERE/'package-e150-staged-v1.json'
    build_path = HERE/f'build-{LABEL}-repro.json'
    base, package, extra = load(source_path), load(package_path), load(build_path)
    assert extra['extra_inputs'] and extra['exact_compile_exit_code'] == extra['host_build_exit_code'] == 0
    assert extra['exact_image'] == base['exact_image']
    assert extra['images'] == package['images']
    host_path = 'src/bin/test_kernels.rs'
    assert {k:v for k,v in extra['source_inputs'].items() if k != host_path} == {
        k:v for k,v in base['source_inputs'].items() if k != host_path}
    original_art = ROOT/'moa-dev/experiments/artifacts'/HERE.name/base['label']
    original_host = (original_art/'crate'/host_path).read_bytes()
    assert digest(original_host) == base['source_inputs'][host_path]
    changed_host = (ART/'crate'/host_path).read_bytes()
    assert changed_host == extended_harness(original_host)
    assert digest(changed_host) == extra['source_inputs'][host_path]
    inputs = {str(p.relative_to(ART/'crate')):digest(p.read_bytes()) for p in (ART/'crate').rglob('*')
              if p.is_file() and (p.is_relative_to(ART/'crate/src') or p.name in {'Cargo.toml','Cargo.lock','rust-toolchain.toml'})}
    assert inputs == extra['source_inputs']
    files = {p.name:digest(p.read_bytes()) for p in (ART/'package').iterdir()}
    assert files == extra['package_files']
    assert (ART/'package/remote_entrypoint.sh').read_bytes() == (original_art/'package/remote_entrypoint.sh').read_bytes()
    host = (ART/'package/test_runtime').read_bytes()
    for name, expected in package['images'].items():
        raw = (ART/'kernels'/f'furiosa_opt_gemma4::ops::{name}.bin').read_bytes()
        assert digest(raw) == {k:expected[k] for k in ('bytes','sha256')}
        assert host.count(raw) == 1
    fixture = verify_fixture((original_art/'package/fixtures.safetensors').read_bytes(),
                             (ART/'package/fixtures.safetensors').read_bytes())
    return {'valid':True, 'build':file(build_path), 'original_build':file(source_path),
            'original_package':file(package_path), 'device_images_identical':True,
            'host_only_change':'RUNS=6, run offset3; PRNG/reference tolerances unchanged',
            'images':extra['images'], 'fixture_schema':fixture, 'package_files':files,
            'scope':'Correctness only, run3..8; extra timing excluded from standard performance.',
            'official_submission':False}


def submit():
    evidence = verify()
    write(VALID/'extra-image-verification.json', evidence)
    remote = f'/tmp/moa-c02-c03/{LABEL}/package'
    for name, expected in evidence['package_files'].items():
        value = subprocess.check_output(['docker','exec','moa2026-dev','sha256sum',remote+'/'+name],text=True).split()[0]
        assert value == expected['sha256']
    path = VALID/'extra-submission.json'
    write(path, {'attempted_at_utc':datetime.now(timezone.utc).isoformat(), 'label':LABEL})
    response = cli('submit',*[remote+'/'+n for n in evidence['package_files']],
                   '--entrypoint','remote_entrypoint.sh','--timeout',70,'--name','beef-e150-staged-v1-extra6')
    record=load(path); record['response']=response
    found = re.findall(r'submitted job (\d+)',response)
    if len(found)==1: record['job_id']=int(found[0])
    path.write_text(json.dumps(record,indent=2)+'\n')
    assert len(found)==1,response
    print(response,end='')


def collect():
    evidence=verify(); assert evidence==load(VALID/'extra-image-verification.json')
    job=load(VALID/'extra-submission.json')['job_id']
    status=json.loads(cli('status',job))
    assert status['status'] in {'SUCCEEDED','COMPLETED','FAILED','CANCELLED','CANCELED'}
    raw=cli('logs',job); log=HERE/f'evidence/arena-extra-{job}.log'
    with log.open('x') as stream: stream.write(raw)
    report={'valid':False,'job':status,'log':file(log),'verification':file(VALID/'extra-image-verification.json'),
            'scope':evidence['scope'],'official_submission':False}
    try:
        remote=f'/tmp/moa-c02-c03/{LABEL}/job-{job}.tar'; archive=ART/f'job-{job}.tar'
        cli('download',job,'--out',remote)
        subprocess.run(['docker','cp','moa2026-dev:'+remote,str(archive)],check=True)
        with tarfile.open(archive) as tar:
            members=[m for m in tar if m.isfile()]
            assert len({m.name for m in members})==len(members)
            files={m.name:digest(tar.extractfile(m).read()) for m in members}
        assert {k:v for k,v in files.items() if k!='job.yaml'}==evidence['package_files']
        parsed=parse_log(raw)
        assert status['status'] in {'SUCCEEDED','COMPLETED'} and status['exit_code']==0
        assert parsed['output_comparisons_passed']==30
        report.update(valid=True,actual_upload_matches=True,package_files=files,archive=file(archive),**parsed)
    except Exception as error:
        report['error']=repr(error)
    write(VALID/'extra-results.json',report)
    write(VALID/'e150-extra-correctness.json',report)
    print(json.dumps(report,indent=2))
    assert report['valid']


def check():
    evidence=verify()
    assert evidence==load(VALID/'extra-image-verification.json')
    report=load(VALID/'extra-results.json')
    assert report==load(VALID/'e150-extra-correctness.json')
    assert report['valid'] and report['actual_upload_matches']
    assert report['job']['id']==load(VALID/'extra-submission.json')['job_id']
    assert report['job']['status'] in {'SUCCEEDED','COMPLETED'} and report['job']['exit_code']==0
    log=ROOT/report['log']['path']; assert file(log)==report['log']
    parsed=parse_log(log.read_text()); assert parsed['output_comparisons_passed']==30
    assert all(report[k]==v for k,v in parsed.items())
    archive=ROOT/report['archive']['path']; assert file(archive)==report['archive']
    with tarfile.open(archive) as tar:
        members=[m for m in tar if m.isfile()]
        assert len({m.name for m in members})==len(members)
        files={m.name:digest(tar.extractfile(m).read()) for m in members}
    assert files==report['package_files']
    assert {k:v for k,v in files.items() if k!='job.yaml'}==evidence['package_files']
    assert report['verification']==file(VALID/'extra-image-verification.json')
    print('PASS extra6 source/image/fixture/runner, original 30 comparisons and actual uploaded archive')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=['verify','submit','collect'])
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        check()
    elif args.action=='verify':
        assert verify()==load(VALID/'extra-image-verification.json')
        print('extra source/image/package validation PASS')
    else:
        globals()[args.action]()
