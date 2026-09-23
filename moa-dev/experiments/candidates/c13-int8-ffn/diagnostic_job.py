#!/usr/bin/env python3
"""Submit/collect an immutable diagnostic payload; never a performance report."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from screen import HERE, ROOT, ARTIFACTS, digest, record

sys.path.insert(0, str(ROOT / 'moa-dev'))
from measure import cli


def submit(label):
    build = json.loads((HERE / f'build-{label}.json').read_text())
    assert build['build_exit_code'] == 0
    package = ARTIFACTS / label / 'package'
    assert {p.name: digest(p.read_bytes()) for p in package.iterdir()} == build['package_files']
    remote = build['container_package']
    for name, info in build['package_files'].items():
        actual = subprocess.check_output(['docker', 'exec', 'moa2026-dev', 'sha256sum', remote + '/' + name], text=True).split()[0]
        assert actual == info['sha256']
    path = HERE / f'{label}-submission.json'
    record(path, {'attempted_at_utc': datetime.now(timezone.utc).isoformat()})
    response = cli('submit', *(remote+'/'+n for n in build['package_files']), '--entrypoint', 'run.sh', '--timeout', 40, '--name', label)
    r = json.loads(path.read_text()); r['response'] = response
    ids = re.findall(r'submitted job (\d+)', response)
    if len(ids) == 1: r['job_id'] = int(ids[0])
    path.write_text(json.dumps(r, indent=2)+'\n')
    assert len(ids) == 1, response
    print(response, end='')


def collect(label):
    build = json.loads((HERE / f'build-{label}.json').read_text())
    job = json.loads((HERE / f'{label}-submission.json').read_text())['job_id']
    status = json.loads(cli('status', job))
    assert status['status'] in ('SUCCEEDED','FAILED','COMPLETED','CANCELLED'), status
    log = cli('logs', job)
    path = HERE / 'evidence' / label / f'arena-{job}.log'
    assert not path.exists(); path.write_text(log)
    remote = f'/tmp/moa-c13/{label}/job-{job}.tar'
    cli('download', job, '--out', remote)
    local = ARTIFACTS / label / f'job-{job}.tar'
    subprocess.run(['docker','cp','moa2026-dev:'+remote,str(local)], check=True)
    with tarfile.open(local) as archive:
        members = [m for m in archive if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        files = {m.name: digest(archive.extractfile(m).read()) for m in members}
    assert {k:v for k,v in files.items() if k!='job.yaml'} == build['package_files']
    complete = re.findall(r'^C13_DIAG_COMPLETE mismatches=(\d+)$', log, re.M)
    integer = re.findall(r'^C13_DIAG_INTEGER weight_mismatches=(\d+) dot_mismatches=(\d+)$', log, re.M)
    quant = re.findall(r'^C13_DIAG_QUANT name=(H|L) case=(\d+) checked=(\d+) mismatches=(\d+)$', log, re.M)
    r = {'job':status, 'build_record':f'build-{label}.json', 'log':str(path.relative_to(ROOT)), 'log_sha256':digest(log.encode())['sha256'], 'uploaded_package_verified':True, 'package_files':files,
         'complete':len(complete)==1, 'integer': [{'weight_mismatches':int(w),'dot_mismatches':int(d)} for w,d in integer],
         'quantization':[{'axis':a,'case':int(c),'checked':int(n),'mismatches':int(b)} for a,c,n,b in quant],
         'valid': status['exit_code']==0 and complete==['0'] and len(integer)==1 and len(quant)==6,
         'scope':'Diagnostic numerical results only; not whole-FFN accuracy or speed.'}
    record(HERE/f'{label}-results.json', r)
    print(log, end='')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['submit','collect']);p.add_argument('label');a=p.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+',a.label)
    (submit if a.action=='submit' else collect)(a.label)
