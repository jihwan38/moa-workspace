#!/usr/bin/env python3
"""Read diagnostic Arena70589 only: never submit, rerun, modify source, or execute NPU."""
import datetime
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
ART=ROOT/'moa-dev/experiments/artifacts/e37-cache-preservation'
JOB=70589
OUT=ROOT/'docs/experiments/e37-cache-preservation-correctness.json'
LOG=ROOT/f'moa-dev/logs/e37-cache-preservation-job-{JOB}.log'
MOA=ROOT/'moa-dev/moa'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('moa_measure',ROOT/'moa-dev/measure.py')
measure=importlib.util.module_from_spec(spec);spec.loader.exec_module(measure)


def cli(*args):
    return subprocess.check_output([str(MOA),'furiosa-arena',*map(str,args)],text=True,stderr=subprocess.STDOUT)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'refuse to overwrite a collected diagnostic'
    deadline=time.monotonic()+1800;previous=None
    while True:
        status=json.loads(cli('status',JOB));state=status['status'].upper()
        current=(state,status.get('queue_ahead'))
        if current!=previous:
            print(f'job{JOB}: {state}, queue_ahead={current[1]}',flush=True);previous=current
        if state in {'SUCCEEDED','FAILED','CANCELLED','CANCELED','COMPLETED'} or status.get('exit_code') is not None:break
        if time.monotonic()>deadline:raise TimeoutError(f'job{JOB} still pending; no new job created')
        time.sleep(5)
    log=cli('logs',JOB);LOG.write_text(log)
    container_tar=f'/tmp/moa-e37-cache-preservation/job-{JOB}.tar'
    cli('download',JOB,'--out',container_tar,'--create-dirs')
    host_tar=ART/f'job-{JOB}.tar'
    subprocess.run(['docker','cp',f'moa2026-dev:{container_tar}',str(host_tar)],check=True)
    package={}
    with tarfile.open(host_tar) as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            assert member.name not in package
            content=tar.extractfile(member).read()
            package[member.name]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    assert set(package)=={'test_runtime','fixtures.safetensors','remote_entrypoint.sh','job.yaml'}
    frozen_package=json.loads((HERE/'package-manifest.json').read_text())['files']
    for name,expected in frozen_package.items():assert package[name]==expected, f'uploaded payload differs: {name}'
    result={'job':status,'log':str(LOG.relative_to(ROOT)),'log_sha256':sha(LOG),'package_files':package}
    record={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'official_score':False,'performance_measurement':False,'candidate':'e37-kernel-specific-rms',
            'build_manifest':str((HERE/'build-manifest.json').relative_to(ROOT)),
            'build_manifest_sha256':sha(HERE/'build-manifest.json'),
            'uploaded_package_matches_frozen_build':True,'downloaded_package_sha256':sha(host_tar),
            'result':result,'scope':'E37 three kernel images are byte-identical to frozen official E37. Custom host sentinel harness. Cycles diagnostic only.',
            'npu_jobs_created_by_collector':0}
    try:
        assert state in {'SUCCEEDED','COMPLETED'} and status.get('exit_code')==0, f'job state {state}, exit{status.get("exit_code")}'
        kernels=measure.parse_log(log);result['kernels']=kernels
        pattern=r'^CACHE_CHECK stage=(before|after) cache=(k|v) run=(\d+) target_slot=(\d+) target_elements=\[(\d+),(\d+)\) target_bytes=\[(\d+),(\d+)\) checked=(\d+) mismatches=(\d+)$'
        keys=['stage','cache','run','target_slot','element_start','element_end','byte_start','byte_end','checked','mismatches']
        checks=[]
        for match in re.findall(pattern,log,re.M):
            check=dict(zip(keys,match))
            for key in keys[2:]:check[key]=int(check[key])
            checks.append(check)
        expected={(stage,cache,run) for stage in ['before','after'] for cache in ['k','v'] for run in range(3)}
        assert len(checks)==12 and {(c['stage'],c['cache'],c['run']) for c in checks}==expected
        for c in checks:
            assert c['target_slot']==[739,693,552][c['run']]
            assert c['element_start']==c['target_slot']*2048
            assert c['element_end']==c['element_start']+2048
            assert c['byte_start']==2*c['element_start'] and c['byte_end']==2*c['element_end']
            assert c['checked']==(2097152 if c['stage']=='before' else 2095104)
            assert c['mismatches']==0
        assert sum(c['checked'] for c in checks if c['stage']=='after')==12570624
        totals=re.findall(r'^CACHE_PRESERVATION total_mismatches=(\d+) status=(PASS|FAIL)$',log,re.M)
        assert totals==[('0','PASS')] and 'CACHE_MISMATCH ' not in log
        assert sum(k['output_comparisons_passed'] for k in kernels.values())==15
        record.update(original_output_comparisons_passed=15,sentinel_checks_passed=12,
                      non_target_elements_checked_after_launch=12570624,cache_mismatches=0,checks=checks,valid=True)
        result['valid']=True
    except (AssertionError,ValueError) as exc:
        record.update(valid=False,error=str(exc));result.update(valid=False,error=str(exc))
    OUT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'valid':record['valid'],
                      'output_comparisons':record.get('original_output_comparisons_passed'),
                      'cache_checks':record.get('sentinel_checks_passed'),
                      'after_elements':record.get('non_target_elements_checked_after_launch'),
                      'error':record.get('error')},indent=2),flush=True)
    return 0 if record['valid'] else 1


if __name__=='__main__':sys.exit(main())
