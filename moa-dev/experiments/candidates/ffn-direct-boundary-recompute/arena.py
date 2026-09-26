#!/usr/bin/env python3
"""Submit/collect immutable diagnostic payloads, preserving failed stages too."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from build import HERE, ROOT, ARTIFACTS, digest
sys.path.insert(0,str(ROOT/'moa-dev'))
from measure import cli

action,label=sys.argv[1:3]
build=json.loads((HERE/f'build-{label}.json').read_text())
package=ARTIFACTS/label/'package'
assert {p.name:digest(p.read_bytes()) for p in package.iterdir()}==build['package_files']
if action=='submit':
    subprocess.run([sys.executable,str(HERE/'verify.py'),label],check=True)
    remote='/tmp/moa-ffn-recompute/'+label+'/package'
    response=cli('submit',*(remote+'/'+name for name in build['package_files']),'--entrypoint','run.sh','--timeout',70,'--name','beef-ffn-recompute-'+label)
    ids=re.findall(r'submitted job (\d+)',response);assert len(ids)==1,response
    with (HERE/f'{label}-submission.json').open('x') as f:
        json.dump({'response':response,'job_id':int(ids[0]),'submitted_at_utc':datetime.now(timezone.utc).isoformat()},f,indent=2);f.write('\n')
    print(response,end='')
elif action=='collect':
    job=json.loads((HERE/f'{label}-submission.json').read_text())['job_id']
    status=json.loads(cli('status',job));assert status['status'] in ['SUCCEEDED','FAILED','CANCELLED','COMPLETED'],status
    raw=cli('logs',job).encode()
    evidence=HERE/'evidence'/label
    (evidence/f'arena-{job}.log').write_bytes(raw)
    remote=f'/tmp/moa-ffn-recompute/job-{job}.tar'
    cli('download',job,'--out',remote)
    archive=ARTIFACTS/f'job-{job}.tar'
    subprocess.run(['docker','cp','moa2026-dev:'+remote,str(archive)],check=True)
    with tarfile.open(archive) as a:
        members=[m for m in a if m.isfile()];assert len({m.name for m in members})==len(members)
        files={m.name:digest(a.extractfile(m).read()) for m in members}
    assert {k:v for k,v in files.items() if k!='job.yaml'}==build['package_files']
    results=[]
    for line in raw.decode().splitlines():
        if line.startswith('BOUNDARY_RESULT '):
            results.append(dict(re.findall(r'(\w+)=([^ ]+)',line)))
    result={'job':status,'log':str((evidence/f'arena-{job}.log').relative_to(ROOT)),'log_hash':digest(raw),
            'actual_upload_verified':True,'package_files':files,'results':results,
            'stage_exits':dict(re.findall(r'BOUNDARY_EXIT_(\w+) (\d+)',raw.decode())),
            'complete':raw.count(b'BOUNDARY_SCREEN_COMPLETE')==1,
            'scope':'Primitive correctness diagnostic; no whole-kernel performance claim'}
    with (HERE/f'{label}-results.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['results','stage_exits','complete']},indent=2))
else:raise ValueError(action)
