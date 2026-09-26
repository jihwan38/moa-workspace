#!/usr/bin/env python3
"""Audit frozen diagnostic sources, actual uploads and primitive comparisons."""
import argparse
import json
from pathlib import Path
import re
import tarfile
from build import HERE, ROOT, ARTIFACTS, digest

def check():
    builds={}
    for p in sorted(HERE.glob('build-*.json')):
        b=json.loads(p.read_text());label=p.stem.removeprefix('build-')
        if 'source_inputs' in b:
            evidence=HERE/'evidence'/label
            for name,expected in b['source_inputs'].items():
                assert digest((evidence/name).read_bytes())==expected,(label,name)
            for name,expected in b['logs'].items():
                assert digest((evidence/name).read_bytes())==expected,(label,name)
            assert digest((HERE/b['variant']/'recipe.json').read_bytes())==b['recipe']
            for dest,name in {'src/lib.rs':'lib.rs','src/probe.rs':'probe.rs','src/bin/boundary_probe.rs':'boundary_probe.rs'}.items():
                assert digest((HERE/b['variant']/name).read_bytes())==b['source_inputs'][dest]
            recipe=json.loads((HERE/b['variant']/'recipe.json').read_text())
            for dest,name in recipe.get('extra_sources',{}).items():
                assert digest((HERE/b['variant']/name).read_bytes())==b['source_inputs'][dest]
        if 'package_files' in b:
            package=ARTIFACTS/label/'package'
            assert {x.name:digest(x.read_bytes()) for x in package.iterdir()}==b['package_files']
            if 'images' in b:
                host=(package/'boundary_probe').read_bytes()
                for name,expected in b['images'].items():
                    paths=list((ARTIFACTS/label/'kernels').rglob(name));assert len(paths)==1
                    image=paths[0].read_bytes()
                    assert digest(image)=={k:expected[k] for k in ['bytes','sha256']}
                    assert host.count(image)==expected['host_occurrences']==1
        builds[label]={'compile':b.get('build_exit_code'),'failure':b.get('failed_exit_code'),
                       'frozen_source_files':len(b.get('source_inputs',{})),
                       'scope':'immutable prior-image replay' if 'sources' in b else 'new diagnostic images'}
    jobs={}
    for p in sorted(HERE.glob('*-results.json')):
        r=json.loads(p.read_text());label=p.stem.removesuffix('-results')
        b=json.loads((HERE/f'build-{label}.json').read_text())
        raw=(ROOT/r['log']).read_bytes();assert digest(raw)==r['log_hash']
        with tarfile.open(ARTIFACTS/f"job-{r['job']['id']}.tar") as a:
            files={m.name:digest(a.extractfile(m).read()) for m in a if m.isfile()}
        assert files==r['package_files']
        assert {k:v for k,v in files.items() if k!='job.yaml'}==b['package_files']
        stages=dict(re.findall(r'BOUNDARY_EXIT_(\w+) (\d+)',raw.decode()))
        assert stages==r['stage_exits']
        values=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in raw.decode().splitlines() if line.startswith('BOUNDARY_RESULT ')]
        assert values==r['results']
        legacy=[{'case':int(c),'projection':n,'checked':int(t),'mismatches':int(m)} for c,n,t,m in
                re.findall(r'C07_PROBE case=(\d+) projection=(\w+) checked=(\d+) mismatches=(\d+)',raw.decode())]
        jobs[label]={'job':r['job']['id'],'status':r['job']['status'],'stage_exits':stages,
                     'comparisons_reported':sum(int(v['checked']) for v in values),
                     'mismatches_reported':sum(int(v['mismatches']) for v in values),
                     'legacy_checks':legacy,'upload_and_log_verified':True,
                     'complete':r['complete']}
    lowering=json.loads((HERE/'lowering-evidence.json').read_text())
    for name,expected in lowering['files'].items():
        assert digest((ROOT/name).read_bytes())==expected,name
    return {'builds':builds,'jobs':jobs,'lowering_files_verified':len(lowering['files']),
            'scope':'Diagnostic evidence only; no whole-FFN speed or all-input correctness claim.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    result=check();p=HERE/'verification.json'
    if args.check:assert json.loads(p.read_text())==result
    else:p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'builds':len(result['builds']),'jobs':{k:{f:v[f] for f in ['job','status','stage_exits']} for k,v in result['jobs'].items()}},indent=2))
