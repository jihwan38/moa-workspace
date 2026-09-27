#!/usr/bin/env python3
"""Recheck actual same-input full-FFN control/failing-prototype/repaired-image evidence."""
import argparse, importlib.util, json, re, subprocess, sys, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fullstress_replay',HERE/'run_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def verify():
    result_path=HERE/'e134-full-ffn-stress-results.json';r=m.load(result_path)
    assert r['valid'] and r['actual_upload_matches'] and r['performance_excluded']
    for key in ['submission','log','archive']:assert m.file(m.ROOT/r[key]['path'])==r[key]
    s=m.load(m.ROOT/r['submission']['path']);assert s['job_id']==r['job']['id']
    assert r['job']['status'] in ['SUCCEEDED','COMPLETED'] and r['job']['exit_code']==0
    assert s['sides']==['e128-recovered','e133','e134']
    for side in s['sides']:
        assert m.file(HERE/f'build-{side}.json')==s['builds'][side]
        m.verified(side)
    with tarfile.open(m.ROOT/r['archive']['path']) as tar:
        members=[x for x in tar if x.isfile()];assert len(members)==len({x.name for x in members})
        files={x.name:m.digest(tar.extractfile(x).read()) for x in members}
    assert files==r['uploaded_files']
    assert {k:v for k,v in files.items() if k!='job.yaml'}==s['package_files']
    raw=(m.ROOT/r['log']['path']).read_text()
    assert re.findall(r'^STRESS_STAGE_(\S+)$',raw,re.M)==s['sides']
    for side in s['sides']:
        block=raw.split('STRESS_STAGE_'+side+'\n',1)[1].split('STRESS_STAGE_',1)[0]
        rows=re.findall(r'^FULL_STRESS_RESULT checked=3840 exact=(\d+) mismatches=(\d+) nonfinite=(\d+) max_abs=(\S+) first=(\S+) second=(\S+) expected=(\S+)$',block,re.M)
        assert len(rows)==1 and block.count('FULL_STRESS_COMPLETE')==1
        exact,bad,nan,delta,first,second,expected=rows[0]
        parsed={'checked':3840,'exact':int(exact),'mismatches':int(bad),'nonfinite':int(nan),
                'max_abs':float(delta),'first':float(first),'second':float(second),'expected':float(expected),
                'passed':int(bad)==0 and int(nan)==0}
        assert parsed==r['stages'][side]
        assert re.findall(r'^FULL_STRESS_INPUT .+$',block,re.M)==[s['input_preparation'].splitlines()[0]]
    assert raw.count('FULL_STRESS_BATCH_COMPLETE')==1
    assert r['stages']['e128-recovered']['passed'] and r['stages']['e134']['passed']
    assert not r['stages']['e133']['passed'] and r['stages']['e133']['mismatches']==1920
    assert all(v['nonfinite']==0 for v in r['stages'].values())
    subprocess.run([sys.executable,str(HERE/'verify_reference.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(HERE/'verify_input_codegen.py'),'--check',
                    '--prepare-log',str(HERE/'evidence/e128/prepare.log'),
                    '--prepare-log',str(HERE/'evidence/e133/prepare.log'),
                    '--prepare-log',str(HERE/'evidence/e134/prepare.log')],check=True,stdout=subprocess.DEVNULL)
    return {'valid':True,'job_id':r['job']['id'],'results':m.file(result_path),
            'builder':m.file(Path(__file__)),'original_tolerance_unchanged':True,
            'official_torch_reference':m.file(HERE/'reference.json'),
            'cross_language_input_evidence':m.file(HERE/'input-codegen-verification.json'),
            'control_and_repair_pass':True,'e133_full_ffn_failure_reproduced':True,
            'scope':r['scope'],'stages':r['stages'],'performance_excluded':True,'official_submission':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();v=verify();dest=HERE/'verification.json'
    if a.check:assert m.load(dest)==v
    else:m.write(dest,v)
    print('PASS same-input public-image fullFFN stress: E128 and E134 pass, E133 fails 1920 outputs; no performance claim')
