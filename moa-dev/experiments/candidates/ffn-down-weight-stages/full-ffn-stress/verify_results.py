#!/usr/bin/env python3
"""Recheck actual E150 same-input full-FFN production-image evidence."""
import argparse, importlib.util, json, re, subprocess, sys, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fullstress_replay',HERE/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def verify():
    result_path=HERE/'e150-full-ffn-stress-results.json';r=m.load(result_path)
    assert r['valid'] and r['actual_upload_matches'] and r['performance_excluded']
    for key in ['submission','log','archive']:assert m.file(m.ROOT/r[key]['path'])==r[key]
    s=m.load(m.ROOT/r['submission']['path']);assert s['job_id']==r['job']['id']
    assert r['job']['status'] in ['SUCCEEDED','COMPLETED'] and r['job']['exit_code']==0
    assert s['sides']==['e150']
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
    assert r['stages']['e150']['passed'] and r['stages']['e150']['exact']==3840
    assert r['stages']['e150']['nonfinite']==0
    prior=m.ROOT/'moa-dev/experiments/candidates/ffn-native-down-weighted-residual/full-ffn-stress'
    for name in ['full_ffn_stress.rs','expected-bf16.bin','reference.json','reference-build.json']:
        assert (HERE/name).read_bytes()==(prior/name).read_bytes()
    subprocess.run([sys.executable,str(prior/'verify_reference.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(prior/'verify_input_codegen.py'),'--check',
                    '--prepare-log',str(HERE/'evidence/e150/prepare.log')],check=True,stdout=subprocess.DEVNULL)
    return {'valid':True,'job_id':r['job']['id'],'results':m.file(result_path),
            'builder':m.file(Path(__file__)),'original_tolerance_unchanged':True,
            'official_torch_reference':m.file(HERE/'reference.json'),
            'cross_language_input_evidence':m.file(prior/'input-codegen-verification.json'),
            'new_candidate_pass':True,'same_input_as_original_e134_regression':True,
            'scope':r['scope'],'stages':r['stages'],'performance_excluded':True,'official_submission':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();v=verify();dest=HERE/'verification.json'
    if a.check:assert m.load(dest)==v
    else:m.write(dest,v)
    print('PASS E150 public-image fullFFN stress, 3840 exact outputs and original13input hashes; no performance claim')
