#!/usr/bin/env python3
"""Check source, metadata-only patches, uploaded payload and complete output gates."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OLD=HERE.parent/'e104-weight-path'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(ROOT/'moa-dev'))
from image_profile import digest,inspect
from measure import parse_log

def main():
    build=json.loads((HERE/'build-profile-v1.json').read_text())
    assert build['exact_compile_exit_code']==0
    assert build['task_bytes_unchanged'] and build['profile_name_bytes_only']
    base=build['source_plan']['source_base_commit']
    for name,value in build['source_inputs'].items():
        raw=(ROOT/'moa-dev/locks/gemma4-Cargo.lock').read_bytes() if name=='Cargo.lock' else subprocess.check_output(
            ['git','show',base+':furiosa-opt-gemma4-12B/'+name],cwd=ROOT)
        assert digest(raw)==value,name
        # This is frozen E104 evidence; later baseline adoption may change live
        # files. Check the recorded source revision, never a future working tree.
    evidence=HERE/'evidence/profile-v1'
    for name,value in {**build['logs'],**build['dumps']}.items():
        assert digest((evidence/name).read_bytes())==value,name
    art=ROOT/'moa-dev/experiments/artifacts/e104-qkv-output-profile'
    original_art=ROOT/'moa-dev/experiments/artifacts/e104-weight-path/profile-v1'
    old=json.loads((OLD/'build-profile-v1.json').read_text())
    original=(original_art/'unlabelled-host').read_bytes()
    assert digest(original)==build['unlabelled_host']==old['unlabelled_host']
    expected=original
    for p in (original_art/'kernels').rglob('*.bin'):
        name=p.name.split('::')[-1].removesuffix('.bin');raw=p.read_bytes()
        assert digest(raw)==build['images'][name]
        if name=='decoder_feedforward':continue
        patched,metadata=inspect(raw,True)
        assert json.loads(json.dumps(metadata))==json.loads((evidence/(name+'.image-profile.json')).read_text())
        allowed={i for row in metadata['spans'] for s in row for i in range(s['name_offset'],s['name_offset']+s['name_bytes'])}
        assert all(a==b or i in allowed for i,(a,b) in enumerate(zip(raw,patched)))
        _,updated=inspect(patched)
        for key in ('tasks','weight','stacks','slots','inputs','outputs','profile_depth'):
            assert metadata[key]==updated[key]
        assert expected.count(raw)==1
        expected=expected.replace(raw,patched)
    assert digest(expected)==build['labelled_host']
    assert expected==(art/'profile-v1/package/probe').read_bytes()
    package=json.loads((HERE/'qo-profile-screen-v1-package.json').read_text())
    count=0;jobs=[]
    for p in sorted(HERE.glob('qo-profile-screen-v1-job-*-results.json')):
        report=json.loads(p.read_text());job=report['job']['id'];jobs.append(job)
        assert report['actual_upload_matches'] and report['screen_complete']
        assert report['job']['exit_code']==0 and report['job']['status'] in {'SUCCEEDED','COMPLETED'}
        raw=(ROOT/report['log']).read_bytes();assert digest(raw)==report['log_digest']
        sections=re.findall(r'^DIAG_STAGE_BEGIN (\w+)\n(.*?)^DIAG_STAGE_END \1$',raw.decode(),re.M|re.S)
        assert [n for n,_ in sections]==[row[0] for row in package['stages']]
        for name,content in sections:
            saved=report['stages'][name]
            assert re.findall(r'^DIAG_STAGE_EXIT '+name+r' (\d+)$',content,re.M)==['0']
            if name.startswith('control'):
                assert parse_log(content)==saved['kernels'];count+=15
            else:
                kernel='sliding_project_qkv' if name.startswith('qkv') else 'sliding_attention_output'
                outputs=('q','k','v') if name.startswith('qkv') else ('',)
                expected_labels={f'{kernel}{" "+o if o else ""} run{run}' for run in range(3) for o in outputs}
                found=re.findall(r'^\[([^]]+)\].*within tol=\s*([\d.]+)%\s*-> (PASS|FAIL)$',content,re.M)
                assert {label.strip() for label,_,_ in found}==expected_labels and len(found)==len(expected_labels)
                assert all(float(tol)==100 and status=='PASS' for _,tol,status in found)
                assert 'DIAG_CHECK_COMPLETE selected=1 runs=3 failures=0' in content
                spans=[json.loads(s) for s in re.findall(r'^DIAG_SPAN (.+)$',content,re.M)]
                assert spans==saved['spans'] and all(s['kernel']==kernel for s in spans)
                assert set(s['run'] for s in spans)=={0,1,2}
                count+=len(found)
        with tarfile.open(art/f'job-{job}.tar') as tar:
            members=[m for m in tar if m.isfile()]
            assert len({m.name for m in members})==len(members)
            files={m.name:digest(tar.extractfile(m).read()) for m in members if m.name!='job.yaml'}
        assert files==package['package_files']
    assert jobs
    result={'status':'PASS','jobs':jobs,'output_comparisons_passed':count,
            'original_task_bytes_unchanged':True,'source_and_uploaded_payload_verified':True,
            'scope':'Diagnostic integrity and unchanged output gates; no optimization claim'}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
