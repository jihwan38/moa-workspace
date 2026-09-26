#!/usr/bin/env python3
"""Exact QKV/O schedules and metadata-only labels on a verified existing host."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'e104-weight-path'
sys.path.insert(0, str(OLD))
from image_profile import digest, inspect

ARTIFACTS = ROOT / 'moa-dev/experiments/artifacts/e104-qkv-output-profile'
TARGETS = ('sliding_project_qkv', 'sliding_attention_output')

def build(label):
    record_path = HERE / f'build-{label}.json'
    assert not record_path.exists()
    spec = importlib.util.spec_from_file_location('planner', ROOT / 'moa-dev/experiments/structural_build.py')
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    _, _, plan = planner.resolve_plan(HERE.name, 'v1')
    out = ARTIFACTS / label
    out.mkdir(parents=True, exist_ok=False)
    crate = out / 'crate'
    archive = subprocess.check_output(['git', 'archive', plan['source_base_commit'], 'furiosa-opt-gemma4-12B'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isfile():
                path = crate / Path(member.name).relative_to('furiosa-opt-gemma4-12B')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
    shutil.copyfile(ROOT / 'moa-dev/locks/gemma4-Cargo.lock', crate / 'Cargo.lock')
    inputs = {name: digest((crate/name).read_bytes()) for name in plan['source_origin_inputs']}
    assert inputs == plan['source_origin_inputs']
    for target, override in plan['source_overrides'].items():
        assert digest((ROOT/override['path']).read_bytes()) == inputs[target]
    remote = '/tmp/moa-e104-qo/' + label
    subprocess.run(['docker','exec','moa2026-dev','mkdir','-p',remote],check=True)
    subprocess.run(['docker','cp',str(crate)+'/.','moa2026-dev:'+remote],check=True)
    copied = subprocess.check_output(['docker','exec','--workdir',remote,'moa2026-dev','tar','-cf','-',*inputs])
    with tarfile.open(fileobj=io.BytesIO(copied)) as tar:
        assert {m.name:digest(tar.extractfile(m).read()) for m in tar if m.isfile()} == inputs
    env = ['docker','exec','--workdir',remote,'-e','CARGO_HOME=/home/moa/.cargo',
           '-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR=/tmp/moa-c01-upgate/target',
           '-e',f'FURIOSA_OPT_OUT_DIR={remote}/kernels','moa2026-dev']
    evidence = HERE/'evidence'/label
    evidence.mkdir(parents=True,exist_ok=False)
    record = {'label':label,'source_plan':plan,'source_inputs':inputs,
              'source_inputs_verified_in_container':True,'builder':digest(Path(__file__).read_bytes()),
              'commands':[],'dumps':{},'images':{},'logs':{},'scope':'Existing arithmetic; profile labels only'}
    try:
        for kernel in TARGETS:
            command = env+['cargo','furiosa-opt','compile','ops::'+kernel,'--exact',
                          '--dump-schedule',kernel+'.json','--dump-ir',kernel+'.ir.txt','--dump-visa',kernel+'.visa.txt']
            record['commands'].append(command)
            print('Compile unchanged',kernel,flush=True)
            log = evidence/(kernel+'.compile.log')
            with log.open('w') as f:
                subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            record['logs'][log.name]=digest(log.read_bytes())
            for suffix in ('.json','.ir.txt','.visa.txt'):
                name=kernel+suffix
                subprocess.run(['docker','cp','moa2026-dev:'+remote+'/'+name,str(evidence/name)],check=True)
                record['dumps'][name]=digest((evidence/name).read_bytes())
        old_build=json.loads((OLD/'build-profile-v1.json').read_text())
        old_art=ROOT/'moa-dev/experiments/artifacts/e104-weight-path/profile-v1'
        original=(old_art/'unlabelled-host').read_bytes()
        assert digest(original)==old_build['unlabelled_host']
        host_source=OLD/'evidence/profile-v1/test_kernels.rs'
        assert digest(host_source.read_bytes())==old_build['source_inputs']['src/bin/test_kernels.rs']
        assert {k:v for k,v in old_build['source_inputs'].items() if k!='src/bin/test_kernels.rs'} == {
            k:v for k,v in inputs.items() if k!='src/bin/test_kernels.rs'}
        binary=original
        for p in sorted((old_art/'kernels').rglob('*.bin')):
            kernel=p.name.split('::')[-1].removesuffix('.bin')
            raw=p.read_bytes()
            assert digest(raw)==old_build['images'][kernel] and binary.count(raw)==1
            record['images'][kernel]=digest(raw)
            if kernel not in TARGETS:
                continue
            labelled,metadata=inspect(raw,True)
            _,checked=inspect(labelled)
            for key in ('tasks','weight','stacks','slots','inputs','outputs','profile_depth'):
                assert metadata[key]==checked[key]
            allowed={i for spans in metadata['spans'] for s in spans for i in range(s['name_offset'],s['name_offset']+s['name_bytes'])}
            assert len(raw)==len(labelled)
            assert all(a==b or i in allowed for i,(a,b) in enumerate(zip(raw,labelled)))
            assert [[(s['begin_marker'],s['end_marker']) for s in row] for row in metadata['spans']] == [
                [(s['begin_marker'],s['end_marker']) for s in row] for row in checked['spans']]
            binary=binary.replace(raw,labelled)
            (evidence/(kernel+'.image-profile.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        assert len(record['images'])==3
        package=out/'package'; package.mkdir()
        (package/'probe').write_bytes(binary); (package/'probe').chmod(0o755)
        record.update(unlabelled_host=digest(original),labelled_host=digest(binary),
                      reused_host_build=str((OLD/'build-profile-v1.json').relative_to(ROOT)),
                      reused_host_source=str(host_source.relative_to(ROOT)),
                      exact_compile_exit_code=0,task_bytes_unchanged=True,profile_name_bytes_only=True)
        print('Verified diagnostic package ready',flush=True)
    except subprocess.CalledProcessError as error:
        record.update(failed_command=error.cmd,failed_exit_code=error.returncode)
        raise
    finally:
        for log in evidence.glob('*.compile.log'):
            record['logs'][log.name]=digest(log.read_bytes())
        record_path.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    build(parser.parse_args().label)
