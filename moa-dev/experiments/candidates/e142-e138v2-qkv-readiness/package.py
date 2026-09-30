#!/usr/bin/env python3
"""Exact E142/E138 native-v2 standard hosts plus names-only QKV profile hosts."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; ARTIFACTS=ROOT/'moa-dev/experiments/artifacts'/HERE.name
sys.path.insert(0,str(HERE))
from image_profile import digest,inspect
from host_transform import host_source
FFN='sliding_project_qkv';KERNELS=(FFN,'sliding_attention_output','decoder_feedforward')
BUILD='e138-native-v2'

def read(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def file_record(p):return {'path':str(p.relative_to(ROOT)),**digest(p.read_bytes())}
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def bindings():
    recipe_path=ROOT/'baseline/e142.json';recipe=read(recipe_path)
    audit_path=ROOT/recipe['source_build_audit'];audit=read(audit_path)
    control_path=ROOT/audit['measured_build'];control=read(control_path)
    assert file_record(control_path)==audit['measured_build_digest']
    manifest=read(ROOT/recipe['compile_inputs'])
    planner=module('e143_source_planner',ROOT/'moa-dev/experiments/structural_build.py')
    assert manifest==control['source_inputs']==planner.source_manifest(recipe['git']['commit'])
    assert len(manifest)==46 and control['images']==audit['images']
    cp=HERE.parent/'qkv-native-consumer-local';bp=cp/f'build-{BUILD}-repro.json';built=read(bp)
    pp=cp/f'package-{BUILD}.json';pkg=read(pp)
    assert built['exact_compile_exit_code']==0 and pkg['status']=='ready'
    assert pkg['compile_record']==file_record(bp) and pkg['source_inputs']==built['source_inputs']
    gate=ROOT/pkg['graph_gate']['record']['path'];assert file_record(gate)==pkg['graph_gate']['record'] and read(gate)['passed']
    changed=sorted(n for n in manifest if manifest[n]!=built['source_inputs'][n])
    assert changed==['src/device/shared/mlp.rs','src/device/sliding/projection.rs','src/ops.rs']
    original=planner.source_manifest(built['base_commit'])
    assert built['base_commit']=='334a7059a82252a310436c6d7e02f8ba577e7f51'
    assert sorted(n for n in original if original[n]!=built['source_inputs'][n])==['src/device/sliding/projection.rs','src/ops.rs']
    for n in ['src/device/sliding/projection.rs','src/ops.rs']:
        assert digest((ROOT/built['frozen_files'][n]).read_bytes())==built['source_inputs'][n]
    cs_path=HERE.parent/'e104-qkv-output-profile/build-profile-v1.json';cs0=read(cs_path)
    assert {k:control['images'][FFN][k] for k in ('bytes','sha256')}==cs0['images'][FFN]
    schedule_path=cs_path.parent/'evidence/profile-v1/sliding_project_qkv.json'
    ir_path=cs_path.parent/'evidence/profile-v1/sliding_project_qkv.ir.txt'
    assert digest(schedule_path.read_bytes())==cs0['dumps'][schedule_path.name]
    assert digest(ir_path.read_bytes())==cs0['dumps'][ir_path.name]
    cs={'exact_image':cs0['images'][FFN], 'schedule':cs0['dumps'][schedule_path.name],
        'graph':cs0['dumps'][ir_path.name], 'frozen_files':{'schedule.json':str(schedule_path.relative_to(ROOT)),
        'graph.json':str(ir_path.relative_to(ROOT))}, 'graph_format':'original compiler textual IR, not JSON graph'}
    sides={'control':{'build':control,'schedule_build':cs,'package':ROOT/control['package_path'],'images':control['standard_image_files']},
           'candidate':{'build':built,'schedule_build':built,'package':ROOT/pkg['package_path'],'images':pkg['image_files']}}
    for side,info in sides.items():
        expected=control['package_files'] if side=='control' else pkg['package_files']
        assert {x.name:digest(x.read_bytes()) for x in info['package'].iterdir() if x.is_file()}==expected
        host=(info['package']/'test_runtime').read_bytes()
        for kernel,item in info['images'].items():
            assert file_record(ROOT/item['path'])==item
            raw=(ROOT/item['path']).read_bytes();assert host.count(raw)==1
            assert digest(raw)=={k:(control if side=='control' else pkg)['images'][kernel][k] for k in ('bytes','sha256')}
        assert digest((ROOT/info['images'][FFN]['path']).read_bytes())==info['schedule_build']['exact_image']
    assert pkg['images']['sliding_attention_output']==control['images']['sliding_attention_output']
    assert pkg['images']['decoder_feedforward']!=control['images']['decoder_feedforward']
    for name in ('fixtures.safetensors','remote_entrypoint.sh'):
        assert (sides['candidate']['package']/name).read_bytes()==(sides['control']['package']/name).read_bytes()
    cr={'recipe':file_record(recipe_path),'source_build_audit':file_record(audit_path),'measured_build':file_record(control_path),'source_commit':recipe['git']['commit']}
    return sides,cr,{'control_build':file_record(control_path),'candidate_build':file_record(bp),
        'candidate_package':file_record(pp),'candidate_graph_gate':pkg['graph_gate'],
        'control_schedule_build':file_record(cs_path),'control_schedule_qkv_image_byte_identical':True,
        'changed_sources_between_standard_packages':changed,'noncomparable_ffn_sources_explicit':True,
        'comparison_scope':'QKV only. Candidate keeps original E137-base/E134 FFN; control is actual E142. No FFN comparison or attribution.'}

def checked_control():
    sides,cr,_=bindings();return read(ROOT/'baseline/e142.json'),sides['control']['build'],sides['control']['package'],cr

def operation_kind(n):
    if n['tpe'].startswith('Dma'):return 'DMA'
    if n['tpe']=='Main':return 'Renegade::TuExec'
    if n['tpe']=='Sub' and n['contexts']!=['Noop']:
        d=n['description']
        if '.to_trf()' in d:return 'Renegade::StoTrf'
        if not d.strip():return 'Renegade::StoTab'
        # This coefficient fetch widens four live lanes to eight physical lanes.
        # Actual image marker is TuExec; a padded collect is not a direct StoVrf.
        if '.to_vrf()' in d and '.collect::<m![1], m![QkvTerm # 8]>()' in d:
            assert '.fetch::<m![1], m![QkvTerm]>()' in d
            return 'Renegade::TuExec'
        if '.to_vrf()' in d and not any(x in d for x in ['fetch_cast','vector_']):return 'Renegade::StoVrf'
        return 'Renegade::TuExec'
    return None

def label_image(raw):
    labelled,meta=inspect(raw,True);_,check=inspect(labelled)
    for key in ('version','chips','pes','header_bytes','tasks','weight','stacks','slots','inputs','outputs','profile_depth'):
        assert meta[key]==check[key],key
    allowed={i for row in meta['spans'] for s in row for i in range(s['name_offset'],s['name_offset']+s['name_bytes'])}
    assert len(raw)==len(labelled) and all(a==b or i in allowed for i,(a,b) in enumerate(zip(raw,labelled)))
    assert [[(s['begin_marker'],s['end_marker']) for s in r] for r in meta['spans']]==[[(s['begin_marker'],s['end_marker']) for s in r] for r in check['spans']]
    return labelled,meta

def map_markers(schedule,meta):
    assert len(meta['spans'])==1
    spans=sorted(meta['spans'][0],key=lambda s:s['begin_marker'])
    markers=[s for s in spans if s['name'] not in {'Task','Core','Cluster'}]
    ops=[n for n in schedule['instructions'] if operation_kind(n)]
    assert len(ops)==len(markers),(len(ops),len(markers))
    assert [s['name'] for s in markers]==[operation_kind(n) for n in ops],[(i,s['name'],n['index'],operation_kind(n)) for i,(s,n) in enumerate(zip(markers,ops)) if s['name']!=operation_kind(n)]
    assert len({s['label'] for s in spans})==len(spans)
    task,=[s for s in spans if s['name']=='Task']
    absent = {'0': [], '1': ['97','107','331','374'] if len(ops)==76 else ['88','293','432','503']}
    return {'cluster_absent':absent,'task_label':task['label'],'spans':spans,'operations':[{'marker':s,'schedule':n} for s,n in zip(markers,ops)],'scope':'Full ordered native-kind/schedule-kind correspondence; no duration-based source attribution.'}

def package(label,target):
    assert re.fullmatch(r'[a-z0-9-]{1,70}',label)
    sides,control,origin=bindings();out=ARTIFACTS/label;out.mkdir(parents=True,exist_ok=False)
    evidence=HERE/'evidence'/label;evidence.mkdir(parents=True,exist_ok=False)
    diag=out/'diagnostic';diag.mkdir()
    record={'label':label,'status':'building','control':control,'origin':origin,'adopted':False,'official_submission':False,'target_dir':target,'commands':[],'sides':{},'tool_sources':{n:file_record(HERE/n) for n in ['package.py','host_transform.py','image_profile.py']}}
    def run(cmd,log=None):
        record['commands'].append(cmd)
        if log:
            with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        else:subprocess.run(cmd,stdout=subprocess.DEVNULL,check=True)
    try:
        # Preserve original whole-stage bytes, never replace them with diagnostic timings.
        for side,info in sides.items():
            shutil.copytree(info['package'],out/(side+'-standard'))
            b=info['build'];sb=info['schedule_build'];source=info['package'].parent/'crate';crate=out/side/'crate';crate.mkdir(parents=True)
            for name,expected in b['source_inputs'].items():
                raw=(source/name).read_bytes();assert digest(raw)==expected
                p=crate/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            hp=crate/'src/bin/test_kernels.rs';original=hp.read_text();hp.write_text(host_source(original))
            inputs={n:digest((crate/n).read_bytes()) for n in b['source_inputs']}
            assert {n:v for n,v in inputs.items() if n!='src/bin/test_kernels.rs'}=={n:v for n,v in b['source_inputs'].items() if n!='src/bin/test_kernels.rs'}
            e=evidence/side;e.mkdir();shutil.copyfile(hp,e/'test_kernels.rs')
            shutil.copyfile(source/'src/device/sliding/projection.rs',e/'production-projection.rs')
            for name in ['schedule.json','graph.json']:
                p=ROOT/sb['frozen_files'][name];assert digest(p.read_bytes())==sb[name[:-5]];shutil.copyfile(p,e/name)
            ff=(ROOT/info['images'][FFN]['path']).read_bytes();assert digest(ff)==sb['exact_image']
            labelled,meta=label_image(ff);mapping=map_markers(read(e/'schedule.json'),meta)
            write(diag/f'{side}-qkv.image-profile.json',meta);write(diag/f'{side}-marker-map.json',mapping)
            remote='/tmp/moa-e143-readiness-profile/'+label+'/'+side
            run(['docker','exec','moa2026-dev','mkdir','-p',remote]);run(['docker','cp',str(crate)+'/.','moa2026-dev:'+remote])
            rawtar=subprocess.check_output(['docker','exec','--workdir',remote,'moa2026-dev','tar','-cf','-',*inputs])
            with tarfile.open(fileobj=io.BytesIO(rawtar)) as tar:assert {m.name:digest(tar.extractfile(m).read()) for m in tar if m.isfile()}==inputs
            env=['docker','exec','--workdir',remote,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+target,'-e','FURIOSA_OPT_OUT_DIR='+remote+'/kernels','moa2026-dev']
            print(f'{side}: building diagnostic host; {len(mapping["operations"])} mapped QKV operations',flush=True)
            run(env+['cargo','clean','--release','-p','furiosa-opt-gemma4'])
            run(env+['cargo','furiosa-opt','build','--release','--locked','--bin','test_kernels'],e/'host-build.log')
            run(['docker','cp','moa2026-dev:'+target+'/release/test_kernels',str(out/side/'unlabelled-host')])
            run(['docker','cp','moa2026-dev:'+remote+'/kernels/furiosa-opt-gemma4',str(out/side/'kernels')])
            host=(out/side/'unlabelled-host').read_bytes();images={}
            for kernel in KERNELS:
                p=out/side/'kernels'/f'furiosa_opt_gemma4::ops::{kernel}.bin';raw=p.read_bytes();expected=ROOT/info['images'][kernel]['path']
                assert raw==expected.read_bytes() and host.count(raw)==1,kernel
                images[kernel]=file_record(p)
            probe=host.replace(ff,labelled);assert len(probe)==len(host) and probe.count(labelled)==1
            (diag/(side+'-probe')).write_bytes(probe);(diag/(side+'-probe')).chmod(0o755)
            record['sides'][side]={'source_inputs':b['source_inputs'],'diagnostic_inputs':inputs,'original_host_source':digest(original.encode()),'diagnostic_host_source':file_record(e/'test_kernels.rs'),'unlabelled_host':file_record(out/side/'unlabelled-host'),'images':images,'standard_files':{p.name:digest(p.read_bytes()) for p in (out/(side+'-standard')).iterdir()},'diagnostic_probe':file_record(diag/(side+'-probe')),'operations':len(mapping['operations']),'schedule':file_record(e/'schedule.json'),'graph':file_record(e/'graph.json'),'log':file_record(e/'host-build.log')}
            print(side+': all3 images match original; QKV names-only patch PASS',flush=True)
        for side in sides:
            for suffix in ['marker-map.json','qkv.image-profile.json']:shutil.copyfile(diag/f'{side}-{suffix}',evidence/f'{side}-{suffix}')
        record.update(status='ready',profile_name_bytes_only=True,task_bytes_unchanged=True,fixture_and_runner_match_e142=True,diagnostic_files={p.name:digest(p.read_bytes()) for p in diag.iterdir()},scope='Complete E142 and frozen E138v2 QKV; code/constants/slots/IO/markers preserved. Each diagnostic keeps its original three device images and source bundle; candidate FFN differs and is not compared. Unchanged standard hosts remain separate performance controls.')
    except Exception as err:
        record.update(status='failed',error=repr(err));raise
    finally:write(HERE/f'package-{label}.json',record)
    print('Profile package ready; no Arena submission',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--target-dir',default='/tmp/moa-c09-validation-probe/target');a=p.parse_args();package(a.label,a.target_dir)
