#!/usr/bin/env python3
"""Exact E137/E138 native-v1 standard hosts plus names-only QKV profile hosts."""
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
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]; ARTIFACTS=ROOT/'moa-dev/experiments/artifacts'/HERE.parent.name/'profile'
sys.path.insert(0,str(HERE))
from image_profile import digest,inspect
from host_transform import host_source
FFN='sliding_project_qkv';KERNELS=(FFN,'sliding_attention_output','decoder_feedforward')
BUILD='e138-native-v1'

def read(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def file_record(p):return {'path':str(p.relative_to(ROOT)),**digest(p.read_bytes())}
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def bindings():
    old=module('e137_frozen_binding',HERE.parent/'package.py')
    pp,pkg,candidate,control=old.checked_package(BUILD)
    recipe,cb,_,cr=old.checked_control()
    bp,built,_,_,_=old.checked_compile(BUILD)
    cp=HERE.parent.parent/'e104-qkv-output-profile/build-profile-v1.json';cs0=read(cp)
    control_qkv=(ROOT/cb['standard_image_files'][FFN]['path']).read_bytes()
    assert digest(control_qkv)==cs0['images'][FFN]
    # Exact device-image identity binds this old QKV schedule; current E137
    # standard/diagnostic hosts retain E137's O and FFN images and source.
    schedule_path=cp.parent/'evidence/profile-v1/sliding_project_qkv.json'
    ir_path=cp.parent/'evidence/profile-v1/sliding_project_qkv.ir.txt'
    assert digest(schedule_path.read_bytes())==cs0['dumps'][schedule_path.name]
    assert digest(ir_path.read_bytes())==cs0['dumps'][ir_path.name]
    cs={'exact_image':cs0['images'][FFN], 'schedule':cs0['dumps'][schedule_path.name],
        'graph':cs0['dumps'][ir_path.name], 'frozen_files':{'schedule.json':str(schedule_path.relative_to(ROOT)),
        'graph.json':str(ir_path.relative_to(ROOT))}, 'graph_format':'original compiler textual IR, not JSON graph'}
    return {'control':{'build':cb,'schedule_build':cs,'package':control,'images':cb['standard_image_files']},
            'candidate':{'build':built,'schedule_build':built,'package':candidate,'images':pkg['image_files']}},cr,{'build':file_record(bp),'package':file_record(pp),'graph_gate':pkg['graph_gate'],'control_schedule_build':file_record(cp),'control_schedule_qkv_image_byte_identical':True,'control_ir_format':cs['graph_format']}

def checked_control():
    sides,cr,_=bindings();return read(ROOT/'baseline/e137.json'),sides['control']['build'],sides['control']['package'],cr

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
    absent = {'0': [], '1': ['97','107','331','374'] if len(ops)==76 else ['158','284','424','494']}
    return {'cluster_absent':absent,'task_label':task['label'],'spans':spans,'operations':[{'marker':s,'schedule':n} for s,n in zip(markers,ops)],'scope':'Full ordered native-kind/schedule-kind correspondence; no duration-based source attribution.'}

def package(label,target):
    assert re.fullmatch(r'[a-z0-9-]{1,70}',label)
    sides,control,origin=bindings();out=ARTIFACTS/label;out.mkdir(parents=True,exist_ok=False)
    evidence=HERE/'evidence'/label;evidence.mkdir(parents=True,exist_ok=False)
    diag=out/'diagnostic';diag.mkdir()
    record={'label':label,'status':'building','control':control,'origin':origin,'adopted':False,'official_submission':False,'target_dir':target,'commands':[],'sides':{},'tool_sources':{n:file_record(HERE/n) for n in ['package_v2.py','host_transform.py','image_profile.py']}}
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
            # Reuse exactly verified diagnostic builds; only pre-NPU visibility
            # classification changes. No source, host or device bytes are rebuilt.
            prior=read(HERE/'package-profile-v1.json');assert prior['status']=='ready'
            previous=prior['sides'][side]
            assert previous['diagnostic_inputs']==inputs and previous['source_inputs']==b['source_inputs']
            oldhost=ROOT/previous['unlabelled_host']['path'];assert file_record(oldhost)==previous['unlabelled_host']
            shutil.copyfile(oldhost,out/side/'unlabelled-host')
            shutil.copytree(oldhost.parent/'kernels',out/side/'kernels')
            oldlog=ROOT/previous['log']['path'];assert file_record(oldlog)==previous['log'];shutil.copyfile(oldlog,e/'host-build.log')
            record['reused_host_build']=file_record(HERE/'package-profile-v1.json')
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
        record.update(status='ready',profile_name_bytes_only=True,task_bytes_unchanged=True,fixture_and_runner_match_e137=True,diagnostic_files={p.name:digest(p.read_bytes()) for p in diag.iterdir()},scope='Complete E137 and frozen E138v1 QKV; code/constants/slots/IO/markers preserved. Same transformed host source except frozen device source. Unchanged standard hosts remain separate performance controls.')
    except Exception as err:
        record.update(status='failed',error=repr(err));raise
    finally:write(HERE/f'package-{label}.json',record)
    print('Profile package ready; no Arena submission',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--target-dir',default='/tmp/moa-c09-validation-probe/target');a=p.parse_args();package(a.label,a.target_dir)
