#!/usr/bin/env python3
"""Exact E147/E148 standard hosts plus names-only O profile hosts."""
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
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]; ARTIFACTS=ROOT/'moa-dev/experiments/artifacts'/HERE.parent.name/'readiness'
sys.path.insert(0,str(HERE))
from image_profile import digest,inspect
from host_transform import host_source
FFN='sliding_attention_output';KERNELS=('sliding_project_qkv',FFN,'decoder_feedforward')
BUILD='e148-direct-v1'

def read(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def file_record(p):return {'path':str(p.relative_to(ROOT)),**digest(p.read_bytes())}
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def bindings():
    old=module('e148_frozen_binding',HERE.parent/'package.py')
    pp,pkg,candidate,control=old.checked_package(BUILD)
    recipe,cb,_,cr=old.checked_control()
    bp,built,_,_,_=old.checked_compile(BUILD)
    cp=HERE.parent.parent/'o-native-activation-resident/build-e137-native-v3-repro.json';cs=read(cp)
    control_o=(ROOT/cb['standard_image_files'][FFN]['path']).read_bytes()
    assert digest(control_o)==cs['exact_image']
    return {'control':{'build':cb,'schedule_build':cs,'package':control,'images':cb['standard_image_files']},
            'candidate':{'build':built,'schedule_build':built,'package':candidate,'images':pkg['image_files']}},cr,{'build':file_record(bp),'package':file_record(pp),'graph_gate':pkg['graph_gate'],'control_schedule_build':file_record(cp),'control_schedule_o_image_byte_identical':True}

def checked_control():
    sides,cr,_=bindings();return read(ROOT/'baseline/e147.json'),sides['control']['build'],sides['control']['package'],cr

def operation_kind(n):
    if n['tpe'].startswith('Dma'):return 'DMA'
    if n['tpe']=='Main':return 'Renegade::TuExec'
    if n['tpe']=='Sub' and n['contexts']!=['Noop']:
        d=n['description']
        if '.to_trf()' in d:return 'Renegade::StoTrf'
        if not d.strip():return 'Renegade::StoTab'
        # This coefficient fetch widens four live lanes to eight physical lanes.
        # Actual image marker is TuExec; a padded collect is not a direct StoVrf.
        if '.to_vrf()' in d and '.collect::<m![1], m![OutputTerm # 8]>()' in d:
            assert '.fetch::<m![1], m![OutputTerm]>()' in d
            return 'Renegade::TuExec'
        if '.to_vrf()' in d and not any(x in d for x in ['fetch_cast','vector_','.switch']):return 'Renegade::StoVrf'
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
    return {'task_label':task['label'],'spans':spans,'operations':[{'marker':s,'schedule':n} for s,n in zip(markers,ops)],'scope':'Full ordered native-kind/schedule-kind correspondence; no duration-based source attribution.'}

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
            shutil.copyfile(ROOT/sb['frozen_files']['src/device/sliding/projection.rs'], e/'schedule-projection.rs')
            for name in ['schedule.json','graph.json']:
                p=ROOT/sb['frozen_files'][name];assert digest(p.read_bytes())==sb[name[:-5]];shutil.copyfile(p,e/name)
            ff=(ROOT/info['images'][FFN]['path']).read_bytes();assert digest(ff)==sb['exact_image']
            labelled,meta=label_image(ff);mapping=map_markers(read(e/'schedule.json'),meta)
            write(diag/f'{side}-o.image-profile.json',meta);write(diag/f'{side}-marker-map.json',mapping)
            remote='/tmp/moa-e148-readiness/'+label+'/'+side
            run(['docker','exec','moa2026-dev','mkdir','-p',remote]);run(['docker','cp',str(crate)+'/.','moa2026-dev:'+remote])
            rawtar=subprocess.check_output(['docker','exec','--workdir',remote,'moa2026-dev','tar','-cf','-',*inputs])
            with tarfile.open(fileobj=io.BytesIO(rawtar)) as tar:assert {m.name:digest(tar.extractfile(m).read()) for m in tar if m.isfile()}==inputs
            env=['docker','exec','--workdir',remote,'-e','CARGO_HOME=/home/moa/.cargo','-e','CARGO_NET_OFFLINE=true','-e','CARGO_TARGET_DIR='+target,'-e','FURIOSA_OPT_OUT_DIR='+remote+'/kernels','moa2026-dev']
            print(f'{side}: building diagnostic host; {len(mapping["operations"])} mapped O operations',flush=True)
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
            if side=='control':
                fbpath=HERE.parent.parent/'ffn-direct-full30-partials/build-e142-full30-v1-repro.json';fb=read(fbpath)
                fk='decoder_feedforward';ffraw=(ROOT/info['images'][fk]['path']).read_bytes()
                assert digest(ffraw)==fb['exact_image']
                assert b['source_inputs']['src/device/shared/mlp.rs']==fb['source_inputs']['src/device/shared/mlp.rs']
                flabelled,fmeta=label_image(ffraw)
                fe=evidence/'ffn';fe.mkdir()
                for fn in ['graph.json','schedule.json']:
                    fsrc=ROOT/fb['frozen_files'][fn];assert digest(fsrc.read_bytes())==fb[fn[:-5]];shutil.copyfile(fsrc,fe/fn)
                shutil.copyfile(ROOT/fb['frozen_files']['src/device/shared/mlp.rs'],fe/'production-mlp.rs')
                fmap=map_markers(read(fe/'schedule.json'),fmeta)
                write(diag/'ffn-marker-map.json',fmap);write(diag/'ffn.image-profile.json',fmeta)
                fprobe=host.replace(ffraw,flabelled);assert len(fprobe)==len(host) and fprobe.count(flabelled)==1
                (diag/'ffn-probe').write_bytes(fprobe);(diag/'ffn-probe').chmod(0o755)
                record['ffn']={'build':file_record(fbpath),'schedule':file_record(fe/'schedule.json'),'graph':file_record(fe/'graph.json'),'source':file_record(fe/'production-mlp.rs'),'image':info['images'][fk],'operations':len(fmap['operations']),'probe':file_record(diag/'ffn-probe'),'scope':'E147 FFN source/image identical to E142; same control host, only FFN marker names patched.'}
            print(side+': all3 images match original; O names-only patch PASS',flush=True)
        for side in sides:
            for suffix in ['marker-map.json','o.image-profile.json']:shutil.copyfile(diag/f'{side}-{suffix}',evidence/f'{side}-{suffix}')
        for fn in ['ffn-marker-map.json','ffn.image-profile.json']:shutil.copyfile(diag/fn,evidence/fn)
        record.update(status='ready',profile_name_bytes_only=True,task_bytes_unchanged=True,fixture_and_runner_match_e147=True,diagnostic_files={p.name:digest(p.read_bytes()) for p in diag.iterdir()},scope='Complete E147 and frozen E148 O plus unchanged E142 FFN; code/constants/slots/IO/markers preserved. Same transformed host source except frozen device source. Unchanged standard hosts remain separate performance controls.')
    except Exception as err:
        record.update(status='failed',error=repr(err));raise
    finally:write(HERE/f'package-{label}.json',record)
    print('Profile package ready; no Arena submission',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--target-dir',default='/tmp/moa-c09-validation-probe/target');a=p.parse_args();package(a.label,a.target_dir)
