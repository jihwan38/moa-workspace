#!/usr/bin/env python3
"""Verify actual uploaded standard/probe bytes and the one source-bound FFN trace."""
import argparse,json,subprocess,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE))
from package import bindings,checked_control,label_image,host_source,read,write,file_record,digest,KERNELS,FFN
from analyze import analyze

def verify(label,job):
    analysis=analyze(label,job);assert read(HERE/'profile-analysis.json')==analysis
    subprocess.run([sys.executable,str(HERE/'independent-review.py'),'--check'],check=True,stdout=subprocess.DEVNULL)
    result=read(HERE/f'{label}-job-{job}-results.json');built=read(ROOT/result['build_record']['path']);package=read(ROOT/built['candidate_package']['path'])
    sides,_,_=bindings();proof={}
    with tarfile.open(ROOT/result['archive']['path']) as tar:uploaded={m.name:tar.extractfile(m).read() for m in tar if m.isfile()}
    _,control_build,control_dir,control_record=checked_control()
    assert control_record==package['control']==built['control']
    assert uploaded['control_test_runtime']==(control_dir/'test_runtime').read_bytes()
    for name in ('fixtures.safetensors','remote_entrypoint.sh'):assert uploaded[name]==(control_dir/name).read_bytes()
    for kernel,image in control_record['images'].items():
        raw=(ROOT/control_build['standard_image_files'][kernel]['path']).read_bytes();assert digest(raw)=={k:image[k] for k in ('bytes','sha256')} and uploaded['control_test_runtime'].count(raw)==1
    for side,info in sides.items():
        record=package['sides'][side]
        for item in package['tool_sources'].values():assert file_record(ROOT/item['path'])==item
        source=ROOT/record['diagnostic_host_source']['path'];original=(info['package'].parent/'crate/src/bin/test_kernels.rs').read_text()
        assert source.read_text()==host_source(original)
        assert record['source_inputs']==info['build']['source_inputs'] and len(record['source_inputs'])==46
        assert {n:v for n,v in record['diagnostic_inputs'].items() if n!='src/bin/test_kernels.rs'}=={n:v for n,v in record['source_inputs'].items() if n!='src/bin/test_kernels.rs'}
        std=uploaded[side+'_test_runtime'];assert std==(info['package']/'test_runtime').read_bytes()
        for name in ('fixtures.safetensors','remote_entrypoint.sh'):assert uploaded[name]==(info['package']/name).read_bytes()
        unlabelled=ROOT/record['unlabelled_host']['path'];assert file_record(unlabelled)==record['unlabelled_host'];host=unlabelled.read_bytes();probe=uploaded[side+'_probe'];checks={}
        for kernel in KERNELS:
            image=(ROOT/info['images'][kernel]['path']).read_bytes();assert std.count(image)==host.count(image)==1
            if kernel==FFN:
                labelled,metadata=label_image(image);assert probe==host.replace(image,labelled) and probe.count(labelled)==1
                checks[kernel]={'original':digest(image),'labelled':digest(labelled),'marker_count':sum(len(r) for r in metadata['spans']),'task_payloads':metadata['tasks'],'profile_names_only':True}
            else:
                assert probe.count(image)==1;checks[kernel]={'original':digest(image),'unchanged':True}
        proof[side]={'source_inputs':record['source_inputs'],'host_transform_exact':True,'uploaded_standard_original_byte_identical':True,'uploaded_diagnostic_only_profile_names_changed':True,'all_three_images':checks}
    return {'valid':True,'diagnostic_only':True,'official_submission':False,'result':file_record(HERE/f'{label}-job-{job}-results.json'),'analysis':file_record(HERE/'profile-analysis.json'),'independent_review':file_record(HERE/'independent-review.json'),'verifier':file_record(Path(__file__)),'correctness_passes':51,'all_stages':result['completed_stage_names'],'complete_operations_per_run_per_cluster':analysis['source_map']['operation_count'],'control':{'recipe':control_record['recipe'],'source_inputs':control_build['source_inputs'],'actual_standard_host_uploaded':True,'images':control_record['images']},'sides':proof,'scope':'Uploaded original E147/E150 three-image standard hosts and one E150 diagnostic host verified. Five serial stages, three original inputs, one allocation. No contemporary E147 internal trace; no pure-service or additive savings attribution.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('job',type=int);p.add_argument('--check',action='store_true');a=p.parse_args();v=verify(a.label,a.job);out=HERE/'verification.json'
    if a.check:assert read(out)==v
    else:write(out,v)
    print('PASS: actual uploaded hosts/images/source46, names-only edits, 51 comparisons and complete 142-operation FFN trace')
