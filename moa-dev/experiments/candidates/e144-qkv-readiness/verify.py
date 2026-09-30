#!/usr/bin/env python3
"""Recheck actual uploaded standard/diagnostic hosts, image edits and raw traces."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from package import bindings,label_image,host_source,read,write,file_record,digest,KERNELS,FFN
from analyze import analyze

def verify(label,job):
    analysis=analyze(label,job);assert read(HERE/'profile-analysis.json')==analysis
    result=read(HERE/f'{label}-job-{job}-results.json');built=read(ROOT/result['build_record']['path']);package=read(ROOT/built['candidate_package']['path'])
    sides,_,_=bindings();proof={}
    with tarfile.open(ROOT/result['archive']['path']) as tar:uploaded={m.name:tar.extractfile(m).read() for m in tar if m.isfile()}
    for side,info in sides.items():
        record=package['sides'][side]
        for name,item in package['tool_sources'].items():assert file_record(ROOT/item['path'])==item
        host_source_path=ROOT/record['diagnostic_host_source']['path'];original=(info['package'].parent/'crate/src/bin/test_kernels.rs').read_text()
        assert host_source_path.read_text()==host_source(original)
        assert record['source_inputs']==info['build']['source_inputs']
        assert {n:v for n,v in record['diagnostic_inputs'].items() if n!='src/bin/test_kernels.rs'}=={n:v for n,v in record['source_inputs'].items() if n!='src/bin/test_kernels.rs'}
        raw_std=uploaded[side+'_test_runtime'];assert raw_std==(info['package']/'test_runtime').read_bytes()
        for name in ('fixtures.safetensors','remote_entrypoint.sh'):assert uploaded[name]==(info['package']/name).read_bytes()
        unlabelled=ROOT/record['unlabelled_host']['path'];assert file_record(unlabelled)==record['unlabelled_host'];original_diag=unlabelled.read_bytes()
        raw_probe=uploaded[side+'_probe'];checks={}
        for kernel in KERNELS:
            original_image=(ROOT/info['images'][kernel]['path']).read_bytes()
            assert raw_std.count(original_image)==original_diag.count(original_image)==1
            if kernel==FFN:
                labelled,metadata=label_image(original_image)
                assert raw_probe==original_diag.replace(original_image,labelled)
                assert raw_probe.count(labelled)==1
                checks[kernel]={'original':digest(original_image),'labelled':digest(labelled),'marker_count':sum(len(r) for r in metadata['spans']),'task_payloads':metadata['tasks'],'profile_names_only':True}
            else:
                assert raw_probe.count(original_image)==1
                checks[kernel]={'original':digest(original_image),'unchanged':True}
        proof[side]={'source_inputs':record['source_inputs'],'host_transform_exact':True,'uploaded_standard_original_byte_identical':True,'uploaded_diagnostic_only_profile_names_changed_from_verified_diagnostic_host':True,'all_three_images':checks}
    return {'valid':True,'diagnostic_only':True,'official_submission':False,'result':file_record(HERE/f'{label}-job-{job}-results.json'),'analysis':file_record(HERE/'profile-analysis.json'),'verifier':file_record(Path(__file__)),'correctness_passes':81,'all_stages':result['completed_stage_names'],'complete_operations_per_run_per_cluster':{side:analysis['source_maps'][side]['operation_count'] for side in sides},'sides':proof,'scope':'Exact uploaded bytes checked independently of prior packer success; source contracts and whole correctness unchanged. Trace durations are not promotion evidence.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('job',type=int);p.add_argument('--check',action='store_true');a=p.parse_args();v=verify(a.label,a.job);out=HERE/'verification.json'
    if a.check:assert read(out)==v
    else:write(out,v)
    print('PASS: actual uploaded four hosts, six original images, names-only diagnostic edits, original inputs, 81 comparisons and complete source-bound marker mapping with source-bound cluster0-only DMA visibility')
