#!/usr/bin/env python3
"""Verify frozen test reuse and preparation only; never build or contact Arena."""
import argparse, ast, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
def record(path):
    raw=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
def check_preparation():
    p=json.loads((HERE/'preparation.json').read_text())
    assert p['status']=='PREPARED_NOT_BUILT_NOT_TESTED'
    for item in p['immutable_reuse']:
        src=ROOT/item['origin']['path'];dst=ROOT/item['copy']['path']
        assert record(src)==item['origin'] and record(dst)==item['copy']
        assert src.read_bytes()==dst.read_bytes()
    for item in p['dependencies']:
        assert record(ROOT/item['path'])==item
    for item in p['prepared_tools']:
        assert record(ROOT/item['path'])==item
        if item['path'].endswith('.py'):ast.parse((ROOT/item['path']).read_text())
    chosen=p['selected_production']
    for key in ['build','projection','ops']:assert record(ROOT/chosen[key]['path'])==chosen[key]
    b=json.loads((ROOT/chosen['build']['path']).read_text())
    assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container']
    assert b['label']==chosen['label'] and b['exact_image']==chosen['exact_image']
    for key,name in [('projection','src/device/sliding/projection.rs'),('ops','src/ops.rs')]:
        assert b['source_inputs'][name]=={k:chosen[key][k] for k in ('bytes','sha256')}
    for name in ['probe.rs','probe_qkv_raw.rs','generate.py']:
        assert (HERE/'raw-src'/name).read_bytes()==(ROOT/'moa-dev/experiments/candidates/qkv-native-consumer-local/boundary-suite-v2/raw-src'/name).read_bytes()
    legacy=ROOT/'moa-dev/experiments/candidates/qkv-boundary-probe-e53'
    cpu=json.loads((legacy/'cpu-references.json').read_text())
    f=record(HERE/'fixtures/boundary.safetensors')
    assert {k:f[k] for k in ['bytes','sha256']}==cpu['payload']
    assert (cpu['cases'],cpu['atol'],cpu['rtol'])==(2,.04,.01)
    r=json.loads((HERE/'raw-src/recipe.json').read_text())
    assert (r['cases'],r['outputs'])==(9,73728)
    assert not any(k in r for k in ['production_source_sha256','production_build_label'])
    return {'status':p['status'],'immutable_reuse_files':len(p['immutable_reuse']),'raw_cases':9,'raw_outputs':73728,'whole_cases':2,'production_build_selected':True,'production_build_label':chosen['label'],'build_or_npu_executed_by_preparation':False}
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--check',action='store_true');a.parse_args()
    print(json.dumps(check_preparation(),indent=2))
