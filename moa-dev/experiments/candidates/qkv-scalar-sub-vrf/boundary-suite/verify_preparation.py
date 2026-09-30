#!/usr/bin/env python3
"""Offline immutable-test/source binding verification; no compile or remote request."""
import argparse, ast, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]
def record(p):
    raw=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def check_preparation(require_selected=False):
    p=json.loads((HERE/'preparation.json').read_text())
    assert p['status']=='PREPARED_NOT_BUILT_NOT_TESTED'
    for item in p['immutable_reuse']:
        src=ROOT/item['origin']['path'];dst=ROOT/item['copy']['path']
        assert record(src)==item['origin'] and record(dst)==item['copy'] and src.read_bytes()==dst.read_bytes()
    for key in ['dependencies','prepared_tools','builder_origins']:
        for item in p[key]:
            assert record(ROOT/item['path'])==item
            if item['path'].endswith('.py'):ast.parse((ROOT/item['path']).read_text())
    cpu=json.loads((ROOT/'moa-dev/experiments/candidates/qkv-boundary-probe-e53/cpu-references.json').read_text())
    fixture=record(HERE/'fixtures/boundary.safetensors')
    assert {k:fixture[k] for k in ['bytes','sha256']}==cpu['payload']
    assert (cpu['cases'],cpu['atol'],cpu['rtol'])==(2,.04,.01)
    recipe=json.loads((HERE/'raw-src/recipe.json').read_text());assert (recipe['cases'],recipe['outputs'])==(9,73728)
    assert not any(k in recipe for k in ['production_source_sha256','production_build_label'])
    path=HERE/'production-selection.json';label=None
    if require_selected:assert path.exists(), 'final variant not selected; await parent signal'
    if path.exists():
        chosen=json.loads(path.read_text());assert chosen['status']=='SELECTED_FOR_BOUNDARY_VALIDATION'
        for key in ['build','projection','ops','exact_image_file','gate','selector']:
            assert record(ROOT/chosen[key]['path'])==chosen[key]
        b=json.loads((ROOT/chosen['build']['path']).read_text())
        assert b['exact_compile_exit_code']==0 and b['source_inputs_verified_in_container']
        assert b['label']==chosen['label'] and b['exact_image']==chosen['exact_image']
        assert len(b['source_inputs'])==46
        for key,name in [('projection','src/device/sliding/projection.rs'),('ops','src/ops.rs')]:
            assert b['source_inputs'][name]=={k:chosen[key][k] for k in ('bytes','sha256')}
        assert {k:chosen['exact_image_file'][k] for k in ('bytes','sha256')}==b['exact_image']
        assert json.loads((ROOT/chosen['gate']['path']).read_text())['passed']
        label=chosen['label']
    return {'status':p['status'],'immutable_reuse_files':len(p['immutable_reuse']),'raw_cases':9,'raw_outputs':73728,'whole_cases':2,'production_build_selected':label is not None,'production_build_label':label,'build_or_npu_executed_by_preparation':False}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--require-selected',action='store_true');a=p.parse_args()
    print(json.dumps(check_preparation(a.require_selected),indent=2))
