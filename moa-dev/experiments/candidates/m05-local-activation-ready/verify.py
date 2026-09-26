#!/usr/bin/env python3
"""Audit exact composition, reused companion, uploaded package and screen logs."""
import argparse,json,re,subprocess,sys,tarfile
from analyze import HERE,ROOT,digest,analyze
LABEL='m05-local-prepare-screen';BUILD='m05-local-prepare-v1'
def check():
    recipe=json.loads((HERE/'local-prepare-v1/recipe.json').read_text());parts=[]
    for c in recipe['source_components']:
        raw=subprocess.check_output(['git','show',c['commit']+':'+c['path']],cwd=ROOT)
        assert digest(raw)['sha256']==c['sha256'];parts.append(raw.decode())
    a,b=parts;candidate=(HERE/'local-prepare-v1/mlp.rs').read_text()
    expected=a[:a.index('type FfnQuantSlices')]+b[b.index('type FfnQuantSlices'):]
    expected=expected.replace('// M03: C14-B ownership with one L stage, materialized native up/gate and F32 cluster reduction.','// M05: M03 local FFN ownership plus M04 producer-local activation preparation; full weight streams.')
    expected=expected.replace('    // Quantize the actual runtime BF16 activation; both projections reuse it.\n    let x: DmTensor<bf16, Chip, m![2], Slice, m![H]> = x.to_dm(&mut device.tdma);\n','    // Prepare in the pre-RMS producing cluster, then replicate prepared terms.\n    // No small cross-cluster activation DMA is required before amax.\n')
    assert expected==candidate
    build=json.loads((HERE/f'build-{BUILD}-repro.json').read_text());plan=build['source_plan'];sys.path.insert(0,str(ROOT/'moa-dev/experiments'))
    from structural_build import source_manifest
    inputs=source_manifest(recipe['base_commit']);assert inputs==plan['source_origin_inputs']
    for dest,value in plan['source_overrides'].items():
        inputs[dest]=digest((ROOT/value['path']).read_bytes());assert inputs[dest]=={k:value[k] for k in ['bytes','sha256']}
    assert inputs==build['source_inputs'];assert build['exact_compile_exit_code']==build['host_build_exit_code']==0
    assert digest((ROOT/plan['recipe']['path']).read_bytes())=={k:plan['recipe'][k] for k in ['bytes','sha256']}
    for name,path in build['frozen_files'].items():assert digest((ROOT/path).read_bytes())==build['source_inputs'].get(name,build['logs'].get(name))
    art=ROOT/'moa-dev/experiments/artifacts'/HERE.name;out=art/BUILD
    assert digest((out/'schedule.json').read_bytes())==build['schedule']==digest((HERE/'evidence'/BUILD/'schedule.json').read_bytes())
    assert digest((out/'exact-kernel.bin').read_bytes())==build['exact_image']
    assert {p.name:digest(p.read_bytes()) for p in (out/'package').iterdir()}==build['package_files']
    reused=json.loads((HERE/'build-m03-reference-repro.json').read_text());info=reused.pop('reused_build_record')
    assert digest((ROOT/info['path']).read_bytes())['sha256']==info['sha256'];assert reused==json.loads((ROOT/info['path']).read_text())
    assert {p.name:digest(p.read_bytes()) for p in (art/'m03-reference/package').iterdir()}==reused['package_files']
    assert {k:v for k,v in inputs.items() if k!='src/device/shared/mlp.rs'}=={k:v for k,v in reused['source_inputs'].items() if k!='src/device/shared/mlp.rs'}
    host=(out/'package/test_runtime').read_bytes()
    for name,value in build['images'].items():
        raw=(out/'kernels'/f'furiosa_opt_gemma4::ops::{name}.bin').read_bytes()
        assert digest(raw)=={k:value[k] for k in ['bytes','sha256']};assert host.count(raw)==value['host_occurrences']==1
        if name==build['target_kernel']:assert digest(raw)==build['exact_image']
        else:assert value['sha256']==reused['images'][name]['sha256']
    assert json.loads((HERE/'static-analysis.json').read_text())==analyze()
    result=json.loads((HERE/f'{LABEL}-results.json').read_text());manifest=json.loads((HERE/f'{LABEL}-build.json').read_text())
    raw=(HERE/f"evidence/arena-{result['job']['id']}.log").read_bytes();assert digest(raw)['sha256']==result['log_sha256']
    with tarfile.open(art/f"job-{result['job']['id']}.tar") as archive:files={m.name:digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
    assert files==result['package_files'];assert {k:v for k,v in files.items() if k!='job.yaml'}==manifest['package_files']
    sys.path.insert(0,str(ROOT/'moa-dev'));from measure import parse_log
    sections=re.findall(r'^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$',raw.decode(),re.M|re.S)
    assert [k for k,_ in sections]==[s['name'] for s in manifest['stages']];assert raw.count(b'STRUCTURAL_COMPLETE')==1
    for name,log in sections:assert f'STRUCTURAL_EXIT {name} 0' in log;assert parse_log(log)==result['stages'][name]['kernels']
    assert result['valid'] and result['job']['status']=='SUCCEEDED'
    return {'component_hashes_and_exact_composition':True,'m03_companion_original_package_reused':True,'source_build_images_upload_logs_verified':True,'schedule_analysis_recomputed':True,'qkv_o_unchanged':True,'job':result['job']['id'],'basic_comparisons_passed':sum(s['comparisons_passed'] for s in result['stages'].values()),'candidate_ffn_inputs':3,'candidate_ffn_elements':11520,'scope':'One screen; static input readiness only, no measured internal NPU interval, no independent repetitions or expanded accuracy.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=check();out=HERE/'verification.json'
    if args.check:assert json.loads(out.read_text())==r
    else:out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
