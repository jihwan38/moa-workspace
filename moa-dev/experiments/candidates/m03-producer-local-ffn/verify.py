#!/usr/bin/env python3
"""Recheck source composition, complete images and actual uploaded NPU evidence."""
import argparse,json,re,subprocess,sys,tarfile
from pathlib import Path
from analyze import HERE,ROOT,FAMILY,BUILD,digest,result
LABEL='m03-native-local-screen'
def section(s,a,b):
    i=s.index(a);return s[i:s.index(b,i)]
def check():
    here=HERE/'native-local-v1';recipe=json.loads((here/'recipe.json').read_text())
    source=(here/'mlp.rs').read_text()
    origin=recipe['source_components'][1]
    e104=subprocess.check_output(['git','show',origin['commit']+':'+origin['source']],cwd=ROOT,text=True)
    assert digest(e104.encode())['sha256']==origin['sha256']
    c14=(ROOT/recipe['source_components'][0]['source']).read_text()
    assert digest(c14.encode())['sha256']==recipe['source_components'][0]['sha256']
    assert section(source,'fn project_rows_with_scale_contraction(', 'pub(crate) fn project_up_and_gate(')==section(e104,'fn project_rows_with_scale_contraction(', 'pub(crate) fn project_up_and_gate(')
    assert source[source.index('type FfnQuantSlices'):]==e104[e104.index('type FfnQuantSlices'):]
    geglu=section(c14,'pub(crate) fn geglu(', 'fn project_down_tile(').replace('UpGateSlices','LocalRows').replace('TileL % 16','TileL % 120').replace('TileL / 8 % 2','TileL / 8 % 15').replace('TileL / 4 % 4','TileL / 4 % 30')
    assert geglu==section(source,'pub(crate) fn geglu(', 'fn project_down_tile(')
    assert section(source,'    let down_global_scale: DmTensor','type FfnQuantSlices')==section(e104,'    let down_global_scale: DmTensor','pub(crate) fn geglu(')
    build=json.loads((HERE/f'build-{BUILD}-repro.json').read_text());plan=build['source_plan']
    sys.path.insert(0,str(ROOT/'moa-dev/experiments'))
    from structural_build import source_manifest
    assert source_manifest(recipe['base_commit'])==plan['source_origin_inputs']
    expected=dict(plan['source_origin_inputs'])
    for dest,item in plan['source_overrides'].items():
        actual=digest((ROOT/item['path']).read_bytes());assert actual=={k:item[k] for k in ['bytes','sha256']}
        expected[dest]=actual
    assert expected==build['source_inputs']
    assert digest((ROOT/plan['recipe']['path']).read_bytes())=={k:plan['recipe'][k] for k in ['bytes','sha256']}
    for dest,path in build['frozen_files'].items():
        assert digest((ROOT/path).read_bytes())==build['source_inputs'].get(dest,build['logs'].get(dest))
    assert build['exact_compile_exit_code']==build['host_build_exit_code']==0
    artifacts=ROOT/'moa-dev/experiments/artifacts'/FAMILY/BUILD
    assert digest((artifacts/'schedule.json').read_bytes())==build['schedule']
    assert digest((artifacts/'exact-kernel.bin').read_bytes())==build['exact_image']
    package=artifacts/'package'
    assert {p.name:digest(p.read_bytes()) for p in package.iterdir()}==build['package_files']
    host=(package/'test_runtime').read_bytes()
    control=json.loads((HERE.parent/'c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json').read_text())
    for name,item in build['images'].items():
        image=(artifacts/'kernels'/f'furiosa_opt_gemma4::ops::{name}.bin').read_bytes()
        assert digest(image)=={k:item[k] for k in ['bytes','sha256']}
        assert host.count(image)==item['host_occurrences']==1
        if name==build['target_kernel']:assert digest(image)==build['exact_image']
        else:assert item['sha256']==control['images'][name]['sha256']
    assert {k:v for k,v in expected.items() if k!='src/device/shared/mlp.rs'}=={k:v for k,v in control['source_inputs'].items() if k!='src/device/shared/mlp.rs'}
    assert json.loads((HERE/'static-analysis.json').read_text())==result()
    assert digest((HERE/'evidence'/BUILD/'schedule.json').read_bytes())==build['schedule']
    screen=json.loads((HERE/f'{LABEL}-results.json').read_text())
    manifest=json.loads((HERE/f'{LABEL}-build.json').read_text())
    raw=(HERE/f"evidence/arena-{screen['job']['id']}.log").read_bytes()
    assert digest(raw)['sha256']==screen['log_sha256']
    with tarfile.open(artifacts.parent/f"job-{screen['job']['id']}.tar") as archive:
        files={m.name:digest(archive.extractfile(m).read()) for m in archive if m.isfile()}
    assert files==screen['package_files']
    assert {k:v for k,v in files.items() if k!='job.yaml'}==manifest['package_files']
    sys.path.insert(0,str(ROOT/'moa-dev'))
    from measure import parse_log
    sections=re.findall(r'^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$',raw.decode(),re.M|re.S)
    assert [n for n,_ in sections]==[s['name'] for s in manifest['stages']]
    assert raw.count(b'STRUCTURAL_COMPLETE')==1
    for name,log in sections:
        assert f'STRUCTURAL_EXIT {name} 0' in log
        assert parse_log(log)==screen['stages'][name]['kernels']
    assert screen['valid'] and screen['job']['status']=='SUCCEEDED'
    return {'source_origin':recipe['base_commit'],'method_source_hashes_verified':True,
            'quantization_and_native_upgate_arithmetic_unchanged':True,
            'geglu_arithmetic_preserved_with_explicit_layout_renaming':True,
            'non_mlp_sources_and_qkv_o_images_equal_e104':True,
            'schedule_and_coordinate_analysis_recomputed':True,
            'compiled_source_and_frozen_logs_verified':True,
            'host_image_and_actual_upload_hashes_verified':True,
            'job':screen['job']['id'],'comparisons_passed':sum(s['comparisons_passed'] for s in screen['stages'].values()),
            'candidate_ffn_inputs':3,'candidate_ffn_elements':11520,
            'scope':'Single basic screen; no independent repeats, expanded numerical claim, adoption or official submission.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=check();out=HERE/'verification.json'
    if args.check:assert json.loads(out.read_text())==r
    else:out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
