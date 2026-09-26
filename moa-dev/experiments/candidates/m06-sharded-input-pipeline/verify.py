#!/usr/bin/env python3
"""Verify successful/failed builds and invalid screens without inventing cycles."""
import argparse,json,re,subprocess,sys,tarfile
from analyze import HERE,ROOT,digest,result as static_result
ART=ROOT/'moa-dev/experiments/artifacts'/HERE.name
sys.path.insert(0,str(ROOT/'moa-dev/experiments'))
from structural_build import source_manifest
sys.path.insert(0,str(ROOT/'moa-dev'))
from measure import parse_log
def section(s,a,b):
    i=s.index(a);return s[i:s.index(b,i)]
def check():
    original=json.loads((HERE.parent/'m03-producer-local-ffn/build-m03-native-local-v1-repro.json').read_text())
    m03=(HERE.parent/'m03-producer-local-ffn/native-local-v1/mlp.rs').read_text()
    m05=(HERE.parent/'m05-local-activation-ready/local-prepare-v1/mlp.rs').read_text()
    builds={}
    for path in sorted(HERE.glob('build-*-repro.json')):
        b=json.loads(path.read_text())
        if 'reused_build_record' in b:
            info=b.pop('reused_build_record');assert digest((ROOT/info['path']).read_bytes())['sha256']==info['sha256']
            assert b==original
            assert {f.name:digest(f.read_bytes()) for f in (ART/'m03-reference/package').iterdir()}==b['package_files']
            continue
        plan=b['source_plan'];recipe=json.loads((ROOT/plan['recipe']['path']).read_text())
        assert digest((ROOT/plan['recipe']['path']).read_bytes())=={k:plan['recipe'][k] for k in ['bytes','sha256']}
        for c in recipe['source_components']:
            raw=subprocess.check_output(['git','show',c['commit']+':'+c['path']],cwd=ROOT)
            assert digest(raw)['sha256']==c['sha256']
        inputs=source_manifest(recipe['base_commit']);assert inputs==plan['source_origin_inputs']
        for dest,item in plan['source_overrides'].items():
            raw=(ROOT/item['path']).read_bytes();inputs[dest]=digest(raw)
            assert inputs[dest]=={k:item[k] for k in ['bytes','sha256']}
        assert inputs==b['source_inputs']
        assert set(plan['source_overrides'])=={'src/ops.rs','src/device/shared/mlp.rs'}
        assert {k:v for k,v in inputs.items() if k not in plan['source_overrides']}=={k:v for k,v in original['source_inputs'].items() if k not in plan['source_overrides']}
        for name,f in b['frozen_files'].items():assert digest((ROOT/f).read_bytes())==inputs.get(name,b['logs'].get(name))
        source=(ROOT/plan['source_overrides']['src/device/shared/mlp.rs']['path']).read_text()
        assert section(source,'fn project_rows_with_scale_contraction(', 'pub(crate) fn feedforward(')==section(m03,'fn project_rows_with_scale_contraction(', 'pub(crate) fn feedforward(')
        assert section(source,'    let (x, x_scale) = prepare_ffn_fp8_split', 'type FfnQuantSlices')==section(m03,'    let (x, x_scale) = prepare_ffn_fp8_split', 'type FfnQuantSlices')
        ops=(ROOT/plan['source_overrides']['src/ops.rs']['path']).read_text()
        old=subprocess.check_output(['git','show',recipe['base_commit']+':furiosa-opt-gemma4-12B/src/ops.rs'],cwd=ROOT,text=True)
        expected=old.replace('    let x = shared::rmsnorm::normalize(device, &residual, pre_ff_rms_weight);','    // Keep pre-RMS output in the exact H480 layout used by the input quantizer.\n    let rms_input: DmTensor<bf16, Chip, Cluster, shared::mlp::DownEpilogueSlices, m![H % 480]> = residual.to_dm(&mut device.tdma);\n    let x = shared::rmsnorm::normalize_sharded_inter_slice(device, &rms_input, pre_ff_rms_weight);')
        assert ops==expected
        if b['variant']=='sharded-dma-terms-v5':
            assert source[source.index('    let hi: DmTensor<f8e5m2'):]==m05[m05.index('    let hi: DmTensor<f8e5m2'):]
        out=ART/b['label'];success=b.get('host_build_exit_code')==0
        if success:
            assert b['exact_compile_exit_code']==0
            assert digest((out/'schedule.json').read_bytes())==b['schedule']==digest((HERE/'evidence'/b['label']/'schedule.json').read_bytes())
            assert digest((out/'exact-kernel.bin').read_bytes())==b['exact_image']
            package=out/'package';assert {f.name:digest(f.read_bytes()) for f in package.iterdir()}==b['package_files']
            host=(package/'test_runtime').read_bytes()
            for kernel,value in b['images'].items():
                image=(out/'kernels'/f'furiosa_opt_gemma4::ops::{kernel}.bin').read_bytes()
                assert digest(image)=={k:value[k] for k in ['bytes','sha256']};assert host.count(image)==value['host_occurrences']==1
                if kernel==b['target_kernel']:assert digest(image)==b['exact_image']
                else:assert value['sha256']==original['images'][kernel]['sha256']
        else:
            assert b['failed_exit_code']!=0
            assert 'sfr_impl.rs:868' in (ROOT/b['frozen_files']['compile.log']).read_text()
        builds[b['label']]={'executable_package':success,'source_files':len(inputs),'frozen_files':len(b['frozen_files'])}
    assert json.loads((HERE/'static-analysis.json').read_text())==static_result()
    for f,expected in json.loads((HERE/'auxiliary-evidence.json').read_text())['files'].items():assert digest((ROOT/f).read_bytes())==expected
    jobs={}
    for label in ['m06-sharded-packed-screen','m06-sharded-dma-screen']:
        r=json.loads((HERE/f'{label}-results.json').read_text());manifest=json.loads((HERE/f'{label}-build.json').read_text())
        raw=(HERE/f"evidence/arena-{r['job']['id']}.log").read_bytes();assert digest(raw)['sha256']==r['log_sha256']
        with tarfile.open(ART/f"job-{r['job']['id']}.tar") as ar:files={m.name:digest(ar.extractfile(m).read()) for m in ar if m.isfile()}
        assert files==r['package_files'];assert {k:v for k,v in files.items() if k!='job.yaml'}==manifest['package_files']
        sections=re.findall(r'^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$',raw.decode(),re.M|re.S)
        assert [n for n,_ in sections]==[s['name'] for s in manifest['stages']];assert raw.count(b'STRUCTURAL_COMPLETE')==1
        for name,log in sections:
            if name=='candidate_0':
                assert not r['stages'][name]['valid'];assert 'the launch ran past its time limit' in log
                assert re.findall(r'STRUCTURAL_EXIT candidate_0 (\d+)',log)==['101']
                prefix,ffn=log.split('==> decoder_feedforward',1)
                assert prefix.count('-> PASS')==12 and '-> PASS' not in ffn
                assert 'median cycles=' not in ffn
            else:
                assert r['stages'][name]['valid'];assert f'STRUCTURAL_EXIT {name} 0' in log
                assert parse_log(log)==r['stages'][name]['kernels']
        assert not r['valid'] and r['job']['status']=='FAILED'
        jobs[str(r['job']['id'])]={'unchanged_control_comparisons_passed':45,'candidate_unchanged_qkv_o_comparisons_passed':12,'candidate_ffn_comparisons':0,'candidate_ffn_performance_valid':False,'failure':'device launch time limit','actual_upload_and_logs_verified':True}
    return {'builds':builds,'jobs':jobs,'shared_projection_geglu_down_source_preserved':True,'ops_scope_only_ffn_input_handoff':True,'static_analysis_recomputed':True,'scope':'Invalid candidates. No FFN hardware performance, accuracy, adoption or official-submission claim. Failure root cause unresolved.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();r=check();out=HERE/'verification.json'
    if args.check:assert json.loads(out.read_text())==r
    else:out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
