#!/usr/bin/env python3
"""Compile actual pure host helpers/tests plus frozen sampler, without SDK/NPU."""
from pathlib import Path
import hashlib
import json
import subprocess

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-continuation-e55'
SCRATCH = '/tmp/moa-model-continuation-e55'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = (C / 'probe_model_continuation.rs').read_text()
    ready = json.loads((C / 'host-source.json').read_text())
    require(sha(source.encode()) == ready['file']['sha256'], 'host_not_frozen')
    require(source.count('#[tokio::main]') == 1, 'ambiguous_main')
    marker = '#[cfg(test)]\nmod tests'
    require(source.count(marker) == 1, 'missing_or_ambiguous_host_tests')
    helpers = source[:source.index('#[tokio::main]')]
    tests = source[source.index(marker):]
    lines = []
    skipping = False
    for line in helpers.splitlines():
        if line.startswith('use furiosa_opt_'):
            skipping = True
        if skipping:
            if ';' in line:
                skipping = False
            continue
        lines.append(line)
    require(not skipping, 'unterminated_sdk_import')
    helpers_no_sdk_imports = '\n'.join(lines) + '\n'
    require('furiosa_opt_' not in helpers_no_sdk_imports, 'unhandled_sdk_reference_in_pure_helpers')
    e55 = A / 'e55/crate'
    original = A / 'original/crate'
    sampling = (e55 / 'src/host/sampling.rs').read_bytes()
    require(sampling == (original / 'src/host/sampling.rs').read_bytes(), 'original_e55_sampler_differs')
    require(sha(sampling) == '71ee4a3dda3662f1242b421636c10c46b0255d6557d85f694b940882db1a6a11', 'frozen_sampler_changed')
    axes = (e55 / 'src/axes.rs').read_text()
    require('E = 32768' in axes and 'W = 262144' in axes, 'context_or_vocabulary_stub_mismatch')
    require('rand = "0.9"' in (e55/'Cargo.toml').read_text(), 'direct_rand_requirement_changed')
    rendered = helpers_no_sdk_imports + '\nmod sampling {\n' + sampling.decode() + '\n}\n'
    rendered += 'use sampling::{sample, SamplingConfig};\n'
    rendered += 'struct E; impl E { const SIZE: usize = 32768; }\n'
    rendered += 'struct W; impl W { const SIZE: usize = 262144; }\n'
    rendered += tests
    output = A / 'guard-tests.rs'
    output.write_text(rendered)
    subprocess.run(['docker','cp',str(output),'moa2026-dev:'+SCRATCH+'/guard-tests.rs'],check=True)
    code = '''from pathlib import Path
import hashlib,json,re,subprocess
root=Path('/tmp/moa-model-continuation-e55')
deps=root/'e55/target/release/deps'
args=['rustc','--edition=2024','--test',str(root/'guard-tests.rs'),'-L','dependency='+str(deps)]
linked={}
for name in ['serde','serde_json','rand']:
    libs=list(deps.glob('lib'+name+'-*.rlib'))
    if name=='rand':
        libs=[p for p in libs if re.search(r'/rand-0\\.9\\.[^/]+/src/lib\\.rs', (deps/(p.stem[3:]+'.d')).read_text())]
    if len(libs)!=1:
        raise RuntimeError('ambiguous_or_missing_library:'+name)
    args+=['--extern',name+'='+str(libs[0])]
    linked[name]={'file':libs[0].name,'sha256':hashlib.sha256(libs[0].read_bytes()).hexdigest(),'bytes':libs[0].stat().st_size}
args+=['-o',str(root/'guard-tests')]
subprocess.run(args,cwd=root/'e55',check=True)
test=root/'guard-tests'
(root/'guard-test-build.json').write_text(json.dumps({'command':args,'linked_libraries':linked,'test_binary':{'sha256':hashlib.sha256(test.read_bytes()).hexdigest(),'bytes':test.stat().st_size},'model_or_npu_executed':False},indent=2)+'\\n')
subprocess.run([str(test),'--nocapture'],check=True)
'''
    result = subprocess.run(['docker','exec','moa2026-dev','python3','-c',code],text=True,capture_output=True)
    log = A / 'guard-tests.log'
    log.write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr,end='')
    result.check_returncode()
    subprocess.run(['docker','cp','moa2026-dev:'+SCRATCH+'/guard-test-build.json',str(A/'guard-test-build.json')],check=True)
    report = {'status':'CPU_GUARD_TESTS_PASS','host_source_sha256':sha(source.encode()),
        'original_helpers_prefix_sha256':sha(helpers.encode()), 'exact_test_module_sha256':sha(tests.encode()),
        'rendered_test_source_sha256':sha(rendered.encode()),'frozen_sampling_source_sha256':sha(sampling),
        'axis_stubs':{'E':32768,'W':262144},'axis_stubs_verified_against_frozen_source':True,
        'method':'Only SDK imports removed; original pure helper bodies/test module and original sampling.rs preserved. No custom sampler.',
        'log_sha256':sha(log.read_bytes()),'build_record_sha256':sha((A/'guard-test-build.json').read_bytes()),
        'model_or_npu_executed':False}
    (C/'cpu-test-evidence.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
