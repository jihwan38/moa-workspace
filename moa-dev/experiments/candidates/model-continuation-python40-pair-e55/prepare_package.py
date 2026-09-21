#!/usr/bin/env python3
"""Freeze the unchanged-model continuation hosts and audited inspector for Arena."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil

C = Path(__file__).resolve().parent
R = C.parents[3]
B = R/'moa-dev/experiments/candidates/model-continuation-python40-e55'
A = R/'moa-dev/experiments/artifacts/model-continuation-python40-e55'
OUT = R/'moa-dev/experiments/artifacts/model-continuation-python40-pair-e55/package'

def check(condition, message):
    if not condition:
        raise ValueError(message)

def fp(path):
    return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    check(not OUT.exists(), 'Refusing to replace an existing package')
    check(not (C/'package-manifest.json').exists(), 'Refusing to replace an existing manifest')
    spec = importlib.util.spec_from_file_location('registry', C.parent/'model-prefix-suite-e55/verify_registry.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    sources, audits = {}, {}
    for variant in ('original','e55'):
        host = A/variant/'probe_model_continuation'
        entries, info = verifier.registry(host)
        reference, reference_info = verifier.registry(R/'moa-dev/experiments/artifacts/model-prefix-suite-e55'/variant/'probe_model_prefix_suite')
        check(entries == reference and len(entries) == 10, variant+' text kernel images changed')
        expected = json.loads((B/(variant+'-source-manifest.json')).read_text())['files']
        compiled = json.loads((B/(variant+'-compiled-source-manifest.json')).read_text())['files']
        actual = {str(p.relative_to(A/variant/'crate')):fp(p) for p in (A/variant/'crate').rglob('*') if p.is_file()}
        check(len(actual) == 47 and actual == expected == compiled, variant+' compiler input mismatch')
        probe = 'src/bin/probe_model_continuation.rs'
        check(actual[probe] == fp(B/'probe_model_continuation.rs'), variant+' host source changed')
        frozen = R/'moa-dev/experiments/artifacts/model-prefix-suite-e55'/variant/'crate'
        for name in actual:
            if name != probe:
                check(actual[name] == fp(frozen/name), variant+' changed an existing model input: '+name)
        sources['probe_model_'+variant] = host
        audits[variant] = {'host':info,'kernel_reference':reference_info,'all_ten_images_unchanged':True,
                           'compiler_inputs':47,'existing_model_inputs_unchanged':46,'new_host_source':actual[probe]}
    inspector = R/'moa-dev/experiments/artifacts/arena-model-assets-rust-inspector/moa-model-asset-inspector'
    check(fp(inspector) == {'bytes':880872,'sha256':'1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11'}, 'Inspector differs')
    sources['moa-model-asset-inspector'] = inspector
    sources['remote_entrypoint.sh'] = C/'remote_entrypoint.sh'
    OUT.mkdir(parents=True)
    for name,source in sources.items():
        shutil.copyfile(source,OUT/name)
        (OUT/name).chmod(0o755)
    manifest = {'schema':'MOA_MODEL_CONTINUATION_PYTHON40_PAIR_PACKAGE_V1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'package_path':str(OUT.relative_to(R)),'files':{name:{'source':str(source.relative_to(R)),**fp(OUT/name)} for name,source in sources.items()},
        'hosts':audits,'file_count':4,'reference_directory':'reference','reference_manifest':'reference/manifest.json',
        'reference_policy':'Original legal-greedy continuation; E55 replays original tokens even after candidate disagreement or EOS.',
        'max_new_tokens_per_case':40,'max_checkpoints':40,'accuracy_acceptance_threshold':None,'accuracy_pass':None,
        'npu_executed_during_preparation':False}
    (C/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'package':str(OUT),'manifest':str(C/'package-manifest.json'),'unchanged_model_source_and_images':True}))

if __name__ == '__main__':
    main()
