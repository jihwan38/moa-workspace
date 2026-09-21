#!/usr/bin/env python3
"""Validate actual uploaded control images, then collect a labeled prefix comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile

import collect_model_prefix_suite as suite

R = suite.ROOT
TARGETS = {'qkv': 'sliding_project_qkv', 'sliding-o': 'sliding_attention_output',
           'ffn': 'decoder_feedforward', 'full-o': 'full_attention_output'}

def collect(status, log, archive, package):
    manifest = suite.read_json(package)
    suite.require(manifest['schema'] == 'MOA_MODEL_SINGLE_KERNEL_CONTROL_PACKAGE_V1', 'Wrong control package schema')
    spec = importlib.util.spec_from_file_location('registry', R/'moa-dev/experiments/candidates/model-prefix-suite-e55/verify_registry.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    base = R/'moa-dev/experiments/artifacts/model-prefix-suite-e55'
    original, _ = verifier.registry(base/'original/probe_model_prefix_suite')
    e55, _ = verifier.registry(base/'e55/probe_model_prefix_suite')
    target = (manifest['target_image'], ())
    suite.require(manifest['control'] in TARGETS and manifest['target_image'] ==
                  'furiosa_opt_gemma4::ops::' + TARGETS[manifest['control']], 'Control label/target mismatch')
    suite.require(manifest['candidate_label'] == 'model-control-' + manifest['control'] + '-original-e55',
                  'Candidate label mismatch')
    suite.require(manifest['reference_label'] in ('original', 'e55') and target in original, 'Invalid reference or target')
    with tarfile.open(archive) as tar, tempfile.TemporaryDirectory(prefix='moa-control-registry-') as temporary:
        inventories = {}
        for name in ('probe_model_reference', 'probe_model_control'):
            members = [m for m in tar.getmembers() if m.name == name]
            suite.require(len(members) == 1 and members[0].isfile(), 'Missing, duplicated or non-regular host')
            path = Path(temporary)/name
            path.write_bytes(tar.extractfile(members[0]).read())
            inventories[name], _ = verifier.registry(path)
        control = inventories['probe_model_control']
        expected_reference = original if manifest['reference_label'] == 'original' else e55
        suite.require(inventories['probe_model_reference'] == expected_reference, 'Uploaded reference images changed')
        suite.require(set(control) == set(e55) and len(control) == 10, 'Wrong control registry inventory')
        suite.require(control[target] == original[target], 'Uploaded target image differs from original')
        suite.require(all(image == e55[key] for key, image in control.items() if key != target), 'Uploaded non-target image differs from E55')
    result = suite.collect(status, log, archive, package_path=package)
    reference_source = base/manifest['reference_label']/'probe_model_prefix_suite'
    reference_fp = suite.fingerprint(reference_source)
    suite.require(manifest['files']['probe_model_reference']['sha256'] == reference_fp['sha256']
                  and manifest['files']['probe_model_reference']['bytes'] == reference_fp['bytes'],
                  'Reference host differs from frozen suite')
    result['reference_label'] = manifest['reference_label']
    result['candidate_label'] = manifest['candidate_label']
    result['control'] = manifest['control']
    result['target_image'] = target[0]
    result['actual_uploaded_registry_single_kernel_identity_pass'] = True
    result['base_collector'] = result.pop('collector')
    result['collector'] = suite.fingerprint(Path(__file__).resolve())
    result['package_manifest_fingerprint'] = suite.fingerprint(package)
    result['package_manifest'] = str(package.relative_to(R))
    for summary in result['case_summaries']:
        summary['candidate_top'] = summary.pop('e55_top')
        summary['reference_decode_and_read_seconds'] = summary.pop('original_decode_and_read_seconds')
        summary['candidate_decode_and_read_seconds'] = summary.pop('e55_decode_and_read_seconds')
    prior = suite.read_json(R/'docs/experiments/e55-model-prefix-suite.json')
    result['reference_equals_job_71239_original_by_reported_hash'] = (
        manifest['reference_label'] == 'original' and result['reference_file_sha256_reported_in_job'] == prior['reference_file_sha256_reported_in_job'])
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('status','log','archive','package'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.status.resolve(),args.log.resolve(),args.archive.resolve(),args.package.resolve()),indent=2,ensure_ascii=False,allow_nan=False))

if __name__ == '__main__':
    main()
