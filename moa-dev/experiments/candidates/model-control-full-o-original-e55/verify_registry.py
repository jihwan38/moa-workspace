#!/usr/bin/env python3
"""Prove target original image + nine E55 images from actual ELF registries."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-full-o-original-e55'
SUITE_C = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
SUITE_A = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
TARGET = 'furiosa_opt_gemma4::ops::full_attention_output'


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    # The frozen suite parser includes an explicit Python -O rejection, bounded
    # ELF/bincode reads, unique entries, empty-key checks at callers, and complete
    # sorted-registry consumption. Importing it does not invoke its main().
    reader = SUITE_C / 'verify_registry.py'
    check(hashlib.sha256(reader.read_bytes()).hexdigest() == '163d326d0904ea4a435584e045e12ae5dabc04b35e1cd796584669f51b1363cb', 'frozen_registry_parser_changed')
    check(hashlib.sha256((SUITE_C / 'registry-summary.json').read_bytes()).hexdigest() == '06ab42da3d1e5eb20621035f13c396e41a4a107423ffefe4a30ddf9f65495805', 'frozen_registry_reference_changed')
    spec = importlib.util.spec_from_file_location('frozen_suite_registry', reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected_set = {(f'furiosa_opt_gemma4::ops::{name}', ()) for name in module.OPS}
    summaries = json.loads((SUITE_C / 'registry-summary.json').read_text())
    frozen = {}
    provenance = {}
    for variant in ('original', 'e55'):
        binary = SUITE_A / variant / 'probe_model_prefix_suite'
        entries, info = module.registry(binary)
        check(set(entries) == expected_set, variant + '_registry_set_mismatch')
        check(info['binary'] == summaries['hosts'][variant]['binary'], variant + '_frozen_binary_hash_mismatch')
        check(info['registry_section'] == summaries['hosts'][variant]['registry_section'], variant + '_frozen_registry_hash_mismatch')
        for (name, key), image in entries.items():
            check(key == (), 'nonempty_generic_key')
            check(module.fingerprint(image) == summaries['hosts'][variant]['entries'][name], variant + '_frozen_image_hash_mismatch')
        frozen[variant] = entries
        provenance[variant] = {'path':str(binary.relative_to(R)), **info}
    control, info = module.registry(A / 'probe_model_prefix_suite')
    check(set(control) == expected_set, 'control_registry_set_mismatch')
    images = {}
    for (name, key), image in control.items():
        check(key == (), 'control_nonempty_generic_key')
        check(image == (A / 'kernels' / (name + '.bin')).read_bytes(), 'standalone_image_mismatch:' + name)
        reference = 'original' if name == TARGET else 'e55'
        check(image == frozen[reference][name, key], 'image_identity_requirement_failed:' + name)
        images[name] = {**module.fingerprint(image), 'required_reference':reference,
            'required_reference_byte_identical':True, 'standalone_bin_byte_identical':True}
    changed = [name for name, key in control if control[name,key] != frozen['e55'][name,key]]
    check(changed == [TARGET], 'not_exactly_one_changed_image')
    check(images[TARGET]['sha256'] == '00f4ef931a38f9223ef27563468e663fb7e58c080fa53112dc96911ab1773daf', 'unexpected_original_full_o_image')
    source = json.loads((C / 'source-manifest.json').read_text())['files']
    compiled = json.loads((C / 'compiled-source-manifest.json').read_text())['files']
    actual = {str(p.relative_to(A / 'crate')):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        for p in (A / 'crate').rglob('*') if p.is_file()}
    check(len(actual) == 47 and actual == source == compiled, 'source_manifest_mismatch_after_build')
    report = {'status':'SINGLE_FULL_O_IMAGE_CONTROL_IDENTITY_PASS',
        'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'host_path':str((A/'probe_model_prefix_suite').relative_to(R)),
        **info, 'entry_count':len(control), 'entries':images, 'changed_from_e55':changed,
        'target_image_exact_original':True, 'other_nine_images_exact_e55':True,
        'all_47_source_inputs_match_actual_build':True, 'unchanged_suite_host_source_sha256':source['src/bin/probe_model_prefix_suite.rs']['sha256'],
        'registry_parser_path':str(reader.relative_to(R)), 'registry_parser_sha256':hashlib.sha256(reader.read_bytes()).hexdigest(),
        'reference_hosts':provenance, 'npu_run':False, 'logits_accuracy_validated':False}
    (C / 'registry-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status':report['status'], 'binary':info['binary'], 'changed_from_e55':changed}))


if __name__ == '__main__':
    main()
