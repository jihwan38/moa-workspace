#!/usr/bin/env python3
"""Copy the existing E55 CLI and inspector after source/ELF provenance checks.

No compiler, model, or NPU is executed. The historical build's 46-input
manifest is checked against preserved sources; this is not a fresh build.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
if not __debug__:
    raise RuntimeError('The imported ELF parser requires assertions; Python -O is unsupported.')

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-cli-python40-e55'
OUT = A / 'package'
BUILD = R / 'moa-dev/experiments/artifacts/e55-model-host-build'
FROZEN = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/crate'
BUILD_AUDIT = R / 'docs/experiments/e55-model-host-build.json'
INPUT_MANIFEST = R / 'moa-dev/experiments/candidates/e55-model-host-build/compiled-source-manifest.json'
VERIFIER = R / 'moa-dev/experiments/candidates/e55-model-host-build/verify_registry.py'
PROMPT = 'Complete the Python expression: len([1, 2, 3])'
CLI_HASH = '7e801ed65016b5d0faa71da3fd3f3f9d012db393adc47863ad8e4665532678eb'
INSPECTOR_HASH = '1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def record(path):
    return {'path': str(path.relative_to(R)), **fingerprint(path)}


def main():
    require(not OUT.exists(), 'Refusing to overwrite existing package')
    require(not (C / 'package-manifest.json').exists(), 'Refusing to overwrite package manifest')
    require(fingerprint(BUILD_AUDIT) == {
        'bytes': 8114, 'sha256': '08bf518ec817e60c1d303d51256e6fe176bd4f4b55f825f85eb635e5e7af16ca'
    }, 'Historical model-build audit changed')
    require(fingerprint(INPUT_MANIFEST) == {
        'bytes': 6162, 'sha256': 'b9a9bdf504bf6fc778a4f005e460b4e17575521834ee1e79a3fe8b31190c5a8f'
    }, 'Historical compiled-input manifest changed')
    audit = json.loads(BUILD_AUDIT.read_text())
    require(audit['status'] == 'ISOLATED_MODEL_HOST_BUILD_AND_REGISTRY_PASS'
            and audit['actual_compiled_input_count'] == 46
            and audit['actual_inputs_match_frozen_e55'] is True, 'Historical build audit scope mismatch')
    expected_inputs = json.loads(INPUT_MANIFEST.read_text())
    parent_probe = 'src/bin/probe_model_prefix_suite.rs'
    input_names = {str(path.relative_to(FROZEN)) for path in FROZEN.rglob('*') if path.is_file()}
    require(parent_probe in input_names, 'Preserved parent is not the audited prefix-suite crate')
    input_names.remove(parent_probe)
    require(len(input_names) == 46 and input_names == set(expected_inputs), 'Model-source file set mismatch')
    actual_inputs = {name: fingerprint(FROZEN / name) for name in sorted(input_names)}
    require(actual_inputs == expected_inputs, 'Preserved model sources differ from compiled 46 inputs')
    require(fingerprint(VERIFIER)['sha256'] == audit['verifier']['sha256'], 'Frozen ELF parser changed')
    spec = importlib.util.spec_from_file_location('frozen_e55_cli_registry', VERIFIER)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    cli = BUILD / 'gemma4'
    require(fingerprint(cli) == {'bytes': 12378864, 'sha256': CLI_HASH}, 'E55 CLI bytes changed')
    entries, info = verifier.registry(cli)
    require(set(entries) == {(name, ()) for name in verifier.EXPECTED} and len(entries) == 14,
            'CLI registry does not contain the audited 14 images')
    require(info['binary'] == audit['hosts']['gemma4']['binary']
            and info['registry_section'] == audit['hosts']['gemma4']['registry_section'],
            'Actual CLI ELF differs from historical build audit')
    inventory = []
    for (name, key), image in sorted(entries.items()):
        require(image == (BUILD / 'kernels' / (name + '.bin')).read_bytes(),
                'CLI image differs from isolated compiler output: ' + name)
        inventory.append({'path': name, 'generic_key': list(key),
                          'bytes': len(image), 'sha256': hashlib.sha256(image).hexdigest()})
    require(inventory == audit['shared_device_registry'], 'Actual 14-image inventory differs from build audit')
    prefix_host = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55/e55/probe_model_prefix_suite'
    text_entries, text_info = verifier.registry(prefix_host)
    require(set(text_entries) == {(f'furiosa_opt_gemma4::ops::{name}', ()) for name in verifier.OPS},
            'Frozen text-host registry differs from ten text callers')
    require(all(entries[key] == image for key, image in text_entries.items()),
            'CLI text images differ from measured E55 prefix suite')

    inspector = R / 'moa-dev/experiments/artifacts/arena-model-assets-rust-inspector/moa-model-asset-inspector'
    require(fingerprint(inspector) == {'bytes': 880872, 'sha256': INSPECTOR_HASH}, 'Inspector bytes changed')
    sources = {'gemma4': cli, 'moa-model-asset-inspector': inspector,
               'remote_entrypoint.sh': C / 'remote_entrypoint.sh'}
    OUT.mkdir(parents=True)
    for name, source in sources.items():
        destination = OUT / name
        shutil.copyfile(source, destination)
        destination.chmod(0o755)
        require(destination.read_bytes() == source.read_bytes(), 'Copy changed payload: ' + name)
    manifest = {
        'schema': 'MOA_MODEL_CLI_PYTHON40_PACKAGE_V1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'package_path': str(OUT.relative_to(R)), 'file_count': 3,
        'files': {name: {'source': str(source.relative_to(R)), **fingerprint(OUT / name)}
                  for name, source in sources.items()},
        'prompt': PROMPT, 'argv': ['./gemma4', PROMPT],
        'max_new_tokens': 40, 'timeout_sec': 70,
        'command_environment': {'GEMMA4_MAX_NEW_TOKENS': '40', 'FURIOSA_OPT_PROFILE': 'warn'},
        'model_directory_environment': 'Inherit the Arena-provided RNGD_MODEL_DIR; no override or path output.',
        'sampling': {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95, 'rng': 'unseeded rand::rng()'},
        'greedy_reference_equivalence_expected': False,
        'source_checkpoint': audit['source_checkpoint'], 'model_build_audit': record(BUILD_AUDIT),
        'historical_compiled_input_manifest': record(INPUT_MANIFEST),
        'preserved_source_root': str(FROZEN.relative_to(R)),
        'preserved_model_input_count': len(actual_inputs), 'all_46_match_historical_compiler_inputs': True,
        'preserved_model_inputs': actual_inputs, 'source_changes': False,
        'cli_registry': {**info, 'entry_count': 14, 'images': inventory},
        'all_fourteen_images_match_isolated_compiler_outputs': True,
        'text_reference': {'path': str(prefix_host.relative_to(R)), **text_info,
                           'ten_text_images_byte_identical': True},
        'registry_verifier': record(VERIFIER),
        'prepare_script': record(Path(__file__).resolve()),
        'reference_notes': 'CLI stdout includes a prompt echo; distinguish its first exact prompt line from subsequent generated text. Stop gives no exact EOS ID.',
        'failure_policy': 'Retain CLI return code over subsequent inspector failure; overall completion only after both succeed. External timeout may prevent EXIT-trap output.',
        'compiler_executed': False, 'model_executed_during_preparation': False,
        'npu_executed_during_preparation': False, 'official_evaluation': False,
        'accuracy_acceptance_threshold': None, 'accuracy_pass': None,
    }
    with (C / 'package-manifest.json').open('x') as writer:
        json.dump(manifest, writer, indent=2)
        writer.write('\n')
    print(json.dumps({'status': 'FROZEN_CLI_PACKAGE_PREPARED_NO_MODEL_EXECUTION',
                      'package_path': str(OUT), 'files': manifest['files'],
                      'verified_model_inputs': 46, 'verified_registry_images': 14}))


if __name__ == '__main__':
    main()
