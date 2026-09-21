#!/usr/bin/env python3
"""Package a source-built single-kernel control after an independent image audit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

C = Path(__file__).resolve().parent
R = C.parents[3]
BASE = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
TARGETS = {'qkv': 'sliding_project_qkv', 'sliding-o': 'sliding_attention_output',
           'ffn': 'decoder_feedforward', 'full-o': 'full_attention_output'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def fp(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', choices=TARGETS, required=True)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--reference', choices=('original', 'e55'), required=True)
    args = parser.parse_args()
    host = args.host.resolve()
    name = 'model-control-' + args.control + '-original-e55'
    out = R / 'moa-dev/experiments/artifacts' / name / 'package'
    require(not out.exists(), 'Refusing to replace an existing package')
    spec = importlib.util.spec_from_file_location('registry', C.parent/'model-prefix-suite-e55/verify_registry.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    control, host_info = verifier.registry(host)
    original, _ = verifier.registry(BASE/'original/probe_model_prefix_suite')
    e55, _ = verifier.registry(BASE/'e55/probe_model_prefix_suite')
    target = ('furiosa_opt_gemma4::ops::' + TARGETS[args.control], ())
    require(set(control) == set(original) == set(e55) and len(control) == 10, 'Registry inventory mismatch')
    require(control[target] == original[target], 'Target image differs from original')
    require(all(image == e55[key] for key, image in control.items() if key != target), 'A non-target image differs from E55')
    inspector = R/'moa-dev/experiments/artifacts/arena-model-assets-rust-inspector/moa-model-asset-inspector'
    require(fp(inspector) == {'sha256':'1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11','bytes':880872}, 'Inspector changed')
    reference = BASE/args.reference/'probe_model_prefix_suite'
    sources = {'probe_model_reference':reference, 'probe_model_control':host,
               'moa-model-asset-inspector':inspector, 'remote_entrypoint.sh':C/'remote_entrypoint.sh'}
    out.mkdir()
    for basename, source in sources.items():
        shutil.copyfile(source, out/basename)
        (out/basename).chmod(0o755)
    manifest = {'schema':'MOA_MODEL_SINGLE_KERNEL_CONTROL_PACKAGE_V1', 'control':args.control,
        'candidate_label':name, 'reference_label':args.reference, 'target_image':target[0],
        'target_image_equals_original':True, 'other_nine_images_equal_e55':True,
        'candidate_host_registry':host_info,
        'reference_host_sha256':fp(reference)['sha256'],
        'files':{basename:{'source':str(source.relative_to(R)),**fp(out/basename)} for basename,source in sources.items()},
        'package_path':str(out.relative_to(R)), 'file_count':4,
        'accuracy_pass':None, 'accuracy_acceptance_threshold':None, 'npu_run':False}
    path = C/(args.control+'-package-manifest.json')
    require(not path.exists(), 'Refusing to replace an existing manifest')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'package':str(out), 'manifest':str(path), 'image_identity_pass':True}))

if __name__ == '__main__':
    main()
