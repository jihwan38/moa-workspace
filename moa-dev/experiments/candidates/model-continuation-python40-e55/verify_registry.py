#!/usr/bin/env python3
"""Verify both actual continuation ELFs keep their frozen ten text images."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json

C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-continuation-python40-e55'
S = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
B = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    reader = S / 'verify_registry.py'
    require(hashlib.sha256(reader.read_bytes()).hexdigest() == '163d326d0904ea4a435584e045e12ae5dabc04b35e1cd796584669f51b1363cb','registry_reader_changed')
    require(hashlib.sha256((S/'registry-summary.json').read_bytes()).hexdigest() == '06ab42da3d1e5eb20621035f13c396e41a4a107423ffefe4a30ddf9f65495805','frozen_registry_reference_changed')
    spec = importlib.util.spec_from_file_location('frozen_registry', reader)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier) # Includes explicit Python -O rejection.
    frozen_summary = json.loads((S/'registry-summary.json').read_text())
    expected_keys = {(f'furiosa_opt_gemma4::ops::{name}',()) for name in verifier.OPS}
    hosts = {}
    inventories = {}
    for variant in ('original','e55'):
        baseline_path = B / variant / 'probe_model_prefix_suite'
        baseline, baseline_info = verifier.registry(baseline_path)
        require(baseline_info['binary'] == frozen_summary['hosts'][variant]['binary'],variant+'_reference_binary_changed')
        require(baseline_info['registry_section'] == frozen_summary['hosts'][variant]['registry_section'],variant+'_reference_registry_changed')
        host = A / variant / 'probe_model_continuation'
        entries, info = verifier.registry(host)
        require(set(entries) == set(baseline) == expected_keys,variant+'_registry_inventory_mismatch')
        images = {}
        for (name,key), image in entries.items():
            require(key == (),variant+'_nonempty_generic_key')
            require(image == baseline[name,key],variant+'_image_changed:'+name)
            require(image == (A/variant/'kernels'/(name+'.bin')).read_bytes(),variant+'_standalone_image_mismatch:'+name)
            require(verifier.fingerprint(image) == frozen_summary['hosts'][variant]['entries'][name],variant+'_frozen_image_hash_mismatch:'+name)
            images[name] = verifier.fingerprint(image)
        inventories[variant] = entries
        hosts[variant] = {'host_path':str(host.relative_to(R)),**info,'entry_count':len(entries),'entries':images,
            'all_ten_images_equal_prefix_suite_host':True,'all_standalone_images_equal_registry':True,
            'reference_host_path':str(baseline_path.relative_to(R)),'reference_host':baseline_info}
    changed = sorted(name for name,key in inventories['original'] if inventories['original'][name,key] != inventories['e55'][name,key])
    require(changed == frozen_summary['images_changed_from_original_to_e55'],'original_e55_image_change_set_mismatch')
    report = {'status':'CONTINUATION_HOST_IMAGE_IDENTITY_PASS','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'hosts':hosts,'images_changed_from_original_to_e55':changed,'all_ten_text_kernels_present':True,
        'all_keys_empty':True,'registry_fully_consumed_and_sorted':True,'npu_run':False,'logits_accuracy_validated':False}
    (C/'registry-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'hosts':{k:v['binary'] for k,v in hosts.items()}}))


if __name__ == '__main__':
    main()
