#!/usr/bin/env python3
"""Require original FFN and exactly the other nine E55 text kernel images."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if not __debug__:
    raise RuntimeError('Registry checks require normal Python; -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-control-ffn-original-e55'
S = R / 'moa-dev/experiments/candidates/model-prefix-suite-e55'
spec = importlib.util.spec_from_file_location('suite_registry', S / 'verify_registry.py')
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)
target = 'furiosa_opt_gemma4::ops::decoder_feedforward'
entries, host_info = registry.registry(A / 'probe_model_prefix_suite')
parents = {}
parent_info = {}
audited_parents = json.loads((S / 'registry-summary.json').read_text())['hosts']
for variant in ('original', 'e55'):
    path = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55' / variant / 'probe_model_prefix_suite'
    parents[variant], parent_info[variant] = registry.registry(path)
    assert parent_info[variant]['binary'] == audited_parents[variant]['binary'], 'parent host differs from audited suite binary'
expected_keys = {(f'furiosa_opt_gemma4::ops::{name}', ()) for name in registry.OPS}
assert set(entries) == set(parents['original']) == set(parents['e55']) == expected_keys
report = {}
for (name, key), image in entries.items():
    variant = 'original' if name == target else 'e55'
    matches = image == parents[variant][name, key]
    standalone_matches = image == (A / 'kernels' / (name + '.bin')).read_bytes()
    report[name] = {**registry.fingerprint(image), 'expected_variant': variant,
                    'byte_identical_expected': matches, 'embedded_equals_standalone': standalone_matches}
changed = sorted(name for name, key in entries if entries[name, key] != parents['e55'][name, key])
passed = all(v['byte_identical_expected'] and v['embedded_equals_standalone'] for v in report.values()) and changed == [target]
output = {'schema': 'MOA_SINGLE_KERNEL_REGISTRY_CONTROL_V1',
          'status': 'EXACT_SINGLE_KERNEL_CONTROL_PASS' if passed else 'EXACT_SINGLE_KERNEL_CONTROL_FAIL',
          'checked_at_utc': datetime.now(timezone.utc).isoformat(),
          'target': target, 'host': host_info, 'parents': parent_info,
          'entry_count': len(entries), 'entries': report, 'changed_images_from_e55': changed,
          'target_image_exact_original': report[target]['byte_identical_expected'],
          'other_nine_images_exact_e55': all(v['byte_identical_expected'] for k,v in report.items() if k != target),
          'npu_executed': False, 'logits_accuracy_validated': False}
print(json.dumps(output, indent=2))
raise SystemExit(0 if passed else 1)
