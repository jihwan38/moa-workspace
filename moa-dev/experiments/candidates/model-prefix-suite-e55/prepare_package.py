#!/usr/bin/env python3
"""Copy verified host/inspector artifacts and the runner; never execute them."""
from pathlib import Path
import hashlib
import json
import shutil
if not __debug__:
    raise RuntimeError('Package verification requires assertions; Python -O is forbidden.')
C = Path(__file__).resolve().parent
R = C.parents[3]
A = R / 'moa-dev/experiments/artifacts/model-prefix-suite-e55'
P = A / 'package'
P.mkdir(exist_ok=True)
fingerprint = lambda p: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
registry = json.loads((C / 'registry-summary.json').read_text())
assert registry['status'] == 'ELF_REGISTRY_IDENTITY_ONLY_PASS'
sources = {}
for variant in ('original', 'e55'):
    source = A / variant / 'probe_model_prefix_suite'
    assert fingerprint(source) == registry['hosts'][variant]['binary']
    assert registry['hosts'][variant]['all_ten_images_equal_fixed_prefix_host']
    sources[f'probe_model_{variant}'] = source
inspector = R / 'moa-dev/experiments/artifacts/arena-model-assets-rust-inspector/moa-model-asset-inspector'
assert fingerprint(inspector) == {'bytes': 880872, 'sha256': '1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11'}
sources['moa-model-asset-inspector'] = inspector
sources['remote_entrypoint.sh'] = C / 'remote_entrypoint.sh'
for basename, source in sources.items():
    shutil.copyfile(source, P / basename)
    (P / basename).chmod(0o755)
assert {p.name for p in P.iterdir()} == set(sources)
files = {name: {'source': str(source.relative_to(R)), **fingerprint(P / name)} for name, source in sources.items()}
manifest = {'schema': 'MOA_MODEL_PREFIX_SUITE_PACKAGE_V1', 'files': files,
            'package_path': str(P.relative_to(R)), 'file_count': 4,
            'order': ['asset_inspector_full_hash', 'original_emit', 'reference_sha256',
                      'e55_compare', 'asset_inspector_header_only'],
            'npu_executed_during_preparation': False, 'accuracy_acceptance_threshold': None, 'accuracy_pass': None}
(C / 'package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
