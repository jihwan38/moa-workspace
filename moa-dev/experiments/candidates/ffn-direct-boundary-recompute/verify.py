#!/usr/bin/env python3
"""Recheck frozen compile/upload provenance and exact diagnostic oracle coverage."""
import json
import re
import sys
from build import HERE, ROOT, ARTIFACTS, digest
from audit_descriptors import audit

label = sys.argv[1]
build = json.loads((HERE / f'build-{label}.json').read_text())
variant = HERE / build['variant']
recipe = json.loads((variant / 'recipe.json').read_text())
assert digest((variant / 'recipe.json').read_bytes()) == build['recipe']
assert build['container_inputs_verified'] and build['diagnostic_only']
assert build['exact_compile_exit_code'] == build['build_exit_code'] == 0
assert build['exact_compiles'] == {k: 0 for k in recipe['exact_kernels']}
artifact = ARTIFACTS / label
evidence = HERE / 'evidence' / label
for name, expected in build['source_inputs'].items():
    assert digest((artifact / 'crate' / name).read_bytes()) == expected, name
    assert digest((evidence / name).read_bytes()) == expected, name
for dest, src in {'src/lib.rs': 'lib.rs', 'src/probe.rs': 'probe.rs', 'src/bin/boundary_probe.rs': 'boundary_probe.rs', **recipe['extra_sources']}.items():
    assert digest((variant / src).read_bytes()) == build['source_inputs'][dest], src
for group in ['logs', 'dumps']:
    for name, expected in build[group].items():
        assert digest((artifact / name).read_bytes()) == expected, name
        if group == 'logs' or name.endswith('.json'):
            assert digest((evidence / name).read_bytes()) == expected, name
assert {p.name: digest(p.read_bytes()) for p in (artifact / 'package').iterdir()} == build['package_files']
host = (artifact / 'package/boundary_probe').read_bytes()
images = list((artifact / 'kernels').rglob('*.bin'))
assert len(images) == recipe['expected_images']
for path in images:
    data = path.read_bytes()
    assert {**digest(data), 'host_occurrences': host.count(data)} == build['images'][path.name]
    assert host.count(data) == 1
assert json.loads((HERE / f'descriptor-audit-{label}.json').read_text()) == audit(label)
result_path = HERE / f'{label}-results.json'
report = {'label': label, 'source_build_descriptors': 'PASS', 'scope': 'Primitive correctness only; no full FFN performance claim'}
if result_path.exists():
    result = json.loads(result_path.read_text())
    log = (ROOT / result['log']).read_bytes()
    assert digest(log) == result['log_hash']
    assert result['actual_upload_verified']
    assert {k: v for k, v in result['package_files'].items() if k != 'job.yaml'} == build['package_files']
    assert result['results'] == [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in log.decode().splitlines() if line.startswith('BOUNDARY_RESULT ')]
    assert result['stage_exits'] == dict(re.findall(r'BOUNDARY_EXIT_(\w+) (\d+)', log.decode()))
    modes = {}
    for mode in recipe['modes']:
        rows = [row for row in result['results'] if row['mode'] == mode]
        expected = {(str(case), str(proj)) for case in range(4) for proj in range(2)}
        coverage = len(rows) == 8 and {(r['case'], r['projection']) for r in rows} == expected and all(int(r['checked']) == 15360 for r in rows)
        passed = coverage and result['stage_exits'].get(mode) == '0' and all(int(r['mismatches']) == int(r['nonfinite']) == 0 for r in rows)
        modes[mode] = {'passed': passed, 'complete_coverage': coverage, 'rows_checked': sum(int(r['checked']) for r in rows), 'mismatches': sum(int(r['mismatches']) for r in rows), 'exit': result['stage_exits'].get(mode)}
    report.update(job_id=result['job']['job_id'] if 'job_id' in result['job'] else result['job'].get('id'), modes=modes, screen_complete=result['complete'])
    if '--write' in sys.argv:
        with (HERE / f'verification-{label}.json').open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
else:
    report['arena'] = 'NOT_RUN'
print(json.dumps(report, indent=2))
