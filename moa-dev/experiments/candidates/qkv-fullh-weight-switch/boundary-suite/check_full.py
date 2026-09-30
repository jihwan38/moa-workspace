#!/usr/bin/env python3
"""Offline binding/results recheck; no build, remote request or NPU launch."""
import hashlib
import json
import re
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]

def load(path):
    return json.loads(path.read_text())

def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

cfg = load(HERE/'collector-config.json')
build = load(ROOT/cfg['build_record'])
result = load(ROOT/cfg['output'])
assert digest((ROOT/build['selected_build']).read_bytes()) == build['selected_build_digest']
assert result['config_sha256'] == digest((HERE/'collector-config.json').read_bytes())['sha256']
assert result['verdict'] == 'PASS' and result['job']['id'] == cfg['job_id']
assert result['job']['exit_code'] == 0
assert cfg['package'] == build['package']
with tarfile.open(ROOT/cfg['archive']) as archive:
    files = [m for m in archive if m.isfile()]
    assert len(files) == len({m.name for m in files})
    blobs = {m.name: archive.extractfile(m).read() for m in files}
assert {n: digest(v) for n, v in blobs.items()} == result['package']
assert {n: digest(v) for n, v in blobs.items() if n != 'job.yaml'} == cfg['package']
image = (ROOT/cfg['image']).read_bytes()
assert digest(image)['sha256'] == cfg['image_sha256'] == result['qkv_image_sha256']
assert blobs['test_runtime'].count(image) == 1
cpu_path = ROOT/build['cpu_reference_source']
assert digest(cpu_path.read_bytes()) == build['cpu_reference_digest']
cpu = load(cpu_path)
assert digest(blobs['boundary.safetensors'])['sha256'] == cpu['payload']['sha256']
assert (cpu['atol'], cpu['rtol'], cpu['cases']) == (.04, .01, 2)
log = (ROOT/cfg['log']).read_text()
records = [json.loads(s) for s in log.splitlines() if s.startswith('{')]
assert [r for r in records if r['kind'] == 'qkv_result'] == result['results']
assert [r for r in records if r['kind'] == 'cache_check'] == result['cache_checks']
assert len(result['results']) == 6 and sum(r['count'] for r in result['results']) == 16384
assert len(result['cache_checks']) == 8
assert all(r['fail'] == 0 and r['status'] == 'PASS' for r in records)
assert sum(r['count'] for r in result['cache_checks'] if r['phase'] == 'after') == 8380416
checks = re.findall(r'^INPUT_CHECK case=(\d+) tensor=(\w+) bytes=(\d+) checksum=([0-9a-f]{8}) status=PASS$', log, re.M)
assert len(checks) == len({(c, n) for c, n, _, _ in checks}) == 38
for c, n, _, value in checks:
    assert int(value, 16) == cpu['records'][int(c)]['checksums'][n]
assert log.count('QKV_SUITE_RESULT completed=2 planned=2 total_failed_values=0 cache_failed_values=0 status=PASS') == 1
assert build['production_inputs_match'] and build['host_check']['full_input_sha256_passed'] == 38
print('PASS: frozen whole-QKV image, original fixture, uploaded bytes, 16384 outputs, 8380416 untouched cache entries')
