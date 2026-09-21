#!/usr/bin/env python3
"""Read one QKV diagnostic job; verify its frozen package and complete results.

No submission, retry, build or source change. An exit1 numerical failure is valid
diagnostic evidence only when both cases and all output/cache checks completed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[4]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    job = config['job_id']
    assert isinstance(job, int) and job > 0
    out = ROOT / config['output']
    assert not out.exists(), 'refuse to overwrite an observation'

    def cli(action, *rest):
        assert action in {'status', 'logs', 'download'}
        return subprocess.check_output([str(ROOT/'moa-dev/moa'), 'furiosa-arena',
                                        action, str(job), *map(str, rest)],
                                       text=True, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + 1800
    observations, previous = [], None
    while True:
        status = json.loads(cli('status'))
        assert status['id'] == job
        current = (status['status'].upper(), status.get('queue_ahead'))
        if current != previous:
            observations.append({'at_utc': datetime.now(timezone.utc).isoformat(),
                                 'status': current[0], 'queue_ahead': current[1]})
            print(f'job {job}: {current[0]}, ahead={current[1]}', flush=True)
            previous = current
        if current[0] in {'SUCCEEDED', 'FAILED', 'COMPLETED', 'CANCELLED', 'CANCELED'} or status.get('exit_code') is not None:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('same job remains pending; no restart or submission')
        time.sleep(10)

    log = cli('logs')
    (ROOT/config['log']).write_text(log)
    tar_path = ROOT/config['archive']
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    remote_tar = f'/tmp/qkv-diagnostic-downloads/job-{job}.tar'
    cli('download', '--out', remote_tar, '--create-dirs')
    subprocess.run(['docker', 'cp', f'moa2026-dev:{remote_tar}', str(tar_path)], check=True)
    blobs = {}
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if member.isfile():
                assert member.name not in blobs
                blobs[member.name] = archive.extractfile(member).read()
    assert set(blobs) == {'test_runtime', 'boundary.safetensors', 'remote_entrypoint.sh', 'job.yaml'}
    package = {name: {'bytes': len(raw), 'sha256': sha(raw)} for name, raw in blobs.items()}
    assert set(config['package']) == set(blobs)-{'job.yaml'}
    for name, expected in config['package'].items():
        assert package[name] == expected, f'uploaded {name} differs from frozen package'
    manifest = yaml.safe_load(blobs['job.yaml'])
    assert manifest['name'] == config['expected_job_name']
    assert manifest['entrypoint'] == './remote_entrypoint.sh'
    assert manifest.get('args', []) == [] and manifest.get('env', {}) == {}
    assert manifest['timeout_sec'] == 70
    image = (ROOT/config['image']).read_bytes()
    assert sha(image) == config['image_sha256']
    assert blobs['test_runtime'].count(image) == 1
    cpu = json.loads((ROOT/config['cpu']).read_text())
    assert sha(blobs['boundary.safetensors']) == cpu['payload']['sha256']
    assert cpu['cases'] == 2 and cpu['atol'] == .04 and cpu['rtol'] == .01
    fixture = blobs['boundary.safetensors']
    size = struct.unpack_from('<Q', fixture)[0]
    header = json.loads(fixture[8:8+size])
    vectors = {}
    for case in range(2):
        for key, count in [('q', 4096), ('k', 2048), ('v', 2048)]:
            t = header[f'case{case}.expected.{key}']
            assert t['dtype'] == 'F32' and t['shape'] == [count]
            start, end = t['data_offsets']
            vectors[case, key] = struct.unpack(f'<{count}f', fixture[8+size+start:8+size+end])
            assert all(map(math.isfinite, vectors[case, key]))

    begin = f'QKV_SUITE_BEGIN cases=2 atol=0.04 rtol=0.01 diagnostic=true official_performance=false host_only=false pos={cpu["pos"]} slot={cpu["slot"]}'
    assert log.splitlines().count(begin) == 1 and 'host_only=true' not in log
    sizes = {'x': 7680, 'input_rms_weight': 7680, 'q_weight': 4096*3840,
             'k_weight': 2048*3840, 'v_weight': 2048*3840,
             'q_weight_scale': 8192, 'k_weight_scale': 4096, 'v_weight_scale': 4096,
             'q_rms_weight': 512, 'k_rms_weight': 512, 'cos_row': 512, 'sin_row': 512,
             'cos': 32768*256*2, 'sin': 32768*256*2, 'rope_offset': 4, 'kv_offset': 4,
             'k_cache_init': 1024*2048*2, 'v_cache_init': 1024*2048*2, 'q_out_init': 8192}
    checks = {}
    for c, name, n, value in re.findall(r'^INPUT_CHECK case=(\d+) tensor=(\w+) bytes=(\d+) checksum=([0-9a-f]{8}) status=PASS$', log, re.M):
        case = int(c)
        assert case in range(2) and (case, name) not in checks
        assert int(n) == sizes[name] and int(value, 16) == cpu['records'][case]['checksums'][name]
        checks[case, name] = value
    assert set(checks) == {(case, name) for case in range(2) for name in sizes}
    assert sum(line.startswith('INPUT_CHECK ') for line in log.splitlines()) == 38
    launches = re.findall(r'^QKV_LAUNCH_BEGIN case=(\d+) name=(\w+)$', log, re.M)
    assert launches == [(str(c), cpu['records'][c]['name']) for c in range(2)]
    records = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
    results, caches = {}, {}
    for r in records:
        case = r['case']
        assert case in range(2)
        assert (r['status'] == 'PASS') == (r['fail'] == 0)
        assert r['status'] in {'PASS', 'FAIL'}
        if r['kind'] == 'qkv_result':
            key = (case, r['output'])
            assert key not in results and key in vectors
            assert r['name'] == cpu['records'][case]['name']
            assert r['count'] == len(vectors[key])
            assert 0 <= r['nonfinite'] <= r['fail'] <= r['count']
            assert 0 <= r['exact'] <= r['count']-r['fail']
            assert math.isfinite(r['max_error_finite']) and r['max_error_finite'] >= 0
            mismatch = r['first_mismatch']
            assert (mismatch is None) == (r['fail'] == 0)
            if mismatch:
                i = mismatch['index']
                assert 0 <= i < r['count'] and mismatch['want'] == vectors[key][i]
                got = struct.unpack('<f', struct.pack('<I', int(mismatch['got_bf16'], 16) << 16))[0]
                if math.isfinite(got):
                    assert got == mismatch['got']
                    assert abs(got-mismatch['want']) > .04+.01*abs(mismatch['want'])
            results[key] = r
        elif r['kind'] == 'cache_check':
            key = (case, r['cache'], r['phase'])
            assert key not in caches and r['cache'] in {'k', 'v'} and r['phase'] in {'before', 'after'}
            after = r['phase'] == 'after'
            assert r['count'] == 1024*2048-(2048 if after else 0)
            assert r['excluded_start'] == (cpu['slot']*2048 if after else 0)
            assert r['excluded_end'] == ((cpu['slot']+1)*2048 if after else 0)
            assert r['sentinel_bf16'] == {'k':'3e80', 'v':'bf00'}[r['cache']]
            assert 0 <= r['fail'] <= r['count']
            caches[key] = r
        else:
            raise ValueError(f'unexpected JSON record {r["kind"]}')
    assert set(results) == set(vectors)
    assert set(caches) == {(c,k,p) for c in range(2) for k in ['k','v'] for p in ['before','after']}
    assert len(re.findall(r'^QKV_RESULT ', log, re.M)) == 6
    assert len(re.findall(r'^CACHE_CHECK ', log, re.M)) == 8
    failed = sum(r['fail'] for r in results.values())
    cache_failed = sum(r['fail'] for r in caches.values())
    verdict = 'PASS' if failed+cache_failed == 0 else 'FAIL'
    final = f'QKV_SUITE_RESULT completed=2 planned=2 total_failed_values={failed} cache_failed_values={cache_failed} status={verdict}'
    assert log.splitlines().count(final) == 1
    assert status['exit_code'] == (0 if verdict == 'PASS' else 1)
    record = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'variant': config['variant'], 'job': status, 'observations': observations,
              'package': package, 'job_manifest': manifest, 'qkv_image_sha256': sha(image),
              'image_occurrences': 1, 'input_checks_passed': len(checks),
              'results': list(results.values()), 'cache_checks': list(caches.values()),
              'non_target_cache_elements_checked': sum(r['count'] for r in caches.values() if r['phase']=='after'),
              'total_failed_output_values': failed, 'total_failed_cache_values': cache_failed,
              'verdict': verdict, 'config_sha256': sha(args.config.read_bytes()),
              'scope': 'Two deterministic QKV cancellation diagnostics outside the official QKV input-generator support; not a performance result, rank, or all-input proof.'}
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'verdict': verdict, 'output_failures': failed,
                      'cache_failures': cache_failed, 'uploaded_hashes_verified': True}), flush=True)


if __name__ == '__main__':
    main()
