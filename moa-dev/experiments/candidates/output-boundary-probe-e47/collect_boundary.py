#!/usr/bin/env python3
"""Read status/log/archive of one explicitly authorized boundary probe job.

No submit, retry, source edit or build. A diagnostic exit1 is collected normally.
"""
import ast
import datetime
import hashlib
import json
import math
import re
import struct
import subprocess
import tarfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
import argparse
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--config', type=Path, required=True)
_args = _parser.parse_args()
CONFIG = json.loads(_args.config.read_text())
JOB = CONFIG['job_id']
assert isinstance(JOB, int) and JOB > 0
HERE = ROOT / CONFIG['candidate_directory']
ART = ROOT / CONFIG['artifact_directory']
OUT = ROOT / CONFIG['output_record']
LOG = ROOT / CONFIG['raw_log']
assert not any(path.is_absolute() for path in map(Path, [CONFIG['candidate_directory'], CONFIG['artifact_directory'], CONFIG['output_record'], CONFIG['raw_log']]))


def cli(action, *args):
    assert action in {'status', 'logs', 'download'}
    return subprocess.check_output([str(ROOT/'moa-dev/moa'), 'furiosa-arena', action, str(JOB),
                                    *map(str, args)], text=True, stderr=subprocess.STDOUT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not OUT.exists(), 'refuse to overwrite an existing observation'
    changes, last = [], None
    deadline = time.monotonic() + 1800
    while True:
        status = json.loads(cli('status'))
        assert status['id'] == JOB
        current = (status['status'].upper(), status.get('queue_ahead'))
        if current != last:
            changes.append({'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'status': current[0], 'queue_ahead': current[1]})
            print(f'job{JOB}: {current[0]}, queue_ahead={current[1]}', flush=True)
            last = current
        if current[0] in {'SUCCEEDED', 'COMPLETED', 'FAILED', 'CANCELED', 'CANCELLED'} or status.get('exit_code') is not None:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('job still pending; no new job created')
        time.sleep(10)

    log = cli('logs')
    LOG.write_text(log)
    print(f'job{JOB} terminal exit={status.get("exit_code")}; collecting all diagnostic results', flush=True)
    container_tar = f"{CONFIG['scratch_directory']}/job-{JOB}.tar"
    archive_path = ART / f'job-{JOB}.tar'
    cli('download', '--out', container_tar, '--create-dirs')
    subprocess.run(['docker', 'cp', f'moa2026-dev:{container_tar}', str(archive_path)], check=True)
    package, blobs = {}, {}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            assert member.name not in package
            data = archive.extractfile(member).read()
            package[member.name] = {'bytes': len(data), 'sha256': sha(data)}
            blobs[member.name] = data
    frozen_manifest = json.loads((HERE/'manifest.json').read_text())
    assert set(package) == {'test_runtime', 'boundary.safetensors', 'remote_entrypoint.sh', 'job.yaml'}
    for name, expected in frozen_manifest['package'].items():
        assert package[name] == expected, f'uploaded {name} differs from frozen build'
    job_manifest = yaml.safe_load(blobs['job.yaml'])
    assert job_manifest['name'] == CONFIG['expected_job_name']
    assert job_manifest['entrypoint'] == './remote_entrypoint.sh'
    assert job_manifest.get('args', []) == [] and job_manifest.get('env', {}) == {}
    assert isinstance(job_manifest['timeout_sec'], int) and job_manifest['timeout_sec'] > 0
    image = (ROOT/CONFIG['image_file']).read_bytes()
    assert sha(image) == CONFIG['image_sha256']
    assert blobs['test_runtime'].count(image) == 1
    cpu = json.loads((ROOT/CONFIG['cpu_reference']).read_text())
    assert sha(blobs['boundary.safetensors']) == cpu['payload']['sha256']
    # Read expected output without requiring PyTorch or importing a candidate oracle.
    fixture = blobs['boundary.safetensors']
    nheader = struct.unpack_from('<Q', fixture)[0]
    header = json.loads(fixture[8:8+nheader])
    base = 8+nheader
    expected_vectors = {}
    for case in range(8):
        t = header[f'case{case}.expected']
        assert t['dtype'] == 'F32' and t['shape'] == [3840]
        start, end = t['data_offsets']
        expected_vectors[case] = struct.unpack('<3840f', fixture[base+start:base+end])

    begin = re.findall(r'^O_SUITE_BEGIN cases=8 atol=0\.05 rtol=0\.01 diagnostic=true official_performance=false host_only=false$', log, re.M)
    assert len(begin) == 1 and 'host_only=true' not in log
    checks = {}
    pattern = r'^INPUT_CHECK case=(\d+) tensor=(\w+) bytes=(\d+) checksum=([0-9a-f]{8}) status=PASS$'
    for c, name, count, checksum in re.findall(pattern, log, re.M):
        case = int(c)
        key = (case, name)
        assert 0 <= case < 8 and key not in checks
        expected = cpu['records'][case]['checksums'][name]
        size = {'x':8192, 'o_weight':15728640, 'o_weight_scale':7680,
                'post_attn_rms_weight':7680, 'residual':7680}[name]
        assert int(count) == size and int(checksum, 16) == expected
        checks[key] = {'case':case, 'tensor':name, 'bytes':size, 'checksum_hex':checksum, 'status':'PASS'}
    assert len([s for s in log.splitlines() if s.startswith('INPUT_CHECK ')]) == len(checks)
    launches = []
    for c, name in re.findall(r'^O_LAUNCH_BEGIN case=(\d+) name=(\w+)$', log, re.M):
        case = int(c)
        assert case not in launches and name == cpu['records'][case]['name']
        launches.append(case)
    results = {}
    pattern = (r'^O_RESULT case=(\d+) name=(\w+) checked=(\d+) failures=(\d+) nonfinite=(\d+) '
               r'exact_f32=(\d+) max_abs_finite=([^ ]+) mean_abs_finite=([^ ]+) status=(PASS|FAIL)$')
    for c, name, count, failed, nonfinite, exact, maximum, mean, verdict in re.findall(pattern, log, re.M):
        case = int(c)
        failed, nonfinite, exact, count = map(int, (failed, nonfinite, exact, count))
        maximum, mean = float(maximum), float(mean)
        assert case not in results and name == cpu['records'][case]['name'] and count == 3840
        assert 0 <= nonfinite <= failed <= count and 0 <= exact <= count-failed
        assert math.isfinite(maximum) and math.isfinite(mean) and 0 <= mean <= maximum+1e-8
        assert (verdict == 'PASS') == (failed == 0)
        results[case] = {'case':case, 'name':name, 'checked':count, 'failures':failed,
                         'nonfinite':nonfinite, 'exact_f32':exact, 'max_abs_finite':maximum,
                         'mean_abs_finite':mean, 'status':verdict, 'first_mismatches':[], 'boundary_samples':[]}
    assert len([s for s in log.splitlines() if s.startswith('O_RESULT ')]) == len(results)
    for c, values in re.findall(r'^O_GROUP_FAILURES case=(\d+) rows_per_group=120 counts=(\[[^\]]*\])$', log, re.M):
        counts = ast.literal_eval(values)
        r = results[int(c)]
        assert len(counts) == 32 and all(isinstance(n, int) and 0 <= n <= 120 for n in counts)
        assert sum(counts) == r['failures'] and 'group_failures' not in r
        r['group_failures'] = counts
    for c, row, got, want in re.findall(r'^O_BOUNDARY_SAMPLE case=(\d+) row=(\d+) got_bf16=([0-9a-f]{4}) want_bf16=([0-9a-f]{4})$', log, re.M):
        case, row = int(c), int(row)
        expected_bits = struct.unpack('<I', struct.pack('<f', expected_vectors[case][row]))[0] >> 16
        assert int(want, 16) == expected_bits
        results[case]['boundary_samples'].append({'row':row, 'got_bf16':got, 'want_bf16':want})
    for line in log.splitlines():
        if not line.startswith('O_MISMATCH '):
            continue
        v = dict(re.findall(r'(\w+)=([^ ]+)', line))
        case, row = int(v['case']), int(v['row'])
        assert 0 <= row < 3840
        w = expected_vectors[case][row]
        want_bits = struct.unpack('<I', struct.pack('<f', w))[0] >> 16
        assert int(v['want_bf16'], 16) == want_bits
        got_bits = int(v['got_bf16'], 16)
        g = struct.unpack('<f', struct.pack('<I', got_bits << 16))[0]
        error, limit = abs(g-w), .05+.01*abs(w)
        assert not math.isfinite(g) or error > limit
        assert math.isclose(float(v['want']), w, rel_tol=1e-8, abs_tol=1e-38)
        if math.isfinite(g):
            assert math.isclose(float(v['got']), g, rel_tol=1e-8, abs_tol=1e-38)
            assert math.isclose(float(v['error']), error, rel_tol=1e-8, abs_tol=1e-38)
        results[case]['first_mismatches'].append({'row':row, 'got':g if math.isfinite(g) else str(g),
                                                'want':w, 'got_bf16':v['got_bf16'],
                                                'want_bf16':v['want_bf16'],
                                                'error':error if math.isfinite(error) else str(error), 'limit':limit})
    for r in results.values():
        assert len(r['first_mismatches']) == min(32, r['failures'])
        assert len({s['row'] for s in r['first_mismatches']}) == len(r['first_mismatches'])
        assert [s['row'] for s in r['boundary_samples']] == [0,119,120,239,240,1919,1920,3839]
        assert 'group_failures' in r
    final = re.findall(r'^O_SUITE_RESULT completed=(\d+) planned=8 total_failed_values=(\d+) status=(PASS|FAIL)$', log, re.M)
    complete = set(results) == set(range(8))
    failed_values = sum(r['failures'] for r in results.values())
    failed_cases = [case for case, r in sorted(results.items()) if r['status'] == 'FAIL']
    all_pass = complete and not failed_cases
    if complete:
        assert len(final) == 1 and int(final[0][0]) == 8 and int(final[0][1]) == failed_values
        assert (final[0][2] == 'PASS') == all_pass
        assert len(checks) == 40 and launches == list(range(8))
        assert set(checks) == {(case, name) for case in range(8) for name in cpu['records'][case]['checksums']}
        assert status['exit_code'] == (0 if all_pass else 1)
    record = {'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'official_score':False, 'performance_measurement':False, 'job':status,
              'state_observations':changes, 'log':str(LOG.relative_to(ROOT)), 'log_sha256':sha(LOG.read_bytes()),
              'downloaded_archive':str(archive_path.relative_to(ROOT)), 'downloaded_archive_sha256':sha(archive_path.read_bytes()),
              'uploaded_package_files':package, 'uploaded_job_manifest':job_manifest,
              'all_three_uploaded_files_match_frozen_manifest':True, 'exact_frozen_o_image_occurrences_in_uploaded_host':1, 'o_image_sha256':sha(image), 'variant_label':CONFIG['variant_label'],
              'build_manifest':str((HERE/'manifest.json').relative_to(ROOT)), 'build_manifest_sha256':sha((HERE/'manifest.json').read_bytes()),
              'uploaded_payload_matches_cpu_reference_hash':True, 'host_only':False,
              'input_checks':list(checks.values()), 'input_checks_passed':len(checks), 'input_checksum_values_verified':True,
              'launch_cases':launches, 'probe_complete':complete, 'probe_passed':all_pass,
              'failure_classification':'diagnostic_numerical_comparison_failure' if complete and failed_cases else ('all_comparisons_passed' if all_pass else 'incomplete_execution'),
              'completed_cases':len(results), 'planned_cases':8, 'total_values_compared':sum(r['checked'] for r in results.values()),
              'total_failed_values':failed_values, 'failed_cases':failed_cases,
              'total_nonfinite':sum(r['nonfinite'] for r in results.values()), 'results':[results[c] for c in sorted(results)],
              'comparison_fields_and_histograms_validated':True, 'new_npu_jobs_created_by_collector':0,
              'scope':CONFIG['variant_label'] + ': frozen O image, same eight deterministic boundary fixtures, original CPU F32 matmul/BF16 scale/RMS/residual oracle, atol .05/rtol .01. This is not the official seeded score harness, an exhaustive input proof, or a performance result.', 'collector_config':str(_args.config), 'collector_config_sha256':sha(_args.config.read_bytes()),
              'case6_limit':'Very small expected output is below atol; zero output passes, so this is floor-branch/finite control rather than tiny-value preservation proof.'}
    OUT.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)), 'complete':complete, 'passed':all_pass,
                      'exit_code':status['exit_code'], 'failed_cases':failed_cases, 'failed_values':failed_values,
                      'cases':[{k:r[k] for k in ('case','status','failures','nonfinite','max_abs_finite')} for r in results.values()]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
