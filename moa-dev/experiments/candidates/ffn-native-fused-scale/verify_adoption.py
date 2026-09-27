#!/usr/bin/env python3
"""Recheck adopted E128 provenance and existing NPU evidence; no new jobs/builds."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from package import checked_package, digest, file_record, read
import validation_extra
sys.path.insert(0, str(ROOT / 'moa-dev'))
from extra_seeds import parse_log
from experiment_index import validate_baseline


def verify(check_live):
    for command in [
        ['verify_graph.py', 'e128-six-row-v1', '--check'],
        ['validation_performance.py', '--check'],
        ['probe.py', '--check'],
        ['validation/probe_model_binding.py', '--check'],
    ]:
        subprocess.run([sys.executable, str(HERE / command[0]), *command[1:]], cwd=ROOT, check=True)
    package_path, package, _, _ = checked_package('e128-six-row-v1')
    assert validation_extra.verify() == read(HERE / 'validation/extra-image-verification.json')
    extra = read(HERE / 'validation/extra-results.json')
    assert extra['valid'] and extra['actual_upload_matches']
    assert file_record(ROOT / extra['log']['path']) == extra['log']
    parsed = parse_log((ROOT / extra['log']['path']).read_text())
    assert parsed == {key: extra[key] for key in parsed}
    assert parsed['output_comparisons_passed'] == 30
    archive = ROOT / extra['archive']['path']
    assert file_record(archive) == extra['archive']
    with tarfile.open(archive) as tar:
        members = [member for member in tar if member.isfile()]
        assert len(members) == len({member.name for member in members})
        files = {member.name: digest(tar.extractfile(member).read()) for member in members}
    assert files == extra['package_files']
    assert {key: value for key, value in files.items() if key != 'job.yaml'} == read(
        HERE / 'build-e128-six-row-extra6-repro.json')['package_files']
    baseline = read(ROOT / 'baseline/e128.json')
    validate_baseline(baseline, check_live)
    audit = read(ROOT / baseline['source_build_audit'])
    measured_path = ROOT / audit['measured_build']
    measured = read(measured_path)
    compiled = read(HERE / 'build-e128-six-row-v1-repro.json')
    assert audit['measured_build_digest'] == file_record(measured_path)
    assert measured['source_inputs'] == compiled['source_inputs'] == read(ROOT / baseline['compile_inputs'])
    assert measured['images'] == audit['images'] == package['images']
    assert measured['package_files'] == package['package_files']
    assert measured['original_package_record'] == file_record(package_path)
    assert measured['original_compile_record'] == file_record(HERE / 'build-e128-six-row-v1-repro.json')
    assert audit['adopted_source_commit'] == baseline['git']['commit']
    assert read(HERE / 'decision.json')['adopted'] is True
    for suffix, actual in [('extra', 'extra-results.json'), ('probe', 'probe-results-v2b.json')]:
        assert read(ROOT / f'docs/experiments/e128-{suffix}-correctness.json') == read(HERE / 'validation' / actual)
    evidence = [ROOT / 'baseline/e128.json', ROOT / baseline['source_build_audit'],
                measured_path, ROOT / baseline['compile_inputs'], HERE / 'decision.json',
                ROOT / baseline['performance_report'], ROOT / baseline['recent_control_report'],
                ROOT / 'docs/experiments/e128-extra-correctness.json',
                ROOT / 'docs/experiments/e128-probe-correctness.json', HERE / 'graph-verification.json']
    return {'valid': True, 'baseline': 'E128', 'source_commit': baseline['git']['commit'],
            'source_tag_manifest_matches': True, 'compile_inputs': len(compiled['source_inputs']),
            'standard_candidate_comparisons': 45, 'extra_comparisons': 30, 'probe_output_elements': 245760,
            'independent_performance_allocations': 3, 'evidence': [file_record(path) for path in evidence],
            'official_submission': False, 'full_model_tested': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-live', action='store_true')
    args = parser.parse_args()
    result = verify(args.check_live)
    path = HERE / 'validation/final-verification.json'
    if args.check:
        assert read(path) == result
    else:
        with path.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print('E128 adoption/source/performance/extra/probe verification PASS')
