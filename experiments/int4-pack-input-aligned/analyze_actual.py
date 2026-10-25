#!/usr/bin/env python3
"""Replay job323012 raw checks and same-job ABBA; no build/network/device work."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
TREE = HERE.parent.parent
ROOT = TREE.parent
RAW = HERE/'actual-323012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(values):
    assert values and all(x > 0 for x in values)
    return dict(samples_us=values, mean_us=statistics.mean(values),
                median_us=statistics.median(values), min_us=min(values), max_us=max(values))


def main():
    manifest = json.loads((RAW/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(RAW/name) == expected, name
    status = json.loads((RAW/'arena-status.json').read_text())
    assert (status['id'], status['status'], status['exit_code']) == (323012, 'SUCCEEDED', 0)
    run = json.loads((RAW/'run.json').read_text())
    assert run['commit'] == manifest['run_evidence_commit'] and run['exit_code'] == 0
    build_audit = load_module('aligned_build_audit', HERE/'audit_build.py')
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        build_audit.main()
    build = json.loads(captured.getvalue())
    assert build['source_commit'] == manifest['source_commit']
    source_hashes = json.loads((RAW/'source-hashes.json').read_text())
    built_hashes = json.loads((HERE/'runs/a004-int4-pack-input-aligned-build/source-hashes.json').read_text())
    assert source_hashes == built_hashes and len(source_hashes) == 405
    for relative, digest in source_hashes.items():
        assert sha(TREE/relative) == digest, relative
    submitted = ROOT/'runs/int4-pack-input-aligned-20261026/submitted/test_int4_pack_input_aligned'
    assert sha(submitted) == manifest['submitted_elf_sha256'] == run['submitted_artifact']['sha256']
    registry = load_module('aligned_registry', Path(build['registry_parser']['path']))
    bound = registry.compare(registry.read_artifact(submitted), build['registry_comparison']['right'])
    assert bound['equal_common_count'] == 6 and bound['different_common_count'] == 0
    assert bound['left_only'] == bound['right_only'] == []

    log = (RAW/'arena.log').read_text()
    cases = {}
    for match in re.finditer(r'^PACK_P512_CASE width=(\d+) case=(\d+) codes=(\d+) packed_bytes=(\d+) whole_owner_exact=true typed_raw_canonical_exact=true$', log, re.M):
        width, case, codes, packed = map(int, match.groups())
        assert (width, case) not in cases and codes == 512*width and packed == codes//2
        cases[width, case] = dict(codes=codes, packed_bytes=packed)
    assert set(cases) == {(w, c) for w in [3840, 15360] for c in range(5)}
    totals = dict(codes=sum(c['codes'] for c in cases.values()),
                  packed_bytes=sum(c['packed_bytes'] for c in cases.values()),
                  owner_rejection_checks=6, independent_original_packed_bytes=24576000)
    assert totals['codes'] == 49152000 and totals['packed_bytes'] == 24576000
    assert log.count('PASS_PACK_P512 ') == log.count('PASS_PACK_ALIGNED ') == 1
    assert 'PASS_PACK_P512 codes=49152000 packed_bytes=24576000 owner_rejection_checks=6 scope=integer_packing_and_raw_blocked_layout_only no_quantizer_gemm_model_gain_claim=true' in log
    assert 'PASS_PACK_ALIGNED original_packed_bytes=24576000 aligned_input_executed=true hidden_staged_executed=true failed_four_row_input_executed=false complete_FFN_or_model_claim=false' in log
    assert log.index('PASS_PACK_P512 ') > log.rindex('PACK_P512_CASE ')
    assert log.index('PACK_ALIGNED_PERFORMANCE ') > log.index('PASS_PACK_ALIGNED ')
    assert 'PACK_ALIGNED_PERFORMANCE required_each_H_case_gain=0.20 pass=true correctness_complete=true' in log

    paired = {}
    pattern = r'^PACK_ALIGNED_ABBA width=(\d+) case=(\d+) samples=16 candidate_us=(\[[^\]]+\]) original_us=(\[[^\]]+\]) input_same_function_control=false all_real_DMA_charged=true$'
    for match in re.finditer(pattern, log, re.M):
        width, case = map(int, match.group(1, 2))
        candidate, original = map(json.loads, match.group(3, 4))
        assert (width, case) not in paired and len(candidate) == len(original) == 2
        paired[width, case] = dict(candidate=summarize(candidate), original=summarize(original),
            mean_fraction_saved=1-statistics.mean(candidate)/statistics.mean(original),
            conservative_fraction_saved=1-max(candidate)/min(original))
    assert set(paired) == {(w, c) for w in [3840, 15360] for c in [0, 4]}
    gates = {}
    for match in re.finditer(r'^PACK_ALIGNED_H_GATE case=(\d+) candidate_worst_us=([^ ]+) original_best_us=([^ ]+) required_gain=0.20 pass=true$', log, re.M):
        case = int(match.group(1)); candidate, original = map(float, match.group(2, 3))
        pair = paired[3840, case]
        assert case not in gates and candidate == pair['candidate']['max_us'] and original == pair['original']['min_us']
        assert candidate <= 0.8*original
        gates[case] = dict(pass_gate=True, required_fraction_saved=0.20,
                          measured_conservative_fraction_saved=1-candidate/original)
    assert set(gates) == {0, 4}

    standalone = {}
    pattern = r'^PACK_P512_TIMING width=(\d+) case=(\d+) repetitions=16 pack_us=(\[[^\]]+\]) pack_plus_raw_copy_us=(\[[^\]]+\]) complete_awaited_calls=true excludes_quantizer_gemm_setup_readback=true$'
    for match in re.finditer(pattern, log, re.M):
        width, case = map(int, match.group(1, 2))
        pack, copied = map(json.loads, match.group(3, 4))
        assert (width, case) not in standalone and len(pack) == len(copied) == 2
        standalone[width, case] = dict(pack=summarize(pack), pack_plus_raw_layout_copy=summarize(copied))
    assert set(standalone) == set(paired)
    by_width = {}
    for width in [3840, 15360]:
        candidate = [x for c in [0, 4] for x in paired[width, c]['candidate']['samples_us']]
        original = [x for c in [0, 4] for x in paired[width, c]['original']['samples_us']]
        copied = [x for c in [0, 4] for x in standalone[width, c]['pack_plus_raw_layout_copy']['samples_us']]
        by_width[width] = dict(candidate=summarize(candidate), original=summarize(original),
            mean_fraction_saved=1-statistics.mean(candidate)/statistics.mean(original),
            pack_plus_raw_layout_copy=summarize(copied))
    spans = len(re.findall(r'^NPU_SPAN', log, re.M))
    markers = len(re.findall(r'^PACK_P512_PROFILE', log, re.M))
    assert json.loads((RAW/'arena-records.json').read_text()) == [] and spans == markers == 0
    resource = json.loads((HERE/'resource-result.json').read_text())
    result = dict(
        job=323012, status=status, source_commit=manifest['source_commit'],
        submit_evidence_commit=run['commit'], raw_manifest_sha256=sha(RAW/'manifest.json'),
        raw_files=manifest['files'], checked_source_inputs=405,
        source_map_sha256=sha(RAW/'source-hashes.json'),
        captured_and_submitted_elf_sha256=sha(submitted), submitted_registry=bound['left'],
        fresh_images=6, prior_control_images_unchanged=5,
        new_H_image_matches_resource=True, correctness=totals,
        fixture_scope='Five cases/width: mixed unique512-row tags, all-8, all7, zero, exact row/channel permutation; all top nibbles and full owner/typed/raw checks. Full post-reuse immutability checks precede final speed assertion.',
        preregistered_H_gates=gates, by_width=by_width,
        paired_cases={f'{w}:{c}': v for (w, c), v in paired.items()},
        separate_pack_copy_cases={f'{w}:{c}': v for (w, c), v in standalone.items()},
        profiling=dict(status='NOT_COLLECTED', npu_span_count=spans, profile_marker_count=markers,
                       note='This successful run contains wall timings only; no per-operation runtime DMA/TU attribution or additional profile rerun.'),
        prior_H_resource_holds=resource['preserved_holds'],
        interpretation='PASS for complete standalone pack and checked raw layout-copy correctness; H per-case20percent gate passes. L is the unchanged staged hidden control. No quantizer/GEMM/completeFFN/model accuracy or throughput claim.',
        device_work_in_this_analysis=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
