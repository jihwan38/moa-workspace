#!/usr/bin/env python3
"""Replay job323067 from frozen logs; no compilation, model or device calls."""
import argparse
import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ELF = 'bd51ab5f4c0cf7e20918e022c023c7e42fcbad4fa11b30da382c33297088e61f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(line):
    out = {}
    for key, value in re.findall(r'(\w+)=([^\s]+)', line):
        if key.endswith('_fnv') or key == 'reciprocal_global_bits':
            out[key] = value
        elif value in ('true', 'false'):
            out[key] = value == 'true'
        elif re.fullmatch(r'-?\d+', value):
            out[key] = int(value)
        elif re.fullmatch(r'-?\d+\.\d+(?:e[+-]?\d+)?', value):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def read_json(path):
    return json.loads(path.read_text())


def analyze(verify_local=False):
    raw = HERE / 'actual-323067'
    hashes = read_json(raw / 'hashes.json')
    for name, expected in hashes.items():
        assert sha(raw / name) == expected['sha256'], name
        assert (raw / name).stat().st_size == expected['bytes'], name
    status = read_json(raw / 'arena-status.json')
    assert status['id'] == 323067 and status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    run = read_json(raw / 'run.json')
    artifact = run['submitted_artifact']
    assert artifact['sha256'] == ELF and artifact['size'] == 6496968
    assert run['commit'] == '2392c7ba1a515db0d2ab685c582f30cbfb57a7c4'
    assert run['command'][-4:] == ['--timeout', '70', '--name', 'beef-int4-complete-ffn-actual-20261026']
    assert 'FURIOSA_OPT_PROFILE=1' in run['command']
    build_hashes = read_json(HERE / 'a005-int4-complete-ffn-device-build/source-hashes.json')
    submit_hashes = read_json(raw / 'source-hashes.json')
    assert submit_hashes == build_hashes and len(build_hashes) == 411
    assert build_hashes['Cargo.lock'] == 'a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
    assert build_hashes['Cargo.toml'] == 'a4154d908d25a512d031dea859b1b013794f821a47deb739334332bd0e4fdf85'
    image_audit = read_json(HERE / 'a005-image-audit.json')
    assert image_audit['current']['elf_sha256'] == ELF
    assert len(image_audit['comparisons']) == 18
    assert all(x['status'] == 'equal' for x in image_audit['comparisons'])
    if verify_local:
        tree = HERE.parents[1]
        for name, expected in build_hashes.items():
            assert sha(tree / name) == expected, name
        for name in ['path', 'source_path']:
            assert sha(Path(artifact[name])) == ELF

    lines = (raw / 'arena.log').read_text().splitlines()
    def rows(prefix):
        return [record(line) for line in lines if line.startswith(prefix + ' ')]
    timings = rows('I4_COMPLETE_TIME')
    gates = rows('I4_COMPLETE_GATE')
    exact = rows('I4_COMPLETE_EXACT')
    f64 = rows('I4_COMPLETE_F64')
    drift = rows('I4_COMPLETE_DRIFT')
    startup = rows('I4_COMPLETE_STARTUP')
    footprint, = rows('I4_COMPLETE_WORKING_SET')
    total, = rows('PASS_I4_COMPLETE_CORRECTNESS')
    final, = rows('I4_COMPLETE_FINAL')
    assert len(timings) == 8 and len(gates) == 2 and len(exact) == 4 and len(f64) == 12 and len(drift) == 12
    assert {(x['round'], x['step']) for x in exact} == {(a, b) for a in range(2) for b in range(2)}
    assert all(x['all_packed_bytes'] and x['tail_included'] and x['scale_bits'] == 1024 for x in exact)
    expected = {'layer': 0, 'intended_bf16_values': 196608000, 'independent_i64_F64_samples': 6144,
                'permutation_values': 78645248, 'candidate_input_scale_bits': 4096,
                'candidate_packed_bytes': 19660800, 'immutable_bytes_compared': 599464072,
                'production_immutable_fingerprint_bytes': 394728988, 'post_reuse_bytes': 226115584}
    for key, value in expected.items():
        assert total[key] == value, (key, total[key], value)
    assert sum(x['intended_bf16_values'] for x in exact) == total['intended_bf16_values']
    assert sum(x['independent_i64_F64_samples'] for x in exact) == sum(x['samples'] for x in f64) == 6144
    assert sum((x['input_codes'] + x['hidden_codes']) // 2 for x in exact) == 19660800
    assert all(x['samples'] == 512 for x in f64)
    assert final == {'performance_gate_passed': False, 'both_row_orders_required': True,
                     'DOWN_resource_HOLD': True, 'model_quality_gate_pending': True}

    aggregates = []
    for round_index in range(2):
        group = [x for x in timings if x['round'] == round_index]
        assert [x['block'] for x in group] == [0, 1, 2, 3]
        assert [x['candidate'] for x in group] == [True, False, False, True]
        assert all(x['samples'] == 8 for x in group)
        assert all(x['control'] == 'current_production_E4_grouped_norm_store_graph' for x in group)
        assert all(x['excludes'] == 'reset_upload_download_startup_oracle' for x in group)
        means = {arm: sum(Decimal(str(x['complete_us'])) for x in group if x['candidate'] == arm) / 2
                 for arm in [False, True]}
        gain = 100 * (1 - means[True] / means[False])
        gate, = [x for x in gates if x['round'] == round_index]
        assert abs(means[False] - Decimal(str(gate['control_us']))) <= Decimal('0.000001')
        assert abs(means[True] - Decimal(str(gate['candidate_us']))) <= Decimal('0.000001')
        assert abs(gain - Decimal(str(gate['gain_percent']))) <= Decimal('0.000001')
        assert gate['threshold_percent'] == 10 and gate['passed'] is False and gain < 0
        aggregates.append({'round': round_index, 'control_mean_us': float(means[False]),
                           'candidate_mean_us': float(means[True]), 'candidate_slower_percent': float(-gain),
                           'candidate_added_us': float(means[True] - means[False]),
                           'gain_percent': float(gain), 'threshold_percent': 10, 'passed': False,
                           'complete_samples_per_arm': 16})
    # A true permutation preserves aggregate drift, while selected i64/F64 sample rows differ.
    for step in range(2):
        for stage in ['joined', 'down', 'final_residual']:
            a, = [x for x in drift if (x['round'], x['step'], x['stage']) == (0, step, stage)]
            b, = [x for x in drift if (x['round'], x['step'], x['stage']) == (1, step, stage)]
            assert {k: v for k, v in a.items() if k != 'round'} == {k: v for k, v in b.items() if k != 'round'}
    for x in drift:
        x['input_relation'] = 'matched_initial_residual' if x['step'] == 0 else 'independently_evolved_feedback_residuals'
    weights = []
    for line in lines:
        if line.startswith('I4_COMPLETE_WEIGHT '):
            head, sources = line.split(' source=', 1)
            item = record(head)
            item['provided_source_fingerprints'] = ast.literal_eval(sources)
            weights.append(item)
    assert [x['projection'] for x in weights] == ['UP', 'GATE', 'DOWN']
    weight_names = ['production_retained_weight_bytes', 'candidate_weight_bytes', 'intended_reference_weight_bytes']
    scratch_names = ['candidate_scratch_bytes', 'production_scratch_bytes']
    assert sum(footprint[k] for k in weight_names + scratch_names) == 774083104
    assert footprint['candidate_weight_bytes'] == 89210884 and footprint['candidate_scratch_bytes'] == 63901696
    assert sum(x['samples'] for x in timings) == 64
    assert total['post_reuse_bytes'] == 2 * sum(footprint[k] for k in scratch_names)
    assert total['immutable_bytes_compared'] == 4*(512*(3840+15360)*3//2+4096) + 2*(89210884+177085440+3932160+4160)
    assert len(startup) == 3 and startup[0]['values'] == 176947200
    startup_ms = sum(value for row in startup for key, value in row.items() if key.endswith('_ms'))
    profile_markers = [line for line in lines if line.startswith('PROFILE_I4_COMPLETE ')]
    assert profile_markers == ['PROFILE_I4_COMPLETE candidate=false', 'PROFILE_I4_COMPLETE candidate=true']
    assert read_json(raw / 'arena-records.json') == []
    return {
        'job': status, 'submission_commit': run['commit'], 'artifact': artifact,
        'build_commit': image_audit['build_commit'], 'functional_source_commit': '70f4c15f',
        'verified_source_config_hashes': 411, 'equal_prior_kernel_images': 18, 'embedded_images': 19,
        'raw_hashes': hashes, 'correctness': total, 'per_feedback_checks': exact, 'f64_samples': f64,
        'timing_blocks': timings, 'recomputed_gates': aggregates, 'printed_gates': gates, 'final_gate': final,
        'startup_records': startup, 'reported_startup_sum_ms': startup_ms,
        'declared_payload': footprint, 'declared_payload_subtotal_bytes': 774083104,
        'weights': weights, 'current_E4_drift': drift,
        'profile': {'requested': True, 'markers': profile_markers, 'arena_records': [],
                    'limitation': 'No device trace payload in collected records; no per-stage bottleneck attribution.'},
        'decision': 'HOLD: complete-chain performance gate failed both row orders; NO model promotion.',
        'scope': [
            'Actual supplied layer0 weights/norms/globals; synthetic finite projected/residual inputs, all512 rows.',
            '196608000 counts comparison events, including repeated feedback and typed/reference checks; not distinct model logits.',
            'Intended rational A4 parity is not equality to the production E4 model.',
            'Candidate timer includes seven full awaited calls, public raw-owner validation, both BF16 canonical writes/reads and tail.',
            'Production timer is the unchanged q020 one-call grouped-store E4 complete graph, not the older per960 F4 prototype.',
            'Eight samples per block, ABBA, two row orders;64 total timed complete chains,32 per arm.',
            'CPU encoding, all startup transfers/prepack, residual restores, validation and downloads are outside both timers.',
            'Startup221.836340ms includes candidate and intended-reference E4 upload; not a candidate-only startup measurement.',
            'Payload is declared retained tensors, excludes temporary reference device buffers, CPU snapshots, allocator and kernel DM/TRF.',
            'Original production immutability is full-read FNV fingerprint equality, separate from bytewise immutable checks.',
            'Step0 drift is matched input; step1 is cumulative separate residual feedback. Neither is a model/task-quality result.',
            'No full-attention prefix, padded tails, fullmodel, HTTP, task-quality or official gain gate was run.',
            'DOWN resource245684>240000 remains HOLD. Correctness-success process exit did not pass the separate speed gate.'
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-local-artifacts', action='store_true')
    args = parser.parse_args()
    print(json.dumps(analyze(args.verify_local_artifacts), indent=2))
