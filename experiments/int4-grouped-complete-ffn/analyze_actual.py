#!/usr/bin/env python3
"""Replay actual323213 from retained evidence. No builds or device invocation."""
import argparse
import ast
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ELF = '3444ef561e2d39d6a15ec6340548d07e65b873d2884b62ac04f9ea2ccd66cdd4'
PHASES = ['prefix_us', 'input_quant_pack_us', 'upgate_us', 'geglu_bridge_us',
          'hidden_quant_pack_us', 'down_us', 'tail_us']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def record(line):
    out = {}
    for key, value in re.findall(r'(\w+)=([^\s]+)', line):
        if key.endswith('_fnv') or key == 'reciprocal_global_bits':
            out[key] = value
        elif value in ('true', 'false'):
            out[key] = value == 'true'
        elif re.fullmatch(r'-?\d+', value):
            out[key] = int(value)
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def stats(values):
    return {'count': len(values), 'min': min(values), 'mean': statistics.mean(values), 'max': max(values)}


def analyze(verify_local=False):
    raw = HERE / 'actual-323213'
    hashes = read(raw / 'hashes.json')
    for name, expected in hashes.items():
        assert sha(raw / name) == expected['sha256'], name
        assert (raw / name).stat().st_size == expected['bytes'], name
    status = read(raw / 'arena-status.json')
    run = read(raw / 'run.json')
    artifact = run['submitted_artifact']
    assert status['id'] == 323213 and status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert run['exit_code'] == 0 and run['commit'] == '0a7863f37cda228a1fbb05eaae8f982d756fcb62'
    assert run['command'][-4:] == ['--timeout', '70', '--name', 'beef-int4-grouped-complete-actual-20261026']
    assert artifact['sha256'] == ELF and artifact['size'] == 6550888
    built = read(HERE / 'a002-device-build/source-hashes.json')
    submitted = read(raw / 'source-hashes.json')
    assert submitted == built and len(built) == 421
    assert built['Cargo.lock'] == 'a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e'
    assert built['Cargo.toml'] == 'a4154d908d25a512d031dea859b1b013794f821a47deb739334332bd0e4fdf85'
    audit = read(HERE / 'a002-image-audit.json')
    assert audit['registry']['elf_sha256'] == ELF and audit['registry']['record_count'] == 19
    assert len(audit['comparisons']) == 19 and audit['q020_whole_graph_control_exact']
    assert all(x['prior_image_equal'] and x['fresh_out_bin_equal'] for x in audit['comparisons'])
    assert {name: sum(x['prior_actual'] == name for x in audit['comparisons']) for name in
            ['actual323067_complete', 'actual323173_quantizer', 'actual323172_projection']} == {
                'actual323067_complete': 13, 'actual323173_quantizer': 2, 'actual323172_projection': 4}
    if verify_local:
        tree = HERE.parents[1]
        for name, expected in built.items():
            assert sha(tree / name) == expected, name
        for name in ['path', 'source_path']:
            assert sha(Path(artifact[name])) == ELF, name
        replay = json.loads(subprocess.check_output([sys.executable, str(HERE / 'audit_build.py')]))
        assert replay == audit

    lines = (raw / 'arena.log').read_text().splitlines()
    def rows(prefix):
        return [record(line) for line in lines if line.startswith(prefix + ' ')]
    timing = rows('I4_GROUPED_COMPLETE_TIME')
    phase = rows('I4_GROUPED_COMPLETE_PHASE')
    gates = rows('I4_GROUPED_COMPLETE_GATE')
    exact = rows('I4_GROUPED_COMPLETE_EXACT')
    f64 = rows('I4_GROUPED_COMPLETE_F64')
    drift = rows('I4_GROUPED_COMPLETE_DRIFT')
    activation = rows('I4_GROUPED_COMPLETE_ACTIVATION')
    production_activation = rows('I4_GROUPED_COMPLETE_PRODUCTION_ACTIVATION')
    startup = rows('I4_GROUPED_COMPLETE_STARTUP')
    footprint, = rows('I4_GROUPED_COMPLETE_WORKING_SET')
    total, = rows('PASS_I4_GROUPED_COMPLETE_CORRECTNESS')
    final, = rows('I4_GROUPED_COMPLETE_FINAL')
    assert (len(timing), len(phase), len(gates), len(exact), len(f64), len(drift),
            len(activation), len(production_activation), len(startup)) == (8, 8, 2, 4, 12, 12, 8, 8, 3)
    nh, nl = 512 * 3840, 512 * 15360
    groups = (nh + nl) // 960
    weight_values = 3 * 3840 * 15360
    candidate_weights = weight_values // 2 + weight_values // 960 * 4 + 4
    candidate_scratch = 6 * (nh + nl) + (nh + nl) // 2 + 8 * groups
    production_scratch = 5 * (nh + nl) + 4096
    expected = {
        'layer': 0, 'intended_bf16_values': 4 * (5 * nh + 5 * nl),
        'independent_i64_F64_samples': 4 * 3 * 512,
        'permutation_values': 2 * (4 * (nh + nl) + 2 * groups),
        'candidate_input_scale_bits': 4 * groups, 'candidate_offset_values': 4 * groups,
        'candidate_packed_bytes': 4 * (nh + nl) // 2,
        'immutable_bytes_compared': 4 * (3 * (nh + nl) // 2 + 4 * groups) +
                                    2 * (2 * nh + 4160 + candidate_weights + weight_values),
        'production_immutable_fingerprint_bytes': 394728988,
        'post_reuse_bytes': 2 * (candidate_scratch + production_scratch),
        'current_model_drift_informational': True, 'no_model_quality_claim': True,
    }
    assert total == expected, (total, expected)
    assert {(x['round'], x['step']) for x in exact} == {(r, s) for r in range(2) for s in range(2)}
    assert all(x['all_packed_bytes'] and x['tail_included'] and x['scale_bits'] == groups and
               x['offset_values'] == groups and x['input_codes'] == nh and x['hidden_codes'] == nl for x in exact)
    assert sum(x['intended_bf16_values'] for x in exact) == total['intended_bf16_values']
    assert sum(x['independent_i64_F64_samples'] for x in exact) == sum(x['samples'] for x in f64) == 6144
    assert all(x['samples'] == 512 and x['max_preBF16_radius'] >= 0 for x in f64)
    assert {(x['round'], x['step'], x['projection']) for x in f64} == {
        (r, s, p) for r in range(2) for s in range(2) for p in ['UP', 'GATE', 'DOWN']}
    assert footprint['candidate_weight_bytes'] == candidate_weights
    assert footprint['intended_reference_weight_bytes'] == weight_values
    assert footprint['candidate_scratch_bytes'] == candidate_scratch
    assert footprint['production_scratch_bytes'] == production_scratch
    assert footprint['production_retained_weight_bytes'] == total['production_immutable_fingerprint_bytes']

    aggregates = []
    for r in range(2):
        group = [x for x in timing if x['round'] == r]
        assert [x['block'] for x in group] == [0, 1, 2, 3]
        assert [x['candidate'] for x in group] == [True, False, False, True]
        assert all(x['samples'] == 8 and x['control'] == 'current_production_E4_grouped_norm_store_graph'
                   and x['excludes'] == 'reset_upload_download_startup_oracle' for x in group)
        mean = {arm: sum(Decimal(str(x['complete_us'])) for x in group if x['candidate'] == arm) / 2
                for arm in [False, True]}
        gain = 100 * (1 - mean[True] / mean[False])
        gate, = [x for x in gates if x['round'] == r]
        assert abs(mean[False] - Decimal(str(gate['control_us']))) <= Decimal('0.000001')
        assert abs(mean[True] - Decimal(str(gate['candidate_us']))) <= Decimal('0.000001')
        assert abs(gain - Decimal(str(gate['gain_percent']))) <= Decimal('0.000001')
        assert gate['threshold_percent'] == 10 and gate['passed'] == (gain >= 10)
        aggregates.append({'round': r, 'production_mean_us': float(mean[False]),
                           'candidate_mean_us': float(mean[True]), 'candidate_added_us': float(mean[True] - mean[False]),
                           'gain_percent': float(gain), 'threshold_percent': 10,
                           'passed': gain >= 10, 'complete_samples_per_arm': 16})
    assert final == {'performance_gate_passed': all(x['passed'] for x in aggregates),
                     'both_row_orders_required': True, 'prior_ungrouped_DOWN_resource_HOLD': True,
                     'model_quality_gate_pending': True}

    # Phase instrumentation is separate from the preregistered ABBA gate.
    assert {(x['round'], x['repeat']) for x in phase} == {(r, i) for r in range(2) for i in range(4)}
    for x in phase:
        assert abs(sum(x[p] for p in PHASES) - x['sum_us']) <= 0.00001
        assert abs(x['sum_us'] + x['gap_us'] - x['total_us']) <= 0.00001 and x['gap_us'] >= 0
        assert x['scope'] == 'completed_await_wall_not_device_trace'
        assert x['excludes'] == 'reset_download_validation_startup'
    phase_summary = []
    for r in [0, 1, None]:
        selected = [x for x in phase if r is None or x['round'] == r]
        phase_summary.append({'round': r, 'stages_us': {p: stats([x[p] for x in selected]) for p in PHASES},
                              **{p: stats([x[p] for x in selected]) for p in ['sum_us', 'total_us', 'gap_us', 'production_us']},
                              'gate_usage': 'diagnostic only; cannot replace matched ABBA'})

    for entries in [activation, production_activation]:
        assert {(x['round'], x['step'], x['stage']) for x in entries} == {
            (r, s, p) for r in range(2) for s in range(2) for p in ['prefix', 'hidden']}
        for x in entries:
            assert x['values'] == (nh if x['stage'] == 'prefix' else nl)
            assert 0 <= x['zero_codes'] <= x['values']
            assert abs(x['zero_codes'] / x['values'] - x['zero_fraction']) <= 0.000000000001
            if 'bf16_ftz_zeros' in x:
                assert 0 <= x['bf16_ftz_zeros'] <= x['values'] and math.isfinite(x['scalar_f32_reconstruction_l2'])
    assert {(x['round'], x['step'], x['stage']) for x in drift} == {
        (r, s, p) for r in range(2) for s in range(2) for p in ['joined', 'down', 'final_residual']}
    for x in drift:
        assert x['matched_inputs'] == (x['step'] == 0) and x['no_model_quality_claim']
        assert x['bf16_values'] == (2 * nl if x['stage'] == 'joined' else nh)
        x['input_relation'] = 'matched_entry_residual' if x['step'] == 0 else 'independently_evolved_feedback'
    # True row permutation preserves aggregate metrics at the printed precision.
    for entries in [activation, production_activation, drift]:
        for x in [x for x in entries if x['round'] == 0]:
            y, = [y for y in entries if y['round'] == 1 and y['step'] == x['step'] and y['stage'] == x['stage']]
            assert {k: v for k, v in x.items() if k != 'round'} == {k: v for k, v in y.items() if k != 'round'}

    weights = []
    for line in lines:
        if line.startswith('I4_GROUPED_COMPLETE_WEIGHT '):
            head, sources = line.split(' source=', 1)
            item = record(head)
            item['provided_source_fingerprints'] = ast.literal_eval(sources)
            weights.append(item)
    assert [x['projection'] for x in weights] == ['UP', 'GATE', 'DOWN']
    old_log = HERE.parent / 'int4-complete-ffn/actual-323067/arena.log'
    old_weights, old_drift = [], []
    for line in old_log.read_text().splitlines():
        if line.startswith('I4_COMPLETE_WEIGHT '):
            head, sources = line.split(' source=', 1)
            x = record(head); x['provided_source_fingerprints'] = ast.literal_eval(sources); old_weights.append(x)
        if line.startswith('I4_COMPLETE_DRIFT '):
            x = record(line)
            if x['round'] == 0 and x['step'] == 0:
                old_drift.append(x)
    assert weights == old_weights, 'Provided per960 weight encoding must remain identical to original complete probe'
    assert startup[0]['values'] == weight_values and startup[0]['no_native_E4_row_dequant']
    markers = [line for line in lines if line.startswith('PROFILE_I4_GROUPED_COMPLETE ')]
    records = read(raw / 'arena-records.json')
    assert markers == [] and records == []
    assert not any('FURIOSA_OPT_PROFILE' in x for x in run['command'])
    return {
        'job': status, 'submission_commit': run['commit'], 'build_commit': audit['source_commit'],
        'artifact': artifact, 'raw_hashes': hashes, 'verified_source_config_hashes': 421,
        'exact_prior_images': 19, 'q020_production_control_image_exact': True,
        'correctness': total, 'per_feedback_checks': exact, 'f64_samples': f64,
        'timing_blocks': timing, 'recomputed_ABBA_gates': aggregates, 'printed_gates': gates, 'final_gate': final,
        'raw_phase_clock_records': phase, 'phase_clock_summary': phase_summary,
        'own_input_A4_reconstruction': activation, 'production_E4_own_input_zeros': production_activation,
        'current_production_drift': drift, 'historical_ungrouped_step0_drift': old_drift,
        'historical_comparison': {'raw_log_sha256': sha(old_log), 'weight_encodings_identical': True,
                                  'scope': 'Same provided layer0 and synthetic entry fixture; grouped activation and scale ordering both differ. Hidden inputs differ; not a factorial error decomposition.'},
        'startup_records': startup, 'weights': weights, 'declared_payload': footprint,
        'declared_payload_subtotal_bytes': sum(footprint[k] for k in ['production_retained_weight_bytes', 'candidate_weight_bytes',
            'intended_reference_weight_bytes', 'candidate_scratch_bytes', 'production_scratch_bytes']),
        'device_trace': {'requested': False, 'markers': markers, 'records': records, 'available': False},
        'decision': 'HOLD: both complete-chain 10% speed gates failed; no model promotion.',
        'scope': [
            'Supplied layer0 weights/norms/globals and synthetic finite512-row projected/residual inputs, two true row orders and two feedback steps.',
            'BF16 comparison counts include repeated typed/raw/reference checks; not distinct model logits or task accuracy.',
            'All grouped scales/relative offsets and packed bytes checked independently against the BF16 rational oracle.',
            'Matched ABBA includes seven awaits, checked raw binding, all BF16/packed/metadata stores and rereads, and final norm/residual/layer gate.',
            'Original q020 one-call production graph is the performance control. Intended dense-E4 code references are correctness controls only.',
            'Phase clocks are eight separately restored diagnostic repeats, not device per-op timings and not an alternative speed gate.',
            'No observed trace explains timing changes; do not infer a per-launch constant, warmup cause, or subtract standalone component times.',
            'Reconstruction metrics use scalar RN32(code*group_scale) then F64 SSE on each candidate BF16 input; they are not a dot-product error decomposition.',
            'Prefix BF16 is matched at step0; hidden already differs through the approximate UP/GATE path. Step1 residual feedback diverges in both arms.',
            'Startup includes CPU audit and intended-reference uploads; not an optimized candidate-only or48-layer startup bound.',
            'Payload subtotal excludes temporary oracle buffers, CPU copies, allocator overhead and device DM/TRF; not server peak or model-fit evidence.',
            'Original production immutable tensors use full-read FNV fingerprints; candidate/input/reference tensors use bytewise equality.',
            'No full-attention prefix, padded tail, whole model, task-quality, HTTP or official score gate was run.',
        ],
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-local-artifacts', action='store_true')
    args = ap.parse_args()
    print(json.dumps(analyze(args.verify_local_artifacts), indent=2))
