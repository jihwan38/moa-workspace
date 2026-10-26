#!/usr/bin/env python3
"""Strict saved-log replay for job323462; no build, network or device action."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import audit_build

HERE = Path(__file__).resolve().parent
RAW = HERE / 'actual-323462'
P, H, L = 512, 3840, 15360


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def fields(text):
    out = {}
    for token in text.split():
        k, v = token.split('=', 1)
        assert k not in out
        if v in ('true', 'false'):
            v = v == 'true'
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out[k] = v
    return out


def finite(v):
    if isinstance(v, dict):
        for x in v.values(): finite(x)
    elif isinstance(v, list):
        for x in v: finite(x)
    elif isinstance(v, float):
        assert math.isfinite(v)


def analyze(verify_local=False):
    assert sha(RAW / 'raw-hashes.json') == 'f3db97af5a4fdcc00f4611ed28a88459a524193b25f6b2f19b0c70b682426ece'
    hashes = read(RAW / 'raw-hashes.json')
    for n, s in hashes.items(): assert sha(RAW / n) == s, n
    status, run = read(RAW / 'arena-status.json'), read(RAW / 'run.json')
    assert status['id'] == 323462 and status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert run['name'] == 'int4-hadamard64-complete-actual-20261026'
    assert run['result'] == 'command_succeeded' and run['exit_code'] == 0 and run['status'] == ''
    assert run['commit'] == 'f9bcf8e48602dc10fb6643d22bc9ac7261a4a916'
    assert (RAW / 'source.diff').read_bytes() == b'' and read(RAW / 'arena-records.json') == []
    inputs = read(RAW / 'source-hashes.json')
    assert len(inputs) == 432 and inputs == read(HERE / 'a003-device-build/source-hashes.json')
    build = read(HERE / 'build-audit.json')
    artifact = run['submitted_artifact']
    assert artifact['sha256'] == build['artifact']['sha256'] == '96ce0228ce20f0afda25206d2bc4ec1e181eb45d9a77e48da759be4d430ce34d'
    assert artifact['size'] == build['artifact']['size'] == 5686792
    assert build['all19_prior_and_fresh_equal'] and build['q020_control_exact'] and len(build['comparisons']) == 19
    if verify_local:
        assert audit_build.main() == build
        for field in ('path', 'source_path'):
            p = Path(artifact[field])
            assert sha(p) == artifact['sha256'] and p.stat().st_size == artifact['size']
    anchor = read(HERE / 'original-grouped-anchor.json')
    assert anchor['source_job'] == 323322 and anchor['source_log_sha256'] == '21412fc68f0fcf1a999caec0a43540059bc6700515d1301559ce77885c6b7360'
    anchors = {(r['round'], r['step'], r['comparison']): r for r in anchor['records']}
    assert len(anchors) == 16
    lines = (RAW / 'arena.log').read_text().splitlines()
    assert len(lines) == 91
    at = 0

    def take(marker):
        nonlocal at
        prefix = marker + ' '
        assert at < len(lines) and lines[at].startswith(prefix), (at, marker)
        payload = lines[at][len(prefix):]
        at += 1
        value = json.loads(payload) if payload.startswith('{') else fields(payload)
        finite(value)
        return value

    startup = [take('I4_H64_COMPLETE_SETUP')]
    assert startup[0]['original_decode_encode_ms'] > 0
    weights = []
    for name in ('up_proj', 'gate_proj', 'down_proj'):
        w = take('I4_H64_COMPLETE_WEIGHTS')
        assert (w['projection'], w['values'], w['groups']) == (name, H * L, H * L // 960)
        assert all(w[k] is True for k in ('all_row_independent_rotation', 'global_unchanged',
            'no_native_row_dequant', 'provided_BF16_then_F32_rotation_then_BF16_then_rational', 'per_row_scratch_only'))
        assert [s[0] for s in w['source']] == [f'model.language_model.layers.0.mlp.{name}.{suffix}'
            for suffix in ('weight_packed', 'weight_scale', 'weight_global_scale')]
        assert all(isinstance(h, int) and 0 <= h < 2**64 for _, h in w['source'])
        weights.append(w)
    startup.append(take('I4_H64_COMPLETE_SETUP'))
    assert startup[1]['rotated_decode_encode_allrow_independent_audit_ms'] > 0
    assert all(r['layer'] == 0 and r['no_all48_or_runtime_claim'] is True for r in startup)
    owners = take('I4_H64_COMPLETE_OWNERS')
    assert owners == {'additional_packed': 3 * H * L // 2, 'additional_dense_reference': 3 * H * L,
        'additional_scales': 3 * H * L // 960 * 4 + 4, 'activation_rotation': 'CPU_D2H_H2D', 'no_timing': True}
    steps = []
    gates = []
    for order in range(2):
        for step in range(2):
            coordinate = {'round': order, 'step': step}
            intended = []
            for arm in ('rotated', 'original'):
                assert take('I4_H64_COMPLETE_INTENDED_BEGIN') == {**coordinate, 'arm': arm}
                activations, samples = [], []
                stage_names = ('rotated_prefix_own_coordinates', 'rotated_candidate_own_hidden_coordinates') if arm == 'rotated' else ('original_prefix', 'original_candidate_own_hidden')
                for stage, width, projections in zip(stage_names, (H, L), (('UP', 'GATE'), ('DOWN',))):
                    a = take('I4_GROUPED_COMPLETE_ACTIVATION')
                    assert all(a[k] == v for k, v in coordinate.items())
                    assert (a['stage'], a['values']) == (stage, P * width)
                    assert 0 <= a['zero_codes'] <= P * width and 0 <= a['bf16_ftz_zeros'] <= P * width
                    assert math.isclose(a['zero_fraction'], a['zero_codes'] / (P * width), abs_tol=1e-12)
                    assert a['scalar_f32_reconstruction_l2'] >= 0 and a['maxabs'] >= 0
                    assert a['scope'] == 'own_BF16_input_group960_not_output_error_decomposition'
                    activations.append(a)
                    for projection in projections:
                        f = take('I4_H64_COMPLETE_F64')
                        assert all(f[k] == v for k, v in coordinate.items())
                        assert f['projection'] == projection and f['samples'] == P
                        assert f['max_preBF16_radius'] >= 0 and f['oracle'] == 'i64_local_F64_envelope'
                        samples.append(f)
                intended.append({'arm': arm, 'activation_metrics': activations, 'sample_checks': samples})
            drift = []
            byname = {}
            for stage, count in (('joined', 2 * P * L), ('hidden', P * L), ('down', P * H), ('residual', P * H)):
                pair = []
                for arm in ('rotated', 'original'):
                    d = take('I4_H64_COMPLETE_DRIFT')
                    assert all(d[k] == v for k, v in coordinate.items())
                    assert d['comparison'] == arm + '_vs_q020_' + stage and d['bf16_values'] == count
                    assert 0 <= d['changed'] <= count and d['reference_ss'] > 0
                    assert min(d['relative_l2'], d['sse'], d['maxabs']) >= 0
                    assert math.isclose(d['relative_l2'], math.sqrt(d['sse'] / d['reference_ss']), rel_tol=2e-15)
                    assert d['maxabs']**2 <= d['sse'] * (1 + 1e-12)
                    assert d['sse'] <= count * d['maxabs']**2 * (1 + 1e-12)
                    assert d['matched_incoming_inputs'] is True and d['no_task_quality_or_speed_claim'] is True
                    assert d['common_feedback'] == 'original q020 residual'
                    if arm == 'original':
                        prior = anchors[(order, step, d['comparison'])]
                        for k in ('bf16_values', 'changed', 'maxabs'): assert d[k] == prior[k]
                        assert abs(d['relative_l2'] - prior['relative_l2']) <= 2e-12
                    pair.append(d); drift.append(d); byname[d['comparison']] = d
                assert pair[0]['reference_ss'] == pair[1]['reference_ss']
            criteria = {}
            for stage, threshold in (('joined', 1.0), ('down', 1.0), ('residual', 0.75)):
                old = byname['original_vs_q020_' + stage]['relative_l2']
                new = byname['rotated_vs_q020_' + stage]['relative_l2']
                criteria[stage] = {'original_l2': old, 'rotated_l2': new, 'maximum_fraction': threshold,
                    'passed': new <= threshold * old}
            gate = all(c['passed'] for c in criteria.values())
            assert take('I4_H64_COMPLETE_STEP_PASS') == {**coordinate, 'intended_code_parity': True,
                'matched_incoming': True, 'common_q020_feedback': True, 'numerical_gate_passed': gate}
            gates.append(gate)
            steps.append({**coordinate, 'intended_references': intended, 'drift': drift, 'criteria': criteria, 'numerical_gate_passed': gate})
    final = take('PASS_I4_H64_COMPLETE')
    assert at == len(lines)
    candidate_bytes = 17 * P * (H + L) // 2 + 8 * P * ((H + L) // 960)
    production_bytes = 5 * P * (H + L) + 8 * P
    intended_input_bytes = 3 * P * (H + L) // 2 + 4 * P * ((H + L) // 960)
    weight_bytes = 9 * H * L // 2 + 4 * (3 * H * L // 960 + 1)
    replay = 4 * (2 * candidate_bytes + production_bytes + 2 * P * L)
    expected = {'layer': 0, 'intended_bf16_values': 8 * 6 * P * (H + L), 'i64_F64_samples': 8 * 3 * P,
        'permutation_compared_owner_bytes': 4 * candidate_bytes, 'same_raw_input_bytes': 4 * 4 * P * H,
        'immutable_exact_bytes': 8 * intended_input_bytes + replay + 4 * 2 * P * H + 2 * (4096 + 64 + 2 * weight_bytes),
        'reuse_bytes': replay, 'numerical_gate_passed': all(gates),
        'decision': 'FULL_FFN_NUMERICAL_GATE_PASS_ONLY' if all(gates) else 'HOLD',
        'no_timing': True, 'CPU_rotation_roundtrips': True, 'no_model_quality_claim': True, 'no_promotion': True}
    assert final == expected, (final, expected)
    # All owner bytes are compared in the binary. Reported aggregates can differ
    # in the last bits because accumulation follows the physical row order.
    for a, b in zip(steps[:2], steps[2:]):
        for x, y in zip(a['drift'], b['drift']):
            for k in ('comparison', 'bf16_values', 'changed', 'maxabs'): assert x[k] == y[k]
            assert abs(x['relative_l2'] - y['relative_l2']) <= 2e-12
            for k in ('sse', 'reference_ss'): assert math.isclose(x[k], y[k], rel_tol=2e-12)
        assert a['numerical_gate_passed'] is b['numerical_gate_passed']
        for aa, bb in zip(a['intended_references'], b['intended_references']):
            for x, y in zip(aa['activation_metrics'], bb['activation_metrics']):
                varying = {'round', 'scalar_f32_reconstruction_l2'}
                assert {k:v for k,v in x.items() if k not in varying} == {k:v for k,v in y.items() if k not in varying}
                assert abs(x['scalar_f32_reconstruction_l2'] - y['scalar_f32_reconstruction_l2']) <= 2e-12
    return {'decision': final['decision'], 'no_promotion': True, 'job': status, 'strict_records': at,
        'raw_hashes': hashes, 'raw_manifest_sha256': sha(RAW / 'raw-hashes.json'),
        'lineage': {'source_commit': run['commit'], 'functional_source': '70812693', 'inputs': 432,
            'prior_429_inputs_unchanged': True, 'unchanged_device_images': 19, 'q020_control_exact': True,
            'artifact': artifact, 'build_audit_sha256': sha(HERE / 'build-audit.json'),
            'control_anchor_file_sha256': sha(HERE / 'original-grouped-anchor.json'), 'all16_original_control_anchors_reproduced': True},
        'CPU_startup_including_decode_encode_and_audit': startup, 'weight_records': weights,
        'additional_layer0_diagnostic_owners': owners, 'steps': steps, 'final_counts': final,
        'source_derived_counts': {'candidate_scratch_fields': 13, 'candidate_scratch_bytes': candidate_bytes,
            'CPU_checked_activation_packed_bytes': 8 * P * (H + L) // 2,
            'CPU_checked_activation_scale_values': 8 * P * ((H + L) // 960),
            'CPU_checked_activation_offset_values': 8 * P * ((H + L) // 960),
            'all_weight_rotated_BF16_values': 3 * H * L, 'all_weight_group_scales': 3 * H * L // 960},
        'limitations': [
            'Provided layer0 weights/norms and synthetic512-row projected/residual inputs; bothsteps and bothorders use common unchanged q020 incoming feedback. Not an autonomous approximate multilayer history.',
            'All weight rows are rotated/encoded. Candidate own GeGLU BF16 is rotated before DOWN; this closes the previous sampled fixed-production-hidden gap.',
            'Both fixed numerical criteria and intended-code correctness pass here; these do not establish whole-model logits, public task accuracy or serving performance. Official q020 remains unchanged.',
            'Activation metrics compare codes with each arm\'s own BF16 input in its own coordinates. Rotated and original hidden differ, so these metrics are not a common-input output-error decomposition.',
            'Activation rotation uses CPU downloads/transforms/uploads. No kernel/FFN runtime timing was collected; job duration and startup audit times are not serving or all48-layer startup measurements.',
            'Replay compares values of all13 fields and unchanged neighboring owners. residual, rotated_prefix and rotated_hidden are reallocated; same-allocation reuse is not established for those3.',
            'Exact-byte totals count repeated comparisons, not distinct memory footprint. Full retained production/source fingerprints are an additional check, not included in the exact-byte counter.'
        ]}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-local-artifacts', action='store_true')
    args = ap.parse_args()
    print(json.dumps(analyze(args.verify_local_artifacts), indent=2))
