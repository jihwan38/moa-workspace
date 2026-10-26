#!/usr/bin/env python3
"""Strict replay of job323322, including source/ELF/image binding. No execution."""
from pathlib import Path
import json
import math
import audit_build

HERE = Path(__file__).resolve().parent
RAW = HERE / 'actual-323322'
P, H, L = 512, 3840, 15360


def read(path):
    return json.loads(path.read_text())


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


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value)


def main():
    assert audit_build.sha(RAW / 'raw-hashes.json') == '8a3b1068b76c0889bbde8daebfbe48b319095fdd9d60a86ea35dc20d367e6006'
    hashes = read(RAW / 'raw-hashes.json')
    for name, digest in hashes.items(): assert audit_build.sha(RAW / name) == digest, name
    status, run = read(RAW / 'arena-status.json'), read(RAW / 'run.json')
    assert status['id'] == 323322 and status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert run['name'] == 'int4-weight-mse-actual-20261026' and run['exit_code'] == 0
    assert run['result'] == 'command_succeeded' and (RAW / 'source.diff').read_bytes() == b''
    assert set(run['status'].splitlines()) == {
        '?? experiments/int4-weight-mse-diagnostic/' + name for name in
        ('a004-device-build/', 'audit_build.py', 'build-audit.json')}
    build = audit_build.main()
    assert build == read(HERE / 'build-audit.json')
    assert run['commit'] == build['source_commit']
    inputs = read(RAW / 'source-hashes.json')
    assert len(inputs) == 427 and inputs == read(HERE / 'a004-device-build/source-hashes.json')
    artifact = run['submitted_artifact']
    assert artifact['sha256'] == build['artifact']['sha256'] and artifact['size'] == build['artifact']['size']
    for name in ('source_path', 'path'):
        path = Path(artifact[name])
        assert audit_build.sha(path) == artifact['sha256'] and path.stat().st_size == artifact['size']
    assert read(RAW / 'arena-records.json') == []
    lines = (RAW / 'arena.log').read_text().splitlines()
    assert len(lines) == 108
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

    setup = take('I4_WEIGHT_MSE_SETUP')
    assert setup['domain_magnitudes'] == 238 and setup['no_model_data_in_table'] is True
    original_start = take('I4_WEIGHT_MSE_STARTUP')
    assert original_start['layer'] == 0 and original_start['no_model_startup_claim'] is True
    weights = []
    for name in ('up_proj', 'gate_proj', 'down_proj'):
        w = take('I4_WEIGHT_MSE_ENCODING')
        assert w['projection'] == name and w['provided_bf16_values'] == H * L
        assert w['groups'] == w['strictly_improved_groups'] == H * L // 960
        choices = w['candidate_choice_counts']
        assert len(choices) == 26 and all(isinstance(x, int) and x >= 0 for x in choices)
        assert sum(choices) == w['groups'] and choices[0] == w['allzero_original_groups'] == 0
        assert w['global_unchanged'] is True and w['original_pair_included'] is True
        assert w['squared_grid_exponent'] == -76
        baseline = int(w['exact_real_baseline_sse_grid_units'])
        fitted = int(w['exact_real_selected_sse_grid_units'])
        norm = int(w['exact_source_ss_grid_units'])
        assert 0 < fitted < baseline < (1 << 126) and 0 < norm < (1 << 126)
        for label, exact in (('baseline', baseline), ('selected', fitted)):
            assert math.isclose(w[f'exact_real_{label}_relative_l2'], math.sqrt(exact / norm), rel_tol=1e-14)
            scalar = w['scalar_RN32_' + label]
            assert scalar['values'] == H * L and 0 <= scalar['reconstructed_zeros'] <= H * L
            assert scalar['sse'] >= 0 and scalar['reference_ss'] > 0 and scalar['maxabs'] >= 0
            assert math.isclose(scalar['relative_l2'], math.sqrt(scalar['sse'] / scalar['reference_ss']), rel_tol=1e-14)
            assert math.isclose(scalar['reference_ss'], math.ldexp(float(norm), -76), rel_tol=1e-12)
        assert [s[0] for s in w['source']] == [f'model.language_model.layers.0.mlp.{name}.{suffix}'
                for suffix in ('weight_packed', 'weight_scale', 'weight_global_scale')]
        weights.append(w)
    fitted_start = take('I4_WEIGHT_MSE_STARTUP')
    assert fitted_start['layer'] == 0 and fitted_start['includes_RN32_metrics'] is True
    assert fitted_start['not_encode_only_or_model_startup'] is True
    owners = take('I4_WEIGHT_MSE_OWNERS')
    assert owners == {'additional_fitted_packed_bytes': 3 * H * L // 2,
                     'additional_fitted_dense_reference_bytes': 3 * H * L,
                     'additional_fitted_scale_bytes': 3 * H * L // 960 * 4 + 4,
                     'no_model_capacity_claim': True}
    steps = []
    for row_order in range(2):
        for step in range(2):
            coordinate = {'round': row_order, 'step': step}

            def check_coordinate(v):
                assert all(v[k] == value for k, value in coordinate.items())

            intended = []
            for arm in ('fitted', 'original'):
                assert take('I4_WEIGHT_MSE_INTENDED_BEGIN') == {**coordinate, 'arm': arm}
                activations, samples = [], []
                for stage, width, projections in (('prefix', H, ('UP', 'GATE')), ('hidden', L, ('DOWN',))):
                    a = take('I4_GROUPED_COMPLETE_ACTIVATION')
                    check_coordinate(a)
                    assert a['stage'] == stage and a['values'] == P * width
                    assert 0 <= a['zero_codes'] <= P * width and 0 <= a['bf16_ftz_zeros'] <= P * width
                    assert math.isclose(a['zero_fraction'], a['zero_codes'] / (P * width), abs_tol=1e-12)
                    assert a['scalar_f32_reconstruction_l2'] >= 0 and a['maxabs'] >= 0
                    assert a['scope'] == 'own_BF16_input_group960_not_output_error_decomposition'
                    activations.append(a)
                    for projection in projections:
                        f = take('I4_GROUPED_COMPLETE_F64')
                        check_coordinate(f)
                        assert f['projection'] == projection and f['samples'] == P
                        assert f['max_preBF16_radius'] >= 0 and f['oracle'] == 'i64_local_F64_envelope'
                        samples.append(f)
                intended.append({'arm': arm, 'activation_metrics': activations, 'sample_checks': samples})
            assert intended[0]['activation_metrics'][0] == intended[1]['activation_metrics'][0]
            drift = []
            for stage, count in (('joined', 2 * P * L), ('hidden', P * L), ('down', P * H), ('residual', P * H)):
                for comparison in ('MSE_vs_q020', 'original_vs_q020', 'MSE_vs_original'):
                    d = take('I4_WEIGHT_MSE_DRIFT')
                    check_coordinate(d)
                    assert d['comparison'] == comparison + '_' + stage and d['bf16_values'] == count
                    assert 0 <= d['changed'] <= count and d['relative_l2'] >= 0 and d['maxabs'] >= 0
                    assert d['matched_incoming_inputs'] is True and d['no_model_quality_claim'] is True
                    assert d['common_feedback'] == 'original production residual'
                    drift.append(d)
            assert take('I4_WEIGHT_MSE_STEP_PASS') == {**coordinate, **{k: True for k in
                ('matched_incoming', 'original_production_feedback', 'intended_code_parity', 'weight_SSE_not_task_quality')}}
            steps.append({**coordinate, 'intended_references': intended, 'drift': drift})
    final = take('PASS_I4_WEIGHT_MSE_DIAGNOSTIC')
    candidate_bytes = 13 * P * (H + L) // 2 + 8 * P * ((H + L) // 960)
    production_bytes = 5 * P * (H + L) + 8 * P
    intended_input_bytes = 3 * P * (H + L) // 2 + 4 * P * ((H + L) // 960)
    weight_bytes = 9 * H * L // 2 + 4 * (3 * H * L // 960 + 1)
    expected = {'layer': 0, 'intended_bf16_values': 8 * 5 * P * (H + L),
                'sampled_quantized_outputs': 8 * 3 * P,
                'permutation_values': 4 * (4 * P * (H + L) + 2 * P * ((H + L) // 960)),
                'same_input_bytes': 4 * (4 * P * H + P * H // 2 + 8 * P * (H // 960)),
                'immutable_bytes_compared': 8 * intended_input_bytes + 4 * 2 * P * H + 2 * (4096 + 64 + 2 * weight_bytes),
                'reuse_bytes': 4 * (2 * candidate_bytes + production_bytes + 2 * P * L),
                'no_timing_gate': True, 'no_task_quality_claim': True}
    assert final == expected, (final, expected)
    assert at == len(lines), 'Unparsed record, possible failure or additional result'
    # Binary compares full row permutation. Independently verify reported aggregate invariants.
    for a, b in zip(steps[:2], steps[2:]):
        for x, y in zip(a['drift'], b['drift']):
            assert x['comparison'] == y['comparison'] and x['changed'] == y['changed'] and x['maxabs'] == y['maxabs']
            assert abs(x['relative_l2'] - y['relative_l2']) <= 2e-12
        for aa, bb in zip(a['intended_references'], b['intended_references']):
            for x, y in zip(aa['activation_metrics'], bb['activation_metrics']):
                varying = {'round', 'scalar_f32_reconstruction_l2'}
                assert {k:v for k,v in x.items() if k not in varying} == {k:v for k,v in y.items() if k not in varying}
                assert abs(x['scalar_f32_reconstruction_l2'] - y['scalar_f32_reconstruction_l2']) <= 2e-12
    comparisons = []
    for s in steps:
        d = {x['comparison']: x for x in s['drift']}
        for stage in ('joined', 'hidden', 'down', 'residual'):
            old, new = d['original_vs_q020_' + stage], d['MSE_vs_q020_' + stage]
            comparisons.append({'round': s['round'], 'step': s['step'], 'stage': stage,
                                'original_relative_l2': old['relative_l2'], 'MSE_relative_l2': new['relative_l2'],
                                'original_maxabs': old['maxabs'], 'MSE_maxabs': new['maxabs']})
        assert d['MSE_vs_q020_residual']['relative_l2'] > d['original_vs_q020_residual']['relative_l2']
        assert d['MSE_vs_q020_hidden']['relative_l2'] < d['original_vs_q020_hidden']['relative_l2']
    return {'decision': 'NO_PROMOTION_FIXED_WEIGHT_MSE', 'execution': status,
            'source_commit': run['commit'], 'functional_source': '4d3dd463',
            'raw_files': hashes, 'strict_records': at, 'run_record': run, 'submitted_artifact': artifact,
            'source_inputs': len(inputs), 'same_prior_actual323213_device_images': 19,
            'build_replay': {'result': 'PASS', 'script_sha256': audit_build.sha(HERE / 'audit_build.py'),
                             'result_sha256': audit_build.sha(HERE / 'build-audit.json')},
            'immutable_table_setup': setup, 'original_cpu_diagnostic_startup': original_start,
            'fitted_cpu_diagnostic_startup': fitted_start, 'weight_metrics': weights,
            'additional_layer0_diagnostic_owners': owners, 'steps': steps, 'final_counts': final,
            'matched_output_comparisons': comparisons,
            'scope': [
                'Provided layer0 weights/norms and synthetic512-row projected/residual inputs; two row orders and two steps. Both steps share the same original production incoming residual across all three arms.',
                'All184320 groups strictly improve exact-real weight SSE, including the original code+stored-scale pair. RN32 scalar reconstruction L2 also improves here, but weight maximum error increases; no additive output-error decomposition follows.',
                'Hidden BF16 relative L2 slightly improves against q020. Joined, DOWN and final residual relative L2 worsen. Own-hidden scalar reconstruction compares each arm with its own distinct hidden tensor, not a common hidden baseline.',
                'Intended-code/reference, sampled i64/F64 bounds, prefix equality, permutation, immutable-owner and full-reuse gates pass. This is no task-quality or full-model-logit test.',
                'No timed complete-chain gate or per-operation profile was collected. The26.859s job duration includes CPU preparation, device/reference launches and downloads. CPU times include full encoding and audits, not isolated encoder or48-layer startup performance.',
                'Permutation counts combine BF16/group metadata/unpacked codes. Immutable/reuse totals count repeated byte comparisons, not distinct allocation footprint. Additional owners are diagnostic layer0 buffers, not serving capacity.',
                'Existing threegraph10percent performance HOLD, gamma-policy HOLD, official q020 and final repository remain unchanged.'
            ]}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))
