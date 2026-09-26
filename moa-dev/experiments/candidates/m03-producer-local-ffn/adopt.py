#!/usr/bin/env python3
"""Checkpoint validated M03 as E122, then bind its recipe to an immutable tag."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = str(HERE.relative_to(ROOT))
CRATE = 'furiosa-opt-gemma4-12B'
PERFORMANCE = 'docs/experiments/e122-m03-adoption-validation.json'
CONTROL = 'docs/experiments/e122-m03-e104-paired-control.json'
EXTRA = 'docs/experiments/e122-m03-extra-correctness.json'
PROBE = 'docs/experiments/e122-m03-probe-correctness.json'


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def source():
    from validation_verify import verify
    checked = verify()
    assert checked == read(PREFIX + '/validation/verification.json')
    assert checked['valid'] and checked['projection_output_elements'] == 241920
    build = read(PREFIX + '/build-m03-native-local-v1-repro.json')
    control = read('moa-dev/experiments/candidates/c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json')
    manifest = build['source_inputs']
    mlp = 'src/device/shared/mlp.rs'
    assert {k: v for k, v in manifest.items() if k != mlp} == {
        k: v for k, v in control['source_inputs'].items() if k != mlp}
    candidate = (HERE / 'native-local-v1/mlp.rs').read_bytes()
    assert digest(candidate) == manifest[mlp]
    for name, expected in manifest.items():
        raw = candidate if name == mlp else (ROOT / CRATE / name).read_bytes()
        assert digest(raw) == expected, name
    for name in ('sliding_project_qkv', 'sliding_attention_output'):
        assert build['images'][name] == control['images'][name]
    perf = read(PERFORMANCE)
    assert perf['valid'] and perf['independent_arena_jobs'] == 6
    assert all(p['delta_percent'] < 0 for p in perf['paired_observations'])
    (ROOT / CRATE / mlp).write_bytes(candidate)
    write(PREFIX + '/compiled-source-manifest.json', manifest)
    write(PREFIX + '/source-build-audit.json', {
        'candidate': 'E122', 'adopted_source_candidate': 'M03 native-local-v1',
        'source_origin': build['base_commit'], 'performance_control': 'baseline/e104.json',
        'measured_build': PREFIX + '/build-m03-native-local-v1-repro.json',
        'compile_inputs': len(manifest), 'live_compile_inputs_match_frozen_build': True,
        'changed_runtime_files': [mlp], 'public_device_entrypoints_unchanged': True,
        'qkv_output_images_unchanged': True, 'basic_and_extra_images_identical': True,
        'images': build['images'], 'validation': [PERFORMANCE, CONTROL, EXTRA, PROBE],
        'verification': PREFIX + '/validation/verification.json',
        'median_paired_ffn_delta_percent': perf['median_paired_delta_percent'],
        'adoption_budget_exception': perf['adoption_budget_exception'],
        'full_model_tested': False, 'official_submission': False,
        'scope': 'Exact existing M03 source; nine distinct standard inputs and seven separate projection cases. Native input representation remains approximate.'})
    print('E122 source checkpoint ready; commit and tag baseline-e122 before recipe generation.')


def recipe():
    commit = subprocess.check_output(['git', 'rev-parse', 'baseline-e122^{commit}'], cwd=ROOT, text=True).strip()
    for name, expected in read(PREFIX + '/compiled-source-manifest.json').items():
        path = 'moa-dev/locks/gemma4-Cargo.lock' if name == 'Cargo.lock' else CRATE + '/' + name
        assert digest(subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)) == expected
    data = read('baseline/e104.json')
    evidence = [PREFIX + '/README.md', PERFORMANCE, EXTRA, PROBE]
    data.update(id='E122', git={'tag': 'baseline-e122', 'commit': commit},
        compile_inputs=PREFIX + '/compiled-source-manifest.json',
        source_build_audit=PREFIX + '/source-build-audit.json',
        performance_report=PERFORMANCE, recent_control_report=CONTROL,
        correctness_evidence=[EXTRA, PROBE, 'docs/experiments/e55-qkv-boundary-probe.json', 'docs/experiments/output-boundary-comparison.md'],
        scope='E122 adopts exact M03: E104 native materialized up/gate, cluster-local GeGLU, L-owned four-column down, F32 cluster-partial sum before BF16. QKV/O and public interfaces unchanged. Native input representation remains approximate; no all-input or full-model equivalence claim.')
    replacements = {
        'F1': ('L 절반의 소유권을 up/gate·GeGLU·down까지 유지',
            '각 cluster가 L7680을 소유한다. Native up/gate의30행 F32 출력을 같은 cluster 안에서120행으로 모아 BF16으로 바꾸고 GeGLU를 계산한다. Down 입력은 그 L 절반 안에서 compact DMA와 ring32로 네 column에 방송한다. 전체 L의 cross-cluster gather/복제를 제거한다.',
            'up: l=7680*c+30*s+r; GeGLU: l=7680*c+120*r+t, slice=4*r; down: h=60*r+t, l=7680*c+1920*k+u, slice=4*r+k',
            'Weight/scale/activation의 동일 L-half 소유권을 유지한다. 네 column의 F32 합 뒤 두 cluster의 H partial을 더하며 합 이전 BF16 rounding을 추가하지 않는다.',
            ['project_up_and_gate', 'cast_gathered_rows', 'geglu', 'project_down_tile', 'finish_down']),
        'F2': ('Materialized native up/gate와 BF16 down DPE',
            'E104의 FP4→E5M2 DM→세 항 native dot→F32 block partial을 그대로 쓴다. Down은 각 cluster의 L 절반에 대해 FP4→E4M3 DM→BF16 lookup/dot으로16원소 F32 block partial을 만든다.',
            'up/gate30×240 F32 partial/slice; down60×120 F32 partial/slice, 네 L1920 columns/cluster',
            '이 채택본은 FP8 weight materialization을 유지한다. 직접 decode/dot 후보는 별도 검증이며 과거 실패를 일반 하드웨어 한계로 간주하지 않는다. Block partial은 F32를 유지한다.',
            ['project_up_and_gate', 'project_rows_with_scale_contraction', 'project_down_tile']),
        'F3': ('Block scale과 column·cluster 합산을 F32로 유지',
            'Up/gate는 input scale을 복원한 F32 block dot에 E4M3 weight scale을 곱하고 H를 합산한다. Down은 block scale→local L1920 합→네 column 합→두 L-half cluster 합을 F32로 수행한 뒤 처음 BF16으로 변환한다.',
            'block16; 두 H3840 F32 partial의30720 logical bytes와 gather/reduction 포함',
            '두 cluster partial을 각각 BF16으로 바꾸면 상쇄 결과가 달라진다. finish_down의 Dummy2 합 뒤에만 첫 BF16 변환을 둔다. 합산 순서 변경은 별도 정확도 검사를 요구한다.',
            ['project_rows_with_scale_contraction', 'project_down_tile', 'finish_down']),
        'F4': ('30/60행 dot과6/12행 scale tile',
            'Up/gate는30행 native dot 뒤6행 scale/reduction을5회, down은60행 BF16 dot 뒤12행 scale/reduction을5회 수행한다. Up/gate30행 F32 producer 네 개를 cluster 안에서120행으로 모아 BF16 commit 정렬을 충족한다.',
            'up scale6×240×4=5760B/slice; down scale12×120×4=5760B/slice',
            '타일 offset, weight/scale 대응과 VRF lifetime을 함께 검증한다. Down의 네 column 합과 두 cluster 합을 타일링 비용에서 빠뜨리지 않는다.',
            ['static_passes', 'project_rows_with_scale_contraction', 'cast_gathered_rows', 'project_down_tile']),
    }
    for method in data['methods']:
        if method['id'] in replacements:
            name, behavior, parameters, invariant, symbols = replacements[method['id']]
            method.update(name=name, behavior=behavior, parameters=parameters, invariant=invariant,
                implementation=[{'path': CRATE + '/src/device/shared/mlp.rs', 'symbols': symbols}], evidence=evidence)
        elif method['id'] == 'F5':
            method['behavior'] = '두 cluster의 F32 H partial을 H480×8로 모아 finish_down에서 합친 뒤 BF16으로 바꾼다. 이후 global scale, post-RMS, residual 덧셈, layer scale, 최종 저장은 E104 경계를 유지한다. Host sliding/full 두 경로가 같은 decoder_feedforward를 사용한다.'
            method['implementation'][0]['symbols'].append('finish_down')
            method['evidence'] = evidence
        elif method['id'] == 'F6':
            method['evidence'] = list(dict.fromkeys(method['evidence'] + evidence))
    write('baseline/e122.json', data)
    write('baseline/current.json', {'recipe': 'baseline/e122.json'})
    print('E122 recipe points to', commit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['source', 'recipe'])
    args = parser.parse_args()
    source() if args.action == 'source' else recipe()
