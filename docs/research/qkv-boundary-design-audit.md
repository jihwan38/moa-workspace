# E47 QKV의 작은 activation 소실: CPU 반례

2026-09-21. **실제 pre-RMS와 head-RMS 및 공식 RoPE table을 사용한 CPU 계산에서, finite BF16 입력의 Q/K/V projection이 현재 FP8 분해 모델에서는 모두 0이 되는 반례를 확인했다.** 이 입력은 현재 공식 QKV generator의 x 지원 범위 밖이다. 공식 QKV 입력 실패나 NPU 결과라고 주장하지 않는다. kernel 구현/compile/NPU 제출 없이 CPU와 source만 검토했다.

## 실제 caller와 입력 범위

이 절은 동결 E47/E53 QKV를 분석하며, 이후 E54에서 제거한 FP8 경로를 현재 구현으로 설명하지 않는다. 동결 [ops::sliding_project_qkv](../../moa-dev/experiments/candidates/qkv-bf16-f32-scale-e53/ops.rs.original)는 x를 H480×8 배치로 옮긴 뒤 `normalize_sharded_inter_slice`로 BF16 pre-RMS를 만든다. 동결 [projection::prepare_fp8_split](../../moa-dev/experiments/candidates/qkv-bf16-f32-scale-e53/projection.rs.original)은 local H480 amax를 다시 8-way reduce하므로 **input scale은 H3840 전체에 대한 공통 값**이다. hi/lo를 각 shard에서 만든 후 `broadcast_quantized`가 전체 H 순서로 복제한다. H480마다 독립 scale을 쓰는 구현이 아니다.

Q/K는 두-lane paired native FP8 dot을 사용하고, V는 두 개의 native dot을 계산한다. 세 경로 모두 `(dot_hi + dot_lo/16) * input_scale`이라는 같은 복원식을 쓴다. 이후 channel scale → head RMS를 적용하며 Q/K만 RoPE를 거친다. K/V는 지정 offset의 cache에 write하고 Q는 q_out에 쓴다. 이 CPU 감사는 계산값만 다루며 cache 주소/layout의 하드웨어 동작을 검증하지 않는다.

현재 [gen_sliding_project_qkv](../../furiosa-opt-gemma4-12B/scripts/generate_references.py)는 임의 uniform activation을 생성하지 않는다. `input_rms_weight`는 0.75..1.25 범위이고, x는 `BF16(signs / input_rms_weight)`다. 따라서 pregamma=1일 때 x는 ±1이고, 아래 2^-23 값은 이 generator의 개별 x 지원 범위에도 속하지 않는다. weight/channel scale의 값은 허용 band 안이지만 이것으로 전체 입력이 공식 generator 범위 안이 되지는 않는다. future/E2E 입력 범위는 이 감사에서 가정하지 않는다.

## 구체적인 유한 반례

8개 H480 chunk마다 다음을 반복하고 모든 Q/K/V output row에 같은 weight row를 둔다.

```text
x = [+1, -1, 478 copies of 2^-23]
input_rms_weight = 1 for all H
W_q[row] = W_k[row] = W_v[row] = [1, 1, 478 copies of 240] repeated 8 times
q/k/v channel scale = 2^-10
q_rms_weight = k_rms_weight = 1
```

모든 값은 finite BF16/FP8다. x의 BF16 bits는 +1=`0x3f80`, −1=`0xbf80`, 2^-23=`0x3400`이고 weight의 FP8 bits는 1=`0x38`, 240=`0x77`이다. RoPE는 원래 `rope_tables(256,10000,1,pos)`를 사용했고 pos는 공식 run0의 `_run_pos(0)=1763`으로 정했다. 원래 reference가 사용하는 BF16 cos/sin과 `apply_rotary_pos_emb`를 그대로 호출했다.

설치된 PyTorch 2.14.0+cpu에서 실제 `rms_norm`, `scaled_projection`, `apply_rotary_pos_emb`를 계산한 중간값은 다음과 같다.

| 경계 | 값 |
|---|---:|
| 원래 x의 F32 mean-square | 0.004166666883975267 |
| BF16 pre-RMS 큰 항 | ±15.5 (`0x4178` / `0xc178`) |
| BF16 pre-RMS 작은 항 | 1.8477439880371094e-6 (`0x35f8`) |
| 공통 FP8 input scale | 0.0625 = 2^-4 |
| 큰 양수 항의 hi / lo | 256 / −128 |
| 작은 항의 x/scale | 2.956390380859375e-5 |
| 작은 항의 lo 변환 전 값 | 0.0004730224609375 |

작은 lo 입력은 최소 E4M3 subnormal 2^-9의 절반인 2^-10보다 작아 RN에서 0이 된다. 따라서 작은 3824개 항을 잃는다. 큰 ±15.5 항은 각각 정확히 복원되지만 같은 weight 1에 곱해 서로 상쇄된다. Q/K paired 경로와 V two-dot 경로 모두 raw projection은 0이다.

원래 raw projection은 `3824 × 240 × 1.8477439880371094e-6 = 1.6956024169921875`다. channel scale 후 BF16은 0.00165557861328125이며 reference의 모든 중간값과 출력은 유한하다. 원래 V head RMS는 0.85546875인데 native CPU 모델의 V는 0이다. zero는 이후 RMS와 RoPE를 거쳐도 zero다.

| 출력 | 원래 reference 범위 | native CPU 모델 | tolerance PASS 개수 | 최대 절대 오차 |
|---|---|---:|---:|---:|
| Q, 4096개 | −1.2109375..1.2109375 | 0 | 80/4096 | 1.2109375 |
| K, 2048개 | −1.2109375..1.2109375 | 0 | 40/2048 | 1.2109375 |
| V, 2048개 | 모두 0.85546875 | 0 | 0/2048 | 0.85546875 |

비교는 QKV의 기존 `atol=.04, rtol=.01`을 사용했다. Q/K에서 소수 값이 통과하는 것은 해당 RoPE 좌표의 원래 값이 0 근처이기 때문이며 전체 kernel PASS를 뜻하지 않는다. 원래 Q/K의 첫 8개 값은 `[-.265625,.1171875,.94921875,.10546875,-1.171875,1.09375,-.6171875,-.484375]`였다.

pre-RMS의 큰 항 제곱은 16개이며 작은 항 제곱은 F32 mean에 영향을 주지 않을 정도로 작다. CPU 결과는 BF16 경계를 포함해 계산했다. SDK rsqrt/sqrt/div 및 native accumulator를 bit-exact하게 에뮬레이션한 것은 아니지만, 작은 lo 값이 rounding threshold보다 충분히 낮고 V의 오차가 0.85546875인 만큼 별도의 general-input 정확도 위험을 보여 준다. NPU 검증 전에는 장치에서 이 값이 관측됐다고 기록하지 않는다.

## 재현과 해석

최소 재현의 핵심은 x를 곧바로 양자화하지 않고 **실제 BF16 pre-RMS를 먼저 적용**하는 것이다.

```python
h = ref.rms_norm(3840, torch.ones(3840, dtype=torch.bfloat16))(x)
projected = ref.scaled_projection(h, weights, channel_scale)
v_reference = ref.rms_norm(256, None)(projected.view(8, 256))

s = 2.0**-4
z = h.float() / s
hi = z.to(torch.float8_e4m3fn).float()
lo = ((z - hi) * 16).to(torch.float8_e4m3fn).float()
raw_candidate = ((hi @ weights.T) + (lo @ weights.T)/16) * s
```

자료형상 가능한 모든 입력에서 tolerance를 만족해야 한다는 새 공식 규칙을 이 반례에서 추론하지 않는다. 공식 generator 범위가 좁은 것과 모델 전체의 numerical behavior를 보존하려는 목표를 구분한다. BF16 projection 대안을 검토할 수 있는 근거지만, 해당 대안의 성능·전체 정확도·NPU 결과는 별도로 검증해야 한다.

## 후속 NPU 대조 — 2026-09-21

위 본문은 NPU 실행 전의 CPU 검토 기록이다. 이후 별도 host harness를 만들어 **E53 job70943, upstream 원본 job70944, BF16 QKV 대안 E54 job70958에서 두 진단 입력을 실제 실행**했다. 각 frozen device source와 공식 harness는 probe를 위해 수정하지 않았고, E53에 남아 있던 QKV image는 위 CPU 감사의 E47 QKV image와 동일하다. 상세한 package/image/입력 검증과 세 구현 비교표는 [NPU 비교 기록](../experiments/qkv-boundary-comparison.md), 원시 판정은 [E53 JSON](../experiments/e53-qkv-boundary-probe.json), [원본 JSON](../experiments/original-qkv-boundary-probe.json), [E54 JSON](../experiments/e54-qkv-boundary-probe.json)에 있다. E54의 생성 시점 package 증거는 [build manifest](../../moa-dev/experiments/candidates/qkv-boundary-probe-e53/e54-manifest.json)다.

case0은 위 `2^-23` 입력, case1은 같은 배열에서 작은 항만 `2^-12`로 올린 control이다. E53는 case0에서 Q4016/4096, K2008/2048, V2048/2048개가 실패해 총8,072개가 tolerance를 벗어났다. 각 출력 첫 불일치의 actual은0이고, 최대 절대 오차는 Q/K1.2109375와 V0.85546875였다. 실패 개수는 위 CPU 모델 예측과 일치한다. E53 case1은 Q/K/V 모두 expected와 정확히 일치했다.

원본과 E54는 각각 두 case의 Q/K/V가 모두 tolerance PASS였다. 두 구현 모두 case0의 Q/K 최대 절대 오차는0.0078125, expected와 정확히 일치한 개수는 Q2928/K1464/V2048이었다. 이 집계의 일치가 원본과 E54 전체 출력 배열의 동일성을 입증하지는 않는다. case1은 세 구현 모두 Q/K/V8,192개가 expected와 정확히 일치했다. 세 job 모두2/2 case가 끝났고 nonfinite=0이었다. E53의 exit1은 수치 비교 실패로, timeout이나 미완료 실행을 뜻하지 않는다.

세 host는 같은 probe source·CPU payload·runner를 사용했고, 실제 업로드 package hash와 서로 다른 원본/E53/E54 device image의1회 포함을 각각 검증했다. E53·원본의 준비 단계에서는 생성한38개 입력 전체 SHA256이 Python과 같았으며 E54의 host-only checksum38개도 통과했다. 실제 NPU에서는 세 job 각각 checksum38/38과 cache 검사8/8이 통과했고, 비대상 K/V cache **8,380,416개 BF16 bits 보존**, mismatch0이었다. 일정 sentinel 검사와 한 위치의 결과이므로 모든 cache 손상을 배제하는 증명은 아니다.

CPU oracle을 실제 fullshape 행렬로 생성하면서 F32 raw dot은 **1.69500732421875**로 계산됐다. 앞선 단일 행/해석적 값1.6956024169921875와의 합산 순서 차이를 장치에 맞춰 보정하지 않았다. 두 raw의 channel scale 뒤 BF16과 최종 Q/K/V BF16은 같으므로 위 최종 expected 값은 바뀌지 않았다. 원본 NPU의 Q/K가 bit-exact하지 않은 이유를 이 raw 차이만으로 설명하지 않는다.

두 진단 입력 모두 현재 공식 QKV generator의 x 지원 범위 밖이라는 한계는 그대로다. 이 관측을 공식 seeded 평가 실패나 모든 유한 입력에 대한 새 규칙으로 해석하지 않는다. 같은 입력의 CPU 예측·E53 실패·원본/E54 PASS는 일반 입력 정확도 대안을 검토할 근거다. 별도의 [E54 공식 harness 동일 패키지 반복](../experiments/e54-bf16-qkv-repeats.json)도 job70933/70974/70987 모두15개 비교를 통과하며 완료됐다. 반복 중앙값은 Q108700/O50970/FFN382268 cycles이고 QKV는 이전 E53 반복 중앙값보다13.68% 높다. 이 성능 수치는 본 정확도 진단에서 나온 값이 아니며 공식 순위나 모델 전체 처리량을 뜻하지 않는다.

추가 후속 [E55 job71032](../experiments/e55-qkv-boundary-probe.json)도 같은 probe·payload의2개 case와 Q/K/V6개 비교를 모두 통과했다. case0 Q/K 최대 절대 오차0.0078125·exact count Q2928/K1464/V2048, case1의8,192개 모두 exact로 원본/E54와 집계가 같지만 전체 출력 배열의 동일성을 뜻하지 않는다. 입력 checksum38/38, cache8/8 PASS, 비대상 cache8,380,416개 BF16 bits 보존과 nonfinite=0을 확인했다. [Package와 image 근거](../experiments/qkv-boundary-comparison.md#e55-후속--job71032)에 동결 hash를 연결했다. 공식 generator 범위 밖의 두 입력이라는 한계는 유지하며, E55 성능·추가 seed 검증과 분리한다.
