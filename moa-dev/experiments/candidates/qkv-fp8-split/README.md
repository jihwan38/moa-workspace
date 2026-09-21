# QKV BF16 input → 두 FP8 항, 독립 prototype

2026-09-21, compiler/std 0.8.1. 공개 fork 구현을 읽거나 복사하지 않고 공식 SDK와 주어진 수식에서 독립적으로 도출했다. **컴파일 성공. 후속 root 통합 [E19 job70261](../../../../docs/experiments/e19-qkv-fp8-split-two-dot.json)에서 공식 3개 설정의 oracle가 모두 통과했으며 QKV median 110,479 cycles다.** 이 결과가 전체 입력 범위의 오차 보장을 의미하지는 않는다. `candidate.patch`는 현재 head-local QKV를 기준으로 ops 연결과 projection만 변경한다. O, input/head RMSNorm 및 RoPE 산술은 그대로다.

| exact QKV kernel | static cycles |
| --- | ---: |
| 기존 BF16 activation, FP8 weight lookup | **52,971** |
| Q만 FP8 두 항, K/V 기존 유지 | 60,543 |
| Q/K/V 모두 공통 FP8 두 항 | 54,866 |

최종 source는 `ops.rs`, `projection.rs`이며 두 FP8 내적을 **별도로** 수행하는 버전이다. main dot 구간은 Q 5,063→1,225×2, K/V 각각 2,663→745×2로 줄지만 quantization, 새 DMA/synchronization, 두 FP8 broadcast가 비용을 추가한다. 이 첫 구현은 baseline보다 3.58% 느린 static schedule이다. 후속 two-lane 연구와 구분하여 보존한다.

## 수학과 범위

입력은 기존 input RMSNorm이 반환한 BF16 벡터다. 유한 입력에 대해 다음을 계산한다.

```text
A = max(abs(x))
e = 2^floor(log2(max(A, 2^-119)))
s = e / 128
z = x / s
hi = RN_E4M3(z)
lo = RN_E4M3(16 * (z - hi))
x_hat = s * (hi + lo/16)
dot_hat = (dot(weight,hi) + dot(weight,lo)/16) * s
```

`s`는 [2^-126,2^120]의 정규 FP32 power-of-two다. 모든 유한 BF16 입력에서 |z|<256, |hi|≤256, |16(z-hi)|≤128이므로 hi/lo의 FP8 overflow가 없다. A=0도 양수 floor 때문에 divide-by-zero가 없다. `amax`를 제곱으로 계산하지 않아 제곱 overflow를 만들지 않는다. NaN/Inf까지 원본과 동일하게 전파된다는 주장은 하지 않는다.

power-of-two로 scaling한 BF16 z가 |z|≥2^-6이면 8-bit BF16 significand를 두 FP8 항으로 정확히 표현할 수 있다. 더 작은 구간은 residual FP8 subnormal grid의 한계 때문에 값이 사라질 수 있다. nearest-even 및 FP8 subnormal 보존 모델에서 전체 scalar 절대오차는 **s·2^-14 이하**다. 이 bound는 tight하다. 예를 들어 s=1, z=2^-14이면 두 항이 모두 0으로 round되어 오차가 2^-14다.

dot의 입력 근사 오차는 `||weight_row||₁ · s · 2^-14` 이하이며 FP32 accumulation 오차는 별도다. cancellation 때문에 출력 상대오차의 유한한 일률적 상한은 없다. `[1,1,2^-22]·[1,-1,1]`은 실제 2^-22인데 이 global-scale 방법은 0을 반환한다. 근사 최적화이며 모든 입력에 exact하다고 주장하지 않는다.

**scale은 두 FP32 dot의 weighted sum 뒤에 적용한다.** BF16max=255·2^120은 s=2^120, hi=256, lo=-16으로 분해된다. `s*hi`를 먼저 계산하면 2^128로 overflow하지만 `(hi+lo/16)*s`는 유한한 원래 값이다. 구현은 low-dot을 먼저 /16하고 high-dot에 더한 뒤 s를 곱한다. 이어 기존 위치에서 BF16으로 cast하고 원래 channel scale을 적용한다. 두 FP32 dot 분리 자체로 reduction grouping은 바뀐다.

## CPU 검증

`analyze.py`는 fixture/model 값을 사용하지 않는다. E4M3 bit patterns를 독립적으로 열거하고 nearest-even 이웃 선택으로 변환을 모델링한다. BF16 RNE와 BF16/F32 subnormal flush도 명시적으로 처리한다. NumPy matmul은 RNGD의 reduction order를 재현하지 않는다.

- |z|≤256인 BF16 encoding 34,562개 전수 검사에서 위 scalar bound 및 |z|≥2^-6의 exact reconstruction을 확인했다.
- Gaussian 세 scale에서 H3840 × 64개 FP8 weight rows를 검사했고 BF16 projection 비교 mismatch는 각각 0이었다.
- 한 입력만 2^20인 outlier 사례는 64 outputs 중 BF16 mismatch 2개다.
- 위 cancellation 사례는 오차 반례로 확인했다.
- all-zero, BF16 최소 normal, BF16 최대 finite, 양 극단이 섞인 입력을 검사했다. 결과와 seed는 `analysis.json`에 있다.

이 결과는 FP8 cast/native multiplier의 실제 NPU rounding 및 subnormal 동작 증명이 아니다. 특히 SDK functional cast 지원과 NPU 오차 검증을 구별해야 한다.

## 실제 SDK 구현 경로

`prepare_fp8_split`은 각 cluster의 canonical BF16 H3840 배열에서 `vector_reinterpret::<i32>()` + `vector_logic(BitAnd,0x7fffffff)`로 절댓값을 만들고 intra-slice Max한다. floor 이후 다음 pass에서 exponent mask `0x7f800000`와 FP32 /128로 s를 만든다. log/exp의 근사값이나 CPU 데이터 분기를 사용하지 않는다.

quantization은 H480씩 8 slices에 나누어 수행한다. BF16→FP32 `fetch_cast`, scale division, FP8 `cast`로 hi를 저장한다. hi를 FP32 VRF로 읽고 원래 x/s에서 뺀 뒤 ×16하여 lo를 cast한다. 이 VRF는 480×4=1920B이며 scale은 32B다. H3840 전체를 FP32 VRF에 넣는 15,360B overflow를 피한다. stash는 사용하지 않는다.

hi/lo 각각 typed DMA로 8 anchor slices에 모은 뒤 기존 hier32 방식으로 두 cluster의 512 slices에 복제한다. scale도 typed DMA로 복제한다. Q/K/V의 모든 weight는 원래 FP8 그대로 두고 `TrfTensor<f8e4m3,...>` activation과 native FP8 contraction한다. low dot FP32 DM/VRF를 high dot에서 합친다. 완성된 BF16 projection result 이후 channel scale 및 head-local norm/RoPE 연결은 기존 그대로다.

공식 근거: [Cast Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/cast-engine.html), [contraction operand pairs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/cast.rs), [vector bit reinterpret/logic](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/vector/tensor/vector_tensor.rs), [E4M3 host conversion](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/float/f8e4.rs), [BF16 FTZ conversion](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/float/bf16.rs). 접근일 2026-09-21.

## 재현과 범위

Docker `/tmp/moa-qkv-fp8-split`에서 기존 source/manifests를 복사하고 전용 target/output을 지정했다. 최초 baseline도 동일 scratch에서 재컴파일했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-fp8-split; export CARGO_TARGET_DIR=/tmp/moa-qkv-fp8-split/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-fp8-split/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/all-qkv.json 2>&1 | tee logs/all-qkv.log'
```

현재 QKV 전체 source와 성공한 scratch hashes가 일치한다. `evidence.json`에 source/schedule hashes와 stage durations를 기록했다. `projection.q-only.rs`/`ops.q-only.rs`는 중간 성공안이며, `quantizer.rs.fragment`는 초기 struct 초안이다. 사용 시 최종 완성 파일/patch를 기준으로 삼는다. custom struct field의 mapping const가 compiler ICE를 일으켜 tuple/개별 인자로 바꿨고, device 함수 안의 u32 `from_bits` 식은 지원되지 않아 동등한 정확한 F32 상수를 사용했다.

공식 3개 설정의 NPU oracle는 후속 E19에서 통과했다. 추가 입력 범위와 two-lane 변형의 검증은 별도다. 라이브 source, Git, NPU 제출은 이 실험에서 수행하지 않았다.
