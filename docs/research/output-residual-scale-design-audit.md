# O FP8 잔차 독립 scale: API 가능성과 수치 한계

이 문서는 E47 FP8 O 근사의 대안을 평가한 기록이다. 최종 E53은 BF16 입력을 보존하며 아래 FP8 잔차 재분해를 채택하지 않았다.

2026-09-21. CPU와 설치된 SDK source만 검토했다. 새 kernel 구현, compile, NPU 실행은 하지 않았다. **잔차 scale을 독립 계산하면 [E47에서 실제 확인된 cancellation 반례](output-boundary-design-audit.md)는 해결되지만, 입력에 따라 같은 문제가 다시 생긴다. 공통 high scale을 마지막에 곱하는 형태가 별도 partial scale보다 안전하며, 어느 형태도 일반 BF16 입력에 대한 정확 경로로 입증되지 않았다.**

## 두-term 대안과 SDK 표현

현재 `s_hi`와 `hi=RN_E4M3(x/s_hi)`는 유지한다. 후보는 잔차의 amax에 따라 다시 power-of-two scale을 정한다.

```text
z = x / s_hi
r = (z - hi) * s_hi
s_lo = 2^(floor(log2(max(max(abs(r)), 2^-119))) - 7)
lo = RN_E4M3(r / s_lo)
```

이는 fixture, seed, 특정 값에 대한 분기가 없는 데이터 의존 알고리즘이다. scale은 O의 각 256-column group마다 계산한다. 기존처럼 native DPE의 두 lane은 hi와 lo에 대응한다.

설치된 `furiosa-opt-std-0.8.1/src/engine/vector/`에서 다음을 확인했다.

| source 위치 | 검증한 계약 |
|---|---|
| `tensor/vector_tensor_pair.rs:600` | `vector_narrow_split`은 두 group을 함께 Way8→Way4로 변환한다. pair의 `narrow_trim`은 지원하지 않는다. |
| `tensor/vector_tensor_pair.rs:689` | `vector_fp_binary(op, group0_operand, group1_operand)`가 두 operand를 따로 받는다. |
| `operand.rs:1077` | 각 operand는 같은 Chip/Cluster/Slice의 `&VrfTensor`를 받을 수 있다. stash/branched operand는 pair에서 제외된다. |
| `tensor/vector_tensor_pair.rs:179,214` | 두 group의 동일 op는 공유 ALU를 한 번 예약한다. group마다 다른 scale VRF를 쓰는 수학적 표현이 가능하다. |
| `tensor/vector_tensor_pair.rs:1235` | `vector_fp_zip(AddF)`는 두 group을 더한 뒤 single Zipped stream을 반환한다. |

따라서 `.vector_fp_binary(Mul0, &hi_scale_vrf, &lo_scale_vrf).vector_fp_zip(AddF)`는 공개 API가 허용하는 operand 형태다. 정확히 이 shape의 MIR/backend compile을 실행한 것은 아니다. 기존 pair layout인 Time `[H%120,1#2]`, Packet `[1#4]`와 scalar `[1#8]` VRF broadcast를 보존하는 설계다.

다만 별도 곱 `dot_hi*s_hi + dot_lo*s_lo`는 피하는 편이 낫다. x=maxBF16(`0x7f7f`)이면 s_hi=2^120, z=255, hi=256이다. `hi*s_hi=2^128`은 F32 overflow다. `(z-hi)*s_hi`는 이 **잔차 생성** overflow를 피하지만 **DPE partial**을 각각 scale하는 단계는 여전히 overflow할 수 있다. x 하나와 weight 1, channel scale 2^-120, gamma 1, residual 0인 유한 reference는 final=1이다. 별도 partial scale CPU 모델은 Inf→NaN을 만든 반면 현재 fixed16 모델은 1이다.

더 나은 결합은 현재의 공통 scale 위치를 유지하는 것이다.

```text
ratio = s_lo / s_hi
partial = (dot_hi + dot_lo * ratio) * s_hi
```

SDK에서 기존 group1 상수 1/16을 ratio VRF로 바꾸면 된다. `Mul0((), &ratio_vrf)` → `zip(AddF)` → `Mul1(&hi_scale_vrf)` 형태다. 이 모델은 위 maxBF16 예에서 final=1을 유지했다. 그러나 각각의 scale이 유한하더라도 ratio가 최저 2^-246까지 작아질 수 있어 F32 underflow할 수 있고, normalized residual을 계산할 때도 `x/s_hi`가 너무 작으면 이미 소실된다. 이 변경으로 모든 finite 입력의 안전성을 주장하지 않는다.

## CPU에서 개선된 범위와 남은 반례

PyTorch 2.14.0+cpu에서 실제 frozen oracle의 `scaled_projection`과 `RMSNorm`을 사용했다. [별도 8-case generator](../../moa-dev/experiments/candidates/output-boundary-probe-e47/generate_suite.py)의 입력과 공식 분포 run0..8을 읽어 fixed16 모델과 독립 scale 모델을 비교했다. CPU 분석의 SDK rounding/FTZ 및 reduction emulation 한계는 [원래 감사](output-boundary-design-audit.md)와 같다.

| 입력 | fixed16 | 독립 residual scale |
|---|---|---|
| probe0..4,6 | activation 정확 복원; 최종 tolerance PASS | 같음 |
| probe5, 작은 값 경계 | activation 2688개 차이, 최대 2^-21; 최종 PASS | 1344개 차이, 최대 2^-28; 최종 PASS |
| probe7, `[+1,-1,254×2^-22]` | final 0.5, 원래 0.609375와 불일치 | final 0.609375, 정확 일치 |
| 공식 분포 run0..8 | 9개 모두 3840 outputs PASS | 9개 모두 3840 outputs PASS |

두 모델의 공식 분포 출력 최대 오차는 각 run에서 0.03125였다. 독립 scale은 run6/7/8의 단일 tiny activation 소실을 복원했고, run4의 scalar 오차를 약 3.576e-7→1.192e-7로 줄였지만 final tolerance 결과가 달라진 것은 아니다. 표의 전체 비교는 별도 partial scale 모델로 계산했고, common-scale 모델도 probe5/probe7 및 아래 두 반례에서 추가 확인했다. common-scale 모델의 공식 9개 입력을 따로 재실행했다고 주장하지 않는다.

**두-term 방식에 남는 범위 내 반례:** 각 group을 다음처럼 바꾸면 잔차 자체에 큰 값이 생긴다.

```text
x = [+0.53125, -0.53125, 254 copies of 2^-23]
W = [1, 1, 254 copies of 240]
channel_scale = 2^-10, gamma = 0.875, residual = 0.5
```

s_hi=2^-8이고 큰 값의 hi 복원은 ±0.5라 잔차가 ±0.03125다. s_lo=2^-12, 작은 잔차의 normalized 값은 2^-11이므로 다시 FP8 zero로 간다. fixed16, separate-scale, common-scale 세 CPU 모델 모두 final=0.5, 원래 reference는 0.59765625다. absdiff=0.09765625로 tolerance 밖이다. 개별 값은 공식 O generator 지원 범위 안이며 실제 PRNG joint input이라고 주장하지 않는다. 이 반례는 아직 NPU에서 실행하지 않았다.

## 3/4-term 분해를 exact BF16이라고 부를 수 있는가

반복해서 잔차 amax와 scale을 만들고 FP8 항을 추출하는 greedy 분해는 좁은 범위에서 유용하지만, 일반 BF16을 3개나 4개 항으로 정확히 보존한다는 보장은 없다. 예를 들어 같은 group에 다음 다섯 값을 둔다.

```text
[1, 2^-20, 2^-40, 2^-60, 2^-80, 0, ...]
```

각 stage에서는 가장 큰 남은 power-of-two 하나만 표현되고 다음 값은 FP8 범위 아래로 사라진다. CPU 계산에서 2/3/4 terms는 각각 처음 2/3/4개만 복원했고 5 terms가 다섯 값을 모두 복원했다. 마지막 값에만 weight=1, channel scale=2^80을 적용하면 reference는 유한한 1이지만 4-term 모델은 0이다. 이 배열은 BF16 자료형상 유효하지만 tiny 값들은 공식 O generator의 최소 nonzero 격자 2^-23 밖이다. 이 반례는 **검토한 greedy shared-scale 분해**의 한계이며 모든 가능한 FP8 표현 알고리즘의 불가능성 증명은 아니다.

정해진 범위의 가능성은 따로 보았다. PyTorch generator seed470032로 24-bit numerator 262,144개를 뽑아 공식 O와 같은 `BF16(-1+2*k/2^24)`를 만들었다. 독립 residual 2 terms는 8개 activation에서 차이가 났고 max error는 2^-22였다. 3/4 terms는 이 표본에서 모두 정확했다. 위 ±0.53125 반례도 3 terms에서 activation을 정확히 복원했다. 이것은 **표본 확인**이며 전체 입력, matmul reduction, BF16 cast/RMS의 동치 증명은 아니다. future/E2E activation은 이 24-bit 격자 제약을 갖는다고 가정하지 않는다.

## 구현 전 비용 판단

현 O quantizer는 amax → scale → hi → fixed lo의 Main pass 4개다. fusion을 가정하지 않는 보수적 설계에서는 hi까지 3개, 이후 각 residual term에 residual F32 생성 → residual amax → scale → FP8 quantize의 Main pass 4개가 필요하다.

- 독립 2 terms: 7 Main passes, 현재보다 +3. common-scale 결합용 ratio를 별도 scalar stream에서 만들면 +1 Main이다.
- 3/4 terms: 각각 11/15 Main passes이며 각 추가 ratio 생성이나 term merge는 별도다. 아직 실제 compile된 instruction 수나 cycle 예측이 아니다.
- residual F32 DM은 slice당 256×4=1024B다. write 한 번과 amax/low quantize의 read 두 번이 추가된다. low scale/ratio에는 각각 32B scalar DM/VRF가 필요하며 lifetime에 따라 copy와 Sub pass가 늘 수 있다. allocator의 실제 peak는 compile 확인 전이다.
- 2 terms는 현재 2-lane DPE와 weight DM을 그대로 쓸 수 있다. 3/4 terms의 packed/TRF 유효 입력은 slice당 768/1024B로 증가한다. 공개 pair API는 2-group이므로 보수적인 설계는 2+1 또는 2+2 dot과 F32 partial merge를 추가한다. 4-lane 한 번의 contraction 뒤 합산을 같은 pass에 배치할 수 있다고 확인한 것은 아니다.
- 기존 O16 schedule에서 quantized TRF는 5184에 준비되고 weight DMA는 16709에 끝난다. 일부 새 Main 연산은 이 사이에 숨을 가능성이 있지만 추가 DM/scale 의존성이 weight DMA 시작을 늦출 수도 있다. pass 수를 cycle이나 wall-time 증가율로 바꾸지 않는다.

일반적인 정확도 기준선에는 검증된 BF16 projection 경로가 더 직접적이다. 독립 residual/common-scale이나 3-term FP8은 별도 근사 후보로 취급하고, 이름이나 문서에서 “exact BF16”을 주장하지 않는다. 현재 확인된 두 종류의 cancellation과 overflow 예를 없앤다는 근거 없이 두-term scale만 바꿔 해결됐다고 판정하지 않는다. 이 감사 이후 구현이나 NPU 검증은 수행하지 않았다.
