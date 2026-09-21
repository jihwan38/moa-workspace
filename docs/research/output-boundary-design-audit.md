# E47 O16 입력 경계: 독립 CPU 수치 감사

2026-09-21. 아래의 device/현재 구현은 동결 E47을 뜻하며, 이후 채택한 E53의 BF16 O와 구분한다. **E47 O16 양자화는 검증된 9개 입력 구성에서 통과했지만, 공식 생성기가 개별적으로 표현할 수 있는 값들을 상관시켜 구성한 반례에서는 CPU 모델과 실제 NPU 모두 원래 reference의 0.609375 대신 0.5를 만든다.** 이 문서는 별도 경계 probe의 입력을 설계하기 위한 것이다. 수학 감사 이후 root가 별도 경계 probe job70785를 실행해 case7의 수치 실패를 확인했다. 이 감사 작업 자체는 compile/NPU 제출 없이 CPU와 문서만 담당했으며 공식 fixture, oracle, harness, live source, Git을 변경하지 않았다.

검토한 [projection.rs](../../moa-dev/experiments/candidates/output-fp8-sixteen-columns/projection.rs)의 SHA256은 `db048b93d9b768bb4cb9cb6f8fd603783789d4a5eb609e42a00a23c69a330718`이다. Docker `moa2026-dev`의 설치된 PyTorch `2.14.0+cpu`에서 실제 [generate_references.py](../../furiosa-opt-gemma4-12B/scripts/generate_references.py)의 `scaled_projection`, `rms_norm` 및 [gemma4.RMSNorm.forward](../../furiosa-opt-gemma4-12B/scripts/reference/gemma4.py)를 호출했다. 비교할 native 경로는 CPU FP8 RN-E4M3 변환과 F32 연산으로 모델링했다. SDK의 FP8 rounding/FTZ, NPU contraction 순서, RMS의 sqrt/div 구현을 bit-exact하게 에뮬레이션한 것은 아니다.

## 실제 경로와 평가 범위

[ops::sliding_attention_output](../../furiosa-opt-gemma4-12B/src/ops.rs)는 `[Ns,Gs,Ds]`를 4096개 Qs로 flatten하고 `project_output` → `normalize_output_fused` → `add_output_residual` → HBM write를 호출한다. 현재 O는 연속 256개씩 16개 column group으로 input scale을 독립 계산한다. 두 cluster와 rowgroup에 복제된 입력은 같은 column scale을 갖는다.

column j에 대해 다음과 같다.

```text
a_j = max(max(abs(x_j)), 2^-119)
s_j = 2^(floor(log2(a_j)) - 7)
z_j = x_j / s_j
hi_j = RN_E4M3(z_j)
lo_j = RN_E4M3(16 * (z_j - hi_j))
partial[h,j] = (dot_f32(W[h,j], hi_j) + dot_f32(W[h,j], lo_j)/16) * s_j
device_projected[h] = BF16(BF16(sum_f32_j(partial[h,j])) * channel_scale[h])
```

실제 CPU oracle의 `scaled_projection`은 **unquantized BF16 x를 F32로 확장한 matmul → F32 channel scale → BF16**이다. 즉 oracle은 scale 전에 BF16 raw-dot 경계를 두지 않는다. 현재 device의 경계를 CPU 비교 모델에 포함했다. 아래 핵심 반례는 power-of-two channel scale을 사용하고 BF16 exponent가 정상 범위여서 두 BF16 경계의 차이로 설명되지 않는다.

reference의 나머지 경로는 `projected.float()` → `mean(projected²)+1e-6` → `rsqrt`와 원소별 gamma 곱 → BF16 → residual BF16 add다. 현재 device RMS는 H480 partial sum을 gather한 뒤 sqrt/div를 쓴다. 따라서 CPU 예상값을 NPU의 정확한 최종 bitpattern이라고 단정하지 않는다. 다만 아래 0.109375 또는 1.0 크기의 차이는 양자화 소실 자체로 설명된다.

공식 [README](../../furiosa-opt-gemma4-12B/README.md)는 Stage 1의 제공 harness와 3개의 independently seeded configurations를 정확도·성능 판단 기준으로 명시하고, 도입부에서는 numerical behavior 보존을 요구한다. 공개 기준만으로 “모든 finite BF16 조합에 대해 bitwise exact여야 한다”는 별도 규칙을 만들지 않는다. 아래 셋을 구분한다.

- **현재 실제 검증:** 공식 run0..2와 개발용 extra run3..8. 동일 package 반복은 새 입력 구성으로 세지 않는다.
- **자료형상 유효한 입력:** finite BF16 activation/scale/gamma/residual과 finite E4M3 weight. 원래 reference의 중간값과 최종값도 유한한지 별도로 확인한다.
- **공식 생성기의 개별 값 지원 범위:** x는 F32 uniform[-1,1) 후 BF16 RNE, weight는 exponent field 7..14인 ±1..240, channel scale은 uniform[0,1e-3) 후 BF16, gamma/residual은 uniform[0,1) 후 BF16이다. 아래 correlated 배열 전체가 실제 run의 PRNG 출력이라는 뜻은 아니다. 아직 공개되지 않은 future/E2E 입력 분포도 추정하지 않는다.

## 우선 probe: 생성기 개별 값 지원 범위 안의 cancellation

모든 16개 group에서 같은 배열을 놓고, 모든 3840 output row에 같은 weight row를 사용한다.

```text
x[256*j + 0] = +1
x[256*j + 1] = -1
x[256*j + t] = 2^-22       for t=2..255
W[h,256*j + 0] = W[h,256*j + 1] = 1
W[h,256*j + t] = 240       for t=2..255
channel_scale[h] = 2^-10
gamma[h] = residual[h] = 1/2
```

| 값 | 저장 bitpattern |
|---|---|
| BF16 +1 / −1 | `0x3f80` / `0xbf80` |
| BF16 2^-22 | `0x3480` |
| BF16 2^-10 | `0x3a80` |
| BF16 1/2 | `0x3f00` |
| FP8 1 / 240 | `0x38` / `0x77` |

각 group의 s는 2^-7이다. 작은 값은 z=2^-15이므로 hi=0, lo 변환 전 값은 2^-11이다. E4M3의 최소 양수 subnormal은 2^-9이며 2^-11은 그 절반보다 작다. 따라서 tie 규칙에 기대지 않고 lo=0이 된다. 큰 항은 ±128로 정확히 변환되고 서로 상쇄된다. 후보의 raw projection은 0이다.

원래 raw projection은 `16×254×240×2^-22 = 0.2325439453125`다. 각 작은 곱은 `15×2^-18`이고 ±1 항을 포함한 모든 가능한 부분합의 크기는 32 미만이다. 이 구성의 합은 F32 격자에서 정확하므로 반례가 dot의 합산 순서에 의존하지 않는다. source의 row/column permutation도 이 모든-row-동일 구성에서는 실패 원인을 설명하지 못한다.

CPU 실계산 결과는 다음과 같다.

| 경계 | 원래 reference | native CPU 모델 |
|---|---:|---:|
| F32 raw projection | 0.2325439453125 | 0 |
| scale 적용 후 BF16 | 0.0002269744873046875 | 0 |
| RMS + gamma + BF16 residual 최종 | 0.609375 (`0x3f1c`) | 0.5 (`0x3f00`) |
| 절대 오차 / 허용치 | 0.109375 / 0.05609375 | 3840개 전부 tolerance 밖 |

모든 reference 중간값과 결과는 유한하다. 작은 값 하나의 activation 오차는 2^-22에 불과하지만, 같은 방향으로 4064개가 누적되고 EPS 부근 RMS가 이를 확대한다. scalar reconstruction bound만으로 최종 tolerance를 보장할 수 없는 사례다.

`fixture_prng.u01`의 24-bit numerator를 직접 넣어 개별 지원 여부도 확인했다. x+1은 k=16777215에서 F32 0.99999988…를 BF16으로 반올림하여 만들 수 있고, x−1은 k=0, tiny는 k=8388610, gamma/residual은 k=8388608, channel scale은 k=16384000에서 정확히 위 bitpattern을 얻는다. weight 1과 240은 허용 exponent band 안이다. **이 numerator들의 전체 상관 배열을 만드는 실제 PRNG seed를 찾거나 주장한 것은 아니다.**

## 256-column 경계를 확인하는 실패/성공 대조

[이전 512-column 감사](../../moa-dev/experiments/candidates/output-native-fp8-two-lane/review.md)의 반례를 현재 256-column에 맞춘 것이다. 모든 row에서 지정하지 않은 weight와 activation은 0, channel scale=gamma=1, residual=0으로 둔다. FP8 zero는 유효한 code지만 현재 공식 weight generator의 exponent band 밖이라는 점을 구분한다.

- **같은 group:** `x[255]=2^20`, `x[254]=1/4`, `W[h,254]=1`, `W[h,255]=0`. s=2^13, 작은 z=2^-15가 소실된다. 원래 final은 BF16 1, native CPU 모델은 0이다.
- **서로 다른 group:** 작은 값을 `x[256]=1/4`, weight를 `W[h,256]=1`로 옮긴다. group0의 s=2^13, group1의 s=2^-9가 독립이다. 작은 값은 정확히 복원되어 양쪽 final이 1이다.
- **zero-weight에 의존하지 않는 대조:** 같은 group에 `x[252]=+2^20`, `x[253]=-2^20`, `x[254]=1/4`, 모든 weight를 1로 둔다. 원래 raw dot 1/4는 F32에서 정확하고 native는 작은 항을 잃는다. final은 역시 1 대 0이다.

2^20의 BF16 bits는 `0x4980`, −2^20은 `0xc980`, 1/4는 `0x3e80`이다. 이 세 경우의 reference는 유한하지만 큰 activation과 channel scale=1은 현재 공식 생성기 범위 밖이다. 255/256 대조는 scale grouping 검증에 유용하고, 정해진 3-seed 결과의 반례로 표시하지 않는다.

## zero, floor 및 rounding 경계

**All-zero control.** x=0, weight=1, channel scale=2^-10, gamma=residual=1/2에서 floor 때문에 s=2^-126이며 hi=lo=0이다. RMS의 EPS는 분모를 양수로 유지한다. CPU reference/native 둘 다 final=0.5, 오차 0이다. ±0의 부호 차이는 현재 절대/상대 오차 비교로 구별하지 않는다.

**Floor는 곧바로 소실 구간을 뜻하지 않는다.** BF16의 최소 양수 subnormal은 2^-133(`0x0001`), 최소 normal은 2^-126(`0x0080`), floor 2^-119는 `0x0400`이다. amax≤2^-119이면 s=2^-126으로 고정되고 모든 BF16 x는 2^-133 격자이므로 z는 2^-7 격자다. 이 격자는 FP8 pair의 작은 값 표현 해상도보다 거칠다. positive BF16 code `0x0000..0x0400`의 **1025개를 전수 CPU 계산해 모두 정확 복원**했고 음수는 대칭이다. 이것은 NPU의 subnormal/FTZ 처리 검증을 대신하지 않는다.

단순히 tiny x를 넣고 기존 absolute tolerance만 검사하면 실제 소실도 통과할 수 있다. 큰 유한 channel scale을 쓰는 별도 진단은 reference를 유한하게 유지하면서 floor 처리의 차이를 드러낼 수 있다.

| 단일 nonzero x, 해당 weight=1 | channel scale | 양쪽 CPU final |
|---|---:|---:|
| 2^-133 (`0x0001`) | 2^120 (`0x7b80`) | 0.12109375 (`0x3df8`) |
| 2^-119 (`0x0400`) | 2^112 | 0.9921875 (`0x3f7e`) |

나머지 x/weight는 0, gamma=1, residual=0이다. 두 경우 모두 intermediate/reference finite, CPU error=0이다. 큰 channel scale은 공식 분포 밖이다. 장치에서 다르게 나오더라도 FTZ나 driver 문제라고 곧바로 결론 내리지 말고 해당 stage를 분리해야 한다.

**Underflow rounding tie.** amax=1이면 s=2^-7이다. x가 s×2^-14=2^-21 부근일 때 hi는 0이고 lo가 첫 subnormal로 넘어간다.

| BF16 x bits | x | CPU hi | CPU lo | 복원 x |
|---|---:|---:|---:|---:|
| `0x34ff` | 2^-21−2^-29 | 0 | 0 | 0 |
| `0x3500` | 2^-21 | 0 | 0 (nearest-even tie) | 0 |
| `0x3501` | 2^-21+2^-28 | 0 | 2^-9 | 2^-20 |

finite BF16에서 |x/s|≥2^-6인 영역은 두 FP8 항의 정확 복원 조건으로 사용할 수 있다. 그보다 작은 영역의 scalar 절대 오차 상계는 s×2^-14이지만, floor처럼 BF16 격자 때문에 더 작은 값도 정확한 예외가 있다. 이 충분조건을 필요조건으로 해석하지 않는다.

**Amax의 power-of-two 경계.** weight가 0인 outlier `x[0]`를 BF16 1 직전 값 `0x3f7f`(0.99609375)에서 1(`0x3f80`)로 올리고, 기여하는 `x[1]=2^-21`, `W[h,1]=1`을 고정한다. s는 2^-8→2^-7로 바뀐다. 앞 경우 작은 값은 정확 복원되고 뒤 경우 0이 된다. channel scale=2^12, gamma=1, residual=0에서 원래 final은 두 경우 모두 0.890625이나 native CPU 모델은 0.890625→0이다. 큰 scale은 공식 범위 밖이다. 이 대조는 outlier가 raw projection에 기여하지 않아도 input quantization만 바뀔 수 있음을 보여 준다.

## 재현 계산과 probe 판정

아래는 설치된 container에서 실행한 핵심 반례의 최소 재현이다. 공식 generator 함수 자체를 import하며 별도 reference 정의로 oracle을 대체하지 않는다. 16-way native CPU sum은 하드웨어 packet 순서를 재현하지 않지만 이 반례는 위의 정수 격자 계산으로 합산 순서에 의존하지 않는다.

```python
import sys, torch
sys.path.insert(0, "/workspace/scripts")
import generate_references as ref

H, Q = 3840, 4096
x = torch.full((16, 256), 2.0**-22, dtype=torch.bfloat16)
x[:, 0], x[:, 1] = 1, -1
w = torch.full((16, 256), 240.0)
w[:, :2] = 1
w = w.to(torch.float8_e4m3fn).float().reshape(1, Q).expand(H, Q)
scale = torch.full((H,), 2.0**-10, dtype=torch.bfloat16)
gamma = residual = torch.full((H,), 0.5, dtype=torch.bfloat16)

projected = ref.scaled_projection(x.flatten(), w, scale)
expected = (ref.rms_norm(H, gamma)(projected) + residual).bfloat16()
a = x.float().abs().amax(1).clamp_min(2.0**-119)
s = (a.view(torch.int32) & 0x7f800000).view(torch.float32) / 128
z = x.float() / s[:, None]
hi = z.to(torch.float8_e4m3fn).float()
lo = ((z - hi) * 16).to(torch.float8_e4m3fn).float()
wr = w.reshape(H, 16, 256)
partial = (torch.einsum("ck,hck->hc", hi, wr)
           + torch.einsum("ck,hck->hc", lo, wr) / 16) * s
raw = partial.sum(1)
projected_cpu = (raw.bfloat16().float() * scale.float()).bfloat16()
actual_cpu = (ref.rms_norm(H, gamma)(projected_cpu) + residual).bfloat16()
assert torch.isfinite(expected).all() and torch.isfinite(actual_cpu).all()
assert float(expected[0]) == 0.609375 and float(actual_cpu[0]) == 0.5
```

실계산에서 쓴 원본 파일 SHA256:

- `generate_references.py`: `eb3454334609f30796be776230e324eedc1354fe81371b057dac914332fb1860`
- `scripts/reference/gemma4.py`: `0095a36a51dca34bd162a0c41aeb9fdbe1a779f91228ee398f2b655922264775`
- `scripts/fixture_prng.py`: `a07e785fbc0c452e7933d6a6184bc6625b0ae0d81b5f0fd20b0eb1f8d1469d6d`

전달한 우선순위는 zero/정확복원 대조 → column boundary와 power-of-two 경계 → rounding/floor → 범위 안 correlated cancellation이다. root가 별도로 실행한 장치 probe 결과는 아래에 기록한다. 그 결과는 공식 score나 기존 latency aggregate에 합치지 않는다. 확인된 실패는 해당 근사에 필요한 입력 제약 또는 더 정확한 경로를 검토할 근거이며, 현재 9개 구성 PASS를 모든 future 입력의 보장으로 확대하지 않는다.

## 후속 NPU 확인: job70785

[측정·package 검증 기록](../experiments/e47-output-boundary-probe.json)과 [원본 로그](../../moa-dev/logs/e47-output-boundary-probe-job-70785.log)에서 frozen E47 O image로 8개 case가 모두 실행됐음을 확인했다. 40개 input checksum이 통과했고 총 30,720개 출력 중 case7의 3,840개만 tolerance 밖이다. 모든 case의 nonfinite count는 0이다. case7은 모든 row에서 got=0.5(`0x3f00`), want=0.609375(`0x3f1c`), error=0.109375, limit=0.05609375로 위 CPU 반례와 일치한다. Job은 수치 비교 실패에 따라 exit1/FAILED이며 timeout이 아니다.

case0..6은 tolerance PASS였다. 특히 case6은 일부 값의 bitpattern이 다르고 max absolute difference가 약 1.016e-35이지만 허용치보다 훨씬 작다. floor case의 PASS를 tiny 값의 정확 보존이나 FTZ 부재 증거로 해석하지 않는다. 원본 BF16 control의 결과는 이 문서 작성 시점에 아직 기록하지 않았다.
