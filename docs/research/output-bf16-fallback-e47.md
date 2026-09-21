E47 O의 FP8 activation hi/lo 경로는 실제 경계 입력에서 오차 기준을 넘겼다. 가장 직접적인 수정은 **원래 BF16 activation과 정확히 BF16으로 확장한 FP8 weight를 한 번 내적하는 것**이다. BF16 입력을 두 BF16 항으로 분해할 필요가 없다. SDK에서 이미 사용·검증한 Main lookup + BF16 DPE를 쓰고, E47의 빠른 row/column layout과 epilogue는 유지할 수 있다.

## 관측과 과거 선택지

[70785 결과](../experiments/e47-output-boundary-probe.json)는 공식 seeded 성능 harness와 별도의 8-case 진단이다. 모든 입력 checksum 40개가 맞고 case0–6은 통과했지만, case7의 3,840개 출력이 모두 `got=.5`, `want=.609375`, error `.109375`로 실패했다. Nonfinite는 없었다. 각 column의 `[+1,-1,254×2^-22]`에서 작은 항이 hi/lo FP8 모두 0이 되는 [CPU 반례](output-boundary-design-audit.md)가 실제 NPU에서도 관측된 것이다. 순위나 성능 측정 결과는 아니다.

| 경로 | 실제 O cycles | 알려진 범위·차이 |
|---|---:|---|
| [초기 원본 반복](../experiments/baseline-rngd.json) | 404,222 | 기존 tile/materialization·복사 비용이 큰 구현 |
| [E14 BF16 8-column](../experiments/e14-headnorm-output-columns.json) | 65,694 | BF16 x, FP8→BF16 lookup, 8×512 FP32 partial 합. 공식3입력 PASS |
| [E18 반복](../experiments/e18-best-repeats.json) | 65,662 | 같은 BF16 O 계열. 세 job의 O 중앙값65,139/65,662/65,919 |
| [E21 native FP8](../experiments/e21-row-fp8-qkv-native-output.json) | 59,016 | O 공식3입력 PASS지만 전체 job은 QKV V 실패. Activation 근사 도입 |
| [E26 병렬 epilogue](../experiments/e26-output-parallel-epilogue.json) | 52,579 | Native 근사 유지, O 후단 복사 감소 |
| [E47 16-column 반복](../experiments/e47-output-sixteen-columns-repeats.json) | 44,588 | Native 근사 유지. 공식/extra 입력은 통과했으나 70785 경계 실패 |

따라서 BF16 입력을 사용한다고 404k 원본으로 돌아가야 하는 것은 아니다. 이미 실제 BF16 경로가 약65.7k까지 내려갔다. 다만 옛 BF16 경로는 E47보다 후단 copy/RMS 비용이 컸고 raw-dot에 BF16 cast를 더 두었으므로, 그 실측을 새 정확성 수정의 예상 시간으로 그대로 옮길 수 없다.

검사한 O 후보에는 별도의 “두 BF16 activation 항 분해” 구현을 찾지 못했다. Native FP8 two-lane은 두 **FP8** 항이다. 입력 자체가 BF16이므로 BF16 한 항으로 그대로 표현되며, 두 BF16 dot은 이 근사를 제거하기 위해 필요한 작업이 아니다. 기존 BF16 fused lookup+dot+column sum의 정적 비용5,063을 기계적으로 두 pass 반복하면 Main 작업 합의 출발점은 약10,126 cycles이며 추가 합산/중간 buffer 비용이 생긴다. 이는 미구현 비용 모델일 뿐 compile·실측 예측이 아니고, 한 BF16 dot보다 우선할 근거가 없다.

## 공식 수치 경계와 지원 API

동결 [reference 생성기](../../moa-dev/experiments/artifacts/e47-output-sixteen-columns-extra/crate/scripts/generate_references.py)의 `scaled_projection`(265행)은 다음이다.

```python
((x.float() @ codes.T) * scale.float()).to(torch.bfloat16)
```

따라서 raw dot을 먼저 BF16으로 만들고 channel scale을 곱하는 과거 device 경계도 제거해야 한다. 원래 activation BF16/weight FP8 값을 유지하고, FP32 dot·column sum → FP32 channel scale → BF16 순서를 구현한다. 이는 누적 순서를 CPU matmul과 bit-identical하게 만든다는 뜻은 아니다.

설치 SDK 0.8.1의 `src/engine/fetch_adapter.rs:185–215`는 Main `fetch_table_lookup<bf16>`를 제공하며 두 FP8 dtype의 BF16 lookup을 설명한다. `CanApplyCollect for PositionFetchTableLookup`도 구현되어 있다. 실제 [기존 O8 소스](../../moa-dev/experiments/candidates/output-two-cluster-column-split/projection.rs)의 `project_output`은 이 Main weight stream에서 `.fetch_table_lookup::<bf16>().collect().contract_outer(...)`를 실행한다. Activation은 Sub BF16→TRF이므로 Sub에서 lookup하려 하지 않는다. Weight 전체 BF16 DM 저장도 필요 없다.

유한 E4M3의 지수/가수 범위는 BF16에 포함된다. 새로운 [scalar 검사](../../moa-dev/experiments/candidates/output-bf16-activation-e47/verification.json)는 유한254개 코드를 BF16으로 정확히 표현할 수 있음을 확인했다. 이것은 LUT table byte 또는 hardware 자체의 전수 검증과는 구분한다. Native FP8의 input-scale/outlier 손실은 없애지만, BF16 DPE의 누적 정밀도·F32 overflow/underflow·FTZ·RMS 구현 차이는 여전히 검증 대상이다.

## 구현한 후보와 정적 비교

[output-bf16-activation-e47](../../moa-dev/experiments/candidates/output-bf16-activation-e47/README.md)는 E47의 O16 mapping을 유지하고 위 경로를 구현했다. 두 compile 중 첫 F32 4행 transpose는 `max_in_rows=2`로 실패했고, F32 2행=8B commit 후 H480×8 typed gather하는 수정이 통과했다. Channel scale 후 첫 BF16 cast를 수행하므로 과거 2행 BF16 packet 조립 문제를 사용하지 않는다. Public signature·QKV·FFN·RMS·residual은 변경하지 않았다.

- 정적 makespan **24,621→27,239(+10.63%)**; 전체32 instructions.
- Weight DMA **13,383** 동일. Lookup+단일 BF16 dot+16-column F32 sum **5,063**; 기존 native pair1,242 + 별도 sum522보다 계산량이 크다.
- Quantizer의 Main 작업과 input/scale copy를 제거한다. LUT 상수 load가 추가되지만 전체 DMA 구간 합은 **18,621→18,229**다.
- 작은 result gather는 BF16에서 F32로 바뀌어 **446→470** cycles다. Main 전체는 **5,196→7,178**이다. 엔진 비용을 합해 전체 실행시간으로 읽으면 안 된다.

Root의 E53 full build에서 image byte 일치를 확인했고, [job70806](../experiments/e53-bf16-output.json)은 공식3입력·15개 비교를 모두 통과했다. O 중앙값은 **50,882 cycles**다. 이는 E47 native의 반복 중앙값44,588보다14.12% 높지만 입력 근사 소실을 제거하는 정확성 수정이다. [Extra6입력 job70814](../experiments/e53-bf16-output-extra-correctness.json)는30개 비교를, [동일 8-case boundary job70810](../experiments/e53-output-boundary-probe.json)은30,720개 출력 비교를 모두 통과했다. 원래 E47이 실패한 case7도 통과했다. 후속 [동일 패키지3회 반복](../experiments/e53-bf16-output-repeats.json)도 모두 PASS, 중앙값은 Q95,621/O50,882/FFN377,880이다. 이는 한정된 입력 검증이다. 원본8559916 kernel의 boundary control은 별도 진단으로 구분한다. 특정 case만 인식하는 branch나 fixture 의존 경로는 넣지 않았다.
