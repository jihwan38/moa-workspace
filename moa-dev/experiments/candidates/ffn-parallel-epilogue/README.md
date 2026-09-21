# FFN 후단 H480×8 유지 — E36 공식·추가 입력 PASS

2026-09-21. 고정된 [E31 source](../../artifacts/e31-switch-gather-rms/src)에 safe E27 FFN 본체를 유지하고, down 결과를 8개 H480 slice에 남겨 global scale → post RMS → residual → HBM까지 처리했다. **같은 scratch의 exact FFN schedule은 167,296 → 162,209 cycles(-5,087, -3.04%)**다. [E36 job70498](../../../../docs/experiments/e36-ffn-parallel-epilogue.json)은 공식15개 비교를 통과했고 FFN median378,021 cycles(Q97,052/O45,141)다. [추가 job70506](../../../../docs/experiments/e36-ffn-epilogue-extra-correctness.json)도 run3–8의30개 비교를 모두 통과했다. [evidence.json](evidence.json)에 43개 전체 소스 hash, 환경 ID, schedule hash와 한계를 기록했다.

적용 파일은 [mlp.rs](mlp.rs), [residual.rs](residual.rs), [ops.rs](ops.rs) 세 개다. [candidate.patch](candidate.patch)는 E31 대비 diff이며 `rmsnorm.rs`는 E31의 기존 RMS-gather helper를 그대로 사용한다. 함께 보관한 [rmsnorm.snapshot.rs](rmsnorm.snapshot.rs)의 SHA256은 `3c3c9525f21fe9916e0f6db00863cbb05e4f0c8cc35501ed08b1bbcdb4635cb0`이다. 다른 snapshot에 적용할 경우 이 의존성을 먼저 확인해야 한다.

## 실제 호출과 변경 범위

`host::runtime::Decode::sliding()`과 `full()`은 공통 `mlp()`를 거쳐 [`ops::decoder_feedforward`](../../artifacts/e31-switch-gather-rms/src/host/runtime.rs)를 launch한다. 공개 함수 이름·인자·HBM alias/update 순서는 그대로다.

```text
canonical residual HBM → pre-FF generic normalize [변경 없음]
→ up/gate safe BF16 blockdot → GeGLU BF16 [변경 없음]
→ down 16-element dot → FP32 local-scale/blocksum → 8-column FP32 sum
→ BF16 [이 경계까지 본문 byte-identical]
→ typed DMA로 H480×8 → F32 down_global_scale 곱 → BF16
→ 기존 E31 normalize_output_fused: RMS-gather → DivF → gamma → BF16
→ canonical original residual을 H480×8로 복사 → 더하기 → BF16
→ layer_scalar 곱 → BF16 → 원래 HBM에 typed store
```

- `mlp::project_down_sharded`는 down의 마지막 두 canonical gather를 H480×8 typed DMA 한 번으로 바꾼다. `feedforward_sharded`는 이 결과에 동일 global-scale 연산을 8개 slice에서 수행한다. 기존 `project_down`/`feedforward`의 signature는 canonical 반환 wrapper로 보존했다. snapshot의 각 기존 직접 caller는 `feedforward`와 `decoder_feedforward` 한 곳이었다.
- `ops::decoder_feedforward`만 새 feedforward와 기존 `normalize_output_fused`, 새 `add_and_scale_sharded`를 호출한다. pre-FF의 generic `normalize`를 바꾸지 않았다.
- `residual::add_and_scale_sharded`는 기존 parallel helper에서 x 재scatter와 마지막 canonical gather만 제거한다. residual 자체의 scatter, BF16 add, F32 refetch/layer multiply, BF16 cast는 유지한다. 기존 `add`, `add_vision`, `scale_by_layer_gate`, `add_and_scale_parallel` 본문은 그대로다.
- generic RMS는 full-QKV/full-O/LM head 등 다른 caller가 있으므로 수정하지 않았다. vision/audio 커널도 수정하지 않았다. 다만 공통 decoder FFN을 호출하는 sliding/full 양쪽은 후속 E2E 검증 대상이다.

[`verify_invariants.py`](verify_invariants.py)는 공개 ops 함수 **10개 header**, 다른 **9개 ops 본문**, up/gate helper·projection·GeGLU 전체, down의 마지막 BF16 commit까지, 기존 residual helper를 비교한다. 결과는 [invariants.json](invariants.json)에 있다.

## 축과 수치 경계

Down reduction 직후 `Cluster=Dummy2`, `Slice=[H/60%32,1#8]`, `Element=H%60`에서 논리 row는 `h=1920*c+60*g+r`, physical slice는 `8*g`다. 기존 high-row cluster 축 이름만 `Dummy2 → H/1920`으로 바꾼 뒤 typed DMA로 `Cluster=m![1#2], Slice=[1#32,H/480], Element=H%480`에 모은다.

목적지는 `slice=h/480=4*c+g/8`, `local=h%480=60*(g%8)+r`다. 2×32×60 모든 row의 중복·누락 없는 bijection과 최종 HBM 순서를 CPU로 확인했다. reshape는 실제 복사를 대체하지 않으며, 재분배는 typed DMA가 담당한다. backend padding/allocator 동작의 증명은 아니다.

**Down BF16 → global-scale BF16 → postnorm BF16 → residual-add BF16 → layer-gate BF16 경계와 순서가 그대로다.** Global scale을 RMS에서 소거하지 않았고, residual add와 gate를 F32 한 번에 합치지 않았다. EPS와 중간 BF16 rounding 때문에 그 두 변형은 같은 계산이 아니다. 새 global-scale/residual 병렬화는 원소별 연산의 분배만 바꾸며 추가 contraction·양자화는 없다.

RMS는 기존 generic helper의 inter-slice FP32 sum에서 E31의 Switch-gather + intra-slice FP32 sum으로 바뀐다. 8개 FP32 mean-square partial을 합한 뒤 EPS, sqrt, DivF, gamma 순서는 같지만 합산 순서 때문에 bitwise 동일하다고 주장하지 않는다. 또한 generic의 Clip Add와 sharded helper의 FP Add 경로가 다르므로 backend 검증이 필요하다.

정상 범위의 비음수 partial과 correctly-rounded FP32 덧셈을 가정하면 8항 합의 순서별 상대 오차는 `gamma_7 = 7u/(1-7u)`, `u=2^-24`로 제한되며 두 순서의 차이는 exact sum 대비 최대 `2*gamma_7 ≈ 8.35e-7`이다. 이는 **overflow/underflow, SDK의 실제 reduction 구현, sqrt/DivF 근사, 최종 BF16 문턱에서의 반올림을 보증하는 수치 오차 한계가 아니다.** 작은 RMS 차이도 BF16 경계나 residual cancellation 근처에서 최종 값을 바꿀 수 있다.

기존 독립 [FP32 order probe](../rmsnorm-switch-gather/check_reduction.py)를 재실행한 [결과](rms-order-probe.json)는 4,096개 비음수 8항 벡터에서 합 차이 892건(최대 2 ULP), RMS 차이 277건(최대 상대 차이 1.572e-7), 32,768개 샘플 BF16 postnorm 출력 차이 0건이다. 이는 가상의 sequential/balanced 순서 비교이며 RNGD oracle이나 전체 FFN 통과 증거가 아니다. 공식 seed나 입력값을 특수 처리하지 않았다.

## 같은 조건의 정적 비교

Fresh E31 baseline과 후보를 `/tmp/moa-ffn-parallel-epilogue`에서 `cargo furiosa-opt compile ops::decoder_feedforward --exact`로 각각 컴파일했다. SDK/lock/source 및 own target/cache 조건이 같다. 처음 scratch staging에서 오래된 outer `src`를 남긴 오류가 1회 있었으며 그 결과는 폐기했다. 전체 source를 다시 복사하여 43개 파일 hash를 확인한 뒤 아래 기준과 후보를 재컴파일했다. **올바른 source에서 kernel/API/backend compile 오류는 0회**였다. [실패 로그](attempt1-stale-staging-error.log)와 [기준](baseline-compile.log)/[후보](candidate-compile.log) 로그를 보존했다.

| 항목 | E31 baseline | 후보 | 차이 |
| --- | ---: | ---: | ---: |
| 전체 makespan | 167,296 | 162,209 | -5,087 |
| 마지막 down dot 종료 | 155,414 | 155,414 | 0 |
| 그 뒤 종료까지 | 11,882 | 6,795 | -5,087 |
| Main instruction duration 합 | 129,652 | 125,274 | -4,378 |
| DmaStos duration 합 | 25,260 | 24,214 | -1,046 |
| DmaLoad duration 합 | 110,705 | 110,745 | +40 |
| 마지막 HBM store | 434 | 648 | +214 |

Duration 합은 서로 겹칠 수 있으며 critical path나 하드웨어 counter가 아니다. Sub 합 19,378과 Core 합 29,708은 같다.

후단의 주요 노드 차이는 다음과 같다.

| 연산 | 기존 cycles | 후보 cycles |
| --- | ---: | ---: |
| Down row gather | 469 + 467 | 469 |
| Global-scale multiply | 1,241 | 401 |
| Post-RMS input scatter | 449 | 제거 |
| RMS 8-partial sum/EPS, sqrt | 2,579 + 282 | gather/local sum 289 + EPS/sqrt 282 + scalar broadcast 319 |
| DivF/gamma | 401 | 401 |
| RMS F32 gather/BF16, residual x scatter | 744 + 449 | gather/scatter 제거; BF16 cast는 유지 |
| Residual add/BF16, gate/BF16 | 341 + 401 | 341 + 401 |
| Final canonical gather | 504 | 제거 |

정적 기준 [baseline.json](baseline.json)과 후보 [candidate.json](candidate.json), 각 graph를 보존했다. 정적 수치를 그대로 하드웨어 speedup으로 해석하지 않는다. 이후 E36 공식15개·추가30개 비교가 통과했고, 같은 FFN device image를 유지한 [E37 반복 측정](../../../../docs/experiments/e37-combined-repeats.json)도 완료됐다. E37은 QKV helper 선택이 달라지는 별도 combined package이므로 package 전체와 FFN image의 동일성을 구분한다.
