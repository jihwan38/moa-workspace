# C02 F32 packet scale / C03 cluster-local BF16 heads

2026-09-22, branch `opt/c02-c03-e55`. 모든 후보는 E55
`873f64ed94229c88cd18d2731bdcca7eea0d6365`에서 독립 생성했다.
Live runtime과 `baseline-e55`는 변경하지 않았다. SDK/compiler는 0.8.1이다.

**완료: C02 세 구성과 C03 모두 미채택, E55 유지.** C03은 대조군과 각각 3회 반복했고,
C02는 job73881에서 E55 앞뒤 대조를 둔 단회 screening을 마쳤다. 기본 출력 비교
165개(대조 포함), 추가 입력 비교 30개, packet 진단 4개, 비대상 cache BF16 원소
25,141,248개 검사가 모두 통과했다. 정확도 통과를 성능 개선으로 간주하지 않는다.
[감사 요약](summary.json), [묶음 원본 결과](batch-results.json).

## C03: head를 계산한 cluster에 유지

`ops::sliding_project_qkv → sliding::projection::{project_query,project_key_value}
→ sliding::rmsnorm::{normalize_query,normalize_key,normalize_value}
→ sliding::rope::apply_rope → Q store / K,V scatter`가 변경 경로다.

E55의 BF16 pre-RMS, BF16 dot, 완전한 F32 dot에 channel scale을 곱한 후 첫 BF16 cast,
row Switch, RMS/RoPE의 연산 순서와 cast 경계를 유지했다. 기존 projection의 cluster별
4개 head를 `[Ns/4]`에 남겨 두고 `[1#64,Ns%4]` slice 배치를 후단까지 전달한다.
Q/K/V의 typed head gather는 여전히 존재하지만 cluster 사이의 이동을 제거한다.
RoPE 마지막 Switch는 cluster 안의 4개 head를 모으도록 `Broadcast1.slice1=4`로 바꾸고,
Q store와 K/V scatter가 분산된 cluster 좌표를 직접 사용하게 했다.
V 마지막 gather는 typed DMA를 유지한다. Dot 자체나 HBM weight load를 줄이지 않는다.

소스: [projection.rs](c03-cluster-heads/projection.rs),
[rmsnorm.rs](c03-cluster-heads/rmsnorm.rs), [rope.rs](c03-cluster-heads/rope.rs),
[ops.rs](c03-cluster-heads/ops.rs). Full attention 경로는 그대로이며 O/FFN image도 E55와 같다.

| 항목 | E55 대조 | C03 |
|---|---:|---:|
| 정적 schedule | 51,293 | 50,370 |
| NPU job 중앙값의 중앙값 | 103,157 | 104,762 |
| job 중앙값 범위 | 102,946–103,978 | 104,034–104,988 |
| 독립 Arena job 수 | 3 | 3 |
| 기본 출력 비교 | 45 PASS | 45 PASS |

정적 시간은 −1.80%인데 NPU에서는 **+1.56% 회귀**했다. 후보의 가장 빠른 job도
관측한 대조군 범위보다 56 cycles 높다. 이 측정 범위에서 미채택이며 모든 환경에서
같은 차이가 난다는 보장은 아니다. 변하지 않은 O/FFN의 cycle 변화는 후보 효과로 해석하지 않는다.
[E68](../../../../docs/experiments/e68-c03-cluster-heads.json),
[E69 대조](../../../../docs/experiments/e69-c02-c03-control.json).

[전송 비용을 다시 대조](qkv-transfer-cost.json)하면 Q/K/V projection의 typed gather는
여전히 각각 **716/575/575 정적 cycles**다. Cluster 사이 이동을 제거한 구조 변경이
이 compiler 비용을 줄이지 않았다. 반대로 cos/sin 복제는 각각 **362→638**,
마지막 V gather는 **362→408**로 늘어 DM→DM 구간 합계가 **5,604→6,202**가 됐다.
전체 schedule −923 cycles에는 실행 중첩·다른 작업 변화도 포함된다. 이 기록으로 실제
NPU의 +1.56%를 instruction별로 정확히 배분할 수는 없지만, 가정했던 gather 절감이
정적 비용에서도 실현되지 않았다는 점은 확인된다.

[edges.py](edges.py)는 공식 PRNG/reference와 tolerance를 유지하면서 run3..8의 위치를
`0,1023,1024,32767,1763,32766`으로 지정한다. 두 cluster의 모든 Q/K/V head를 비교하고,
cache를 비영(非零) BF16 sentinel로 채워 target row 이외의 원소를 bit 단위로 검사한다.
범위는 cache 첫/마지막 row, wrap, RoPE 첫/마지막 row다. O/FFN도 같은 추가 6입력을
실행한다. 이 진단의 cycle은 기본 성능 측정에 합치지 않는다. C03의 edge host가 포함한
세 device image는 기본 host의 image와 모두 byte-identical하다.

## C02: 두 block의 F32 결과를 보존하는 packet

실제 호출은 `host/runtime.rs: Decode::mlp → ops::decoder_feedforward
→ shared::mlp::feedforward_sharded → project_up_and_gate / project_down_sharded`다.
입출력 signature, full-matrix FP4→FP8→BF16 decode, GeGLU, global scale,
post-RMS/residual/layer scalar는 E55를 유지한다. Up/gate와 down을 각각 변경한 구성과
둘을 함께 적용한 구성을 분리했다.

1. FP8 scale을 패딩 없이 연속 F32 DM으로 변환한다.
2. 이미 F32인 scale을 Main sequencer에서 `2#8`로 읽고 `2#4`로 commit한다.
   FP8 cast와 padding 삽입을 한 번에 요구하지 않는다.
3. Sub가 32B씩 읽어서 두 live F32 값 / 네 F32 slot의 VRF를 만든다.
   Up/gate는 4행 × 120쌍 × 4slot × 4B = **7,680B**,
   down은 4행 × 60쌍 × 4slot × 4B = **3,840B**다.
4. 64B BF16 packet에 16원소 block 두 개를 넣는다. `contract_packet`과
   `LaneMode::Sequential`이 두 F32 dot을 따로 유지하고 각각의 scale을 곱한다.
5. SDK 0.8.1이 padded packet 축의 VE reduction을 거부하므로 결과를 compact F32 DM에
   저장한다. 다시 읽은 unpadded F32 block들을 합산한 뒤 E55의 BF16 경계로 돌아간다.
   Down은 column-slice F32 합산 후 BF16으로 변환한다.

추가 BF16 partial 반올림은 없다. F32 합산 배치는 달라지므로 임의 입력의 수치 동등성을
가정하지 않는다. Row tile은 up/gate `8행×4회 → 4행×8회`, down `12행×5회 → 4행×15회`다.

소스: [up/gate](c02-upgate/mlp.rs), [down](c02-down/mlp.rs), [결합](c02-combined/mlp.rs).
`prepare.py`는 E55 원본의 해당 helper/pass만 바꾸고 기존 decode·후단을 복원하거나 대체하지 않는다.

| 구성 | FFN 정적 cycles | NPU cycles | 같은 job 대조 대비 |
|---|---:|---:|---:|
| E55 앞/뒤 | 162,209 | 379,238 / 379,598 | 두 값 중앙값 379,418 기준 |
| [C02 up/gate E70](../../../../docs/experiments/e70-c02-upgate.json) | 182,993 | 432,912 | +14.10% |
| [C02 down E71](../../../../docs/experiments/e71-c02-down.json) | 176,715 | 415,542 | +9.52% |
| [C02 결합 E72](../../../../docs/experiments/e72-c02-combined.json) | 198,453 | 456,372 | +20.28% |
| 최초 typed-DMA scale padding 결합 | 1,599,773 | 미실행 | 정적 회귀로 중단 |

C02 각 구성의 기본 15개 비교는 모두 PASS다. 각 행은 같은 job의 서로 다른 프로세스이며
반복 측정이 아니다. 대조군 앞뒤 차이는 360 cycles(약 0.095%)이고, 후보 세 개는 모두
양쪽 대조보다 크게 느려 추가 반복·추가 입력·모델 전체 검증 없이 중단했다.
분류표의 E69 독립 3-job 대조 378,860을 사용하면 각각 +14.27%, +9.68%, +20.46%다.
두 기준의 백분율을 섞지 않는다. C02+C03 전체 결합은 실행하지 않았다.

최초 typed-DMA 후보는 반복되는 DM→DM 재배치가 컸다. F32 sequencer로 바꾸어 이 비용을
크게 줄였지만, scale 변환·재배치 pass와 F32 partial 저장·재읽기 때문에 최종 정적 시간도
기준보다 길다. Dot packet 수를 줄인 사실만으로 전체 FFN 개선을 주장하지 않는다.
[정적 구간 기록](static-summary.json), [최초 DMA 소스](attempts/c02-combined-dma.rs).

독립 [packet_probe.rs](packet_probe.rs)와 [host](probe_packet.rs)는 네 진단 입력을 만든다.
행·block별 서로 다른 scale, 첫/마지막 scale block, 이웃 block 상쇄, zero input을 검사한다.
F32 scale 3,840개와 출력 16개를 CPU의 정확한 dyadic oracle와 비교한다.
상쇄 입력의 정답은 `[0.234375,0.46875,0.9375,1.875]`이며 BF16 partial 반올림을 넣으면
이 결과를 보존할 수 없다. NPU에서 네 case의 scale 3,840개·출력 16개가 모두 exact PASS였다.
이는 4행 microkernel 검사이며 전체 FFN 성능/정확도 검사를 대체하지 않는다.

이전에 실패한 접근도 지우지 않았다. [probe-attempts.json](probe-attempts.json)과 `attempts/`에
네 번의 compile failure와 두 번의 성공 경로를 보존했다. 실패는 padded reduce,
commit packet trim, transpose의 입력 행 제한으로 구분한다. 실패한 build의 stale executable은
제출하지 않았다.

## 검증과 재현

기본 host/fixture/runner는 동결 E55와 같고, live crate를 편집하거나 재빌드하지 않는다.
[build.py](build.py)는 실제 container의 46개 compile input을 확인한 뒤 exact compile의
종료 코드와 host build 종료 코드를 검사한다. 세 device image가 host에 각각 한 번 포함되고,
변경하지 않은 kernel의 image가 E55와 같은지도 확인한다.

```bash
python3 moa-dev/experiments/candidates/c02-c03-e55/prepare.py
python3 moa-dev/experiments/candidates/c02-c03-e55/build.py c02-upgate --label NEW_LABEL
python3 moa-dev/experiments/candidates/c02-c03-e55/verify.py --check
```

`NEW_LABEL`은 이전 기록과 다른 이름이어야 한다. Compiler target은 후보 사이에 공유하므로
동일 target을 사용하는 host build는 직렬로 실행한다. Kernel 소스·빌드 fingerprint·실패 로그는
Git에 보존하며, 큰 host/image/fixture/schedule 원본은 ignored `experiments/artifacts/c02-c03-e55/`에 둔다.

[batch.py](batch.py)는 독립 packet probe → E55 → C02 up/gate → C02 down → C02 결합 →
C03 경계 → E55 순서의 70초 이내 screening job을 만든다. 각 성능 프로세스는 원래
3-input host와 fixture/runner를 그대로 사용한다. 바깥 실행 script만 추가했으며,
한 job의 여러 stage를 독립 Arena 반복으로 세지 않는다. E55 앞뒤 실행은 시간에 따른
변동을 관찰하는 대조다. 실패한 stage가 있어도 다른 stage의 결과를 수집한다.

## Weight 변환 비용에 대한 후속 확인

[E55 weight 비용 기록](e55-weight-cost.json)은 실제 기준 schedule의 여섯 decode instruction을
소스 위치와 함께 추적한다. 합계 **86,178 정적 cycles**, Main 합계 **125,274 cycles**의
약 **68.8%**다. HBM/DM 전송과 겹치므로 전체 실측의 68.8%가 변환이라는 뜻은 아니다.
세 행렬의 packed weight+scale은 논리 **94.92 MiB**, FP8+BF16 중간 write는
**506.25 MiB**다. 후자는 padding·read traffic·다른 임시 tensor를 제외한다.

C02는 이 full-matrix decode를 제거하지 않는다. 다음 decode staging 검토는 기존
[C05/D02](../../../../docs/research/optimization-candidates-20260922.md)의 실패 경계를 먼저
확인해야 한다. 특히 FP8 중간 tensor만 제거한 E48/E50과, BF16 weight 저장을 제거하고
바로 DPE에 연결하는 경로를 구분한다. C01의 streamed 후보는 scale-DPE와 추가 BF16
partial 변경도 포함했으므로 그 성능을 decode 단독 효과로 간주하지 않는다.
