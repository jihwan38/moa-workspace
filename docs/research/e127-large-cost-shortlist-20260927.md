# E127 이후: 실제로 없앨 중간 저장과 다음 구조 후보

2026-09-27 연구 기록. 조사 시작 main `c924cee`, 성능 비교군은
[E127](../../baseline/e127.json), source tag `baseline-e127`
(`09cf22fcb54349cf5984640d815ade3c4478b3e8`)이다.
새 kernel 구현·compile·Arena·공식 제출은 없다. 아래는 구현 전 shortlist이며 새 E 번호를 부여하지 않는다.

첫 검증 대상으로 **native FFN dot→입력 scale→block scale→F32 합산을 하나의 Main stream으로 연결**하는
조합을 선정한다. Up/gate F32 부분합의 **28.125 MiB write와 같은 양의 read**를 없애는 경로다.
기존 E73의 scale-resident fusion을 현재 native E5M2에 결합해 반복 BF16 lookup을 피하고,
C11의 partial VRF handoff도 없앤다. **국소 데이터 흐름 교체**이며 전체 FFN 구조 탐색을 대신하지 않는다.
현재 profile로 이 부분합 왕복이 전체 실행의 최대 병목이라고 입증된 것은 아니다.

더 큰 구조 연구의 대상은 **up/gate를 물리 cluster별로 소유하게 해 FP8 전체 저장과 joint packed 복사를 함께 제거**하는 안이다.
공개 SDK의 정확한 cluster별 적재·소비 경로와 direct decode 정확도를 먼저 검증해야 하므로
이를 이미 구현 가능한 고성능 해법으로 발표하지 않는다. 두 후보의 이득은 합산하지 않는다.

## 1. 먼저 검증할 조합: 작은 scale을 VRF에 두고 큰 partial은 저장하지 않는다

실제 호출은 `ops::decoder_feedforward → mlp::feedforward_sharded → project_up_and_gate →
project_rows_with_scale_contraction`이다.
[현재 mlp.rs](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)의 36–79행이 제거 대상 경계다.
소스 SHA256은 `f321f960efdbf362dc1ebf20950a2e6b1ae0f0b337bc04b5d4ea508ef64730ff`다.

현재 up/gate 각각의 경로:

```text
compact FP4 HBM → FP4 DM → lookup → 전체 E5M2 weight DM
  → native 3-term dot → F32 입력 scale 복원
  → F32[Row30, H/16=240] DM 저장
  → 5회 partial DM 읽기 + block scale VRF 곱 + H 합산
  → F32 row output → cluster 안에서 120행 compact → 첫 BF16
  → M03 local GeGLU → L-half-owned down → F32 cluster 합 → 기존 후단
```

제안 경로는 앞뒤를 연결한 완성 FFN으로 평가한다:

```text
pre-RMS / 실제 3-term 입력 준비·방송 / FP4→전체 E5M2 DM은 유지
up/gate마다 x TRF와 x_scale VRF를 다시 준비
  각 6행: block scale을 Sub에서 F32 VRF로 준비
          E5M2 weight DM → native dot → narrow_trim4
          → Mul0(x_scale) → Mul1(block_scale) → F32 H 합산
          → F32 두 행씩 정렬 commit, 원래 Row30 output의 해당 6행에 저장
기존 cluster-local compact/BF16 → GeGLU → down → 후단
```

| 항목, up+gate 합 | E127 | 제안 |
|---|---:|---:|
| F32 block partial DM write / read | 28.125 / 28.125 MiB | 0 / 0 |
| 별도 Main block-scale/reduce stream | 10 | 0 |
| Native dot stream 시작 | 2 | 10 |
| Sub block-scale 준비 | 10 | 10 |
| 전체 FP8 weight DM write / read | 112.5 / 112.5 MiB | 유지 |
| FP4 HBM weight read | 56.25 MiB | 유지 |

부분합 계산은 `2×15360×(3840/16)×4 = 29,491,200 B = 28.125 MiB`다.
논리 payload이며 버스 실측량·동시 메모리 peak·절감 cycles가 아니다.
Dot와 scale/reduce의 Main stream 수는 12→10이지만 dot 시작은 2→10으로 늘어난다.
작아진 dot의 setup, scale 준비를 기다리는 시간, Main/VE 점유 변화가 저장 제거 이득을 상쇄할 수 있다.
FP8 materialization·입력 방송·down의 BF16 lookup/부분합 저장은 이 후보에서 없어지지 않는다.

6행은 parameter sweep 결과가 아니다. 현재 scale VRF의 `6×240×4=5760 B` 배치를 재사용한다.
입력 scale 32 B를 합쳐 5792 B로 8 KiB보다 작다. Native 출력의
`Time=[Row6,H/64], Packet=[H/16%4]`는 현재 partial 재읽기의
`collect(Row6,H/128,block8) → narrow_split4`와 같은 유효 block 순서다.
출력은 기존 두 F32/8 B 정렬 commit을 유지한다.
용량 계산과 축 대응은 실제 backend VRF 할당·동시 operand 공급 성공의 증명이 아니다.

설치 SDK **0.8.1**에서 확인한 표현 근거:

- `engine/vector/stage/markers.rs:257,265`: Fp→Fp 및 Fp→IntraSliceReduce 전이 허용.
- `engine/vector/stage/state.rs:31`: 동일 ALU 재사용을 금지하므로 두 곱은 서로 다른 Mul0/Mul1을 쓴다.
- `engine/vector/tensor/vector_tensor.rs:1320`: binary operand로 VRF를 받을 수 있다.
  두 VRF operand와 위 exact shape의 lowering은 미검증이다.
- [공식 Vector Engine 문서](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/vector-engine/intra-slice-chain.html#float-cluster)는
  Float 단계의 Mul0/Mul1과 이어지는 IntraSliceReduce 연결을 설명한다.
  최신 문서의 새 API를 설치 버전에 있다고 가정하지 않았다.

### 과거 시도와 달라지는 비용

| 과거 경로 | 실제 결과 | 이번 조합의 차이 |
|---|---|---|
| [E73](../../moa-dev/experiments/candidates/c05-streamed-decode/README.md), tile별 BF16 dot→scale→합산 | FFN +10.24% 실측 회귀. Up4+gate4+down5번 FP8→BF16 lookup | 현재 materialized E5M2 native dot을 사용해 타일마다 BF16 lookup을 하지 않음. E73의 fusion 원리는 재사용 |
| [C11](../../moa-dev/experiments/candidates/c11-ffn-partial-vrf/README.md), partial을 VRF에 저장한 뒤 새 Main에서 scale·합산 | 정적 +3.55%, NPU 미평가. Native 2→30, scale stream 10→30, Main/Sub/VE handoff | Scale을 VRF에 두고 partial을 같은 stream에서 소비. Native 10개, 별도 scale stream과 partial handoff 없음 |
| E127 현재 | 큰 native dot과 작은 scale pass 사이에 F32 DM 경계 유지 | 곱 순서와 BF16 경계는 유지하면서 그 저장 경계 제거 |

검토한 frozen native 후보에서 이 exact **E5M2 native + 두 scale 곱 + F32 H reduction 단일 stream**의
측정 기록은 찾지 못했다. Fusion 자체를 새 발명이나 새 전체 FFN architecture로 세지 않는다.

### 첫 구현·검증 계약

실제 구현 시작 commit/branch/recipe는 구현 시 별도 기록한다. 제안 source는 E127의
`mlp.rs`와 E73의 scale-resident 방식 조합이며, 성능 control은 별도 E127 recipe로 지정한다.

1. 고정 6행 경로를 whole-shape FFN에 표현한다. Graph에서 10개 fused native stream,
   partial DM/VRF 제거, 원래 FP8 materialization 유지, VRF spill 유무를 검사한다.
   Weight/scale tile sequencer, 출력 destination coverage, padding bounds도 확인한다.
2. 실제 3-term quantizer를 포함해 서로 다른 up/gate block scales, row5/6·29,
   cluster 경계 L7679/7680, K15/16·63/64·3839, zero·상쇄·작은 값의 독립 oracle을 검사한다.
   `((dot*x_scale)*block_scale)` 순서와 최종 BF16 위치를 보존한다.
   기존 3-term 표현 자체가 일반 BF16 exact decomposition이라는 주장은 하지 않는다.
3. 필수 전체 FFN 정확도와 contemporaneous E127 앞뒤 control로 최초 Arena screen을 수행한다.
   정적 차이를 실측 개선/회귀라고 부르지 않는다. Positive면 구현을 동결하고 제한적 반복·확장 검증한다.

구체적인 API/축 오류는 원인을 찾아 지원되는 수리를 한다. 목적한 저장이 남거나 새로운
대형 materialization이 필요하면 그 prototype의 비용 가설을 다시 판단한다.
VRF나 tile 크기를 무작정 바꾸는 series로 확대하지 않는다.

## 2. 더 큰 구조 연구: cluster0=up, cluster1=gate

현재 두 cluster는 각각 L의 절반을 맡아 up과 gate를 모두 계산한다.
대안은 cluster0에 up의 전체 L15360, cluster1에 gate의 전체 L15360을 두어 slice당60행을 소유한다.
각 public FP4 weight를 해당 cluster의 compact DM에 적재하고 **단일 native pipeline**이
각 cluster의 projection을 처리한다. Projection을 Element/interleave 축으로 합치지 않는다.
FP4 DM→lookup E5M2→native dot→F32 block scale/reduction→BF16 row 출력을 거쳐,
필요한 두 half를 M03의 L-half GeGLU/down consumer로 돌려보내야 완성 구조다.

성립하면 up/gate FP8 intermediate의 **112.5 MiB write + 112.5 MiB read**를 제거하면서
E123의 joint packed assembly **56.25 MiB write + 56.25 MiB read**도 만들지 않는다.
후자는 E123 대비 회피하는 비용이며 E127에는 없으므로 두 제거량을 더해 E127 절감량으로 쓰지 않는다.
필수 compact FP4 HBM read 56.25 MiB와 block scale, F32 partial/reduction은 여전히 필요하다.
새 BF16 출력 재분배 대상은 두 projection 합60 KiB이지만 실제 routing·gather·정렬·임시 복사 비용은 미확인이다.
입력/scale lifetime, per-slice 행 수와 두 projection의 겹침도 바뀐다.

`tensor/memory.rs`의 public `cluster_tile` 존재를 확인했지만,
서로 다른 두 HBM source를 해당 cluster에만 적재하고 단일 consumer가 정확히 읽는
descriptor/mask와 native lookup 경로는 아직 만들거나 검증하지 않았다.
**첫 gate는 이 소유권/전송 계약과 독립 두-projection oracle**이다.
직접 decode가 단일 producer여서 정확해질 것이라는 추정은 검증 결과가 아니다.

재실험 중복을 막는 근거:

- Separate compact direct + projection별 TRF/scale reload는 이미 **E103**이다.
  [고정 source를 지정한 build](../../moa-dev/experiments/candidates/c07-native-scale-dpe/build-projection-probe-reload-v1.json)와
  [projection 실패](../../moa-dev/experiments/candidates/c07-native-scale-dpe/projection-probe-reload-v1-results.json)가 있다.
  M03 consumer 결합만으로 이 producer 정확도 실패가 해결되지는 않는다.
- [E123](../../moa-dev/experiments/candidates/ffn-direct-boundary-recompute/README.md)은
  compact load→raw Main copy로 joint packed DM 구성→단일 direct producer 경로다.
  기본 비교 PASS, 단회 M03보다 0.7392% 개선은 남아 있는 positive 근거다.
  작은 개선 때문에 실패한 architecture라고 취급하지 않는다. 추가 검증·M03 consumer 결합은 별도 결정이다.
- `begin_interleaved`도 이미 [v7b](../../moa-dev/experiments/candidates/fp4-native-boundary/v7b/probe.rs)에서 시도했다.
  [로그](../../moa-dev/experiments/candidates/fp4-native-boundary/evidence/boundary-v7b/build.log)의
  declared `Projection` / pipeline 내부 `interleave` 축 mismatch를 고칠 공개 stream rename을 찾지 못했다.
  [QKV interleaved 기록](../../moa-dev/experiments/candidates/qkv-bf16-interleaved-kv-e55/README.md)도 동일 문제다.
  근거 없이 API 이름만 바꾼 재시도를 권하지 않는다.

## 3. QKV/O에서 당장 큰 삭제라고 부를 수 없는 것

QKV의 source는 이미 [ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)에서 Q/K/V projection을
Q/K/V RMS·RoPE보다 먼저 호출한다. 단순 호출 순서 변경은 새 해법이 아니다.
[동일 QKV image의 profile](../../moa-dev/experiments/candidates/e104-qkv-output-profile/README.md)에서
V dot이 Q RoPE 직후 시작하지만 V weight가 준비된 뒤 남은 구간은 관측223–2107 cycles다.
약15.6k의 Q/K head RMS→Q RoPE 전부를 제거 가능한 대기로 계산하면 weight 전송과의 중첩을 무시한다.

후속 architecture 질문은 C0=Q4096행/C1=K2048+V2048행으로 head 후단까지 소유하는 구성,
또는 Main 순수 F32 dot producer + Sub scale/BF16/head consumer 분리다.
전자는 cluster마다 총4096행·15 MiB weight로 현재와 같아 계산량이나 HBM byte가 줄지 않는다.
후자는 논리 F32 raw projection 32 KiB write/read와 소비자 작업을 추가한다.
현재 dot 자체의 VE scale를 남긴 채 후처리만 Sub로 옮기면 공유 VE 때문에 겹침을 보장할 수 없다.
[공식 Schedule](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)과
[Commit Adapter](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/commit-adapter.html)의 자원 조건을 따른다.
정확한 scheduling·allocator lifetime 변화부터 확인할 조건부 안이며 큰 이득 후보로 확정하지 않는다.

O의 full-K row owner + F32 scale fusion + BF16 직접 gather는
[과거 full-K family](../../moa-dev/experiments/candidates/output-two-clusters/README.md)의 후단까지 바꾸는 조합이다.
K16 합산을 없애도 FP8 HBM15 MiB와 lookup은 남고, activation 목적지 payload는
256 KiB→3.75 MiB(유효480 slices)로 늘며 gather도 32×480B F32에서 예상480×16B BF16로 쪼개진다.
E127의 두 stage overlap을 버리는 비용도 함께 보아야 한다. 이번 우선 후보에서 제외한다.

공개 SDK0.8.1의 `TuContext::begin`은 DM view만 받고, direct FP8→BF16 fetch cast와
재사용 가능한 lookup table handle은 없다. HBM→Fetch 직결이나 무료 table 재사용을 해법으로 세지 않는다.
[기존 API 감사](fp8-lookup-table-sharing-audit.md)와
[공식 tensor 이동 문서](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/index.html)를 대조했다.

## 결정과 성능 근거의 한계

- **채택:** 새 구현/성능 결과가 없으므로 E127 유지. E123 단회 positive의 검증 가능성도 보존한다.
- **추가 튜닝:** tile-size/배치 sweep은 시작하지 않는다. 첫 후보는 현재6행 scale 배치로 고정한다.
  향후 positive가 5% 미만이어도 채택 검증을 자동 차단하지 않는다.
- **주요 구조 연구:** 첫 bounded feasibility는 native dot/scale/reduce의 partial 저장 제거.
  더 큰 연구 대상은 projection별 cluster 소유권으로 FP8 materialization과 joint 복사를 함께 없애는 경로다.
  QKV/O를 점수 전략에서 제외한 것은 아니며 공유 자원/ownership의 구체적인 삭제 효과가 확인되어야 한다.

[E104 FFN NPU profile](../../moa-dev/experiments/candidates/e104-weight-path/README.md)에서
up/gate별 FP4 적재 약48k, FP8 변환·저장 약20k, native dot·입력 scale·partial 저장 약12k가 관측됐다.
마지막12k 전체가 partial 저장 비용인 것은 아니다. 구간 중첩·대기를 포함하며 현재 E127의
M03 후단/스케줄과도 구분해야 한다. 이 수치나 논리 byte 감소만으로 큰 speedup·10위 진입을 약속하지 않는다.
첫 후보는 실제 저장 경계 하나를 없앨 구체적 경로이고, 큰 순위 격차를 닫는 충분조건은 아직 없다.
