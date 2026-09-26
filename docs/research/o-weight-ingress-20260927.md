# O weight 전송·배치 재검토 — 2026-09-27

후속 실행 완료: [E125/M08와 E126/동결 C09의 NPU 결과](o-weight-ingress-execution-20260927.md).
M08은+13.55% 회귀로 종료했다. C09는 최초 단회−2.50% 뒤 종료했으나,
후속 [E127 검증](../../moa-dev/experiments/candidates/c09-output-first-npu/validation/README.md)에서
세 allocation 모두 개선(변화율 중앙값−3.8255%)과 확장 정확도를 확인해 채택했다.
아래는 실행 전 연구 당시의 근거와 우선순위이며, 미실행 표시는 후속 보고가 갱신한다.

**우선순위는 추가 weight 복사 없이 읽기 순서를 바꾸되, 결과를 모으는 비용까지 함께
검사하는 것이다. 첫 진단 후보는 ‘행쌍을 보존한 4-way row interleave + 기존 16-way K 합산
+ 직접 F32 gather’다.** 전송만 빨라지는지를 보고 채택하지 않는다. 두 번째 선택지는
정적 분석만 하고 닫았던 C09의 첫 NPU 측정이다. 큰 개선이 입증된 새 설계는 아직 없다.

시작 commit `48bea3dea8d3b98b4f66a15981c30646d40ddfd0`, 조사 branch
`research/o-weight-ingress-20260927`. 비교 runtime은 [E122](../../baseline/e122.json),
O는 E104와 동일한 image다. 이 작업은 구현·기존 graph·원본 로그·설치 SDK 0.8.1·1차 문헌을
대조한 연구다. 새 kernel compile, Arena 실행, runtime 변경, 공식 제출은 없다.

세부 근거는 [kernel/descriptor 감사](o-weight-ingress-20260927/kernel-audit.md),
[실험 원본 재검증](o-weight-ingress-20260927/prior-audit.md),
[1차 문헌 10개 비교](o-weight-ingress-20260927/literature.md),
[조건부 pipeline 계산](o-weight-ingress-20260927/pipeline-model.json)에 보존한다.

## 1. 지연되는 경계는 확인했지만, 전송이 느린 물리 원인은 아직 구분하지 못했다

[ops::sliding_attention_output](../../furiosa-opt-gemma4-12B/src/ops.rs)의 경로는 다음과 같다.
실제 구현은 [project_output](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs),
[normalize_output_fused](../../furiosa-opt-gemma4-12B/src/device/shared/rmsnorm.rs)를 확인했다.

```text
BF16 x[4096] HBM ─→ K256씩 DM에 복제 ─→ Sub → BF16 TRF
FP8 W[3840,4096] HBM ─→ 각 slice의 120×256 FP8 DM 전체 적재
                         └→ Main: lookup BF16 → dot → F32 16-way K 합산
                                                   ↓
                                    F32 H3840 결과 → H480×8 gather
                                                   ↓
channel scale HBM → VRF ───────────→ F32 scale → 첫 BF16 변환
                                                   ↓
                                    post-RMS → residual add → HBM
```

Weight는 15,728,640 B(15 MiB), 각 slice는 30,720 B다. 2 clusters×256 slices를 모두 쓴다.
고유 입력 x는 8,192 B지만 DM 복제 payload는 262,144 B다. 현재 weight는 이미
**FP8으로 DM에 저장하고 dot 직전에만 BF16으로 변환**한다. 전체 BF16 weight를 따로
저장하는 낭비를 없애자는 제안은 현재 O에 새로 적용할 최적화가 아니다.

[기존 NPU trace](../../moa-dev/experiments/candidates/e104-qkv-output-profile/profile-analysis.json)는
2개 Arena job, 3 runs×2 clusters씩 총 12개 관측을 갖는다.

| 관측 | 중앙값 | 의미 |
|---|---:|---|
| task 시작 → x TRF 준비 | 7,902 cycles | 입력은 dot보다 일찍 준비됨 |
| x 준비 → dot 시작 | 20,955 | 입력이 준비돼도 weight를 기다림 |
| weight DMA marker 구간 | 26,736 | 대기 포함 구간; 순수 버스 service time은 아님 |
| weight 종료 → dot 시작 | **2, 모든 12개 관측 동일** | 확인한 마지막 준비 경계는 weight |
| lookup table 준비 → dot 시작 | 21,419 | 현재 단일 dot의 시작을 늦추는 table은 아님 |
| dot + 16-slice 합산 | 12,014 | 순수 곱셈 시간으로 분리하지 않음 |
| F32 결과 gather | 1,218 | 배치를 바꾸면 증가할 수 있는 실제 비용 |
| dot 종료 → task 종료 | 9,389.5 | gather·scale·RMS·residual·wrapper를 포함 |

각 중앙값은 별도 분포다. 한 실행의 waterfall처럼 합산하지 않는다. 이 증거는
전체 weight 완료를 기다리는 dependency를 보여주지만, HBM 포화·row conflict·DM bank
충돌·engine queue 중 무엇이 26.7k를 만들었는지는 알려주지 않는다.

## 2. 현재 배치는 기본 정렬과 DMN 분산을 이미 갖추고 있다

보존된 `output-bf16-activation-e47/candidate.bin`의 SHA-256은 현재 O와 동일한
`359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b`다.
동반 graph `24d45fc706964d6dbfb7eb3a52a29381dd59347cd714d681ed2ab603f506cee0`에서
weight ingress와 F32 gather를 확인했다. 단순히 과거 같은 이름의 helper를 비교한 것이 아니다.

현재 좌표는 `c<2, r<16, t<120, q<16, u<256`에 대해 다음과 같다.

```text
physical slice s = 16*r + q
output row h    = 1920*c + 120*r + t
input column k  = 256*q + u
HBM offset     = 4096*h + k
```

Weight DMA의 한 descriptor 안에서 packet 바깥 루프는 다음 크기/stride를 갖는다.
표는 바깥→안쪽 순서이며, 마지막 DMN-pair 축이 먼저 변한다.

| loop 크기 | HBM stride | DM stride | 역할 |
|---:|---:|---:|---|
| 120 | 4,096 B | 256 B | slice의 local output row |
| 2 | 491,520 B | 67,108,864 B | 같은 DMN 안의 row group |
| 16 | 256 B | 4,194,304 B | K shard / slice |
| 2 | 983,040 B | 134,217,728 B | DMA engine에 연결된 DMN pair |
| packet 256 B | 1 B | 1 B | FP8 K256 |

이는 DM **주소 공간 stride**이며 SRAM 용량이나 복사량이 아니다. 8개 descriptor를
합쳐 61,440개 packet, 원본 15 MiB를 정확히 한 번 읽는다. 목적지 512 slices의
120×256 좌표도 일치한다. descriptor 순회상 DMN pair가 매 packet 교대하고,
각 DMN의 32 slices를 방문하며, HBM stack bit 8도 바뀐다.

따라서 ‘한 DMN만 사용해서 느림’, ‘256B 정렬을 놓침’, ‘한 HBM stack만 읽음’을
현재 구현의 확인된 문제로 쓰면 틀린다. 다음 질문은 **DMN 분산을 지키면서 HBM 주소
순서와 최종 소비 배치를 더 유리하게 만들 수 있는가**다. 각 controller의 실제 issue 순서,
절대 HBM base, channel XOR, queue 재정렬까지는 이 열거로 증명하지 않는다.

공식 [DMA 설명](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html)은
source/destination mapping에서 sequencer를 유도한다고 설명한다.
[메모리 설명](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)은
정렬 외에 stack/channel·DMN·slice 분산을 구분한다. 문서의 일부 bandwidth/topology 요약은
단위·범위가 모호하므로 광고 peak로 실효율이나 회수 가능한 cycles를 계산하지 않았다.
설치 SDK의 `to_dm`에는 stream 순서·engine 번호·절대 bank 주소를 지정하는 공개 인자가
확인되지 않았다. 문서의 저수준 설명을 즉시 사용할 Rust API로 가정하지 않는다.

## 3. 이전 실험은 무엇을 배제하며, 무엇을 아직 배제하지 못하는가

과거 결과의 숫자는 당시 동시 비교군 또는 기록에 명시된 비교 방식이다. 서로 다른
산술·epilogue를 갖는 실험의 raw cycles를 현재 E122에 직접 순위 매기지 않는다.

| 선행 시도 | 확인 결과 | 이번 판단 |
|---|---|---|
| E24 기반 row2/row4 | weight DMA 정적 13,512로 동일. 전체 32,252→33,475/34,435. canonical BF16 Switch 추가. 독립 NPU 없음 | 행 interleave 자체가 새 기법은 아님. 출력 복구 비용을 바꿔야 재검토 가치가 있음 |
| E31 + row2, E34 | NPU PASS, 당시 E31 반복 중앙값 대비 약 +4.31% | 기존 조합은 실패. 현재 BF16 입력·F32 gather 조합 전체의 실패 증거는 아님 |
| E64 column-major weight 적재 후 전체 weight 재배치 | 정확하지만 O +149.61%. weight DMA 정적 시간도 줄지 않음 | 15 MiB weight 재복사를 추가하는 방향의 우선순위가 낮음 |
| M01 적재 배치에서 dot 후 partial만 이동 | DMA/Switch 경로 각각 +8.74%/+18.65% | 작은 intermediate도 현재 경로에 없던 저장·이동이면 비쌀 수 있음 |
| M02 cluster가 K-half 소유 | 두 완성 경로 +55.00%/+75.06% | reduction 감소와 큰 연속 chunk만으로 입력 복제·partial gather 증가를 상쇄하지 못함 |
| C09 두 60-row stage | 정적 27,239→27,432(+0.71%), 실제 정적 overlap 1,825. **NPU 미실행** | ‘하드웨어에서 실패한 pipeline’으로 분류하면 안 됨. 첫 NPU 진단은 남아 있음 |
| C12 Main dot / Sub reduction | +2.46%. 의도했던 다음 dot과의 정적 overlap 0 | context 이동만으로 다음 weight 준비가 앞당겨지지는 않음 |
| C10 lookup→weight TRF | **QKV**의 StoTrf lowering 실패, O는 그대로 | O 성능 실험이 아님. 같은 lookup→TRF 경로가 필요한 대안의 API 위험 근거 |
| E123/E124 전송 계약 수정 | padded destination으로 source 전송이 확대되는 사례 확인·수정 | 새 tile도 수학적 coverage에 더해 실제 descriptor 범위·중복·packet을 검사해야 함 |

[prior 감사](o-weight-ingress-20260927/prior-audit.md)는 원본 report/log와 보존 graph를
재검증한다. C09의 정적 작은 차이를 NPU 회귀로, C10의 lowering 실패를 O 아키텍처
전체의 불가능으로 확대하지 않는다. 반대로 실제 회귀 결과를 논문의 일반론으로 무시하지 않는다.

## 4. 문헌에서 가져올 원칙과 이미 충족한 조건

| 1차 자료 | O에 적용할 원칙 | 그대로 가져오면 안 되는 전제 |
|---|---|---|
| [Gemma3 FusedDQP](https://arxiv.org/html/2602.06063v2), §3.2 | 작은 weight 영역을 그 소유자가 변환·소비하고 local 누산 | AMD AIE API, Q4NX INT4 배치. 현재 O의 lookup+dot은 이미 fusion되어 있음 |
| [TileFuse](https://arxiv.org/html/2606.11357v2), §4.2–4.4 | weight 공급 단위와 출력 소유권을 함께 설계 | offline pre-tiling, 기존 8→32 core 활용 개선. O는 이미 512 slices 사용 |
| [Marlin 저자 구현](https://github.com/IST-DASLab/marlin) | 작은 activation 재사용, compact weight 공급, 변환·계산 중첩 | 무료 offline reshuffle, CUDA 비동기 copy와 cache 제어 |
| [FLUTE](https://arxiv.org/html/2407.10960v4), §3 | lookup 준비 수명과 staged 공급을 함께 설계 | 사전 packing, 사용자 LUT 공유. 설치 SDK에 prepared-table handle 없음 |
| [JAX Pallas TPU matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) | stage 누산을 F32로 유지하고 마지막에 변환 | TPU pipeline API·GEMM 처리량을 batch1 RNGD에 대입 |
| [GemLite](https://pytorch.org/blog/accelerating-llm-inference/) | batch1/non-packed에 맞춰 K 분할과 reduction 비용 선택 | packed INT4 scale/zero 재사용을 FP8 channel scale의 이득으로 계산 |
| [FlashDecoding++](https://arxiv.org/html/2311.01282v4), §4–5 | shape별 공급/계산 비율과 pipeline의 처음·끝 비용 비교 | buffer를 늘릴수록 항상 빠르다는 가정 |
| [T-MAN](https://arxiv.org/html/2511.11248v1), §4–5 | LUT·register 용량·DMA를 한 경로로 계산 | HVX runtime LUT를 RNGD의 고정 dtype lookup과 동일시 |

공통 원칙은 **읽기 좋은 배치와 계산·결과 저장에 좋은 배치를 함께 결정**하는 것이다.
현재 O의 BF16 activation TRF 재사용, FP8 weight 유지, 소비 직전 변환은 이 원칙과 맞는다.
미해결 부분은 실제 읽기 순서와 전체 weight 완료 의존성이다. 문헌의 배속을 우리의
예상 개선율로 옮길 근거는 없다. 확인 절·버전·추가 비용은 문헌 부록에 기록했다.

## 5. 1순위: 행쌍 보존 interleave의 공급 이득과 gather 손실을 함께 판별

새 독립 아키텍처가 아니라 **기존 row-interleave family의 생산·소비 배치 재설계**다.
산술은 현재 BF16/F32 경로에서 시작하고, 성능 비교군은 E122다. 아직 구현하지 않았다.

```text
c < 2, g < 4, r < 4, t < 60, j < 2, q < 16, u < 256
s = 64*g + 16*r + q
h = 1920*c + 480*g + 8*t + 2*r + j
k = 256*q + u
```

한 DMA-engine pair가 소유하는 H480 영역과 16개 K shard를 보존한다. 각 slice는
여전히 FP8 120×256과 BF16 x256을 처리한다. 달라지는 것은 120행이 연속인지,
두 행씩 묶여 H480 안에 교차 배치되는지다. 원본 HBM을 pack하는 단계는 두지 않는다.

같은 루프 순서를 생성한다고 가정하면 DMN-pair 사이 HBM 주소 차이는
983,040→16,384 B로 바뀐다. 대신 local t stride는 커진다. 이는 **주소 순서 가설**이며
bank conflict 감소나 runtime bandwidth 개선의 증명이 아니다. 256B 정렬과 stack bit 8
분포를 새로 개선하는 것도 아니다.

완성 경로는 `원본 FP8 HBM → 새 DM 배치 → 기존 lookup/dot/16K F32 합 → 직접 canonical
H480 F32 gather → 기존 scale/BF16/RMS/residual`이다. x TRF는 dot 동안 유지하며
weight DM은 dot 종료까지 유지한다. 새 intermediate weight·partial tensor를 추가하지 않는다.
F32 결과는 gather가 끝날 때까지, scale VRF는 scale 종료까지 유지한다.

**두 행씩 묶는 이유는 마지막 저장 계약이다.** 한 행 F32는 4B라 scatter 목적지의
8B 단위를 만족시키기 어렵다. 두 인접 행은 8B다. local output `2*t+j`에 대해
source byte는 `8*t+4*j`, canonical destination은 `32*t+8*r+4*j`다.
`j=0,1`을 한 packet으로 보내면 8B 정렬을 보존한다. 예전 BF16 canonical Switch를
별도 stage로 되살리지 않는 설계상의 차이다.

그러나 비용은 무료가 아니다.

| 항목 | 현재 | 제안 경로의 요구/예상 |
|---|---:|---:|
| 고유 FP8 weight 읽기 | 15 MiB | 동일, 중복 읽기 금지 |
| 추가 weight DM copy | 없음 | 없음이어야 함 |
| lookup/dot stream 수 | 1 | 1 유지 |
| K 합산 | 16-way, stream 내부 F32 | 유지 |
| 최종 F32 gather payload | 15,360 B | 동일 |
| gather sequencer packet 모양 | **32×480 B, 실제 graph** | **1,920×8 B 예상**, 새 graph로 확인 필요 |
| 추가 gather stage | 없음 | 추가하지 않는 것이 목표이나 packet 처리비용 증가 가능 |

packet 60배는 latency 60배나 command 60배라는 뜻이 아니다. aggregate 내부 반복과
명령 startup을 구분해야 한다. 이 tradeoff 때문에 ‘더 연속된 HBM’만 보고 바로 전체
후보를 확장하지 않는다.

이 1순위는 공급/소비 배치의 원인을 분리하려는 탐색 순서다. 예상 성능이 더 좋다는
순위는 아니다. 이미 구현된 C09의 최초 측정은 새 배치 구현보다 준비 비용이 낮을 수 있다.

다음 실행의 판별 절차는 한 비교쌍으로 제한한다.

1. **전송 계약:** baseline과 위 mapping 한 개만 생성한다. 새 descriptor의 source
   coverage·bounds·목적지 좌표·256B 구간·DMN 순서, F32 gather의 8B packet을 검사한다.
   결과가 같은 descriptor라면 공급 순서 가설을 구현하지 못한 것이다.
2. **분리 진단:** 같은 row-major weight의 HBM→DM 적재를 profile한다. 모든 byte를
   검증할 수 있는 왕복 출력 등 실제 consumer를 두고, optimizer가 load를 지우지 않았는지
   확인한다. 왕복 HBM store는 진단용 추가 비용이며 해당 marker를 ingress 측정에서 구분한다.
   별도로 F32 결과→canonical H480 gather도 확인한다. 원본/후보 입력·형식·payload를 고정한다.
3. **전체 O:** isolated ingress가 짧아져도, 변경된 gather를 포함한 O에서 contemporaneous
   E122 앞뒤 비교가 이겨야 한다. Main/Sub 동시 접근 조건이 달라지므로 isolated 차이를
   전체 cycles에서 빼서 예상값을 만들지 않는다.
4. **판정:** isolated ingress가 그대로면 그 조건에서 개선 근거를 얻지 못한 것이다.
   전체 경로의 동시 접근에서 달라질 구체적 근거도 없다면 해당 prototype을 예산상 종료한다.
   ingress가 빨라도 gather·전체 O에서 이득을 잃으면 이 조합을 닫는다. 기본 정확도 및 전체 O 개선이 유망할 때만 상쇄·작은 값·
   zero 등 경계 입력과 반복 검증으로 진행한다. 기본 5%는 탐색 예산 기준이지 물리적 성능 한계가 아니다.

수동 DM bank offset이나 DMA priority 설정은 설치 API에서 확인되지 않아 이 절차의
전제로 넣지 않는다. 문서에 나온 기능을 쓸 수 있다고 가정한 가짜 실행 계획을 피한다.

## 6. 2순위: C09의 첫 NPU 진단, 그리고 staging 이득의 한계

C09는 첫 stage가 두 번째 weight 적재와 겹치도록 이미 구현됐다. 반면 반복 lookup/table
준비, DMA 명령, 작은 dot의 시작/종료 비용이 늘었다. 정적 +0.71%는 당시 예산상의 종료
근거였으며 NPU 성능 판정은 아니다. 새 O trace는 비교군 weight 구간이 정적 13,383 대
NPU 26,736, dot/reduce가 정적 5,063 대 NPU 12,014임을 보여준다. 서로 다른 비율이므로
정적 작은 차이를 정수 배로 바꿔 C09 NPU 성능을 예측할 수도 없다.

따라서 필요하면 **동결 C09의 최초 NPU correctness·profile 한 번**으로 실제 첫 dot 시각,
다음 weight와의 중첩, 반복 table 준비, 전체 O cycles를 확인할 가치가 있다. source·image를
새 이름으로 포장하거나 곧바로 4/8-stage sweep을 하자는 뜻은 아니다. 기존 logical coverage에
더해 실제 DMA bounds/stride를 먼저 감사해야 한다.

두 동일 stage의 이상적인 산술을 보면, weight 공급시간 W가 dot/reduce D보다 클 때
serial `W+D`가 `W+D/2`가 된다. 추가 비용이 없고 공급 처리량이 변하지 않는 가정에서만
기존 marker 구간이 정확히 절반씩 나뉜다는 이상모형의 **D/2 민감도는 약 6,007 cycles**다.
이는 C09에서 실제 회수할 수 있는 구간이나 후보 우선순위를 입증하지 않는다.
26,736 cycles 전체를 숨기는 계산이 아니다. [계산 스크립트](o-weight-ingress-20260927/analyze_pipeline.py)는
각 run/cluster의 입력·table 준비 시각을 사용하며, 새 table 준비·contention·startup이 없다는
가정을 결과에 명시한다. 측정된 speedup이나 하드웨어 상한으로 사용하지 않는다.

[공식 scheduling 문서](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)도
context를 나누는 것과 실제 중첩을 구분한다. `fetch_table_lookup`에는 공개 prepared-table
handle이 없으므로 ‘table 한 번 준비해서 여러 stage에서 재사용’은 즉시 구현 가능한
해법으로 제시할 수 없다. [설치 API 감사](fp8-lookup-table-sharing-audit.md).

Weight를 BF16 DM에 한 번 변환해 반복 table을 피하는 대안은 **30 MiB write + 30 MiB read**의
중간 weight traffic과 추가 Main 작업을 만든다. 현재 inline lookup에 없는 비용이다.
Weight TRF 전환도 재사용이 거의 없는 batch1 weight를 위해 입력 재사용 경로를 바꾸며,
같은 Main lookup→StoTrf lowering 위험이 있다. 둘 다 현 증거에서는 우선 후보가 아니다.

이 단계에서 권하는 변경은 입력 dtype·합산 정밀도를 다시 낮추는 것이 아니다. E53에서
확립한 BF16 입력, F32 K 합산, channel scale 뒤 첫 BF16 경계를 유지하면서 공급/소비 배치를
판별하는 편이 원인을 분리하기 쉽다. O 입력·weight는 공식 평가가 제공하므로 prepacking을
host나 이전 kernel로 옮겨 비용을 평가 밖에 둘 수도 없다.
[대회 skeleton contract](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B#stage-1-rules-skeleton-contract).

순위권 성능을 보장하는 해법은 아직 아니다. 이번 조사의 실질적인 진전은 현재 배치가
이미 지키는 조건, 재시도할 때 반드시 없애야 하는 과거 추가 비용, 그리고 다음 실행에서
어떤 결과가 설계 선택을 바꾸는지를 구체화한 데 있다.

재현 검증은 아래 세 분석기와 baseline 검사를 통과했다. 기존 graph·log를 읽는 검사이며
새 compiler/NPU 실행을 포함하지 않는다. `kernel-audit`와 `prior-audit`는 이 workspace의
보존 artifact cache가 필요하고, SDK 감사에 사용한 원본 경로·해시는 부록 JSON에 있다.

```bash
python3 docs/research/o-weight-ingress-20260927/kernel-audit.py --check
python3 docs/research/o-weight-ingress-20260927/prior-audit.py --check
python3 docs/research/o-weight-ingress-20260927/analyze_pipeline.py --check
python3 moa-dev/experiment_index.py --check --check-live
```
