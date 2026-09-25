# Gemma 디코더용 RNGD 데이터플로·메모리 매핑 후보

2026-09-25 조사. 브랜치 `research/rngd-dataflow-20260925`, 조사 시작 commit
`f125ca66901e58492541affda9ba7058c656ff53`, 설치 SDK `furiosa-opt-std 0.8.1`.
**설계·좌표 검사까지 완료했으며 후보 Rust 구현, compile, Arena 측정은 아직 없다.**
위 상태는 조사 commit `8109a27` 시점이다. 이후 사용자 지정으로 M01을 먼저 실행했다.
**후속 M01 검증:** DMA/Switch 모두 compile·기본 NPU PASS, 각각+8.74%/+18.65% 회귀로
종료했다. [구현·실측·원인·판정](../../moa-dev/experiments/candidates/m01-partial-remap/README.md).
**후속 M02 검증:** K-half 소유권의 두 완성 경로도 compile·기본 NPU PASS 후
+55.00%/+75.06% 회귀로 종료했다. [설계 수정·실측·정적/NPU 역전](../../moa-dev/experiments/candidates/m02-cluster-k/README.md).
M03은 아직 미실행이다. 아래의 사전 설계·가설과 후속 실측을 구분한다.
현재 실행본 [E104](../../baseline/e104.json)는 성능 비교군이다.
[후보 JSON](rngd-dataflow-candidates-20260925.json),
[출처·해시·열람 범위](rngd-dataflow-evidence-20260925.json),
[좌표·바이트 계산](rngd-dataflow-accounting-20260925.json)을 함께 보관한다.

이번 조사에서 바꿀 설계 단위는 **weight를 가져오는 배치부터 마지막 합산까지의 경로**다.
조사 당시 우선순위는 **M02: O의 cluster 소유권 변경 → M03: FFN 생산·소비 배치 결합**이다.
M01/M02 실측 종료 이후 남은 사전 후보는 M03이다.
M01은 E64의 재배치 실패를 다른 경로로 푸는 조건부 후보다. 과거에 weight 적재 자체가
빨라졌다는 증거가 없어, 먼저 작은 판별 실험에서 이득을 보여야 한다.
세 항목을 모두 새로운 독립 아키텍처로 세지 않는다. 특히 M03은 C14 계열의 조합 재설계다.
상위권 수준의 개선이나 예상 speedup을 입증한 후보는 아직 없다.

## 1. 실제 실행 경로와 입력 계약

[host/runtime.rs](../../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 `Decode::sliding`은
`sliding_project_qkv → sliding_attention → sliding_attention_output → mlp`를 실행한다.
`mlp`은 `decoder_feedforward`를 호출한다. 현재 Stage 1 대상은 그중 QKV/O/FFN 세 개다.
QK/softmax attention 자체는 이 세 점수에 직접 포함되지 않는다.
[공식 README](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/blob/main/README.md),
[대회 안내](https://micro2026-moa.github.io/index.html).

설계의 shape 근거는 모델 소개가 아니라 실제
[axes.rs](../../furiosa-opt-gemma4-12B/src/axes.rs)와
[ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)의 공개 인자다.
단일 토큰 GEMV이며 `H=3840`, `L=15360`, `Qs=4096`, `Ps=2048`이다.
다른 Gemma 변형이나 prefill GEMM의 배치를 그대로 적용하지 않는다.
[Google의 Gemma 4 12B 소개](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/)는 모델 맥락 확인에 사용했다.

| 대상 | 행렬·수치 형식 | 호출 1회 고유 weight/scale payload | 현재 중요한 배치 |
|---|---|---:|---|
| QKV | Q:4096×3840, K/V:각2048×3840, E4M3 + BF16 channel scale | 31,457,280 + 16,384 B | cluster가 출력 행 분할, full-H BF16 입력 복제, 8-way row interleave |
| O | 3840×4096, E4M3 + BF16 channel scale | 15,728,640 + 7,680 B | cluster가 H 반씩 소유, cluster당 16행그룹×16 K열그룹, slice당120×256 |
| FFN | up/gate:각15360×3840, down:3840×15360; E2M1 + 16개당 E4M3 scale | 88,473,600 + 11,059,200 B | up/gate는 L 소유, down은 H 소유; 사이에 전체 L gather/재분배 |

payload에는 작은 global scalar·RMS 파라미터·입출력은 제외했다. FFN 합계는
99,532,800 B(94.921875 MiB)다. 이 값은 논리량이며 실제 버스 전송량·DMA transaction
수·메모리 stall 시간은 아니다. 원래 dense 연산량/weight byte만으로 병목을 판정하지 않는다.

현 구현의 변환·연산도 구분해야 한다.

- QKV/O: `fetch_table_lookup::<bf16>() → BF16 DPE → F32 합/scale → BF16`.
  O는 현재 16열 합산을 dot stream 안에서 수행한다. 합산 전 DM partial tensor는 없다.
- FFN up/gate: 입력의 runtime scale과 E5M2 세 항을 만들고 복제한다. FP4 weight를
  E5M2 DM에 저장한 뒤 native FP8 DPE, F32 block partial, block scale/reduce를 수행한다.
- FFN down: FP4→E4M3 DM 저장, BF16 lookup을 DPE stream에 결합, F32 block partial,
  block scale/reduce 및 8열 합산을 거친다. BF16 weight 전체 저장은 현재 경로에 없다.
- 따라서 초기 baseline의 weight 변환 비용 분석을 현재 E104에 그대로 가져오면 틀린다.
  근거는 `project_output`, `project_up_and_gate`, `project_rows_with_scale_contraction`,
  `project_down_sharded`다.
  [projection.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs),
  [mlp.rs](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs).

Stage 1은 고정 입력 형식과 공식 harness를 사용한다. host에서 weight를 미리 재배치하고
그 비용을 평가 밖으로 빼는 제안은 이 후보군에 넣지 않는다. runtime 재배치는 전부
해당 kernel의 비용에 포함한다. 정확도 허용오차·입력 생성·공개 signature는 유지한다.

## 2. 논문에서 가져올 원리와 가져올 수 없는 전제

전체 문헌을 망라했다는 주장은 아니다. 아래는 원문 중 관련 설계 절, 공식 문서, 실제
SDK를 대조한 결과다. ZigZag/Timeloop은 이번에 초록·공식 소개 수준으로 사용했다.

| 1차 자료 | 이번 설계에 사용할 원리 | RNGD/이번 평가에 바로 옮길 수 없는 전제 |
|---|---|---|
| [TCP, ISCA 2024](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-isca24.pdf) | spatial/temporal partition, register 재사용, 통신을 포함한 contraction 매핑 | 논문의 INT8 Llama 예제가 우리 FP4+block-scale FFN의 최적 배치라는 뜻은 아님 |
| [RNGD, IEEE Micro 2025](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-micro25.pdf) | operand 배치·순서가 재사용과 자원 활용을 함께 결정 | 칩의 peak 수치를 현재 kernel의 실효 bandwidth로 사용하지 않음 |
| [MARLIN](https://arxiv.org/html/2408.11743v1) | 낮은 bit weight의 공급·변환·계산을 함께 설계; scale 배치 포함 | GPU warp/Tensor Core 경로와 offline weight reshuffle, 논문의 배속을 이식할 수 없음 |
| [FLUTE](https://arxiv.org/html/2407.10960v1) | 연산기의 소비 순서에 맞춘 weight/metadata layout; lookup 자원 재사용 | 사전 packing과 GPU의 사용자 LUT를 RNGD typed lookup과 동일시하지 않음 |
| [TileFuse, 2026 preprint](https://arxiv.org/html/2606.11357v1) | 생산자가 전달하는 tile과 소비 tile을 함께 설계; 작은 GEMV의 공간 분할 | AMD XDNA2 memory/compute tile, offline pretiling 전제. 이미 512 slices를 쓰는 우리 구현에 논문의 활용률 개선 배율을 적용하지 않음 |
| [T-MAC](https://arxiv.org/html/2407.00088v1) | activation 관련 LUT/bitplane으로 곱셈 경로 자체를 교체하는 대안 | CPU shuffle과 runtime LUT 접근이 필요. FP4는 uniform INT4가 아니며 현재 SDK 지원도 미확인 |
| [ZigZag](https://arxiv.org/abs/2007.11360), [Timeloop](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation) | operand별로 다른 tiling·재사용·메모리 용량을 비교하는 후보 표현 | RNGD로 보정하지 않은 분석기의 cycles는 예측 근거가 아님 |

이 자료들로부터 도출한 **우리 설계 가설**은 다음과 같다. GEMV의 weight는 각 dot에서
한 번 쓰이는 반면 activation은 출력 행마다 재사용된다. 따라서 weight를 TRF에 오래
유지하는 기법 자체보다, 고정 HBM 배열을 효율적으로 읽고 필요한 변환을 거쳐 DPE에
공급하는 경로와 activation 재사용·합산 비용을 함께 비교해야 한다. 어느 하나가
현재 NPU의 지배 병목인지는 [정적 분석](../../moa-dev/experiments/diagnostics/e104-cost-attribution/README.md)만으로 아직 확정할 수 없다.

## 3. 공식 지침을 SDK 0.8.1에 대조한 제한

| 확인 사항 | 설계에 미치는 영향 |
|---|---|
| 2 clusters×256 slices; slice당 DM512KiB, TRF64KiB, VRF8KiB | 논리 크기가 들어가도 allocator 예약·lifetime·padding 검사가 별도로 필요 |
| HBM↔DM의 8B 주소/전송 정렬과 256B 권장 구간은 다른 조건 | 목적지 padding만으로 원래 HBM의 960B weight/120B scale 조각을 고칠 수 없음 |
| DM bank 선택에 주소 bit3…6이 관여 | 상대 주소·동시 엔진 접근을 보아야 함. bank conflict 감소는 실측 전 가설 |
| float DPE는 BF16/BF16, E4M3/E4M3, E5M2/E5M2 조합 | FP8 weight×BF16 input을 변환 없이 바로 넣는 설계는 현재 API에 없음 |
| `fetch_*_lift`는 기존 복제 위치를 주소 계산으로 선택 | private DM 간 복사를 대신하지 않음; 복제되지 않은 tensor를 `reshape`해서 방송으로 만들 수 없음 |
| Switch `Transpose`는 slice 축을 교환; fetch 쪽에서 적용 | dot 뒤 공짜 축 교환은 아님. 먼저 partial 저장/재읽기, ring 비용을 포함해야 함 |
| Main/Sub가 자원을 공유하며 Sub에는 DPE가 없음 | context 분리만으로 겹침이나 처리량 두 배를 주장할 수 없음 |
| `fetch_table_lookup<OutD>()`는 typed 변환; 사용자 runtime table 인자가 없음 | T-MAC/FLUTE식 동적 LUT를 이 함수로 표현할 수 있다고 가정하지 않음 |

근거: [memory performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html),
[DMA](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html),
[fetch/lifting](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/fetch-engine.html),
[DPE](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html),
[register](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html),
[switch](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/switch-engine.html),
[fetch adapter](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/fetch-adapter.html),
[tuning](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html).

[kernel design 문서](https://developer.furiosa.ai/furiosa-opt/book/quick-start/kernel-design.html)의
명시적 배치 `_at` 설명과 설치 std 사이에는 확인이 필요한 차이가 있다. 조사한 0.8.1
`src/`에서 `new_at/to_dm_at/commit_at` 공개 메서드를 찾지 못했고 `DmTensor` 주소 필드도
비공개다. 수동 bank 주소 제어는 **설치 API 확인 전 보류**한다. 하드웨어에서 불가능하다는 뜻은 아니다.
SDK source 8개를 실행 container와 대조한 해시는 evidence JSON에 있다.

## 4. M02 — O의 두 cluster가 출력 행 대신 K 반쪽을 소유

후속 실행 완료: [M02 기록](../../moa-dev/experiments/candidates/m02-cluster-k/README.md).
원래 H%256→H%480 직접 gather가 factorization에 실패해 full-H 중간 집결(v3)을
구현했고, H480 안에서32-way row interleave하는 v4도 기본 NPU까지 검사했다.
두 결과는+55.00%/+75.06% 회귀, 미채택이다. 아래는 실행 전 가설을 보존한 것이다.

**우선 실행 설계.** 기존 E104/E53 산술과 scale-before-BF16 경계는 참고하되
`project_output`의 입력 분배·weight 매핑·합산 경로를 통째로 교체한다.
옛 `output-two-cluster-column-split`도 H를 cluster에 나눈 구조였다. 이 제안처럼
두 cluster가 모든 H를 만들고 서로 다른 K를 계산한 구조와 구분한다.
다만 [이전 O 두열 source](../../moa-dev/experiments/candidates/output-column-sweep/projection.2.rs)도
slice당15×2048을 사용했다. 새로 검증할 부분은 그 tile 크기가 아니라 cluster 간 K 소유권,
입력의 leader 방송, 행 interleave와 F32 cluster 합산의 조합이다. 이전 두열 sweep은
weight DMA 정적 시간이 줄지 않고 입력/compaction 비용이 늘었으므로, 큰 segment만으로
이번 적재가 빨라진다고 예상하지 않는다.

```text
HBM x[4096], W[3840,4096]
  → cluster0: K[0:2048] / cluster1: K[2048:4096]
  → 각 cluster 내부 x 방송 → BF16 TRF
  → 각 slice의 FP8 W → BF16 lookup → 2048항 dot, F32
  → cluster별 H[3840] partial gather
  → 두 partial의 F32 합 → channel scale → 첫 BF16
  → 기존 post-RMS → residual → HBM
```

`c∈[0,2), s∈[0,256), t∈[0,15), u∈[0,2048)`에서
`h=256*t+s`, `k=2048*c+u`. HBM byte offset은 `4096*h+2048*c+u`다.
각 slice는 15행×2048열을 읽는다. weight 구간의 시작/길이는 모두 256B 정렬이다.

| 중간값 | 소유·메모리·배치 | lifetime / 소비자 |
|---|---|---|
| x의 K 반쪽 | cluster당2048 BF16, ring32 leader DM→cluster마다256 slices 복제; TRF4096B/slice | 15개 dot 동안 유지 / DPE |
| W 조각 | 각 slice FP8 DM30,720B, H행 interleave | 해당 dot 완료까지 / Main lookup+DPE |
| H partial | F32 15개/slice; 우선 각 값에 8B slot을 계획 | cluster gather까지 / DMA; 유효값 외 padding은 읽지 않음 |
| 두 H 벡터 | 총F32 30,720B; H480×8 epilogue에 2항으로 모음 | 합·scale 완료까지 / VE; F32 VRF 두480항은 논리3840B |

줄이는 작업: slice 사이 16열 합산과 그 소유권 제약. 행별 부분합 개수는 16→2다.
바꾸는 공급: 256B×61,440 row segments → 2048B×7,680 segments.
**segment 개수는 compiler DMA descriptor나 실제 transaction 개수가 아니다.**

추가 비용: activation replica payload가 262,144→2,097,152B(8배), ring 방송,
새 partial 저장/읽기 및 cluster 간 합산. 첫 partial의 8B slot 저장 공간은 61,440B다.
HBM에서 leader에만 적재하면 leader payload는 65,536B지만 뒤의 방송을 빼면 안 된다.
현재 16 partial은 stream 안에 있으므로 “partial이 8분의1 → 전송도8분의1”은 틀리다.

**첫 판별:** 실제 shape에서 `HBM 공급 + activation 방송 + dot + 모든 partial 이동/합산`
경로를 완성한다. 초기 mapping 검사 후 O 전체에 붙여 contemporaneous E104와 비교한다.
긴 연속 적재가 늘어난 방송/합산을 상쇄하지 못하면 이 매핑을 닫는다. 작은 column sweep으로
계속 바꾸지 않는다. dtype는 BF16을 유지하지만 합산 순서가 달라지므로 기존 검증을 상속하지 않는다.

## 5. M03 — native up/gate와 같은 cluster에서 GeGLU·down을 연결

**두 번째 실행 설계. C14 가족의 재조합이며 새 가족으로 세지 않는다.**
실제 시작 구조는 [C14-B v4](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/bf16-stage-compact-v4/mlp.rs)다.
여기에 C07의 native up/gate 산술과 E79의 down block 처리법을 조합한다.
C14의 BF16 up/gate 및 두 L stage를 제거하고 **한 L stage**로 만든다.

```text
pre-RMS → runtime 입력 표현·복제
  → cluster0가 L[0:7680], cluster1이 L[7680:15360] up/gate 생산
  → 각 cluster에서 30행 F32 결과를 60행으로 compact → BF16 → GeGLU
  → 자기 L 반쪽만 자기 cluster의 down 입력으로 분배
  → 각 cluster가 전체 H의 F32 partial 생산
  → 두 H partial 합 → BF16 → global scale → post-RMS/residual/layer scale
```

Up/gate producer `l=7680*c+30*s+t`. 두 인접 slice의 30행을 묶으면 cluster 안에서
60행 BF16 packet을 만들 수 있다. Down은 `s=4*r+k`, `h=60*r+t`,
`l=7680*c+1920*k+u` (`r<64,k<4,t<60,u<1920`)로 배치한다.
즉 cluster당 **64행그룹×4 L열그룹**, slice당 down60×1920이다.

| 중간값 | 소유·메모리·배치 | lifetime / 소비자 |
|---|---|---|
| up/gate 입력·weight | native E5M2 입력 세 항 TRF, FP4→E5M2 DM | projection별 재로딩; term 수는 설계 선택이며 이 첫 조합만3항 |
| up/gate block partial | F32 DM30×240/slice, scale VRF6×240×4=5760B | 각 scale/reduce가 소비한 뒤 해제 가능 |
| up/gate 최종값 | 각 cluster에 자기 L 절반만; F32 compact 후 BF16 60행 단위 | GeGLU까지 / VE; global scale와 BF16 경계 보존 |
| GeGLU 출력 | cluster-local BF16 L7680, compact packet; down에 맞는 4열 분배 | down 입력 TRF 구성까지 / DMA+Switch |
| down packed/FP8/partial | 각 slice packed57,600B, FP8 115,200B, F32 partial28,800B | lookup→DPE→scale 순; allocator peak는 compile로 확인 |
| down scale·입력 | scale VRF12×120×4=5760B, BF16 TRF3840B | 60행 block 처리 / VE·DPE |
| H partial | cluster당F32 H3840, 2항 epilogue 배치 | 두 cluster 합산 후 BF16; padding 포함 경로 검증 |

줄이는 작업은 전체 L을 한 cluster로 모았다가 두 cluster에 다시 보내는 경로다.
Down ring32의 첫 분배 대상 수는 128→64, leader payload는
491,520→245,760B다. 그러나 **모든 slice에 복제된 activation 총량은 양쪽 모두
1,966,080B**다. ring payload/지연 감소는 별도 확인해야 한다.

추가 비용은 H partial 총30,720B의 저장·gather·F32 cluster 합산이다. FP4 변환량,
down의 960B packed/120B scale HBM 구간, F32 block partial 저장은 남는다.
한 stage는 C14의 여섯 half-matrix decode stream을 세 full-matrix stream으로 바꾼다.
**weight 총량이 절반이 되는 것은 아니며**, setup·stage별 중간값/의존성이 달라지는 것이다.
up/gate 완료 뒤 down이 시작하는 의존성은 유지하므로 early-down pipeline 이득도 주장하지 않는다.

**첫 판별:** local GeGLU→down 입력의 실제 전송과 H partial 합산을 모두 포함한 FFN
완성 경로를 비교한다. native up/gate만의 효과는 E104에 이미 포함되어 있어 다시 더하지 않는다.
C14-B의 +17.58%는 BF16 up/gate·stage 수 등도 함께 다른 결과였으므로 이 재조합의
성능 예측치로 쓰지 않는다. correctness는 native 표현의 작은 값/상쇄와 새 down 합산을 함께 검사한다.

## 6. M01 — O는 weight 적재 배치에서 계산하고 작은 partial만 재배치

**조건부 후보.** [E64](e55-memory-layout-audit.md)는 column-major slice 적재 뒤
전체 weight를 계산 배치로 다시 옮겨 O가 +149.61% 느려졌다. 당시 weight DMA의
정적 시간 자체도 13,383→13,435로 줄지 않았다. 같은 적재 이름을 새 후보로 재포장하지 않는다.

새 경로는 `W 적재 → 그 배치에서 lookup/dot → F32 partial만 row-major slice로 이동
→ 16열 합 → channel scale → BF16 → epilogue`다.
`s=16*k+r`, `h=1920*c+120*r+t`, `j=256*k+u`로 두며 각 slice는 W120×256을 가진다.
계산 후 `s'=16*r+k`로 옮겨 VRU의 최내부 reduction 축 조건을 만족시킨다.

| 중간값 | 배치·메모리·lifetime | 소비자 |
|---|---|---|
| W | FP8 DM30,720B/slice; `(K열, H행그룹)` 순서, repack 없음 | Main lookup/DPE |
| x | BF16 256개 TRF/slice, W의 K열과 일치 | 120행 dot 동안 유지 |
| partial A | compact F32 120개/slice, 총245,760B; 2값씩8B commit | DMA 또는 fetch→Switch |
| partial B | DMA 경로만 추가, `(H행그룹,K열)`; 합산까지 유지 | fetch/VE16열 합산 |
| 최종 H | E104와 같은 F32 H480×8 및 scale VRF | scale 뒤 첫 BF16, 기존 epilogue |

재배치 tensor는 weight보다 논리64배 작다. 하지만 **현재 E104에는 없던 비용**이므로
64배 빠르다는 뜻이 아니다. 가능한 구현 두 가지의 추가 edge를 구분한다.

- Typed DMA: dot의 partial A write + A read/B write + reduce의 B read
  = **983,040B의 추가 유효 DM edge payload**. 이 중 layout copy만491,520B다.
- Fetch→`Transpose{slice1:16,slice0:16}`→VE reduce를 한 stream으로 표현할 수 있으면
  A write/read491,520B에 ring256 switch 비용이 더해진다. B 저장을 생략할 수 있는지는
  새 mapping으로 compile해야 한다. 공개 primitive의 존재가 전체 합성의 성공을 보장하지 않는다.

**첫 판별:** 고정 shape/source에서 E104 적재·소비와 이 적재·소비를 제한된 diagnostic으로
비교한다. 적재만의 우위가 없고 추가 partial 경로를 상쇄할 구체적 근거도 없으면 종료한다.
작은 정적 차이만으로 NPU 승패를 선언하지 않는다. 이 조건을 통과하기 전 M02/M03보다 우선하지 않는다.

## 7. 중복·보류 목록과 검증 순서

| 과거/다른 제안 | 이번 판단 |
|---|---|
| full-K O row ownership, 2/4/8열 sweep | 이미 시도. M02의 cluster K 소유권과 구분; 단순 재실행 제외 |
| E64 column-major ingest + full W repack | 재실행 제외. M01은 repack 대상을 바꾸고 사전 판별 요구 |
| C15 up/gate split-K | NPU +14.93%; 입력 방송 감소와 weight/scale 분절 증가를 함께 기록. split-K 전체가 느리다는 결론은 아님 |
| C14-B 두-stage BF16 whole FFN | NPU +17.58%. M03은 native 산술+한stage+producer/consumer 소유권 조합으로 재검토 |
| local GeGLU만 분산, H-owned down 유지 | 기존 실패 요인 재현 가능. M03은 down 소유권까지 함께 바꿈 |
| C09 staging / C11 VRF partial / C12 Main/Sub 분리 | 국소 구조이며 전체 데이터플로 가족 수를 늘리지 않음 |
| C10 weight→TRF | target lowering 실패; 속도 탈락이 아님. target/API 근거 없이 우선 재시도하지 않음 |
| 기존 QKV lane 병렬, Q/K/V 통합, KV interleave | 이미 compile/정확도/성능 기록 존재. QKV 신규 후보 숫자를 채우기 위해 재명명하지 않음 |
| full pre-scaled BF16 weight materialization | 초기 FFN과 관련 시도 존재; 없던 기법으로 세지 않음 |
| T-MAC/LUT-GEMM 계열, 수동 bank `_at`, 직접 FP8×BF16 | SDK 표현/변환 비용이 먼저 입증돼야 하는 보류 항목 |
| FlashAttention·speculative decode·batch 변경·모델 pruning | 이번 세 GEMV/FFN 공개 계약과 별개의 평가 범위 |

관련 기록: [기존 구조 후보](structural-candidates-20260923.md),
[C14/C15/C12 실행](../experiments/structural-round-20260923.md),
[C15](../../moa-dev/experiments/candidates/c15-upgate-columns/README.md),
[C13](../../moa-dev/experiments/candidates/c13-int8-ffn/README.md),
[이전 O column split](../../moa-dev/experiments/candidates/output-two-cluster-column-split/README.md),
[down 두열](../../moa-dev/experiments/candidates/ffn-down-two-columns/README.md).

좌표 분석은 M01의61,440개, M02의7,680개, M03 down의30,720개 row segment와 scale
구간을 열거했다. 모든 유효 weight를 한 번씩 포함하고 M03의 producer/consumer L 좌표가
일치한다. HBM source의8B 정렬은 확인했지만 destination lowering·padding·수치 정확도는 미확인이다.

```sh
python3 moa-dev/analysis/dataflow_candidates_20260925.py
```

구현 착수 시 후보별 branch와 실제 source 조합을 고정하고 `structural_build.py --plan`으로
확인한다. 이 문서의 JSON은 설계 목록이며 실행 가능한 build recipe가 아니다.
M02/M03의 첫 완성 경로에 기존 정확도 기준을 적용하고, NPU screen은 E104 전후 control로
측정한다. 회귀 또는 대상 kernel 5% 미만 개선은 원칙적으로 짧게 종료하며 예외 근거를 남긴다.
유망한 경우에만 독립 반복·추가 정확도 검증으로 확대한다. compile 오류는 source 축/정렬 오류와
실증된 SDK 제약을 구분하고, 원인이 명확한 수정을 무조건 횟수 제한으로 포기하지 않는다.

다음 판단에 필요한 것은 논문 배속을 합산한 전망이 아니라, **완성된 새 소유권·배치가
추가 통신을 포함하고도 실제 critical path를 줄이는지**에 대한 첫 측정이다.
