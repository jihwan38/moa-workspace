# RNGD / furiosa-opt 조사와 Stage 1 적용 계획

조사일: 2026-09-21 (KST). 기준 소스: 작업 저장소 `90f3c3fe16f4fe445f81e82d30b17b1b951e0082`, `furiosa-opt-std 0.8.1`. 이 문서는 공식 문서와 실제 구현을 연결한 설계 근거다. 아래 실험 후보의 속도 향상은 아직 이 문서에서 검증하지 않았다. 실제 채택 여부는 별도 실험 기록의 RNGD 정확성·사이클 결과로 판단한다.

## 1. 조사 범위와 근거의 강도

- 공식 [Programming Tensor Contraction Processors](https://developer.furiosa.ai/furiosa-opt/book/)의 목차 전체를 확인하고 46개 페이지를 조회했다. 핵심 본문은 Mapping, Moving, Fetch Adapter, Switch, Collect, Register Files, Contraction, Vector, Cast, Scheduling/Tuning, Compiler Tools, Transformer case study를 읽었다. 각 엔진의 모든 예제 실행이나 튜토리얼 영상 시청을 완료했다는 뜻은 아니다.
- 최신 웹 문서의 API 설명을 그대로 믿는 대신 [0.8.1 release source](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/)를 내려받아 `engine/mod.rs`, `engine/fetch_adapter.rs`, `engine/collect.rs`, `engine/contraction/outer/mod.rs`, `cast.rs`, `constraints.rs`에서 stage 연결·정밀도·용량을 확인했다.
- 공식 [RNGD 사양](https://developer.furiosa.ai/latest/en/overview/rngd.html), ISCA 2024 [TCP 논문](https://furiosa.ai/download/FuriosaAI-tensor-contraction-processor-isca24)의 구조·FFN·매핑 교훈을 확인했다. 논문 전체 벤치마크를 재현한 것은 아니다.
- 대회 코드의 `src/ops.rs`에서 대상 진입점을 시작으로 `device/layout.rs`, `device/sliding/projection.rs`, `device/shared/mlp.rs`, `device/shared/rmsnorm.rs`, `src/axes.rs`, `src/bin/test_kernels.rs`를 확인했다. 자세한 커널별 실행 추적과 대회 규정 조사는 별도 문서가 담당한다.

**확인된 사실**은 문서 또는 소스가 직접 보장하는 내용이다. **실험 가설**은 그 사실에서 유도한 후보이며 compiler lowering, 수치 정확성, 하드웨어 측정을 통과해야 한다. 순위는 다른 팀의 제출 및 최종 평가에 달려 있으므로 이 조사만으로 3위를 보장할 수 없다.

## 2. 현재 작업은 모델 재학습보다 커널 매핑 최적화다

세 진입점의 외부 HBM 텐서 계약을 유지하면서 내부 데이터 배치·이동·계산 파이프라인을 줄이는 것이 우선이다. 현재 axes는 `H=3840`, `L=15360`, `Qs=4096`, `Ps=2048`, `Ns=8`, `Gs=2`, `Ds=256`이다. 한 hidden vector를 받는 GEMV 형태이므로 대규모 배치 GEMM을 위한 GPU 기법을 그대로 적용하지 않는다.

| 진입점 | 실제 데이터 흐름 | 현재 정밀도 |
|---|---|---|
| `ops::sliding_project_qkv` | hidden RMSNorm → broadcast → Q/K/V projection → Q/K/V norm → RoPE → Q 반환 및 K/V cache scatter | BF16 activation, FP8 weight, BF16 row scale |
| `ops::sliding_attention_output` | attention 결과 flatten → broadcast → O projection → RMSNorm → residual add | BF16 activation, FP8 weight, BF16 row scale |
| `ops::decoder_feedforward` | RMSNorm → up/gate projection → GeGLU → down projection → RMSNorm → residual add → layer scalar | FP4 packed weight, FP8 scale/16 elements, FP32 global scale, BF16 activation |

`sliding_attention_output`은 softmax attention 자체가 아니다. FlashAttention의 QK/softmax/PV 최적화를 이 진입점에 적용한다고 쓰면 대상 연산을 잘못 이해한 것이다. 실제 softmax attention은 `ops::sliding_attention` / `device/sliding/attention.rs`에 따로 있다.

## 3. RNGD의 실행 단위와 병목을 읽는 방법

공식 사양은 칩당 48 GB HBM3, 1.5 TB/s HBM 대역폭, 256 MB SRAM, BF16 256 TFLOPS / FP8 512 TFLOPS를 제시한다. 이것은 peak spec이며, 현재 커널이 그 처리량을 내거나 서버가 그 비율대로 빨라진다는 뜻은 아니다. Stage 1의 실제 cycle counter가 성능 근거다. [사양](https://developer.furiosa.ai/latest/en/overview/rngd.html)

vISA 관점에서 칩은 2개 cluster, cluster당 256개 slice로 표현된다. 각 slice의 DM은 512 KiB이고, contraction은 최대 8개 lane을 사용한다. Tensor Unit은 HBM을 직접 소비하지 않는다. 일반적인 경로는 다음과 같다.

```text
HBM ─TDMA→ DM ─fetch/adapter/switch/collect→ contraction 또는 vector
                                       └─sub-context store→ TRF / VRF
                  계산 결과 ─cast/commit→ DM ─TDMA→ HBM
```

`device.main`은 주 계산, `device.sub`는 TRF/VRF 적재 등에 쓰고, `device.tdma`는 HBM↔DM 및 DM↔DM 이동을 담당한다. 각 context 안에는 순서가 있으며, 서로 다른 context도 데이터 의존성이나 같은 Vector Engine 자원 때문에 직렬화될 수 있다. 소스 줄을 위아래로 옮겼다고 overlap을 얻었다고 판단하지 않는다. [이동 경로](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/index.html), [스케줄](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)

### 현재 유효 cluster 수

`src/device/layout.rs`는 `Cluster = m![1 # 2]`를 사용한다. 공식 문서는 이 표현을 **active cluster 1개와 padding cluster 1개**로 명시한다. `Slice = m![1 # 256]`도 하나의 유효 slice와 padding을 나타낸다. 이후 `QueryRows=m![Qs/16]` 등으로 256 slices에 작업을 분산하지만, 공통 `Cluster`는 그대로다. [Spatial and Temporal Dimensions](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/spatial-temporal-dimensions.html)

따라서 두 cluster로 유효 output rows를 나누는 것은 중요한 구조적 후보다. 다만 `m![1 # 2]`를 `m![2]`로 교체하면 분할이 아니라 같은 값의 broadcast가 될 수 있다. output-row 축을 Cluster와 Slice에 나누고, 입력 replication과 최종 gathering을 함께 설계해야 한다. 얻을 수 있는 속도 향상은 HBM·환원·gathering 병목에 제한된다.

## 4. 매핑은 shape 설명을 넘어 메모리와 병렬성을 결정한다

`m![A,B]`는 A-major/B-minor 배치를 나타내고 `/`, `%`로 축을 공간과 시간에 분해한다. Cluster/Slice/Lane/Packet은 병렬 작업 위치를, Time은 반복을 정한다. 축을 Slice로 옮겨도 마지막에 비싼 reduction이나 broadcast가 생기면 전체 성능은 나빠질 수 있다. [Mapping Tensors](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/index.html)

다음 두 가지를 반드시 구분한다.

| 표현/연산 | 의미 | 잘못 사용했을 때 |
|---|---|---|
| `unsafe { tensor.reshape() }` | 이미 있는 물리 순서를 다른 축 이름/그룹으로 해석; 복사 없음 | 크기가 같아도 permutation이면 다른 값을 읽는다 |
| `.to_dm(&mut device.tdma)` | 논리값을 보존하며 목적 매핑에 실제 복사 | 새 buffer, DMA 시간, 동시 생존 메모리 증가 |
| `#` / `#{*}` | 접근 가능하지만 값은 임의인 padding | 합에 넣으면 쓰레기 값이나 NaN이 전파될 수 있다 |
| `#{0}` | 0이라는 속성을 표현 | 이 표기 자체가 메모리를 0으로 채우지는 않는다 |
| `#{!}` | 접근해서는 안 되는 padding | tile 밖 write 방지 등에서 필요 |

크기만 맞춘 `reshape`를 성능 최적화로 채택하지 않는다. 모든 유효 물리 위치에서 이전 매핑의 값과 새 매핑의 값이 같아야 한다. 특히 row-major FP4 packed 데이터를 잘못 재해석하면 scale의 16-element group 대응까지 무너진다. [Mapping Expressions](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/mapping-expressions.html), [Tensor Semantics](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/tensor-semantics.html)

## 5. 메모리 이동과 broadcast

### TDMA와 Tensor Unit 중 하나가 항상 우월한 것은 아니다

TDMA는 계산 파이프라인을 차지하지 않고 배치를 바꾼다. 반면 같은 cluster의 DM→DM 복사는 읽기/쓰기가 직렬화될 수 있고, 작은 command마다 startup 비용을 낸다. 공식 DMA 설명은 약 500-cycle startup, 연속 접근과 DMN 균등 분배의 중요성을 설명한다. 합칠 수 있는 작은 DMA를 묶고, 성긴 strided 접근을 줄이는 것이 후보이며 실제 `util`로 확인한다. [DMA Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html)

현재 `broadcast_hidden`과 `broadcast_sliding_heads`는 MainContext의 `CustomBroadcast { ring_size: 256 }`를 사용한다. 공식 Switch 비용 모델은 대략 `ring_size × input Time size × packet flit 수`다. custom bitmap 적재는 DMA와 sub context까지 점유하므로 main 실행 길이만 보면 비용을 놓칠 수 있다. TDMA의 목적 `DmTensor<..., Replicated, ...>`로 바로 복제하거나 regular broadcast로 재구성하는 것을 실험한다. 패킷을 작게 쪼개는 것만으로 위 곱이 자동 감소하지는 않는다. [Switch Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/switch-engine.html)

### 정렬·bank·동시 작업

HBM은 256-byte 정렬 접근을 선호하며, row conflict와 일부 channel에 편중된 접근은 대역폭을 크게 떨어뜨릴 수 있다. DM은 slice당 16개 bank를 사용하고 DMA보다 Main/Sub fetch/commit 우선순위가 높다. 같은 bank의 64회 이상 연속 접근은 DMA starvation 위험과 스케줄 직렬화를 유발할 수 있다. 타일 크기를 바꿀 때 bytes가 아니라 형식만 예쁘게 보이는 매핑을 선택하지 않는다. [Memory Performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)

예컨대 `H=3840` FP8 한 행은 3840 B로 256의 배수이지만, down FP4 행은 `L/2=7680 B`, per-block FP8 scale 행은 `L/16=960 B`다. 모두 같은 접근 특성을 가지지 않는다. 실제 compiler가 어느 축을 DMA packet으로 선택했는지와 padding 비용을 별도로 검사해야 한다. 이 계산은 소스 shape에서 직접 유도한 것이며 측정 결과는 아니다.

## 6. 레지스터와 정밀도: 0.8.1에서 가능한 것

### 용량과 overlap

TRF는 contraction의 고정 operand, VRF는 vector의 별도 operand를 담는다. 문서의 TRF 물리 SRAM 설명은 80 KB이지만, **0.8.1 frontend의 논리 payload 제한은 `TRF_CAPACITY_BYTES=65_536`**이다. 320-bit 물리 row와 유효 payload를 혼동해서 타일을 잡지 않는다. VRF는 slice당 8192 B로 검증된다. 실제 주소 배치와 half 선택은 scheduler가 한다. [Register Files](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html), [release collect.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/collect.rs), [release constraints.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/constraints.rs)

TRF 두 half는 같은 bank를 공유하므로 double buffering이 완전한 compute/load overlap을 보장하지 않는다. narrow/wide read, cache reuse, 동시 live TRF 텐서의 용량을 함께 고려한다. book의 단순 per-lane 표보다 실제 release validation 및 compiler 할당 결과를 우선한다.

### FP8 lookup과 contraction의 연결

확인된 0.8.1 API 특성은 다음과 같다.

- table lookup은 main context에만 있다. `f4e2m1 → f8e4m3/f8e5m2`, `f8 → bf16` decode가 지원된다.
- `fetch_cast`의 `f8 → bf16` 직접 변환은 없다. 이 경우 lookup을 쓴다.
- contraction의 floating operand pair는 동일 dtype만 허용한다. `f8×bf16` 조합은 지원하지 않는다.
- `fetch_table_lookup → collect → contract_outer`는 stage trait 연결상 허용된다. 따라서 이미 TRF에 BF16 activation이 있다면 **streaming FP8 weight를 lookup 직후 BF16 contraction에 넣는 후보**가 있다.
- vector 이후에는 contraction으로 되돌아갈 수 없다. FP4 block scale을 Vector Engine에서 적용한 뒤 같은 파이프라인에서 바로 GEMV를 이어붙이는 설계는 지원 stage 순서를 위반한다.

[Fetch Adapter](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/fetch-adapter.html), [Outer operand types](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html), [release engine/mod.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/mod.rs), [release cast.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/cast.rs)

구체적으로 `sliding/projection.rs`의 `project_query`, `project_one_kv_matrix`, `output_partial`은 FP8 lookup 결과를 BF16 DM에 commit하고 다시 fetch한다. 직접 연결은 외부 dtype·수학식을 바꾸지 않고 큰 중간 buffer와 메모리 왕복을 제거할 가능성이 있다. **API상 가능**과 **compiler가 해당 조합을 성공적으로 낮추며 RNGD에서 빨라짐**은 별개의 검증 단계다.

### FP4를 native INT4처럼 취급하지 않기

FFN의 `f4e2m1`은 정수 INT4가 아니다. 현재는 FP4 decode → FP8 → FP32 local scale 곱 → BF16 weight materialization → BF16 contraction이다. 하드웨어 사양의 INT4 TOPS를 이 경로의 처리량으로 인용하면 안 된다.

`scale/16`을 dot-product 뒤로 옮기는 blockwise accumulation은 별도 설계 후보지만 reduction order 및 BF16 rounding 위치가 달라진다. FP8 activation으로 다시 양자화해 FP8×FP8을 쓰는 것도 정확성 위험이 있는 새로운 알고리즘이다. 둘 다 메모리 이동 제거 실험보다 뒤에 둔다.

### 구현되지 않은 API와 padding

0.8.1 `engine/fetch_adapter.rs`의 `fetch_mask`는 존재하지만 `verify_fetch_mask`가 `todo!()`다. API 이름만 보고 사용 가능한 masking 경로라고 계획하지 않는다. 데이터가 가변이거나 padding이 계산에 들어갈 때는 구현된 경로와 정확성 검증으로 0/무시 의미를 보장한다. [release fetch_adapter.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/fetch_adapter.rs)

## 7. 계산 파이프라인에서 줄일 수 있는 작업

Contraction은 `outer → packet reduction → time reduction → lane fold` 순서다. Collect는 32-byte flit을 만들고 Outer는 32 또는 64-byte operand packet을 사용한다. 현재 BF16 `H%32` contraction은 64-byte packet을 사용하므로 이미 해당 폭을 채운다. 무조건 packet을 키우라는 처방은 이 코드에는 맞지 않는다. [Collect](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/collect-engine.html), [Contraction](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/index.html)

반면 현재 많은 projection은 activation을 `Lane=m![1]` TRF에 두고 output row를 Time으로 순회한다. weight를 여러 lane에 두고 activation을 stream으로 보내는 GEMV 재매핑은 lane 병렬성을 높일 수 있다. 다만 weight staging과 FP4/FP8 decode, TRF capacity, 남은 reduction slot까지 비용이 바뀐다. 먼저 schedule에서 contraction이 실제 병목인 구간을 찾는다. [Outer](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html), [Time Reducer](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/time-reducer.html)

Vector Engine은 FP32/INT32를 처리하며 contraction 결과 뒤에 직접 이어질 수 있다. RMSNorm의 epsilon·sqrt나 residual의 일부 단계를 합치는 것은 작은 값의 DM commit/refetch를 줄이는 후보다. 현재 `shared/rmsnorm.rs`는 mean-square → inter-slice sum+epsilon → sqrt → VRF preload를 여러 pass로 수행한다. `vector_final().to_vrf(...)`는 0.8.1에서 허용되므로 중간값을 직접 VRF에 보내는 경로를 검토할 수 있다. 이 store는 sub context도 점유하며 cast/transpose/commit과 같은 command에서 병행할 수 없다는 제약이 있다. [Vector Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/vector-engine/index.html), [Register Files](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html)

Cast Engine 자체는 매 flit을 처리하므로 단순히 cast 호출 수만 세어 병목으로 단정하지 않는다. 다만 BF16 cast를 없애면 수치의 반올림 위치가 달라지며, cast를 사이에 둔 두 pass를 합칠 때 이 변화는 실제 정확성 검사 대상이다. [Cast Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/cast-engine.html)

## 8. 루프 전개는 명확한 근거가 있는 실험이다

공식 Tuning 문서에 따르면 rolled loop는 반복별 schedule을 반복하므로 다음 iteration의 register staging과 현재 contraction 사이의 overlap을 노출하지 못한다. `#[unroll]`은 compile-time count가 필요하고 중첩 loop에 자동 전파되지 않는다. **커널에 남은 loop가 하나도 없어야 register-file reuse 최적화가 활성화**된다. 문서의 예제 수치는 우리 커널의 예상 speedup으로 사용하지 않는다. [Tuning](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html)

현재 `project_up_and_gate`는 up와 gate 각각 15회 loop, `project_down`은 30회 외부 loop와 2회 내부 loop를 가진다. 후보는 다음 순서로 나눈다.

1. 작은 inner 2회 loop 전개: 낮은 코드 증가로 scheduler가 볼 수 있는 범위가 넓어진다. 전체 loop-free 최적화는 아직 안 켜진다.
2. 2개 pass를 한 scheduling unit에 노출하는 manual pipeline: 메모리 사용량을 제한하면서 overlap을 확인한다.
3. 모든 15/30/2회 loop 전개: compile 시간·IR 크기·live DM/VRF/TRF 증가까지 기록한다.

전개 자체가 spill, 메모리 할당 실패, 긴 compile, 오히려 긴 schedule을 만들면 복구한다. 기존 pass 크기를 바꾸는 실험과 동시에 수행하지 않는다.

## 9. 실행할 실험의 우선순위

번호는 예상 효과를 보장하는 순위가 아니라, 근거·구현량·정확성 위험을 합친 실행 순서다. parent 실험 담당자는 현재 schedule을 보고 조정한다.

| 순서 | 변경 범위 | 검증할 가설 | 유지 조건 / 주요 위험 |
|---|---|---|---|
| E1 | `layout.rs` 두 broadcast helper | TDMA replication이 ring256 및 custom bitmap 비용을 줄임 | 3 kernel 정확성, 동일 입력 사이클 개선; DMA contention이 커지면 폐기 |
| E2 | `sliding/projection.rs` FP8 decode→contraction | BF16 weight 중간 DM commit/refetch 제거로 MainContext·DM 비용 감소 | 먼저 한 projection에 적용; LUT/contract lowering와 수치 확인 |
| E3 | `shared/mlp.rs` loop | unroll/manual 2-pass가 preload와 계산 overlap을 노출 | compile 시간/메모리와 schedule 함께 기록; outer loop 존재 여부 명시 |
| E4 | FFN pass/row layout | 4 rows/pass가 최적인지, down의 2-row decode→재배치→복사 비용 제거 가능성 | 4→8처럼 단순 숫자만 바꾸지 말고 60/120 divisibility·VRF scale 용량 검증 |
| E5 | QKV/O 및 FFN output row의 Cluster 매핑 | padding cluster를 유효 row 계산에 활용 | 2-cluster redistribution 비용까지 포함; 단순 replication으로 착각 금지 |
| E6 | `shared/rmsnorm.rs`, GeGLU, residual epilogue | 같은 dtype인 중간값의 DM 왕복을 제거하고 직접 VRF 전달 | BF16 rounding 경계 보존부터 시작; Vector resource contention 확인 |
| E7 | O projection의 reduction chunk | 현재 1024×4와 BF16 partial-add tree 비용 감소 | 2048×2 또는 단일 reduction은 합 순서 변경으로 오차 위험; hidden 정확성까지 고려 |
| E8 | projection contraction 재매핑 | output rows를 TRF Lane에 배치해 lane 병렬성 활용 | weight decode/staging 비용, TRF 한계와 packet·time accumulator constraints 검증 |
| E9 | FP4 blockwise scale/reduction, FP8 activation | scale/element 계산과 BF16 materialization 축소 | 가장 높은 수치 위험; 규정 확인과 넓은 oracle 통과 전 채택 금지 |

E1/E2/E3을 독립적으로 측정한 뒤 검증된 변경만 조합한다. E5/E8은 구조를 크게 바꾸므로 별도 실험 branch/patch와 기존 최선의 정확한 구현을 유지한다.

## 10. 튼튼한 baseline과 채택 기준

`OPTIMIZATION.md`의 핵심은 schedule이 개발 도구이고 `test_kernels`의 RNGD 실행이 Stage 1 판단 기준이라는 점이다. `--exact`를 붙여 대상 하나씩 dump한다. `--dump-*`는 single-kernel 옵션이다.

```bash
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule target/schedules/sliding_project_qkv.json
```

Compiler는 MIR→vISA→LIR→EDF 순서로 진행한다. `--dump-visa`는 초기 IR, `--dump-graph`는 LIR graph JSON, `--dump-summary`는 directory, `--dump-schedule`은 최종 resource scheduling을 남긴다. 서로 같은 형식으로 취급하지 않는다. [Kernel Optimizer](https://developer.furiosa.ai/furiosa-opt/book/tools/kernel-optimizer.html)

각 후보에 필요한 기록은 다음과 같다.

- 소스 commit/patch hash, compiler/std/toolchain, fixture hash, binary hash.
- 변경 가설과 대상 함수; 수학식·dtype·shape·reduction order 변경 여부.
- schedule makespan, context별 interval union, longest node, critical predecessor, DM peak/live tensors, DMA utilization.
- 실제 RNGD 3-input correctness 결과와 각 cycle, median; timeout·오류·서버 상태.
- 유지 또는 폐기 결정 및 이전 최선 대비 kernel별 비율. 전체 점수는 공식 산식으로 따로 계산.

정확성은 `abs(actual-expected) <= atol + rtol*abs(expected)`로 검사하며 소스의 `rtol=0.01`, 커널별 `atol=0.04 / 0.05 / 0.01`을 확인했다. 근처까지 오차를 소모한 결과보다 margin이 남는 변경을 선호한다. 공개 fixture만 통과하도록 값이나 seed를 특수 처리하지 않는다. `RUNS=3`의 서로 다른 입력 조건을 모두 유지한다. 정확한 현재 실행 옵션은 `moa-dev/README.md`와 실험 기록을 따른다.

Compilation은 매핑/타입 검증, CPU oracle은 값 검증, schedule은 정적 계획, RNGD는 실제 동작/성능을 검증한다. 어느 한 단계로 나머지를 대체하지 않는다. [Kernel Validation](https://developer.furiosa.ai/furiosa-opt/book/quick-start/kernel-validation.html), [Diagnosis](https://developer.furiosa.ai/furiosa-opt/book/scheduling/diagnosis.html)

## 11. 논문과 다른 모델 예제를 적용할 범위

ISCA TCP 논문의 직접적인 교훈은 연속 operator 사이의 SRAM 배치가 맞지 않으면 로컬 최적화 이득이 재배치 비용으로 사라진다는 점이다. FFN weight를 나누어 다음 부분을 prefetch하고 contraction 뒤 vector activation을 잇는 설계가 제시된다. 이것은 현재 FFN의 pass 단위 스케줄과 epilogue 연결을 검토할 근거다. 논문의 LLaMA-2 SiLU 결과를 Gemma GeGLU/FP4 경로의 측정값으로 옮기지 않는다. [TCP 논문, VI-C 및 VII](https://furiosa.ai/download/FuriosaAI-tensor-contraction-processor-isca24)

book의 Transformer case는 Qwen3-0.6B의 single-token decode이며 hidden=1024, intermediate=3072다. mapping·oracle·schedule 검증 절차는 참고할 수 있지만, Gemma의 `H=3840`, FP4 group scale, GeGLU, RMSNorm 위치를 대체할 구현 정답이 아니다. prefill·sampling 전체 성능을 그 예제로 주장하지 않는다. [Transformer case study](https://developer.furiosa.ai/furiosa-opt/book/kernel-examples/transformer.html)

## 12. 문서에서 확인한 한계와 미확인 항목

- 온라인 book은 버전 고정 문서가 아니다. 제안 API는 0.8.1 source에서 다시 확인했고 실제 compiler lowering 여부는 실험으로 검증해야 한다.
- book의 generic SRAM 용량 표와 TRF architecture 표는 표현 단위가 다르다. 이 문서는 0.8.1 frontend 검증값을 우선 기록했다.
- 최신 SDK 개요의 TDP는 150 W, 제품 사양 페이지는 180 W로 서로 다르다. kernel 최적화에 필요 없는 전력 수치를 목표나 모델에 사용하지 않았다.
- 정적 `util`과 context occupancy로 hardware utilization을 확정하지 않는다. 긴 static span과 실제 cycle 차이는 서버 측 실행을 통해 확인한다.
- 모든 메모리 배치·block scale 변형에 대해 광범위한 입력 검증이 아직 된 것은 아니다. 위 E1–E9은 설계된 가설 목록이며 실험 결과표가 아니다.
- 목표 순위를 판단하려면 최신 리더보드·공식 점수 산식·제출 결과가 필요하다. 이 문서는 hardware/API 적용 근거를 담당한다.
