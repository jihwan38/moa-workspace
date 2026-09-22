# E55 이후 최적화 후보와 실행 순서

2026-09-22. 기준은 [E55 구성표](../../baseline/e55.json)의 `baseline-e55`
(`873f64ed94229c88cd18d2731bdcca7eea0d6365`)다. 최초 작성은 조사·설계 단계였다.
**후속 C01은 구현·Arena 검증을 마쳤고 미채택이며, C02–C07은 미구현 제안이다.**
[C01의 코드·D01 진단·E65–E67 결과](../../moa-dev/experiments/candidates/c01-scale-dpe-e55/README.md)는
별도로 보존했다. E55는 유지하며 공식 제출은 하지 않았다.
과거 시도는 [측정 결과 분류](../experiments/optimization-results.md)에 따로 보존한다.
조사 대상 commit·파일 해시·점수 계산 입력은 [조사 근거 JSON](optimization-candidates-20260922-evidence.json)에 있다.

## “남은 두 개”의 의미와 점수 목표

앞서 나눈 세 방향 중 메모리 정렬·배치는 [E57–E64](e55-memory-layout-audit.md)의
한정된 후보를 검증했다. 그 조사에서 채택할 개선은 없었다. 남은 큰 방향은
**FFN의 block scale·합산을 DPE로 옮기기**, **weight 중간 저장과 전송·계산 대기를 줄이기**다.
두 방향 모두 과거 일부 시도가 있었고, 둘을 끝내면 최적화 전체가 끝나는 것은 아니다.
QKV/O의 데이터 이동과 FFN의 분할·통신 구조도 추가 후보로 남는다.

[공식 리더보드 설명](https://micro2026-moa.github.io/leaderboard.html)은 세 커널의
baseline 대비 speedup을 기하평균한다. 한 커널을 2배 빠르게 하면 전체 점수는
`2^(1/3) = 1.260배`가 된다. 같은 비율의 개선이면 세 커널의 점수 기여는 같다.

아래는 [9월 22일 E55 대조 3회](../experiments/e55-memory-layout-control-repeats.json)와
[9월 21일 리더보드 snapshot](../experiments/leaderboard-20260921-refresh.json)의 비교다.
**서로 다른 측정 조건의 수치를 이용한 개발 목표이며 공식 점수·현재 순위 예측이 아니다.**

| 커널 | E55 Arena 중앙값 | snapshot의 3위 cycles | 그 수준까지 필요한 cycle 감소 |
|---|---:|---:|---:|
| QKV | 103,858 | 80,573 | 22.4% |
| Attention output | 51,445 | 34,010 | 33.9% |
| FFN | 379,073 | 231,258 | 39.0% |

Snapshot의 baseline과 수식에 E55를 대입한 참고 점수는 **5.702×**, 당시 3위는 **8.399×**다.
QKV/O를 그대로 두고 FFN만 2배 빠르게 해도 참고값은 **7.184×**다. FFN 하나만으로
그 격차를 메우려면 약 **3.196배**, 즉 약 **118,607 cycles**가 필요하다.
이는 달성 예측이 아니라 FFN만 개선하는 전략의 부담을 보여 주는 가정 계산이다.
세 커널 각각을 표의 값으로 맞출 필요는 없으며 개선의 곱이 목표를 결정한다.

현재도 [공식 홈페이지](https://micro2026-moa.github.io/index.html)에 제출 서버 점검과
Arena 이용 가능 안내가 있다. 리더보드는 참고용이며 최종 순위는 같은 조건의 재실행으로
결정한다고 명시한다. 이번 leaderboard API 재조회는 연결이 끊겨 위 snapshot을 사용했다.

## 확인한 구현과 병목

실제 진입점은 [ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)의
`sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`다.
채택 구성의 호출·자료형·반올림 경계는 [baseline/methods.md](../../baseline/methods.md)를 따른다.

- **FFN:** `decoder_feedforward → feedforward_sharded → project_up_and_gate / project_down_sharded`
  ([shared/mlp.rs](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)). FP4 weight를
  FP8 DM, BF16 DM으로 차례로 넓힌 뒤 16원소 dot을 한다. Block 결과는 F32로 유지하며
  scale 곱과 block 합산을 VE에서 수행한다. 전체 BF16 weight는 여러 row tile이 재사용한다.
- **QKV:** [projection.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs)의
  `project_query`/`project_one_kv_matrix`는 두 cluster에서 dot·scale·row 복원을 수행한 뒤
  `output.to_dm`으로 `Cluster` 배치에 head를 모은다. 이후
  [rmsnorm.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/rmsnorm.rs)와
  [rope.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/rope.rs)가 정규화·RoPE를 수행하고
  `ops.rs`가 Q 출력과 지정 K/V cache 슬롯을 쓴다. Head별 병렬화 Q6은 이미 적용돼 있다.
- **O:** `project_output`의 두 cluster·16 column partial을 F32로 합산하고 H480×8로
  옮긴다. `apply_output_channel_scale_f32`의 scale 이후 처음 BF16로 반올림하며,
  post-RMS와 `add_output_residual`이 residual HBM을 갱신한다.

[FFN 정적 일정 분석](../../moa-dev/experiments/candidates/ffn-parallel-epilogue/remaining-bottlenecks.md)의
162,209 cycles는 E55와 같은 FFN image의 compiler schedule이다. 실제 Arena cycles와 구분한다.

| 비용 | 정적 cycles / 의미 |
|---|---|
| Main의 두 단계 weight decode | 86,178; 그중 FP8→BF16 lookup 57,189 |
| Main의 up/gate 및 down dot+scale+reduce | 17,624 + 8,925; scale만의 비용은 분리되지 않음 |
| DMA의 packed weight + block scale HBM load | 82,377 + 17,352 = 99,729 |
| 전체 DMA interval union | 135,607; Main과 104,921 cycles 겹침 |

따라서 decode 비용을 전체 지연에서 그대로 뺄 수 없다. **기존 DMA 명령·순서·duration을
고정한 가상 모델**에서는 decode가 공짜여도 속도비 상한이 `162209 / 135607 ≈ 1.196`이다.
이는 RNGD 전체의 하드웨어 성능 한계가 아니다. 더 큰 개선에는 전송 배치·명령 수·activation
복제·실행 의존성도 바꿔야 한다. 세 weight/scale의 논리 HBM payload는 94.92MiB이며,
FP8+BF16 intermediate write는 506.25MiB다. 실제 버스 전송량이나 read까지 포함한 수치가 아니다.

## 우선순위 표

P1은 먼저 가능성을 판별할 후보, P2는 선행 진단이나 비용 분석 후 구현할 후보,
P3는 정확도 위험이 큰 연구 후보다. 구현 부담은 상대적 판단이며 소요 시간 추정이 아니다.
효과 크기는 측정 전 수치로 약속하지 않는다.

| ID | 우선순위 | 후보 / 바꾸는 구성 | 기대 효과와 한계 | 부담 / 주위험 |
|---|---|---|---|---|
| C01 | P1 | FFN block scale의 두 번째 DPE; F3 대체 | VE scale·reduction 축소. Partial 저장·추가 반올림 비용 발생 | 큼 / 수치·block 좌표 |
| C02 | P1 | FFN 두 block을 한 packet에 묶고 F32 scale 처리; F3/F4 변경 | 추가 BF16 partial 없이 packet 활용 개선. 전체 dot+scale 구간만 대상 | 중간 / SDK 배치·VRF |
| C03 | P1 | BF16 QKV head를 원래 cluster에 유지; Q6 입출력 배치 변경 | head gather·후단 전송 감소 가능. Dot 자체 비용은 그대로 | 중간 / head 순서·cache 쓰기 |
| C04 | P2 | FFN down의 column 분할과 activation broadcast 동시 재설계; F1 변경 | weight/scale DMA와 activation 전달을 함께 줄이는 구조 후보 | 큼 / 통신 증가·합산 순서 |
| C05 | P2 | FFN staged fused decode와 실제 겹침; F2/F4 변경 | 전체 중간 weight 저장 감소. LUT setup·DMA 대기까지 줄여야 유효 | 큼 / 기존 E56 오동작 |
| C06 | P2 | O 작은 tensor 전송·후단 의존성 조정; O3/R1 주변 | 작은 DMA·Sub 준비 비용 축소. 큰 개선 근거는 아직 없음 | 중간 / 직렬화·BF16 경계 |
| C07 | P3 | Native FP8 FFN과 scale-DPE의 결합; F2/F3 대체 | BF16 weight 변환을 없애는 큰 구조 후보. 비용·정확도 모두 미확인 | 매우 큼 / 입력 근사·상쇄 |

### C01 — Scale-DPE는 partial 진단부터

**실행 결과:** D01에서 compact120→128 전송 후 zero tail이 보존되지 않는 경로를
재현하고, BF16 tail 8칸을 실제로 덮는 수정으로 전수 검사를 통과했다. 수정 후 down
기본 FFN도 PASS다. Up/gate E65는 3회 중앙값 −2.72%이나 대조 범위와 겹쳐 개선
미확인, down E66은 단회 +9.42%, 결합 E67은 단회 +3.66%다. E65의 추가 6입력도 PASS다.
새 BF16 partial rounding의 상쇄 반례는 남는다. [단독·조합 기록](../experiments/combinations/e67-scale-dpe.md).
이하 내용은 최초 가설이며, hi/lo 보정은 이번에 구현하지 않았다. 다음 우선순위는 C02다.

현재 `p[r,b] = sum16(x*w)`를 F32로 유지한 채 VE에서 `sum_b(p[r,b]*scale[r,b])`를
계산한다. 후보는 block 결과와 scale을 두 번째 DPE의 operand로 보내 합산까지 맡긴다.
우리 SDK의 float DPE는 BF16×BF16 또는 같은 FP8 형식을 요구한다. F32 partial을 그대로
두 번째 DPE에 넣는 경로는 없다. [설치 SDK 감사](mixed-dtype-dpe-audit.md),
[공식 contraction 설명](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html).

과거 [E23 up/gate](../../moa-dev/experiments/candidates/ffn-scale-contraction-streamed/README.md)는
FFN 기본 입력만 통과했고 결합 QKV는 실패했다. 단회 FFN 측정을 채택 근거로 삼지 않는다.
[E33 down](../../moa-dev/experiments/candidates/ffn-down-scale-contraction/README.md)은 FFN 세 입력이
모두 실패했다. Partial의 BF16 반올림, 120→128 padding, scale 좌표, DPE를 함께 바꿨으므로
원인을 반올림 하나로 단정할 수 없다.

**첫 판별 D01:** 작은 독립 입력에서 F32 block partial → scale → 합산을 단계별로 출력한다.
원소마다 다른 scale, 한 block만 nonzero, 인접 block 상쇄, 마지막 block 및 실제 zero tail로
16원소/120·128 경계를 확인한다. 좌표와 초기화가 맞은 뒤 BF16 partial 오차를 따로 측정한다.
Up/gate와 down을 별개 후보로 검증하며, up/gate 통과를 down에 승계하지 않는다.

새 설계의 대안은 `hi=bf16(p)`, `lo=bf16(p-f32(hi))`를 따로 scale-DPE에 넣고 F32로 합치는
보정 방식이다. 이는 제안이며 일반 F32 동치가 아니다. 두 번의 dot·저장·변환 비용을 추가한다.
오차가 tolerance를 넘거나 보정 비용이 VE 제거 이득을 지우면 이 경로를 중단하고 C02로 간다.

### C02 — F32 packet scale의 미시도 경로

[기존 packed-lanes 실험](../../moa-dev/experiments/candidates/ffn-blockscale-packed-lanes/README.md)에서
BF16 64B packet에 16원소 block 두 개를 넣는 contraction은 검증 단계를 통과했다.
실패 지점은 그 두 F32 결과에 대응하는 scale VRF의 4연속 operand 배치와
FP8→F32 변환을 padding과 동시에 수행하는 fetch였다. 다섯 변형 모두 compile 실패이며 NPU 결과는 없다.

이번 제안은 **FP8 scale을 연속 F32 DM으로 먼저 변환 → 별도 전송으로 실제 `2#4` 슬롯 배치
→ 정렬된 Sub fetch → VE scale**로 변환과 배치를 분리한다. 기존 기록에 미구현으로 남은 경로다.
Up/gate 4행이면 scale VRF 산정은 `4×120×4×4=7,680B`다. 8KiB 안에 들어가도
추가 VRF operand·stash·compiler 할당까지 확인해야 한다. Down 4행은 3,840B다.

먼저 두 block·서로 다른 scale·4행의 작은 compile과 값 검사를 한다. 통과하면 E55의
`project_eight_rows_scaled_after_dot`를 독립 변형으로 대체한다. Block 결과의 추가 BF16 cast는
없지만 F32 합산 순서가 달라질 수 있다. Scale 재배치 비용과 8→4행으로 늘어난 pass 수를 포함해
전체 schedule을 비교한다. 추가 전송이 이득을 소모하면 tile 숫자 탐색을 확대하지 않는다.

### C03 — QKV의 BF16 head를 두 cluster에 유지

E55의 Q4/Q5, 즉 BF16 activation·완전한 F32 dot·channel scale 뒤 첫 BF16 cast는 유지한다.
Row Switch 이후 이미 `Ns/4`로 나뉜 head를 각 cluster 안에서 gather하고,
그 배치에서 RMS·RoPE·Q 저장·K/V scatter까지 끝내는 후보다.
기존 Q6의 head 병렬화에 **cluster 이동 제거**를 더하는 것이며 단순 row tile 확대가 아니다.

실제 변경 지점은 `project_query`, `project_one_kv_matrix`의 반환 배치와 마지막 `to_dm`,
sliding RMS/RoPE의 tensor 타입, `ops.rs`의 출력·cache 쓰기다. Cos/sin·RMS weight의
복제 비용까지 포함한다. Gather를 Switch로 옮기는 것만으로 자동 개선되는 것은 아니다.
먼저 기존 schedule에서 해당 gather와 후단이 완료 경로에 놓이는지 확인하고 비용 상한을 잡는다.

Q/K/V를 각각 변경해 head·group·dimension 식을 추적한다. 정확도 검사에 모든 head와
cache 처음/마지막 유효 슬롯, RoPE offset, 비대상 cache bit 보존, 기존 상쇄 입력을 포함한다.
RMS/RoPE의 산술 순서와 BF16 경계를 유지하고도 dot/weight DMA가 늦어지면 해당 배치를 제외한다.

### C04 — Down의 weight 전송과 activation 복제를 함께 변경

현재 `project_down_sharded`는 column 8분할, shard당 activation 1,920개를 사용한다.
[기존 2-column 시도](../../moa-dev/experiments/candidates/ffn-down-two-columns/README.md)는
weight/scale DMA를 줄였지만 activation 전달이 9,392→39,136 정적 cycles로 커졌다.
따라서 같은 `to_dm`을 두고 column 수만 줄이는 재시도는 하지 않는다.

제안은 4-column부터 **GeGLU 출력의 필요한 조각을 leader에 전달 → cluster 내부 Switch로
사용 slice에 복제 → 더 연속적인 weight/scale load → column partial 합산**을 함께 설계하는 것이다.
Cluster를 가로지르는 값은 실제 전달 경로를 포함하며 reshape/padding을 데이터 복사로 간주하지 않는다.
먼저 weight/scale/activation의 byte·명령 수와 gather·복제·restore 비용표를 만든다.
그 비용표가 기존 8-column보다 유리할 때만 compile한다. 합산 순서 변경에 따른 정확도 검사도 필요하다.

### C05 — Decode staging은 E56의 실패 경계를 먼저 분리

[E50 down-only fused decode](../../moa-dev/experiments/candidates/ffn-down-fused-decode-e37/README.md)의
첫 작은 이득은 E51 반복에서 재현되지 않았다. [8행 staging](../../moa-dev/experiments/candidates/ffn-up-tiled-fused-decode-e47/README.md)은
DMA/LUT 준비와 겹침 악화로 정적 회귀했다. [E56의 16행 staging](../../moa-dev/experiments/candidates/ffn-up-sixteen-row-e55/README.md)은
정적 −1.83%였지만 정확도 실패 뒤 timeout이 발생했다. 코드상 tensor 두 개는 물리 double buffer의 증거가 아니다.

**선행 D02:** 전체 실제 shape에서 16개 FP4 code, 인접 nibble, row/cluster/chunk마다 구별되는
입력을 사용한다. 8/16행 decode의 단독 값뿐 아니라 뒤의 BF16 DPE consumer까지 연결해
검사한다. 별도 소유 buffer와 의도한 재사용 buffer를 비교하고 경계 sentinel을 확인한다.
이전 단독 변환 probe가 통과했다는 사실로 결합 pipeline의 정확성을 대신하지 않는다.

통과 후에만 lookup setup을 늘리지 않는 tile, 한 번에 살아 있는 실제 DM 영역,
scale의 준비 시점, 다음 weight 전송과 현재 dot의 겹침을 schedule에서 확인한다.
기존 E50/E56 코드를 그대로 재실행하는 것은 새 최적화가 아니다.
[공식 tuning 예제](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html)의
TRF double buffering을 HBM DMA 중첩 보장으로 확대하지 않는다. 7,680B scale VRF 두 개를
8KiB VRF에 동시에 두는 식의 계획은 불가능하다.

### C06 — O의 작은 전송과 후단 준비

대상은 channel scale, post-RMS weight, residual의 DM 배치·Sub 준비·완료 경로다.
동일한 backing tensor의 tile에 작은 데이터를 준비하는 pool 또는 기존 tensor 재사용을
검토하되, HBM 원본들이 별도 tensor인 만큼 하나의 pool이 HBM 명령을 자동으로 합치지는 않는다.
Pool 쓰기/읽기의 의존성이 dot을 직렬화할 수도 있다.

기존 [channel scale을 dot에 결합한 후보](../../moa-dev/experiments/candidates/output-bf16-fused-channel-scale-e53/README.md)는
선행 DMA가 dot을 막아 정적 회귀했다. [scale→RMS API 검토](output-scale-rms-fusion-audit.md)에서는
중간 BF16 경계를 같은 Main에서 유지하는 경로가 미해결이었다.
[Div→reciprocal 실험](../../moa-dev/experiments/candidates/output-rms-reciprocal/README.md)도
당시 reciprocal 단독의 정적 이득이 없었다. 이 세 개를 새 유망 후보로 다시 세지 않는다.

E55 schedule에서 실제 임계 DMA/Sub 명령을 먼저 특정한 뒤 그 의존성만 바꾼다.
F32 scale→BF16→RMS, residual의 BF16 경계를 보존한다. 보존하지 못하는 fusion은 별도의
수치 연구로 분리한다. 대상 비용이 작거나 dot 지연이 늘면 우선순위를 내린다.

### C07 — Native FP8 + scale-DPE는 결합 연구

FP4 weight는 FP8로 정확히 표현할 수 있지만 BF16 activation은 별도 FP8 근사가 필요하다.
[E35](../../moa-dev/experiments/candidates/ffn-native-fp8-full256/README.md)와 후속 native FFN은
실행 가능한 경로를 이미 확인했다. 가장 유리했던 E39도 FFN 415,457 cycles로 현재 기준보다 느렸다.
따라서 packet4 실행 가능성을 다시 확인하는 probe나 native 전환만으로 개선을 기대하지 않는다.

새 가설은 FP8 weight를 그대로 dot에 넣고 C01 계열의 scale-DPE까지 연결해
BF16 weight materialization과 VE 병목을 함께 줄이는 것이다. Activation hi/lo 분해,
block partial 반올림, scale 좌표, zero tail, GeGLU/down 상쇄 오차가 겹친다.
QKV/O의 기존 FP8 상쇄 실패와 [잔차 scale 검토](output-residual-scale-design-audit.md)를 읽고
독립 수치 모델부터 확인한다. CPU 오차 검사를 통과해도 NPU의 합산 결과를 보장하지 않는다.
추가 dot·quantizer·buffer 비용을 포함해 E55보다 유리할 근거가 있을 때만 전체 구현으로 진행한다.

## 공개 구현·다른 대회에서 가져온 근거

이번에는 공개 repository 검색 15건의 branch commit을 확인하고, 그중 관련성이 높은
4개 저장소의 코드·문서 18개 파일을 고정 commit으로 확보하고 관련 계산·호출부를 검토했다.
15개 전부를 상세 감사했다는 뜻은 아니다. 공개 작성자와 리더보드 팀의 대응은 확인되지
않았으므로 임의로 연결하지 않는다.

| 자료 | 실제로 확인한 내용 | 이번 후보에 적용하는 부분 |
|---|---|---|
| [hyunjun1234 `ffn7.rs`](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/device/shared/ffn7.rs)와 [호출부](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/ops.rs) | `block_sums_all → scales_bf16 → apply_scales`; FP8 두 항 activation, BF16 block partial, 두 번째 BF16 DPE. Down도 tiled block/scale contraction | C01/C07의 구조 근거. 우리 정확도·SDK에서 성공했다는 근거는 아님 |
| [Kwon-GiHwan `qkv_head_local.rs`](https://github.com/Kwon-GiHwan/furiosa-opt-gemma4-12B/blob/001ba1b33a4b984c0ece81d5c5ad663322848a4f/src/device/sliding/qkv_head_local.rs)와 [호출부](https://github.com/Kwon-GiHwan/furiosa-opt-gemma4-12B/blob/001ba1b33a4b984c0ece81d5c5ad663322848a4f/src/ops.rs) | `HeadCluster=Ns/4`, head 내부 Switch gather, 그 cluster의 RMS/RoPE와 최종 scatter. Projection은 native FP8 경로 | C03에서는 head 배치 아이디어를 검토하고 E55 BF16 dot·scale 경계를 유지 |
| [hyunjun1234 `pool.rs`](https://github.com/hyunjun1234/furiosa-opt-gemma4-12B/blob/bc9a3cc225604fff119bc6858939becb8a9ccfae/src/device/qkv/pool.rs) | 작은 scale/norm 및 K head/RoPE tile이 하나의 tensor를 공유하며 의존성을 형성 | C06의 의존성 조사 근거. Pool 하나가 곧 DMA 합치기라는 주장은 하지 않음 |
| [thangduchd-cmyk projection](https://github.com/thangduchd-cmyk/furiosa-opt-gemma4-12B/blob/7e414f25952f64b0438bbfebe67ee61386b04b8a/src/device/sliding/projection.rs)와 [실제 진입점](https://github.com/thangduchd-cmyk/furiosa-opt-gemma4-12B/blob/7e414f25952f64b0438bbfebe67ee61386b04b8a/src/ops.rs) | 호출되는 `project_query_dist_f8_hbm`은 한 항 FP8 입력, 2-cluster Q와 HBM 경유 gather. K도 정규화 후 HBM 경유 | 우리 E55에 그대로 적용할 후보로 제외. 함수가 존재하는 것과 실제 호출되는 것을 구분 |
| [hunterhhunter 실험 파이프라인 설명](https://github.com/hunterhhunter/furiosa-opt-gemma4-12B/blob/cd96180ba172dcd4fd8ed7027c56023fc8f61753/pipeline/README.md) | 한 가설의 source·patch·schedule·Arena·KEEP/REJECT를 함께 보관하는 운영 방식이 문서화됨 | 현재 baseline/결과/조합 기록 방향과 유사한 공개 사례. 해당 CLI 구현까지 감사한 것은 아님 |
| [RadeonFlow 기술 보고서](https://github.com/RadeonFlow/RadeonFlow_Kernels/blob/main/TechnicalReport.md) | 연속 전송, split-K, buffer를 늘린 load/compute pipeline을 함께 다룸 | C04/C05: weight 접근 개선의 activation·reduction·buffer 비용도 함께 평가 |
| [NVFP4 grouped GEMM 작업일지](https://mufeezamjad.com/blog/nvfp4-group-gemm) | 단계별 baseline에서 producer/consumer pipeline 및 cluster multicast로 확장 | C04/C05: 한 번 전달한 activation의 재사용과 실제 중첩을 확인. CUDA TMA/warp 명령을 RNGD API로 가정하지 않음 |

Public hyunjun 측정 노트의 SDK는 0.6.0이고 우리 설치 SDK는 0.8.1이다.
공개 수치와 Arena의 차이를 코드만의 차이로 해석하지 않는다. 상세 이력은
[이전 공개 구현 조사](public-solutions-20260921.md), 국내외 사례의 적용 범위는
[competition-techniques.md](competition-techniques.md)에 있다.
이번 변경에는 외부 구현 본문이나 실행 코드를 추가하지 않았다. 자체 후보를 구현할 때도
아이디어 출처·달라진 설계·자체 측정 근거를 연구 기록에 유지한다.

## 실제 작업 순서와 채택 조건

1. **D01과 C02의 작은 판별부터:** E33의 block 좌표·padding·반올림을 분리하고,
   C02의 F32 scale 재배치가 SDK에서 표현되는지 확인한다. 실패한 전체 FFN 재실행을 앞세우지 않는다.
2. **C03의 head 배치 검증과 독립 구현:** E55의 산술을 유지하는 QKV 후보로 점수 개선 경로를 넓힌다.
   첫 schedule에서 대상 gather 비용이 충분하지 않으면 범위를 확대하지 않는다.
3. **C01의 up/gate부터 전체 검증:** D01 결과에 따라 단일 BF16 partial/보정 경로를 선택한다.
   C02와 C01은 같은 F3의 대안이므로 처음부터 함께 적용하지 않는다.
4. **C04의 전송 비용표와 D02를 먼저, 이후 C05:** decode 감소만으로 가려진 DMA 병목까지 다룬다.
   C06은 별도 O 후보로 해당 임계 전송을 확인했을 때 진행한다. C07은 마지막 연구 경로다.
5. 정확도 screen을 통과하고 성능 가능성이 있는 후보를 같은 시점의 E55 대조와 동일 package 최소 3회 측정한다.
   첫 screen에서 큰 회귀가 보이면 반복 없이 중단하고 단회 결과임을 표시한다.
   Source·SDK/lock·fixture·image·host·package hash를 고정하고 공식 입력, 추가 입력,
   변경에 맞는 경계·cache 검사를 수행한다. 결과가 겹치면 개선 미확인으로 남긴다.
6. 단독 개선 후 [조합 양식](../experiments/combination-template.md)으로 C / C+A / C+B / C+A+B를
   비교한다. C04+C05는 layout과 buffer를 함께 다시 설계해야 하며 단순 patch 합치기가 아니다.
   공유 `rmsnorm` 변경은 QKV/O/FFN의 affected image를 확인한다. 전체 정확도·반복 측정을 통과한
   조합만 새 baseline으로 채택하고, 그때 실험 ID와 실제 결과를 결과 분류에 등록한다.
