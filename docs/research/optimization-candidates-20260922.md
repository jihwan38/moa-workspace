# E55 이후 최적화 후보와 실행 순서

2026-09-22. 기준은 [E55 구성표](../../baseline/e55.json)의 `baseline-e55`
(`873f64ed94229c88cd18d2731bdcca7eea0d6365`)다. 최초 작성은 조사·설계 단계였다.
**C01–C03은 구현 후 미채택, C05는 E74, 후속 C04는 E79로 채택했다.
C07은 native FP8 결합 구조를 구현·실행해 E104(FFN 반복 −4.93%)를 채택했다.
C06은 아직 아래 제안대로 구현·검증되지 않았다.**
[C07 구조별 측정·실패·소스 동결](../../moa-dev/experiments/candidates/c07-native-scale-dpe/README.md).
[C04의9개 조합·E79 반복과 정확도](../../moa-dev/experiments/candidates/c04-down-transfer/README.md).
E55/E74는 고정 비교군이며 탐색 구조를 제한하지 않는다. Decode·dot·scale·tile·통신의
결합 설계를 직접 비교할 수 있고, 각 중간 단계의 단독 성능 개선을 요구하지 않는다.
[C05 코드·반복·정확도 결과](../../moa-dev/experiments/candidates/c05-streamed-decode/README.md).
[C01의 코드·D01 진단·E65–E67 결과](../../moa-dev/experiments/candidates/c01-scale-dpe-e55/README.md)는
별도로 보존했다. E55를 유지하며 C01 후보의 공식 제출은 하지 않았다.
후속으로 [E55 공식 제출 `3ab72d66`](../experiments/e55-official-submit-retry-20260922.json)이
채점이 완료돼 **5.6665×**를 받았다. 이 조사에 기록한 과거 snapshot 수치와는 구분한다.
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

최초 조사 때는 [공식 홈페이지](https://micro2026-moa.github.io/index.html)에 제출 서버
점검과 Arena 이용 가능 안내가 있었다. 이후 9월 22일 E55의 실제 제출이 접수돼 재개를 확인했다.
리더보드는 참고용이며 최종 순위는 같은 조건의 재실행으로 결정한다고 명시한다.
위 표는 최초 조사 당시 API 연결 실패로 사용했던 과거 snapshot이며 새 채점 결과가 아니다.

## 확인한 구현과 병목

실제 진입점은 [ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)의
`sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`다.
채택 구성의 호출·자료형·반올림 경계는 [baseline/methods.md](../../baseline/methods.md)를 따른다.

- **FFN, 최초 E55 조사:** `decoder_feedforward → feedforward_sharded → project_up_and_gate / project_down_sharded`
  ([shared/mlp.rs](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)). FP4 weight를
  FP8 DM, BF16 DM으로 차례로 넓힌 뒤 16원소 dot을 한다. Block 결과는 F32로 유지하며
  scale 곱과 block 합산을 VE에서 수행한다. E55의 전체 BF16 weight는 여러 row tile이 재사용했다.
  현재 E79는 C05에서 바꾼 BF16 stream/DPE→F32 partial 저장 경로와 C04 down 입력 방송을 쓴다.
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
| C07 | 완료/E104 | Native FP8 FFN·scale-DPE·F32 VE·down 분할 결합; F2/F3 대체 | 기본 단회 최대 −10.13%; 경계 정확도와 실행 실패는 별도 판정 | 매우 큼 / 입력 근사·상쇄·일부 행 오동작 |

### C01 — Scale-DPE는 partial 진단부터

**실행 결과:** D01에서 compact120→128 전송 후 zero tail이 보존되지 않는 경로를
재현하고, BF16 tail 8칸을 실제로 덮는 수정으로 전수 검사를 통과했다. 수정 후 down
기본 FFN도 PASS다. Up/gate E65는 3회 중앙값 −2.72%이나 대조 범위와 겹쳐 개선
미확인, down E66은 단회 +9.42%, 결합 E67은 단회 +3.66%다. E65의 추가 6입력도 PASS다.
새 BF16 partial rounding의 상쇄 반례는 남는다. [단독·조합 기록](../experiments/combinations/e67-scale-dpe.md).
이하 내용은 최초 가설이며, hi/lo 보정은 구현하지 않았다. 후속 C02/C03도 검증 후 미채택이다.

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

**후속 E70–E72:** F32 변환과 padding을 분리한 sequencer 경로가 compile·NPU 검사에
성공했다. 추가 BF16 partial cast 없이 두 block을 묶지만, padded reduce 제약 때문에
F32 partial의 저장·재읽기를 추가했다. 단회 NPU에서 up/gate **+14.10%**, down **+9.52%**,
결합 **+20.28%**(같은 job E55 앞뒤 중앙값 기준)로 회귀했다. 독립 4-case packet 검사와
각 구성의 기본 15개 출력 비교는 통과했다. [구현·실패 경로·결과](../../moa-dev/experiments/candidates/c02-c03-e55/README.md).
아래는 최초 가설이다. 네 번의 새 compiler 제약을 확인한 후 전체 후보를 검증한 결과와 구분한다.

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

**후속 E68:** 해당 배치를 E55의 산술과 BF16 경계를 유지하며 구현했다.
기본 입력 3회 반복에서 QKV **103,157 → 104,762 cycles (+1.56%)**로 회귀했다.
45개 출력 비교는 모두 PASS다. 정적 −1.80%가 NPU 속도 개선으로 이어지지 않았다.
[C02/C03 구현·원인·검증 기록](../../moa-dev/experiments/candidates/c02-c03-e55/README.md).
아래는 최초 가설과 검증 계획이다.

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

**실행 결과:** E74를 출발점으로2/4/8열, Main/Sub 방송,30행 padding 제거,
선행 full-L gather를 결합한9개 후보를 NPU에서 실행했다. **E79:8열+Main ring32 방송**이
동시기 E74 **369997→353136 cycles(−4.56%)**로 독립3회 개선됐다.
기본45개·추가30개 비교, projection 출력138240개를 통과해 채택했다.
[구현·단회와 반복의 구분](../../moa-dev/experiments/candidates/c04-down-transfer/README.md),
[조합 기록](../experiments/combinations/e79-down-broadcast.md).

E79는 GeGLU 조각을32-slice ring의8개 leader에 DMA하고 같은 column의4개 row group에
Main Switch로 복제한다. 계산 배치와 weight/scale 좌표는8열 E74와 같다.2/4열도 실제로
비교한 뒤 고른 결과이며8열 유지가 탐색 전제는 아니었다.4열30행 E81은 단회−1.65%로
후속 후보에 남지만 반복·추가 진단까지 완료한 개선으로 취급하지 않는다.

[과거2-column](../../moa-dev/experiments/candidates/ffn-down-two-columns/README.md)의
activation9392→39136 정적 cycles 회귀를 피하도록 leader 방송을 결합했다.
다만 E79도 전체 정적 makespan은157822→160203으로 늘었다. 실제 NPU에서는 개선됐으므로
**정적 비용표는 설계·원인 추적에 사용하고, 정적 개선을 compile/NPU 실행의 필수조건으로
삼지 않는다.** 새로운 열 분할은 F32 합산 순서와 VRF/TRF, 출력 복원 비용을 함께 검증한다.

### C05 — BF16 weight 미저장 구조 검증, E56 staging은 별도

**실행 결과:** E73은 row tile마다 lookup/dot/scale을 연결해 반복 FFN +10.24%로 회귀했다.
E74는 전체 lookup→DPE→compact F32 block dot 저장→별도 VE scale/reduce로 바꿔
동시기 E55 **377804→369246 cycles, −2.27%**를 독립3회에서 확인했다.
기본45개·추가30개 비교, 실제 projection 크기의 네 case·138240개 출력도 통과했다.
당시 E74를 채택하고 E55를 보존했으며, 후속 C04의 E79가 현재 코드다. Main/Sub로 dot과 합산을 나누는 다른 설계는
SDK 내부 compiler 오류로 미실행이다. E56의8/16행 HBM staging 오류를 해결한 것은 아니다.

후속 E55 스케줄 재확인에서 weight decode 6개는 **86,178 정적 cycles**,
Main 작업 합계의 **68.8%**였다. FP8+BF16 중간 write는 논리 **506.25 MiB**다.
[소스 위치와 비용 근거](../../moa-dev/experiments/candidates/c02-c03-e55/e55-weight-cost.json).
실측 시간의 68.8%라는 의미는 아니다. HBM/DM 전송과의 중첩, lookup 준비 횟수,
consumer 시작 시점까지 확인해야 하며 이 근거만으로 C05의 성능 개선을 보장하지 않는다.

[E50 down-only fused decode](../../moa-dev/experiments/candidates/ffn-down-fused-decode-e37/README.md)의
첫 작은 이득은 E51 반복에서 재현되지 않았다. [8행 staging](../../moa-dev/experiments/candidates/ffn-up-tiled-fused-decode-e47/README.md)은
DMA/LUT 준비와 겹침 악화로 정적 회귀했다. [E56의 16행 staging](../../moa-dev/experiments/candidates/ffn-up-sixteen-row-e55/README.md)은
정적 −1.83%였지만 정확도 실패 뒤 timeout이 발생했다. 코드상 tensor 두 개는 물리 double buffer의 증거가 아니다.

**E56 방식의 staging 재시도 전 D02:** 전체 실제 shape에서 16개 FP4 code, 인접 nibble, row/cluster/chunk마다 구별되는
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

**후속 구현:** E86–E104의 19개 기본 실행 구성에서 FP4→native FP8, term의 lane/time
배치, contiguous row, BF16 scale-DPE와 F32 VE, 보정 partial, 2/4-column down,
입력 reload와 weight materialization을 비교했다. 가장 빠른 기본 단회 관측은
E93의 FFN **352705 → 316977 cycles (−10.13%)**다. 이 수치는 반복 검증된 채택 결과나
공식 점수가 아니다. E4 입력 근사와 BF16 partial의 상쇄 제약, 일부 E5 projection의
행 오류/timeout을 확인했다. [구성별 결과](../../moa-dev/experiments/candidates/c07-native-scale-dpe/measurements.md)와
[후속 진단·채택 판정](../../moa-dev/experiments/candidates/c07-native-scale-dpe/decision.json)을 따른다.
Direct DMA 입력 복제도 전달 경로를 분리하는 추가 진단 구조로 빌드했다.
이하 문단은 실행 전의 가설이다.

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

## 후속 작업과 채택 조건

C01–C05와 C07의 위 결과는 실행 기록이다. C06/O 후단 전달은 미구현이고 C07/native FP8은
새 구조의 정확도 제약을 확인한 단계다. C04의4열30행 E81, QKV의 다른 배치,
C05의 staging도 후속 탐색이 가능하다.
고정 순서나 후보 수 제한 없이 관측된 병목과 검증 가능성에 따라 다룬다.

E55/E74는 비교군이다. 후보 구조와 출발점은 자유롭게 정하고, 결합 설계를 직접 시험할 수 있다.
중간 단계의 단독 개선을 요구하지 않는다. 기여를 분리할 수 있을 때는 단독/조합 또는
최종 설계에서 하나씩 제거하는 비교를 [조합 양식](../experiments/combination-template.md)에 남긴다.

정확도 screen 후 유망 후보는 동시기 고정 대조와 동일 package 최소3개 독립 jobs에서
비교한다. Source·SDK/lock·fixture·image·host·실제 upload hash를 고정하고 추가/경계 검사를 한다.
단회 회귀 중단, compile 실패, 정확도 실패는 각각 표시한다. Shared helper 변경은 affected
image/caller를 확인하고, 전체 조합 검증 후에 새 recipe/tag/current pointer로 채택한다.
