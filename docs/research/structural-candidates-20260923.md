# 큰 구조 변경 후보: C09–C14

2026-09-23. 이번 작업은 **기존 시도 대조·논문 조사·설계 선별**이다. 아래 여섯 후보의
새 구현, 컴파일, Arena 실행은 하지 않았다. 비교군은 현재 채택본
[E104](../../baseline/e104.json)(`baseline-e104`, `7f891f9`)이며 출발 구조를 제한하지 않는다.
소스 조사 기준 HEAD는 `78b2930`이다. [검색 범위·해시·출처](structural-candidates-20260923-evidence.json).

우선순위는 **C09 → C11 → C10**, 이후 **C12 → C13 → C14**다. 이것은 효과가
입증된 순서가 아니라, 큰 비용을 바꿀 가설을 적은 비용으로 판별할 순서다.
과거와 같은 아이디어를 새 이름으로 세지 않고, 미시도 조합과 이전 실패를 함께 적었다.

| 순서 | 후보 | 바뀌는 실행 구조 | 과거 시도와의 관계 | 첫 종료 판단 |
|---|---|---|---|---|
| 1 | C09 · O 행 구간 pipeline | 전체 weight 전송 완료를 기다리는 대신 다음 구간 DMA와 앞 구간 lookup/dot을 중첩 | 이전 문서에 제안만 했던 O 경로 | 실제 중첩 없이 작은 DMA·lookup 준비만 늘면 컴파일 단계 종료 |
| 2 | C11 · FFN 계산 결과를 VRF에 유지 | F32 block partial을 DM에 쓰고 다시 읽는 대신 VRF에 두고 scale을 stream | native FP8 + computed VRF 조합 미시도; E73의 BF16 반복 lookup과 다름 | 8KiB 한도·배치가 안 맞거나 작은 DPE 반복 비용이 더 크면 종료 |
| 3 | C10 · lookup에서 weight TRF로 직접 전달 | BF16 weight의 DM 중간 저장·재읽기 제거, 여러 출력 행을 DPE lane에서 계산 | weight-TRF 자체는 이미 실패; **중간 DM 없는 경로**가 미시도 | lookup→TRF가 compiler lowering에서 막히면 종료 |
| 4 | C12 · Main DPE / Sub F32 후처리 분리 | Main의 다음 dot과 Sub의 이전 partial 합산·scale을 중첩 | FFN의 Sub reduce 실패 이력 있음; O의 자원 분리 조합은 미시도 | Main도 VE를 계속 점유하거나 같은 compiler 오류면 종료 |
| 5 | C13 · FFN INT8 DPE | FP8 세 항 계산을 정수 weight·입력 항과 I32 누산으로 대체 | 기록에서 integer DPE 경로 미발견 | 입력 양자화 오차·변환 비용·항 수가 불리하면 작은 검사에서 종료 |
| 6 | C14 · up/gate→GeGLU→down 구간 pipeline | 전체 중간 activation 완성을 기다리지 않고 L 구간별 down 누산 시작 | GeGLU 위치 이동은 기시도; down의 조기 소비까지 연결한 구조는 미시도 | weight 재전송·부분합 저장이 늘고 stage 중첩이 없으면 종료 |

## 확인 범위와 이미 해본 것

[실험 catalog](../experiments/catalog.json)의 138개 항목, 후보 폴더 136개와 그 README,
`evidence/` 하위 제외 Rust snapshot 656개를 검색했다. 이 숫자는 독립 최적화 방법 수가
아니다. 복제 소스·실패 변형·진단도 포함한다. 전체 파일을 줄마다 읽었다는 뜻도 아니다.
검색 후 관련 구현·호출부·판정 기록을 직접 대조했다. 현재 진입점은
[ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)의 세 competition 함수다.

| 다시 미시도라고 부르면 안 되는 방법 | 확인한 결과 |
|---|---|
| Weight를 TRF에 놓고 여러 row를 lane 병렬 계산 | [qkv-lane-parallel](../../moa-dev/experiments/candidates/qkv-lane-parallel/README.md): compile PASS, 정적 61,994→70,492. BF16 DM 저장과 TRF 재읽기가 dot 단축보다 큼. Arena 미실행 |
| Up/gate의 두 cluster에서 GeGLU까지 수행 | [ffn-geglu-two-cluster](../../moa-dev/experiments/candidates/ffn-geglu-two-cluster/README.md): 두 설계 모두 compile PASS·정적 회귀. Down 입력 재분배 비용 증가 |
| BF16 lookup→dot→scale 완전 연결 | [C05/E73](../../moa-dev/experiments/candidates/c05-streamed-decode/README.md): row tile마다 lookup을 반복해 FFN 반복 +10.24%. 같은 C05의 Sub F32 reduce는 compiler 내부 오류 |
| Native FP8, scale-DPE, packet·column·방송 조합 | [C07](../../moa-dev/experiments/candidates/c07-native-scale-dpe/README.md), [기존 C01–C08](optimization-candidates-20260922.md): 일부 채택, 나머지 실패/회귀. 단순 크기 변경은 새 구조 후보로 세지 않음 |
| QKV 묶음 계산 / QKV 8-column BF16 dot | [C08](../../moa-dev/experiments/candidates/c08-fused-qkv/README.md): 각각 정확도 실패 / 단회 약 30% 회귀로 종료 |
| HBM load를 소스 앞에 모으기, O column-major load 후 DM 재배치 | [qkv-prefetch](../../moa-dev/experiments/candidates/qkv-prefetch/README.md), [E57–E64](e55-memory-layout-audit.md): 앞당긴 소스 순서만으로 중첩이 생기지 않음; O 재배치도 기시도 |

## 우선 판별할 세 가지

**C09 — O의 weight 전송과 lookup/dot을 출력 행 구간으로 나눈다.**

현재 [project_output](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs)은
FP8 weight 전체를 DM으로 옮기고, Main에서 BF16 lookup→dot→16-column F32 합산을
수행한다. 이후 H480×8 배치로 이동해 channel scale을 곱한 뒤 처음 BF16으로 반올림한다.
후보는 처음에 두 출력 행 구간으로만 나누어 `DMA(n+1) ∥ lookup/dot(n)`의 의존성을
만든다. Reduction 축은 나누지 않으며 기존 scale·RMS·residual 순서를 보존한다.

근거는 RNGD의 비동기 TDMA와 compiler scheduling이며, MARLIN의 메모리 단계별
pipeline도 참고했다. GPU의 구체적 stage 수나 속도비를 RNGD에 대입하지 않는다.
[TCP](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-isca24.pdf),
[MARLIN](https://research-explorer.ista.ac.at/download/19877/19883/2025_PPoPP_Frantar.pdf),
[SDK scheduling](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html).

첫 산출물은 exact O compile과 schedule 한 쌍이다. 두 buffer 선언 자체가 성공 조건은
아니다. 실제 TDMA/Main 중첩, 추가 DMA 수, lookup 준비 반복, 동시에 살아 있는 DM을
확인한다. 전체 weight 대기가 남거나 추가 명령만 늘면 Arena 없이 종료한다. 기존
FFN E56 staging 실패나 QKV prefetch 결과를 이 O 설계의 검증으로 대신하지 않는다.

**C11 — FFN F32 partial을 VRF에 두고 scale을 흘려보낸다.**

현재 [project_rows_with_scale_contraction](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)은
E5M2 native DPE→입력 scale 복원→F32 block partial DM 저장을 수행한다. 다음 pass는
block scale을 VRF에 읽고 partial을 DM에서 다시 읽어 곱하고 합산한다. 후보는 이
소유 관계를 바꾼다: `DPE→입력 scale 복원→partial VRF`, 이어서
`block scale stream × partial VRF→F32 reduce`. BF16 block partial을 새로 만들지 않는다.

SDK 0.8.1의 `PositionVectorFinal`은 `to_vrf(&mut device.sub)`를 제공한다.
Main에서 VRF에 쓰는 동안 Sub도 점유하므로 이 경로 자체가 Main/Sub 중첩은 아니다.
[collect.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/collect.rs),
[register files](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html).
전체 snapshot 검색에서 계산 결과의 Main→VRF 사용은 발견하지 못했다.

물리 padding을 포함하면 3행의 `H/64 × 4#8` F32 partial은
`3 × 60 × 8 × 4 = 5,760B`다. 현재 6행을 그대로 쓰면 11,520B로 VRF 8KiB를 넘는다.
우선 3행을 판별 대상으로 삼고 scale의 대응 packet과 살아 있는 다른 VRF까지 확인한다.
Up/gate의 기존 partial은 합계 28.125MiB의 논리 payload여서 write+read는 56.25MiB다.
**이는 HBM 전송량이나 실측 bus traffic이 아니며, 절약 cycles로 직접 환산할 수 없다.**

E73과 달리 FP4→FP8 materialization은 tile 바깥에서 유지하고 native DPE 결과를 VRF에
보낸다. 첫 검사는 3행·전체 H의 DPE→VRF→scale/reduce compile이다. Padding·VRF
operand 규칙을 만족하지 못하거나, 전체 compile에서 작은 DPE 실행 수가 급증하면
종료한다. Down은 packet 배치가 달라 이 검증을 승계하지 않는다.

**C10 — lookup→weight TRF를 직접 연결한다.**

기존 lane-parallel은 `FP8 weight→BF16 lookup→DM→Sub 재읽기→TRF`였다.
새 가설은 Main의 `fetch_table_lookup::<bf16>()→collect→to_trf()`로 DM 왕복을
없애고 BF16 activation을 stream하여 여러 출력 행을 DPE lane에 둔다. SDK의
`to_trf` frontend가 Main을 배제하지 않는 것은 확인했으나, **이 정확한 lookup 조합의
compiler lowering과 실행은 미검증**이다. BF16 weight TRF 4행은 30,720B, 8행은
61,440B로 SDK의 65,536B 논리 한도 안이다. 이것만으로 bank 충돌·할당 성공·중첩을
보장하지 않는다.

첫 산출물은 작은 exact lookup→TRF compile 하나다. 막히면 API/lowering 제약으로
닫고, 이미 느렸던 BF16 DM materialization으로 되돌려 오랫동안 고치지 않는다.
통과하면 TRF 적재·activation stream·dot을 포함한 QKV 전체 비용으로 비교한다.
과거 후보의 dot 직후 BF16 cast를 복사하지 않고 **E104의 F32 channel scale 이후
첫 BF16 경계**를 유지한다. 근거는 [SDK TRF API](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/collect.rs)와
기존 lane 실험이며, 직접 경로의 개선 가능성은 이번 설계의 추론이다.

## 후속 세 가지

**C12 — Main의 DPE와 Sub의 F32 reduce/scale을 분리한다.**

O의 Main은 다음 행 구간 dot을 하고 Sub는 이전 구간의 16-column F32 partial을
합산·scale 처리하는 구조다. Main의 출력 packet이 VE transpose나 후처리를 요구하면
공유 VE 때문에 중첩이 막힐 수 있다. DPE 결과 저장 경로부터 작게 compile해서 이
자원 충돌을 확인한다. 별도 partial의 DM 왕복 비용도 후보 비용에 포함한다.

[TCP](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-isca24.pdf)의 CE/VE 자원 분리와
[FlashAttention-3](https://arxiv.org/html/2407.08608v2)의 producer/consumer 분리를
설계 근거로 삼았다. Attention 알고리즘이나 GPU warp·TMA를 그대로 이식하는 뜻은 아니다.
SDK의 [commit_cast](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/commit-adapter.html)는
VE를 쓰지 않는 F32→BF16 저장 후보지만, channel scale **전**에 적용하면 현재 반올림
계약을 깨므로 후처리의 합법적인 경계에서만 검토한다. 기존 C05의 Sub reduce 내부
오류가 반복되거나 Main VE 점유를 없애지 못하면 종료한다. C09와 결합은 가능하지만
각 중간 설계의 단독 개선을 필수 조건으로 삼지는 않는다.

**C13 — FFN을 INT8 DPE와 I32 누산으로 계산한다.**

FP4 E2M1 weight는 `0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6`이므로 두 배 값은
INT8에 정확하게 들어간다. FP4 nibble을 INT4로 재해석하는 것은 이 변환과 다르다.
런타임 activation은 정수 상위 항·잔차 항으로 표현하고, INT8 dot의 I32 결과에
activation scale, 1/2, block scale을 F32에서 적용하는 설계다. 항 수와 입력 표현은
아직 결정하지 않았으며 BF16 입력과의 동치를 주장하지 않는다.

SDK에 integer DPE, `vector_fp_to_fxp`, I32→I8 cast가 있으나, 현재 fixed table lookup은
FP4→INT8 직접 변환을 제공하지 않는다. FP4→FP8→F32→정수 변환 비용까지 포함해야 한다.
[cast.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/cast.rs),
[vector tensor API](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/vector/tensor/vector_tensor.rs).
첫 검사는 16개 weight code 변환, activation 범위·상쇄 입력의 오차와 I32 누산 범위,
작은 변환+DPE compile이다. 추가 항·변환 때문에 이득이 사라지거나 오차를 지키지
못하면 종료한다. 칩의 INT4 peak 수치로 이 INT8 경로의 성능을 예상하지 않는다.

**C14 — FFN 전체를 L 구간의 생산·소비 pipeline으로 바꾼다.**

현재 `feedforward_sharded→project_up_and_gate→geglu→project_down_sharded`는 각
중간 tensor가 완성된 뒤 다음 함수를 실행한다. 후보는 L 구간별 up/gate와 GeGLU가
준비되는 즉시 대응 down weight 구간을 읽고 F32 출력 partial에 누산한다. 기존의
global-scale BF16 경계와 GELU 식은 유지하고 RMS·residual은 down 합산 완료 후 수행한다.

GeGLU를 두 cluster에 남기는 이전 실험과 달리 **down을 일찍 시작하는 의존성 변경**이
핵심이다. Up/gate/GeGLU의 BF16 activation은 각각 30,720B로 작으며 이것만 없애서
큰 개선을 예상하지 않는다. Weight HBM payload도 저절로 줄지 않는다. TCP의 연산
연결과 FlashAttention-3의 구간별 생산·소비 원리를 참고한 별도 FFN 가설이다.

처음에는 소수의 큰 L 구간만 사용해 compile하고, 각 down weight가 불필요하게
재전송되는지, partial 저장량·실행 수가 증가하는지 확인한다. Stage 대기가 그대로면
종료한다. 진행할 경우 새 down 합산 순서에 따른 상쇄 오차와 행·column 대응을 검증한다.

## 읽었지만 직접 실행 후보에서 제외한 방법

- [FLUTE](https://arxiv.org/html/2407.10960v2): lookup에 맞춘 offline weight 배치와
  lookup 구성은 참고할 만하지만, 1라운드는 제공된 quantized weight 형식을 받는다.
  Host 사전 재배치로 실행 비용을 감추는 구조는 제출 경로에 적용되지 않는다. Device 안의
  재배치는 가능하더라도 timed kernel 비용에 포함된다. 기존 FP4 paired lookup과
  [공개 table API 제약](fp8-lookup-table-sharing-audit.md)을 넘어선 새 직접 경로는 확인하지 못했다.
- [T-MAC](https://arxiv.org/html/2407.00088v2): 입력 activation의 조합으로 매번 LUT를
  만들고 weight bit pattern으로 조회한다. 현재 SDK의 고정 FP4/FP8 변환 lookup과
  다른 연산이다. 이에 대응하는 programmable lookup API를 확인하지 못해 보류한다.
- FlashAttention의 attention/softmax 알고리즘 교체: 1라운드 세 대상은 QKV projection,
  attention output projection, FFN이다. 논문의 scheduling 원리만 활용한다.
- FP8 weight×BF16 activation을 변환 없이 float DPE에 넣기: 현재 SDK는 해당
  혼합 dtype 조합을 지원하지 않는다. [기존 감사](mixed-dtype-dpe-audit.md).
- O channel scale과 RMS를 무조건 한 pass로 합치기: 첫 BF16 반올림을 보존하는 공개
  경로가 확인되지 않았다. [기존 감사](output-scale-rms-fusion-audit.md).

TCP는 architecture·dataflow 절, MARLIN은 memory pipeline·dequantization 절,
FLUTE/T-MAC/FlashAttention-3는 해당 알고리즘·pipeline 절을 중심으로 읽었다.
[Timeloop](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)은
공식 개요의 mapping/dataflow 탐색 관점을 참고했다. 논문의 benchmark를 재현한 것은 아니다.
논문에서 얻은 원리와 우리 SDK에서 검증한 API, 아직 미검증인 설계 추론을 구분한다.

## 짧게 시도하고 종료하는 운영 기준

1. **가설 한 개와 최소 구현 한 개부터 시작한다.** 의미가 명확한 API/shape 수정은
   한 번 정도로 제한한다. Compiler 내부 오류를 반복 추적하거나 여러 tile 값을
   훑기 전에 그 구조가 줄일 비용과 지원 경로가 있는지 다시 판정한다.
2. **Compile-only 종료도 정상 결과다.** 원하는 중첩·중간 저장 제거가 나오지 않거나
   명백한 회귀면 `compiled-not-measured`로 닫는다. 지원 경로가 없으면 compile 실패
   상태와 원인을 보존하고 `blocked-by-api/lowering`로 닫는다. 빌드 성공을 성능 성공으로
   기록하지 않으며, 무조건 빌드를 통과시키느라 시간을 쓰지 않는다.
3. Compile을 통과해 효과의 근거가 남은 후보만 동일 Arena 할당의 **E104→후보→E104**로
   기본 정확도·총비용을 screen한다. 실패 cycle은 개선으로 세지 않는다. 분리 가능한
   후보는 목표 커널부터 검사하고, 채택 검증에서 조합 전체를 확인한다.
4. 단회에서 회귀하거나 개선이 대상 커널의 5% 미만이면 기본적으로 추가 구조 튜닝을
   종료한다. 5%는 실험 예산을 위한 잠정 분류선이지 통계적 유의성·채택·순위 기준이
   아니다. Compiler 근거로 큰 병목 제거가 확인된 결합 설계는 이유를 적고 예외로 둘 수 있다.
5. 유망한 후보만 동시기 대조를 포함한 반복 측정·정확도 경계 검사로 넘어간다.
   한 번 잘 나온 수치로 채택하지 않는다. 작은 개선을 채택하려면 필요한 검증만
   마치고 닫으며, 디버깅·추가 sweep으로 장기 확장하지 않는다.
6. 후보마다 출발 commit, source/build 해시, 변경 경계, 종료 이유, 다음 후보를 남긴다.
   가설이 다른 재도전은 이전 실패와의 차이를 설명해야 한다. 서로 보완하는 decode·dot·
   scale·layout 변경은 함께 시험할 수 있고, E55/E104 주변에만 탐색을 한정하지 않는다.

## 9월 23일 공식 안내 반영

[공식 홈페이지](https://micro2026-moa.github.io/index.html)는 1라운드 마감을
**9월 30일 23:59 AoE**, 결과 발표·2라운드 시작을 10월 2일로 갱신했다.
최종 평가 대상은 leaderboard 최고점이 아니라 **마감 전 마지막 제출**이다.
따라서 실험용 Arena 작업과 최종 공식 제출을 구분한다. SDK 0.6.0/0.8.1이 지원되며
0.8.1을 권장한다. 마감 전 competition code 공개 금지 안내도 추가됐다.

이번 조사는 우리 private 저장소의 기록, 공식 SDK·대회 저장소, 공개 논문을 사용했다.
다른 참가자의 competition 코드를 이번 후보의 새 근거로 가져오지 않았다.
현재 저장소가 private인 것은 `gh repo view`로 확인했다. 허용 수정 범위는
[공식 과제 README](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B)의
`src/device/`와 `src/ops.rs` 함수 본문이며 공개 인터페이스·입력 계약을 보존한다.
