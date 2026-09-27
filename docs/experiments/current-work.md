# 2026-09-27 E128 native FFN dot/scale/reduce 채택 완료

현재 runtime은 **E128 / `baseline-e128` / `cc82b9c`**다.
[실험](../../moa-dev/experiments/candidates/ffn-native-fused-scale/README.md) · [구성표](../../baseline/e128.json).
`opt/ffn-native-fused-scale`, 실제 시작 `57e6a5b`, source·별도 성능 control E127.
고정6행 native dot→Mul0(input scale)→Mul1(block scale)→F32 H 합산으로
up/gate partial DM28.125MiB write/read가 실제 graph에서 제거됐다. Native 시작은2→10이다.
정적155190 vs154440(+0.486%)와 달리 NPU3allocation은 FFN−0.7879/−0.6569/−0.5564%,
paired중앙−0.6569%다. 후보기본45·추가30비교(9 distinct standard inputs),
별도8case245760 projection출력 PASS. QKV/O image는E127과동일하다.
Positive를동결해검증후채택했으며 tile sweep은없다. 이번큰저장삭제의실측효과는작았다.
추가튜닝은중단하고 FP8 weight 저장/joint packed복사를없앨ownership구조는별도연구로남긴다.
새공식제출없음;공식순위는아래E127기록이다. 검증host의checksumAPI·출력경로수리로그도보존했다.

# 2026-09-27 E127 이후 큰 중간 저장 제거 shortlist

[현재 코드·과거 실패·SDK를 대조한 후보 분석](../research/e127-large-cost-shortlist-20260927.md)을 추가했다.
첫 검증 대상은 native FFN dot→입력 scale→block scale→F32 합산의 단일 Main stream이다.
Up/gate F32 partial28.125MiB write/read를 제거하되 native dot 시작은2→10으로 늘어난다.
E73의 반복 BF16 lookup과 C11의 partial VRF handoff를 피하는 국소 데이터 흐름 교체이며 미구현·미측정이다.
더 큰 구조 연구는 projection별 cluster 소유권으로 FP8 전체 저장과 joint packed 복사를 함께 없애는 안이다.
전송 계약·direct 정확도부터 확인해야 한다. E103/direct reload와 v7b/interleave를 새 후보로 반복하지 않는다.
새 compile/Arena/제출은 없으며 runtime은 아래 E127 그대로다.

# 2026-09-27 E127 공식 제출·리더보드 반영 완료

현재 runtime E127을 공식 제출 **`bc21122c`**로 평가했다. 서버 contract/build/evaluate 완료,
기본15비교 PASS. QKV103186 / O49155 / FFN330948 cycles, **6.0699449×**다.
제출 직전 E1045.9265771× 대비 **+2.4191%**, 공개 리더보드는 **53/69위 →53/69위**다.
새 submissionCode가 실제 leaderboard에 반영됐음을 확인했다.
[제출·원시 로그](e127-official-submit-20260927.json) · [제출 전 순위](leaderboard-e127-before-20260927.json) ·
[제출 후 순위](leaderboard-e127-20260927.json).
QKV 변경은 없으므로 두 공식 실행의 QKV 차이를 새 최적화 효과로 세지 않는다.
아래 채택·실험 기록의 미제출 표시는 해당 작업 완료 당시의 상태다.

# 2026-09-27 E127 / 동결 C09 채택

현재 runtime은 **E127 / `baseline-e127`**다. E122의 QKV/FFN image를 유지하고,
O에 원본 C09의 두 stage 공급을 그대로 채택했다. [채택 구성](../../baseline/e127.json) ·
[성능·확장 정확도 검증](../../moa-dev/experiments/candidates/c09-output-first-npu/validation/README.md).

E126과 새 독립2allocation의 paired O 변화율은 **−2.4960% / −3.8255% / −3.9303%**,
중앙값은 **−3.8255%**다. 실제 stage host/fixture/runner가 동일하며,
표준 candidate45비교와 추가6입력30비교가 모두 PASS했다. 표준 입력은 반복을 제외하면
커널별9개다. 별도 pre-RMS projection probe8case/30720출력도 PASS했다.
Probe는 별도 diagnostic image이며 표준 성능 집계에는 포함하지 않는다.

**E126의 초기 종료·미채택 판단은 E127 후속 검증과 채택으로 대체됐다.**
5%는 추가 parameter tuning의 예산 기준이며, 작은 이득의 제한적 채택 검증을 금지하거나
runtime 채택의 최소 개선율을 뜻하지 않는다. Source/layout/tile을 더 조정하지 않고
기존 구현의 개선을 검증해 회수했다. 큰 구조 개선 탐색은 별개이며 새 공식 제출은 없다.
아래 기록의 runtime/미실행/종료 상태는 각 실험 당시를 가리킨다.

# 2026-09-27 E126 / frozen C09 최초 NPU 완료 — 당시 기록

[C09 최초 NPU](../../moa-dev/experiments/candidates/c09-output-first-npu/README.md)는
과거 exact image를 재현하고57비교 PASS. Arena105820 전체 O49923 vs E122
before51237/after51165로 **단회 −2.496%**를 관측했다. 추가 튜닝·반복·확장 정확도·채택 없이
5% 탐색 예산 기준으로 종료하고 소폭 positive option으로 보존한다. Live E122 유지.
Trace의 첫 dot은28945→17148, dot0/weight1 실제 overlap7392.5다. 다만 weight공급
전체 구간은26660→30073으로 늘어 최종 dot 종료는40953→39542.5의 제한적 개선이다.
정적 마지막dot→gather1206 gap은 NPU에서재현되지않았고 양측모두2cycles였다.
[이번 실행 결과와 다음 설계 조건](../research/o-weight-ingress-execution-20260927.md).
M08 기록617aaef를 main/origin에 통합한 뒤 이 별도 진단을 실행했다. 공식 제출 없음.

# 2026-09-27 E125 / M08 O row-pair4 종료

[M08](../../moa-dev/experiments/candidates/m08-output-pair4-ingress/README.md)는
actual DMA coverage·BF16/F32 경계 검사와57개 기본 비교를 통과했지만,
Arena105767 전체 O58095 vs E122 before51418/after50906으로 **+13.55% 회귀**했다.
Trace에서 weight26769→27190으로 단축이 없고 F32 gather1201→8222로 증가했다.
같은15360B gather의32×480B→1920×8B 분할 비용을 전체 경로에서 확인했다.
추가 sweep 없이 종료; live E122 유지, 공식 제출 없음. 기록을 main에 통합한 뒤
동결 C09 two-stage의 **최초 NPU 진단**으로 이어갔다. 완료된 E126 결과는 상단에 있다.
정적+0.71%만으로 종료했던 prototype의 실제 overlap/whole O를 확인했으며 새 architecture로 세지 않는다.

# 2026-09-27 O weight ingress 연구 완료

[O 전송·배치 종합 분석](../research/o-weight-ingress-20260927.md)을 추가했다.
현재 O와 동일한 image의 graph로 15 MiB의 정확한 source coverage와 DMN 교대를 확인했다.
첫 진단 후보는 행쌍 보존 4-way ingress + 기존 16K F32 합산 + 직접 H480 gather다.
현재 gather 32×480B가 예상 1920×8B로 분할되는 비용까지 전체 경로에서 비교해야 한다.
C09는 정적 +0.71% 후 NPU 미실행이므로 최초 하드웨어 진단의 가치가 남아 있다.
이번에는 새 compile/Arena/runtime 변경·공식 제출이 없다. 채택본은 아래 E122다.

# 2026-09-27 E122 채택 및 전송 계약 재검증

현재 runtime은 **E122 / `baseline-e122` / `c3553ab`**다. M03 원본을 그대로 채택했다.
[검증](../../moa-dev/experiments/candidates/m03-producer-local-ffn/validation/README.md):
동일 실행 stage의6allocation에서 모두개선, paired비율중앙값 FFN−1.4846%,
추가6표준입력과7-case241920 projection출력 PASS. QKV/O image는E104와같다.
5%는탐색예산문턱이며 이미반복검증된작은이득의채택금지가아니라는예외를기록했다.
공식제출은아직E104이며 Arena결과를공식점수로환산해주장하지않는다.

[E123 corrected direct FFN](../../moa-dev/experiments/candidates/ffn-direct-boundary-recompute/README.md)은
전체 기본60비교 PASS, 단회329395cycles로 E104 앞뒤 대비−1.9179%, 동시 M03 대비−0.7392%다.
작은 이득으로 반복·확장 정확도·채택 없이 종료했다. Direct primitive 실패를 일반
native DPE 한계로 보았던 해석은 철회하되 이 단회 결과도 일반 동치로 확대하지 않는다.

E124 corrected joint QKV도 종료했다. Arena105093의45비교가 모두통과했으나
QKV129258 vs 앞뒤 E104103899로 +24.41% 회귀해 미채택했다. [C08 exact 감사](../../moa-dev/experiments/candidates/c08-joint-ingress/README.md)에서
K/V padded destination DMA의 실제 source 범위 초과를 확인했다. 원본과 image가 같다.
이는 과거 family의 전송 계약 수리이며 새로운 독립 architecture가 아니다.
수정 뒤에는 정확하지만 느린 실행으로 판정할 근거가 생겼다. 추가 반복·튜닝은없다.

[이번 실행 전체 결과와 다음 판단 기준](../research/top10-execution-20260927.md).
다음 미실행 질문은 O weight 준비 지연의 DMA 순서/DM 배치 민감도다. 단독 전송
진단을 커널 speedup으로 계산하지 않고, 유망한 배치만 완성 O 경로에서 검증한다.

# 이전 구조 실험 기록 — 아래의 현재/미실행은 당시 상태

작업 폴더: `/home/jihwan38/moa-workspace`. Private GitHub: `jihwan38/moa-workspace`.
최신 조사: `research/rngd-dataflow-20260925` / 시작 commit `f125ca6`.
보완 조사: 2026-09-26 / `opt/m02-cluster-k`의 `a46baa4` 이후.
당시 채택 runtime: E104. 최신 채택본은 문서 상단 및 [baseline/current.json](../../baseline/current.json)의 E127이다.

## 최신 감사 — 10위권 격차와 누적 실험 재평가, 2026-09-27

[코드·실험·논문 재점검](../research/top10-audit-20260927.md)을 추가했다.
9월27일00:41 KST 공식 조회는 BEEF52/69위5.926577×, 10위8.859047×다.
FFN만 바꿔 그 점수를 맞추려면335987→100594 cycles가 필요하므로 QKV/O의
큰 비율 개선 경로도 병행해야 한다. 메모리 배치 전체가 잘못됐다고 단정하지 않는다.

M03의 첫 E117은 단회였지만, 후속 companion을 포함하면 동일 package의5개 allocation에서
앞뒤 E104 대비1.35–1.63% 감소, 감소율 중앙값1.4891%다.
[누적 원본·해시·계산](../research/m03-cumulative-audit-20260927.json).
동일3입력 반복이며 확대 정확도나 새 채택을 뜻하지 않는다. 아래 단회 설명은 첫 실험의
역사 기록이다. 5% 탐색 예산 cutoff와 제한적 채택 검증은 구분할 필요가 있다.
QKV/O 내부 계측, direct-FP4 고정 shape boundary 재계산 후보, M06 timeout 기제 진단의
조건과 중복 여부를 정리했다. 새 compile/Arena/채택/공식 제출은 없고 runtime은E104다.

## 최신 작업 — M07 weight 전송 분할, E121 개선 미확인으로 종료

M06 기록 `0c6dea5`까지 main/origin에 통합하고 `opt/m07-weight-ingress-chunks`에서
[M07](../../moa-dev/experiments/candidates/m07-weight-ingress-chunks/README.md)를 구현했다.
E74 crate + 동결 M03 전체 FFN을 source로 선택하고 세 weight의 HBM 적재만5분할한다.
M04와 달리 FP8 변환·dot은 전체 크기의3개 stream을 유지한다. Local transfer 변경이며
새 전체 FFN 아키텍처 가족으로 세지 않는다.
v1은 compile PASS지만 up/gate 각 tile이30행 전체 descriptor로 넓어져 정적 비용이 컸다.
v2는 compact6행 임시 buffer를 만들었으나 assembly DMA가30행을 읽으려 해 범위 검사 실패.
v3은 해당 assembly만 raw FP4 Sub copy로 고쳐 전체 compile·host PASS다.
v3의15개 HBM weight command는 의도한5분할 payload이며 전체 FP8 변환·dot은3개다.
입력 준비 앞의 상대 대기는 줄지만 pre-RMS 자체가 늦어져 정적 전체는 M03보다 늘었다.
Arena104739: E104335871 → v3 336059 → 원본 M03 330118 → E104335305.
기본60비교 PASS, 후보 FFN3입력/11520출력 PASS다. E104 앞뒤 중앙값 대비+0.14%,
동시 M03 대비+1.80%로 채택할 이득이 없어 추가 튜닝 없이 종료했다. 단회 screen이며
내부 NPU 시각·stall은 계측하지 않았다. [E121](e121-m07-weight-ingress-chunks.json).
v1은 중복 전송 descriptor 비용으로 정적 종료, v2는 compile 실패, v3은 실측 종료다.
세 구현 변형을 독립 아키텍처3개로 세지 않는다. Live E104 유지; 새 채택·공식 제출 없음.

## 최신 작업 — M06 실행 timeout으로 보류, E119/E120

M05 기록 `83d09dd`까지 main/origin에 통합하고 `opt/m06-sharded-input-pipeline`에서
[M06](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/README.md)를 구현했다.
예전 E53 FFN sharded RMS와 QKV/O 직접 packing의 전송 방식을 현재 native3항 입력과
M03 구조에 결합했다. pre-RMS gather/입력 재분배/scale 재분배/3항 packing DMA를
제거하는 경로다. 과거 방법을 새 독립 아키텍처로 세지 않는다.
v1/v2/v3은 SFR lowering에서 실패했다. Packed lo의 cast/add를 Sub→Main으로
옮긴 v4와 별도 term buffer·packing DMA를 복구한 v5는 전체 compile·host build PASS다.
그러나 Arena104566/104588에서 두 후보 모두 첫 FFN 출력 비교 전에 device timeout.
같은 작업의 E104 before/M03/E104 after는 각각45비교 PASS여서 접속 실패가 아니다.
Direct packing만이 원인이라는 증거도 없다. 원인 미확정으로 보류하며 추가 변형은 중단했다.
[실패 E119](e119-m06-sharded-packed-invalid.json) · [실패 E120](e120-m06-sharded-dma-invalid.json).
유효 FFN 성능·정확도·채택·공식 제출 없음. 실행 source는E104다.

## 최신 작업 — M05 입력 준비·M03 조합, E118 개선 미확인으로 종료

M03 기록 `db7cbf4`까지 main/origin에 통합한 뒤 `opt/m05-local-activation-ready`에서
[M05](../../moa-dev/experiments/candidates/m05-local-activation-ready/README.md)를 시험한다.
M03의 local GeGLU/L-owned down 배치에 M04의 producer-local 입력 준비만 결합했다.
M04의15개 row-tile decode/dot 대신3개 전체 weight stream을 유지한다.
첫 activation 복제 DMA 전에 amax를 시작할 수 있는지가 구체적 가설이며,
새 독립 아키텍처 가족이나 타일 크기 sweep으로 세지 않는다.
Arena104458: E104335775 → M05 335317 → 원본 M03 331749 → E104337758.
기본60비교 PASS. E104 앞뒤 중앙값 대비−0.43%, 동시 M03 대비+1.08%여서 종료했다.
정적 amax는 앞당겨졌지만 최종 입력 방송 완료는 M03보다 늦어졌다. 내부 NPU 시각
측정은 아니며, 전체 실측에서도 유의미한 이득이 없다. 추가 튜닝·채택·공식 제출 없음.
실행 source E104 유지. [E118](e118-m05-local-activation-ready.json).

## 최신 작업 — M03 producer-local 전체 FFN, E117 소폭 개선으로 종료

진단 기록 `6ea3664`까지 main/origin에 통합하고 `opt/m03-producer-local-ffn`에서
[M03](../../moa-dev/experiments/candidates/m03-producer-local-ffn/README.md)를 구현했다.
실제 source origin은 E74이며 C14-B의 local GeGLU/L-owned down에 E104의 검증된
materialized native up/gate를 조합했다. C14의 두 L stage는 한 stage로 바뀌었다.
전체 L gather 대신 cluster-local 120행 compact와 local down 입력 분배를 사용하며,
마지막에 두 F32 H partial을 더한다. C14 가족의 재조합이다.

첫 source compile·host build PASS, QKV/O image는 E104와 같다. 정적154440은
E104157045 대비 작은 감소지만 NPU 실측 성능으로 해석하지 않는다.
Arena104289: **E104335669 → M03 330837 → E104335107 cycles**, 앞뒤 비교군 중앙값
대비 **1.36% 감소**다. 총45비교 PASS, 후보 FFN3입력/11520출력 PASS.
단회 screen이며 확대 정확도·독립 반복은 없다. 5% 실험 예산 기준에 따라 추가 타일
튜닝 없이 종료하고 소폭 개선된 조합 선택지로 보존했다. 실행 source E104 유지,
새 채택·공식 제출 없음. [E117](e117-m03-producer-local-ffn.json).
전체 FP8 저장량과 초반 입력 준비 대기는 이 구조에서도 남는다. 이것을 메모리 배치가
전역 최적이라는 근거나 10위권 진입 방법이 확보됐다는 주장으로 해석하지 않는다.

## 최신 진단 — 직접 FP4/native 경로 보류, 정확도 미해결

`c49300d`까지 main/origin에 통합한 뒤 `diag/fp4-native-boundary`에서
[원본·진단 기록](../../moa-dev/experiments/candidates/fp4-native-boundary/README.md)을 남긴다.
현재 목표는10위권이며, 순위 상승이나 새 runtime 채택은 아직 없다.

단일 native projection의 실제 입력 준비·dot·block scale·행 합산은4입력에서
통과했다. 같은 입력 표현을 공유하는 두 projection은 최종 BF16 gather 전에
틀렸고, 한 joint producer로 바꾸면 첫 projection의16행에만 오류가 남았다.
입력 양자화를 수행한 slices0..7의 첫 행과 일치한다. 저수준 원인은 아직 미확정이다.
옛 C07 실패 image도 다시 실행해 실패를 재현했고 materialized 원본은 통과했다.
새 진단의 materialized control에는 timeout이 있어 전체 paired PASS로 세지 않는다.
후속 v8의 scale 경계 이동도 실패했다. v9c/v9d 행 padding은 lowering assertion으로
NPU 미실행이며, v10(job104243)의 첫 projection 전체 warmup도 두 실제 출력에
non-finite 값을 만들었다. 이 경로는 원인 미해결 상태로 보류하고 반복 수정을 중단했다.
새 FFN 성능 결과·채택·공식 제출은 없다. 이 진단을 최적화 가족/성능 실험으로 세지 않는다.
당시 후속이었던 M03의 완료 결과는 위에 있다: 검증된 weight materialization을 유지하면서
producer-local GeGLU와 L-owned down을 한 stage로 연결한다.

## 최신 작업 — M04 입력 준비·weight 타일 공급 결합, E116 종료

진단 기록 `db89a50`까지 main/origin에 통합한 뒤 `opt/m04-tiled-weight-supply`에서
[M04 전체 경로](../../moa-dev/experiments/candidates/m04-tiled-weight-supply/README.md)를 구현했다.
입력 amax/3항 quantization을 원래 cluster에서 먼저 수행하고 완성된 표현을 복제한다.
Up/gate는6행, down은12행씩 HBM 적재→FP8 DM→dot→F32 partial→scale/reduce로 연결했다.
한 파일의 완성 경로 조합이며 기존 row/column 소유권·GeGLU·공개 입력 계약은 유지한다.
M03의 producer-local GeGLU/down 구조를 구현한 것으로 세지 않는다.

**첫 source compile·기본 NPU 정확도 PASS, FFN은 13.67% 느려 미채택**했다.
Arena 103108: E104 334906 → M04 381507 → E104 336369 cycles. 총45비교 PASS이며
후보 FFN은3입력/11520출력을 검사했다. [E116](e116-m04-tiled-weight-supply.json).
확장 정확도·독립 반복·공식 제출은 없고, 채택 runtime은 **E104 유지**다.

정적 graph에서 pre-RMS→amax gap은23250→0이지만 pre-RMS 자체는 더 늦어졌다.
스케줄상 FP8 텐서의 최대 동시 크기는56.25→22.5MiB로 감소했고 전체 FP8 저장량은 그대로다.
DMA 명령35→74, dot3→15, 정적 전체157045→165192다. 실제 NPU 내부 stall 비용을
이 수치로 환산하지 않는다. [판정·해시·원본 로그](../../moa-dev/experiments/candidates/m04-tiled-weight-supply/decision.json)를 남기고,
회귀 prototype의 tile sweep 없이 종료한다. 두 변경 각각이 나쁘다는 판정은 아니다.
M04 종료 당시 M03은 미구현이었다. 후속 재시도는 반복 setup/전송/저장 비용을 바꾸는 구체적 근거가 필요하다.

## 이전 작업 — E104 whole-FFN 내부 NPU 진단 완료

M02와 보완 조사 기록을 main/origin의 `913f9ac`까지 통합하고 `diag/e104-weight-path`에서
[내부 NPU profile 진단](../../moa-dev/experiments/candidates/e104-weight-path/README.md)을 실행했다.
별도 stage 출력 대신 기존 image의 trace marker를 읽어 원래 FFN 흐름을 측정했다.
Arena102797·동일 package 재실행102842 성공, 총78개 출력 비교 PASS다.
Kernel 계산 task bytes는 E104와 동일하며 diagnostic host와 profile 이름만 별도 처리했다.

Weight 적재는 gate약47.6–47.8k/up약48.0–48.2k/down약65.7–66.0k cycles,
FP8 전체 저장은 각각약20–21k, native up/gate dot·부분합은각각약12k다.
초반 큰 gate weight 전송 중 작은 입력 복제 구간이 길어지고, pre-RMS 끝부터
activation amax 시작까지 다른 Main 연산이 없는 구간은 중앙값42.2k였다.
입력 방송도23.4k다. 이 구간들은 겹치고 대기를 포함하므로 더해서 비용 비중을 만들지 않는다.

다음 구조는 weight 공급·작은 입력 준비의 순서/배치와 FP8 전체 중간 저장을 함께 검토한다.
M03은 아직 미구현이며, 채택 baseline은 E104다. 진단 자체를 최적화·순위 개선으로 세지 않는다.

## 이전 조사 — 현재 리더보드 격차와 FFN weight 경로

[코드·정적 근거·관련 대회 자료](../research/e104-bottleneck-20260926.md)를 보완했다.
2026-09-26 18:25 KST의 공식 API 응답에서 BEEF는69팀 중52위, 5.926577×,
3위는9.251393×다. 해당3위 cycle까지 QKV36.01%, O29.73%, FFN41.53% 감소가 필요하다.
FFN weight 적재→FP8 전체 DM 저장→dot 경로가 우선 병목 가설이다.
실제 NPU에서 내부 비용을 분해한 결과는 아직 없으며 기존 정적 진단만 재검증했다.
이번 작업은 조사·기록이며 새 compile/Arena/채택/공식 제출은 없다.
당시 M03은 미실행이었고 최소 비용 진단을 먼저 수행하기로 했다. 실행 결과는 위 최신 작업에 있다.

## 이전 작업 — M02 cluster K-half 소유권 검증 완료

M01의 `ecdeaca`까지 main/origin에 fast-forward 통합하고 `opt/m02-cluster-k`에서
전체 O 경로를 새로 구현했다. [M02 설계·수정·실측](../../moa-dev/experiments/candidates/m02-cluster-k/README.md).
cluster는 H 반쪽 대신 K2048 반쪽을 소유하고, 각 slice가15×2048 dot을 수행한다.
Ring32 leader 입력 방송, F32 partial gather/압축, 두 cluster의 F32 합+scale을 포함한다.
15×2048 tile 자체는 이전 두열 실험에도 있었고 이 조합의 새 ownership을 검증했다.

**두 완성 경로 모두 compile·기본 NPU PASS 후 큰 회귀로 종료했다.**
Arena95914: E10451010 → v3 full-H 집결78981 → v4 H480 묶음89203 → E10450901.
앞뒤 control 중앙값 대비 각각**+55.00%, +75.06%**다.
[E114](e114-m02-full-h-gather.json), [E115](e115-m02-grouped-interleave.json).
총60비교 PASS이며 각 후보 O는3입력·11520출력 원소를 검사했다.

v1 방송 축, v2 H%256→H%480 직접 DMA factorization 오류를 구체적으로 수정했다.
v3는 full-H 집결로 실행 가능성을 확인했고, v4는 producer row를 H480 경계에 맞춰
그 중간 집결을 없앴다. 정적 전체37413→32382로 줄었지만 NPU 순서는 반대로 나타났다.
Weight 적재·dot의 정적 interval은 줄지 않았고 partial 의존성이 추가됐다.
NPU에서 어떤 bank/전송/stall이 얼마를 차지했는지는 측정하지 않았다.

source snapshot4개·exact compile6회(그중 package 재compile2회)·host2개·Arena1회.
한 architecture의 두 경로이며 독립 가족 두 개로 세지 않는다.
추가 sweep·확장 정확도·독립 반복·채택·공식 제출 없이 종료하고 **E104를 유지**한다.
Source/log/hash/실측/판정은 branch에 보존한다. 다음 미실행 사전 후보는 **M03**이다.

## 이전 작업 — M01 partial 재배치 검증 완료

사용자의 실행 순서 지정에 따라 **M01을 먼저** 진행한다. 조사 기록을 main/origin에
`8109a27`까지 fast-forward 통합한 뒤 `opt/m01-partial-remap`을 만들었다.
[M01 설계·두 전송 경로·검증 기록](../../moa-dev/experiments/candidates/m01-partial-remap/README.md).
E64의 column-major 적재 뒤 weight 전체 재배치를 제거하고, 동일 배치에서 dot 후
compact F32 partial을 DMA 또는 Switch로 옮기는 완성 O 경로를 검사한다.
단순 column 수 sweep이 아니며 두 경로를 별도 아키텍처 가족으로 세지 않는다.

**두 경로 모두 compile·기본 NPU 정확도 PASS 후 실측 회귀로 종료했다.**
Arena95636: E10451007 → DMA55608 → Switch60677 → E10451274 cycles.
앞뒤 control 중앙값 대비 DMA+8.74%, Switch+18.65%다. 각 process의15비교가 모두 PASS다.
[E112 DMA](e112-m01-partial-dma.json), [E113 Switch](e113-m01-partial-switch.json).

첫 `dma-v1`의 commit-adapter source 오류는 v2에서 수정했다. 정적 일정에서는
weight 적재13383→13435, dot stream5063→5063으로 줄지 않았고,
새 partial 이동/합산이 직렬로 추가됐다. 정적 makespan은 DMA29824, Switch31995,
E10427239다. NPU stall 분해 측정과 구분한다.
긴 튜닝·추가 반복 없이 종료, E104 실행 소스 유지, 공식 제출 없음.
후속 미실행 설계는 M02와 M03이다.

## 이전 작업 — RNGD 데이터플로·메모리 매핑 조사

논문·공식 programming guide·SDK 0.8.1 및 과거 시도를 대조해
[전체 경로 후보와 비용](../research/rngd-dataflow-candidates-20260925.md)을 정리했다.
[설계 JSON](../research/rngd-dataflow-candidates-20260925.json),
[출처·해시](../research/rngd-dataflow-evidence-20260925.json),
[좌표·논리량 검사](../research/rngd-dataflow-accounting-20260925.json).

다음 구현 우선순위는 **M02 O cluster K 소유권 → M03 producer-local FFN**이다.
M03은 C14 가족을 native up/gate·한 L stage와 조합하는 재설계이며 새 가족으로 세지 않는다.
M01은 E64의 weight repack을 partial 재배치로 바꾸는 조건부 후보로,
적재·소비 경로의 이득을 먼저 제한된 진단으로 확인해야 한다.
후보별 전체 graph, HBM/DM/TRF/VRF 소유권, 추가·제거 작업, 첫 검증/종료 조건을 기록했다.

이번 작업은 조사·설계·정수 좌표 검사까지다. 후보 compile/Arena/제출은 없고,
실행 소스·채택 runtime은 E104다. 후보의 속도 향상이나 순위 상승은 아직 입증하지 않았다.

## 이전 작업 — 세 구조 실행

진행 순서는 **C14-B → C15 → C12-B**다.
[설계 범위·비용 가설·과거 시도 대조](structural-round-20260923.md).
첫 `opt/c14-stage-compact`의 C14-B는 전체 compile·기본 NPU 정확도 PASS 후,
FFN397348 vs 동시 E104336341/339553(+17.58%)로 종료했다.
[source·측정·비용 분석](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/stage-compact.md).
C15도 up/gate column 소유권을 교체해 compile·기본 NPU PASS 후 FFN385621 vs
E104335618/335432(+14.93%)로 종료했다. [전송 비용·수정 기록](../../moa-dev/experiments/candidates/c15-upgate-columns/README.md).
C12-B도 전체 compile·기본 NPU PASS 후 O51862 vs E10450555/50680(관측+2.46%)로
종료했다. [컴파일 분리 진단·완성 경로·측정](../../moa-dev/experiments/candidates/c12-output-main-sub/README.md).
세 후보 모두 기본 NPU까지 실행했으며 채택·공식 제출은 없다. 실행 source는E104다.
세 실험 branch의 기록을 main에 fast-forward 통합했다. 각 branch와 원격 tip도 보존했다.
[통합 이력](structural-round-20260923-integration.json), [전체 재검증](structural-round-20260923-verification.json).

## 이전 작업 — main 통합과 E104 비용 진단

**main과 origin/main을 `873f64e`에서 `f9f905b`로 fast-forward해 push했다.**
`opt/c13-int8-ffn`의 `2c51fe2`까지 26개 commit과 작업 규칙·안내 정리 commit을
포함한다. 확인한 모든 기존 local/origin 실험 브랜치 tip이 이 이력에 포함되어,
별도로 병합할 기존 실험 변경은 없다. [통합 확인 기록](integration-20260923.json).

실행 소스는 채택 E104이며, C13/C14는 후보 snapshot에만 남는다. AGENTS.md와
조합 양식에 최신 비교군의 비용 근거, 제거·추가 작업, 실제 의존성/자원 경쟁,
설계 선택을 바꾸는 최소 진단과 그 한계를 기록하도록 보완했다. Root README의
현재 채택본 표시도 E79에서 E104로 수정했다.

당시 브랜치는 **`diag/e104-cost-attribution`**였다. E104의 기존 schedule 293개
instruction을 source와 대조하고 긴 node 18개·context 중첩을 재현 가능한 JSON으로
남겼다. [첫 정적 분석·다음 판별 질문](../../moa-dev/experiments/diagnostics/e104-cost-attribution/README.md).
Weight 준비와 activation 분배 중 무엇을 먼저 크게 바꿀지 결정하는 최소 NPU
진단은 아직 미구현·미실행이다. 이것은 새 최적화 가족이나 E104 구조에 탐색을 고정하는
규칙이 아니다. C14 compact 전송 수정은 아직 미구현인 후속 선택지이며, C12도 미구현이다.

## 이전 작업 — C13 재개 후 NPU 검증 완료

작업 브랜치는 `opt/c13-int8-ffn`이었다. **C13 v7은 전체 compile과 변경 없는 NPU
기본 정확도 검사를 통과했지만, 동시 E104보다 FFN이 18.58% 느려 미채택**했다.
[전체 구현·초기 실패·수정·실측](../../moa-dev/experiments/candidates/c13-int8-ffn/README.md).

초기 v2를 compile 횟수 한도로 닫은 뒤, 사용자의 지적을 받아 구체적인 구현 오류를
수정했다. 8B scale packing과 명시적 입력 복제로 v6을 compile했다. 첫 Arena
실행 E107은 non-finite FFN 출력으로 실패했다. 분리 진단에서 긴 K=3840 integer
unzip/zip 뒤에 scale 곱셈을 붙이는 조합의 block 대응 오류를 확인했다. v7은 정확한
정수 결과를 F32 DM에 저장한 뒤 별도 tag-zero stream에서 scale을 적용한다.

Arena job 78730: **E104 before 334969 → C13 397318 → E104 after 335156 cycles**.
각 process의 기본 15비교가 모두 통과했으며 FFN 3입력/11520출력도 기존 기준 그대로
PASS다. 단회 screen이며 공식 점수·독립 반복 결과가 아니다. 성능 회귀가 커서 더
튜닝하지 않고 종료했다. E104 실행 소스 유지, 공식 제출 없음.

E74를 출발로 up/gate/down 모두 INT8 두 항/I32 누산으로 구현한 같은 C13 구조다.
7개 source variant와 분리 진단을 여러 독립 구조로 세지 않는다. 0 입력의 raw
restoration scale은 CPU subnormal 대신 0으로 나오는 차이가 있어 진단 전체를
PASS로 처리하지 않았으며, 일반 BF16 입력 동치나 확대 검증은 주장하지 않는다.

AGENTS.md에는 수정 가능한 source/layout 오류를 compile 횟수만으로 포기하지
않도록 보완했다. C14의 기존 padding 전송 실패도 불가능성이나 성능 탈락의 근거는
아니다. [C14 후속 검토](../../moa-dev/experiments/candidates/c13-int8-ffn/c14-assessment.md)의
stage 내부 compact 경로는 아직 미구현이다. C12 역시 미구현이며 기존 제안 범위는
국소 Main/Sub dataflow 변경이다.

## 이전 작업 — C14-A

C14 작업 브랜치는 `opt/c14-ffn-l-pipeline`이다. **C14-A는 producer→down DMA의
layout lowering 실패로 종료**했다. [구현·세 버전·종료 근거](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/README.md).

E74 `9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`에서 출발해 BF16 stream 계산,
분산 GeGLU, L-owned down cluster, 두 L 구간의 F32 partial 합산을 한 경로로
구현했다. E104는 비교 계획에만 사용했다. 기존 전체-stage 대기와 H-owned down
배치를 함께 변경한 whole-FFN 재설계 시도이며, 완성된 image는 생성되지 않았다.

Compile 3회: 최초 tiled-view reshape 실패 → HBM 축 분할 순서 수정 후 factorize
실패 → 별도 cluster 축 표현으로 수정 후 padded producer→down column DMA 실패.
기본 수정 횟수를 한 번 넘긴 구체적 근거와 총 3회 제한은 v3 recipe에 사전 기록했다.
논리 좌표/pitch/GeGLU 연산 순서 검사는 통과했지만 NPU 정확도·성능은 미평가다.
Arena·공식 제출 없음, E104 실행 소스 유지. C14 전체가 느리다는 판정은 아니다.
당시 다음 우선순위였던 C13의 실행 결과는 위 최신 작업에 기록했다. C12는 미구현이다.

## 이전 작업 — C09/C11/C10

사용자가 승인한 실행 순서는 **C09 → C11 → C10**이다.
[설계·과거 시도·종료 기준](../research/structural-candidates-20260923.md).

| 후보 | 브랜치 | 상태 |
|---|---|---|
| C09 | `opt/c09-output-pipeline` | [종료](../../moa-dev/experiments/candidates/c09-output-pipeline/README.md): compile PASS, 정적 +0.71%, Arena 미실행 |
| C11 | `opt/c11-ffn-partial-vrf` | [종료](../../moa-dev/experiments/candidates/c11-ffn-partial-vrf/README.md): compile PASS, 정적 +3.55%, Arena 미실행 |
| C10 | `opt/c10-direct-weight-trf` | [종료](../../moa-dev/experiments/candidates/c10-direct-weight-trf/README.md): StoTrf lowering에서 현재 target 설정 제약으로 compile 실패 |

각 후보는 E104 소스에서 별도로 출발한다. 다음 브랜치는 이전 실험의 기록을 이어받으며,
채택하지 않은 kernel 구현은 후보 snapshot에 보존한다. Main 병합·공식 제출 상태와
Arena 실험 상태를 구분한다.

이번 요청의 세 후보는 각각 **최초 compile 1회**로 종료했다. C09/C11은 E104 control도
각 1회 compile했으며 exact image hash가 채택본과 동일했다. Arena 실행·공식 제출은
없다. 성능표의 +0.71%/+3.55%는 compiler schedule이며 NPU cycle 변화가 아니다.

세 실험의 마지막 브랜치는 `opt/c10-direct-weight-trf`이며 C09 → C11 → C10 순서의 기록을
모두 포함한다. 세 후보 브랜치는 private origin에 각각 push했다. E104 실행 소스는
바꾸지 않았고 `experiment_index.py --check --check-live`로 확인한다.
당시 후속 미구현 후보는 C12(Main/Sub 분리), C13(INT8 DPE), C14(FFN 전체 구간
pipeline)였다. C14-A의 후속 실행 결과는 위 최신 작업에 기록했다.

## 사용자 지적 이후 변경 범위 재검토

실제 candidate diff와 호출 경로를 다시 읽었다. C09/C11을 C10과 묶어서 세 개의 큰
아키텍처를 충분히 시험한 것처럼 설명한 것은 범위를 과장했다. 이번 실험의 정확한
분류는 다음과 같다. 기존 소스·빌드·판정 기록은 그대로 보존한다.

| 후보 | 그대로 남긴 구조 | 실제 변경 범위 |
|---|---|---|
| C09 | 두 cluster·16-column BF16 dot, F32 합산·scale, 후단 RMS/residual | `project_output`의 행별 전송/계산 의존성을 두 구간으로 분리한 국소 dataflow 변경 |
| C11 | E5M2 세 항 입력, FP4→FP8 weight DM, up/gate→GeGLU→down 전체 단계 의존성, down 구현 | `project_rows_with_scale_contraction`의 partial DM을 VRF로 바꾼 국소 dataflow 변경 |
| C10 | Q/K/V 개별 projection, 입력 전체 복제, 기존 head 후처리 | Weight/activation의 TRF·stream 역할과 row/lane 배치를 교체한 projection 재설계 시도. Lowering 실패로 성능 미평가 |

C07 native FP8/scale 계산 조합과 C08 QKV 통합·열 분할은 더 큰 변경을 실제로
시도한 이력이 있다. 그러나 build·parameter 변형 수를 독립 아키텍처 탐색 수로
해석할 수 없다. 이력만으로 현재 구조가 최적에 가깝다거나 상위권의 구현 방식을
안다고 결론 내리지 않는다.

최근 결과가 주는 정보는 제한적이다. C09는 추가 준비/전송이 overlap을 상쇄했고,
C11은 dot 2→30개로 실행 비용이 늘었다. 이들은 그 구현의 compiler schedule 관측이다.
NPU에서 해당 비율로 느려진다고 확정하거나 전체 설계 계열을 폐기하는 근거가 아니다.

다음 큰 설계는 전체 projection 또는 FFN 실행 graph를 먼저 쓰고, 상속한 데이터
소유권·weight 중간 저장·scale 합산·stage 완료 대기 중 무엇을 없앨지 명시한다.
기존 함수 내부 한 곳의 수정으로 설계 범위를 축소하지 않는다. 작은 지원 여부 검사는
그 큰 설계의 일부를 검증하는 용도로 사용한다. 빠른 종료 규칙은 유지한다.
C12도 현재 제안 범위는 국소 자원 배치 변경이므로 자동으로 다음 대형 설계로 세지 않는다.
C14의 FFN 단계 간 의존성 재설계와 C13의 계산 표현 교체처럼 넓은 가설을 우선 검토하되,
실제 개선 여부는 아직 미검증이다. 이번 재검토에서 새 kernel을 구현·실행하지 않았다.

## 소스 선택의 E104 고정 제거

소스 선택 수정 브랜치는 `fix/recipe-driven-experiments`다. 후속 지적을 받고 실제 빌더를
확인한 결과, `structural_build.py`는 JSON의 `base_commit`을 사용하지 않고 E104 상수를
`common.BASE`에 대입하고 있었다. 변경하지 않은 image의 검사도 E104로 고정돼 있었다.
이 고정은 내가 만든 구현 선택이며, 기록만으로 조합 탐색이 수행됐다고 볼 수 없다.

빌더가 recipe의 출발 commit과 variant별 파일 선택을 실제로 사용하도록 수정했다.
성능 비교군은 `comparison_recipe`로 분리했다. 과거 build를 image 기준으로 쓸 때는
선택한 출발 source와 일치하는지 검사한다. 같은 destination의 중복 덮어쓰기는 거부한다.
[실제 입력 형식과 조립 범위](combination-template.md)를 따른다.

검증: E55/E79/E104 source 선택과 E104 비교군의 독립성, variant별 source 선택,
기준 image의 잘못된 승계 거부, 중복 destination 거부, 기존 세 recipe 호환성,
실제 Git archive→변경 파일 복사 결과까지 **6개 테스트 PASS**다. 실제 조립 검사는
첫 Docker 호출 직전에 멈추도록 만든 테스트다. 새 compiler/NPU/공식 제출 실행은 없다.
기존 실험의 frozen source/log/build JSON과 채택 E104는 변경하지 않았다.

이 수정은 선택한 조합을 빌더가 따르게 만든 것이다. 과거 method ID를 자동으로
서로 호환되는 코드로 합치는 기능은 아니다. 다음 실험은 출발 구조·과거 방법·제거할
충돌 요소의 조합부터 정하고, 그 조합을 담은 실제 코드와 JSON이 일치하는지 확인한다.
# 2026-09-27 실행 후속: E104 QKV/O 실측

[D-E104-QO](../../moa-dev/experiments/candidates/e104-qkv-output-profile/README.md)는
동결 E104의 task bytes를 유지한 내부 계측이다. Arena104914/104933의108개 비교가
통과했고76개 QKV/22개 O engine을 source와 대조했다. O dot은 weight 준비 직후,
Q dot은 weight와 table 준비 직후 시작한다. Q 입력 준비 뒤20,118 / O 입력 준비
뒤20,955 cycles 대기가 관측된다. 입력 방송 제거 시간을 전체 절감으로 계산하지
않는다. K/V의 Main 순서와 Q table 준비도 다음 설계에 포함한다.

[M06 timeout 감사](../research/m06-timeout-mechanism-20260927.md)는 generic code4,
공통 global-Max/scalar-broadcast 경계까지 특정했다. Bank starvation을 입증할
장치 증거가 없어 특정 수리 없이 성능 실험을 재개하지 않는다.
