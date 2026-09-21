# RNGD Gemma 최적화 실행 계획

목표는 정확한 구현으로 MOA 상위 3위에 진입하는 것이다. 현재는 1라운드 세 커널의
기준선과 병목을 확보했고, FP8 fusion·O 행 분산·FFN direct layout의 NPU 정확성 통과를 확인했다.
후속 후보는 아래 절차로 독립 구현·검증한다. 목표 달성 여부는 공식
제출 결과와 최신 순위로 판단한다. 조사일은 2026-09-21이다.

## 현재 확보한 기준선

공식 소스 `8559916`의 동일 바이너리·fixture·entrypoint·실행 manifest를 Arena job
`69783`, `69913`, `69931`에서 실행했다. 패키지를 다시 내려받아 파일별 SHA256이
같고, 처음 기록한 바이너리·fixture 해시와도 일치함을 확인했다. 세 job 모두 정확성 통과다.

| 커널 | job별 중앙값의 중앙값 | job 중앙값 최소–최대 | 상대 범위 |
|---|---:|---:|---:|
| QKV | 244,051 | 243,791–244,520 | 0.299% |
| Attention output | 404,222 | 403,700–404,373 | 0.166% |
| FFN | 4,264,297 | 4,264,096–4,268,493 | 0.103% |

[전체 실험 기록](experiments/baseline-rngd.json)에 개별 입력 cycle, 정확성, source provenance,
실제 업로드 package hash, 로그 hash, job 시각을 보존했다. 초기 측정의 작은 차이를
최적화로 오인하지 않도록 이 변동 폭을 기준으로 삼는다.

## 실측 진행 상황

최신 **E37**은 [job70519](experiments/e37-kernel-specific-rms.json)에서
QKV94,094/O45,796/FFN376,040 cycles, 15개출력을모두통과했다.
QKV deviceimage는E29, O/FFN은E36과byte-identical하다. 결합패키지의반복/추가입력검사는별도진행한다.

직전 조합 **E36**은 [job70498](experiments/e36-ffn-parallel-epilogue.json)에서
QKV97,052/O45,141/FFN378,021로15개출력통과, 추가run3..8도30개출력을 모두통과했다.
FFN의down이후값을H480×8로유지해 globalScale/postnorm/residual의중복gather/scatter를줄인다.
행렬계산과BF16경계는유지하지만postnorm합순서는바뀐다. FFN단일screen은E31반복중앙값보다2.91%낮다.
E29재측정Q중앙값94,491을재현해, Q에만이전RMS를쓰는 E37조합을별도평가한다.

반복 검증 후보 **E31**은 [job70441](experiments/e31-switch-gather-rms.json)에서
QKV **98,797** / O **45,355** / FFN **389,335** cycles로 공식15개출력을 통과했다.
RMS 부분합8개를 Switch로 모아 local sum하고 별도 EPS/Sqrt 및 scalar broadcast를 수행한다.
추가run3..8도 [job70448](experiments/e31-rms-extra-correctness.json)에서30개출력을 전부통과했다.
동일패키지3회(70441/70459/70465)가 모두통과했고 중앙값은 **97,085 /45,355 /389,335**다.
원본 반복중앙값 대비 **2.514× /8.912× /10.953×**이며 job중앙값 범위는2.714%/1.965%/0.177%다.
[반복 근거](experiments/e31-rms-repeats.json)에 실제 업로드4파일의 동일SHA를 확인했다.
직전E29(Q94,491/O49,352/FFN389,199)는단일screen이므로 작은Q차이를확정하지않는다.

추가 입력까지 통과한 checkpoint **E27**은 [job70382](experiments/e27-qk-pair-v-two-dot.json)에서
QKV **97,247** / O **52,543** / FFN **386,729** cycles를 기록했고,
공식 harness의 15개 출력 비교를 모두 통과했다. 원본 반복 중앙값 대비
**2.510× / 7.693× / 11.027×**이며 아직 독립 job 반복 전의 단일 측정이다.
같은 snapshot의 별도 개발 harness run3..8도 [job70409](experiments/e27-qkv-fp8-extra-correctness.json)에서
30개 출력 비교를 통과했다. 이 추가 검사는 다른 binary/fixture를 사용하며 공식 cycles 집계에 섞지 않는다.

E27은 Q/K에 native FP8 hi/lo two-lane, V에는 두 별도 dot을 사용한다.
O는 E26의 H480×8 배치에서 channel-scale/RMS/residual을 처리해 중간 gather/scatter와
F32 DM 왕복을 제거한다. FFN은 검증된 BF16 blockdot 경로를 유지한다.
E24·E26도 각각 별도 추가 입력 검사를 완료했다. FP8 분해는 일반 입력에 대해 비동치이고
outlier/cancellation 반례가 있어, 이 통과 결과를 임의 입력의 허용오차 보장으로 확장하지 않는다.

후속 [E28 job70410](experiments/e28-explicit-pair-identity.json)은 QKV/O가
98,428/53,363 cycles로 통과했지만 첫 FFN launch가 timeout으로 종료했다.
FFN 출력·cycles와 전체 유효 점수는 없다. Pair의 group0에 명시적으로 ×1을 넣는 변경으로
timeout이 해결되지 않았으며, rank0 표기는 실패 instruction이나 cluster를 특정하지 않는다.
[E29 job70428](experiments/e29-sharded-qkv-packed-output.json)은 sharded QKV 입력과
direct-packed O quantizer, 기존 safe FFN을 결합해 전체 정확성을 통과한 단일 screen이다.
검증된 checkpoint와 실험 중인 live 소스를 구분한다.

다음은 이전 검증 단계와 실패 후보의 근거다.

2026-09-21 job `70181`에서 QKV **115,780** / O **65,476** / FFN **397,675** cycles와
전체 정확성 통과를 확인했다. 원본 대비 각각 약 **2.11× / 6.17× / 10.72×**다.
QKV는 두 cluster와 head별 정규화를 사용하고, O는 두 cluster·8개 column 분할과
FP32 inter-slice reduction을 사용한다. FFN은 세 행렬을 두 cluster에 배치하고,
FP4→FP8 DM→unscaled BF16 DM 변환을 재사용해 16-element dot 이후 local scale을 적용한다.
후자는 FP32 합산 순서를 바꾸므로 추가 입력 검사를 별도로 수행한다.
동일 패키지 세 job(70181/70194/70200)의 중앙값도 위 수치였고 전부 통과했다.
추가 run3..8의 30개 출력 비교도 job70186에서 통과했다. Job 중앙값 범위는 QKV4.25%,
O2.50%, FFN0.27%여서 작은 QKV/O 개선은 반복 검증한다.
후속 E16은 residual/layer-scale 병렬화로 FFN388,813 cycles를 기록했고 전체 정확성 통과다.
E17의 down2-column/HBM-relay 변형은416,500으로 느려져 제외했다.

이전 E5 four-row snapshot은 동일 패키지 세 job과 추가 run3..8의 30개 출력 비교를 모두
통과했다. 후속 소스의 검증을 이 이전 결과로 대신하지 않는다.
[실험 원장](experiments/README.md)과 개별 JSON이 최신 채택·실패·변동 근거다.

FFN two-row 출력 경로 두 개는 같은 정확성 실패를 보여 제외했다. Blockscale-after-dot은
FP4×FP8의 BF16 표현 가능성을 확인하고 NPU 정확성도 통과했지만, up/gate에만 적용한
버전은 반복 lookup table 적재 때문에 FFN이 E6보다 12.37% 느려 그대로 채택하지 않았다.
QKV lane 병렬화도 weight BF16 materialization과 TRF 적재 때문에 정적 13.7% 회귀했다.
FP4 lookup→F32→BF16 단일-pass decode는 컴파일되지만 NPU FFN 정확도 실패와 timeout이
발생해 제외했다. 수학적 표현 가능성과 실제 compiler/실행 경로의 정확성을 구분한다.
실패 후보와 근거도 보존하며, compiler schedule 개선만으로 실제 speedup을 주장하지 않는다.

E18의 QKV 8-way row interleave는 동일 패키지3회(job70248/70271/70285) 모두 통과했고,
중앙값은 QKV105,462/O65,662/FFN392,608 cycles다. 원본 대비2.314×/6.156×/10.861×이며,
job 중앙값 변동범위는1.599%/1.188%/0.792%다. 추가run3..8도 job70309에서30개출력 전부통과했다.
새 Docker image의 빈 target에서 같은 E18 소스를 빌드한 결과도 기존 NPU 바이너리와 byte-identical이다.
E19의 runtime power-of-two FP8 hi/lo 두 내적도 통과했지만110,479로 E18보다 느리다.
이후 hi/lo 두 TRF lane, row interleave 결합, O normalization fusion은 E20–E27에서
분리해서 검증했다. E20/E21의 two-lane QKV는 Q/K 통과·V 실패였고 E23의 register 재적재도
해결하지 못했다. 별도 진단 job70363은 각 cluster의 첫 8 slices / local row0에 큰 V 오류가
집중되는 패턴을 보였다. [진단 분석](../moa-dev/experiments/candidates/qkv-fp8-two-lane-diagnostics/analysis-job-70363.md)은
producer/allocator 결함을 확정하지 않으며, E27의 두-dot V 우회가 통과해도 원인이 증명되는 것은 아니다.
E21의 O native FP8는3개입력 전부통과,59,016 cycles였지만 결합job은 QKV때문에 실패했다.
이 O 경로를 safe QKV/FFN과 결합한 E24는 전체 및 추가 입력 검사까지 통과했다.
E30의 clean paired QKV control도 통과했으므로 E28의 명시적 ×1은 통과에 필수이지 않았다.
FFN native FP8는 E22/E25/E28 및 pair-free E32의 첫 launch timeout 때문에 채택하지 않았다.
E33의 down BF16 scale-DPE는 큰 정확성 오류(허용오차 통과12.94–16.17%)로 제외했다.
후속은 safe FFN 후단 H480×8 유지와 native FP8의240유효slice/256물리slice 경로 분리다.
실제 Arena job70431에서 driver2026.4.0~rc2와 firmware2026.2.0을 별도로 확인했으며,
이 사실만으로 compiler/driver 결함을 확정하지 않는다.
상위 3위 달성을 입증한 공식 결과는 아직 없다. 공식 제출·순위 확인은 별도 인증 흐름을 따른다.
공개 상위 기록은 9월21일 입력·평가 환경 변경 이전 제출이므로 같은 환경의 성능으로 단정하지 않는다.

## 조사에서 얻은 설계 근거

- [공식 평가 계약·현재 목표선](research/moa-rules-and-target.md)
- [RNGD 구조와 furiosa-opt 0.8.1의 실제 제약](research/rngd-furiosa-opt.md)
- [세 커널의 실제 호출·데이터 흐름과 정적 병목](research/kernel-analysis.md)
- [국내외 대회·수상 구현·논문에서 이전 가능한 기법](research/competition-techniques.md)

다음 수치는 **최적화 전 원본 schedule**의 병목이며 현재 후보의 잔여 비용이 아니다.
QKV의 정적 118,815 cycles 중 activation ring broadcast가 62,215다. O는 정적
195,823 중 broadcast 66,311, FP8 conversion 77,852다. FFN은 정적 1,918,472 중
down weight assembly copy 486,900, layout redistribution 256,500이 명확한 제거 대상이다.
이 값들은 compiler schedule의 구간이며 실제 절감량을 더해서 예상하지 않는다.

## 초기 실험 계보와 현재 우선순위

| 실험 | 한 번에 검증할 가설 | 정확성 위험 | 다음 단계 |
|---|---|---|---|
| E1 TDMA broadcast | QKV/O의 `Slice → Replicated`를 TDMA로 수행해 긴 main ring 작업 제거 | 낮음: 값·dtype·연산식 동일, 실제 broadcast 검증 필요 | exact schedule → 전체 test → 원본 대비 측정 |
| E2 FP8 stream fusion | FP8 lookup 뒤 BF16 DM store/refetch 없이 contraction에 연결 | 낮음: scale·reduction·BF16 경계 유지 | QKV/O 각각 schedule, 전체 test |
| E3 O row distribution | 32개 유효 row groups 대신 256 slices에 output rows 분산, full reduction | 중간: 매핑·tail padding·누적 반올림 변화 | 정확한 축 매핑 확인, oracle 비교, RNGD |
| E4 FFN two-row | down weight 4-row assembly를 없애고 2-row tile 즉시 contract | 낮음~중간: BF16/reduction 유지, 최소 commit packet 대응 | 제거된 copy 확인, 전체 test |
| E5 FFN direct layout | packed weights와 scales부터 column-split 배치로 복호화 | 중간: FP4 nibble·16-element scale group·배치 의미 | split group 동치, schedule, RNGD |
| E6 tile/unroll/overlap | pass 크기·반복 전개로 staging과 계산 중첩 | 중간: capacity·padding·compile 비용 | 작은 탐색표, 컴파일 실패 포함 기록 |
| E7 cluster/epilogue | 현재 padding인 cluster 활용, RMSNorm/RoPE/residual 중간 이동 축소 | 높음: 축 분할·reduction·부수효과 | 앞 단계 병목 재측정 후 별도 설계 |

E1–E7은 구현·정적 비교·NPU 측정을 거쳤고 채택·기각 내역은 실험 원장에 기록했다.
각 후보는 직전 소스 대비 변경을 분리하고, 결합한 바이너리는 다시 실제 NPU에서 검사한다.
개선분의 단순 합이나 곱을 조합 결과로 대신하지 않는다.

현재 후속 작업의 순서는 다음과 같다.

1. **E29 추가 정확성 검사:** 정적 QKV 51,710→50,364 / O 29,088→26,686 후보는
   job70428에서 전체 15개 비교를 통과했고 QKV/O의 단일 측정도 개선됐다.
   이제 그 snapshot에 추가 run3..8을 적용한다. E27 extra 결과로 E29를 검증했다고 쓰지 않는다.
2. **최종 통과 조합의 반복 측정:** E29 또는 이를 앞선 새 조합의 동일 package를 독립 job으로 반복한다.
   E27/E29의 측정 JSON에 있는 `relative_range=0`은 job 하나뿐이라 생긴 값이며 안정성 증거가 아니다.
   작은 개선은 동일 환경의 A/B 측정과 비교한다. 추가 seed의 진단 cycles도 반복 측정 대용이 아니다.
3. **FFN native 경로는 원인 분리부터:** E22/E25/E28 timeout을 반복하는 대형 sweep 대신
   [pair-free 두-dot 진단](../moa-dev/experiments/candidates/ffn-native-fp8-two-dot-diagnostic/README.md)처럼
   native packet4를 유지하면서 Sequential/pair 경로를 제거한 대조로 범위를 좁힌다.
   compiler PASS만으로 성공을 예상하지 않으며 검증된 BF16 FFN checkpoint를 유지한다.
4. **실행 환경 관찰:** [독립 버전 probe](../moa-dev/experiments/candidates/arena-version-probe/README.md)로
   Arena의 driver/firmware를 기록한다. 조회 실패는 미확인이며, 공개 포럼의 과거 사례를 현재 오류의 원인으로 단정하지 않는다.
   probe와 좌표 출력 진단은 공식 harness 성능표에 넣지 않는다.
5. **공식 평가와 다음 병목:** 최종 source diff가 제출 계약을 만족하는지 확인하고 공식 평가를 별도로 확보한다.
   이후 통과 조합의 schedule에서 FFN 후단 이동/RMS/residual과 남은 QKV/O 이동을 다시 추적한다.
   O weight row interleave 2-way/4-way는 scratch compile만 완료했고 동일 DMA에 Switch 비용이 늘어
   정적으로 각각 3.79%/6.77% 회귀했으므로 후순위다. 실제 HBM bank 효과는 아직 측정하지 않았다.

실제 compiler/SDK 제약도 실험표에 남긴다. 예를 들어 BF16 2개를 그대로 commit하면
4-byte packet이 되어 거절되므로 허용되는 padding/commit 경로가 필요하다. 또한 15-row
타일은 논리 축과 물리 padding을 동시에 맞춰야 하며 unsafe reshape로 permutation을 숨기지 않는다.

## 각 실험의 실행 절차

제출 계약은 `src/ops.rs` 함수 본문과 `src/device/` 구현에 한정한다. 기존 `#[device]` 함수의
이름·인자·타입·반환형과 module 경로를 보존한다. [공식 계약 대조](research/moa-rules-and-target.md)의
입력 생성·모든 원소 비교·허용오차를 그대로 사용하며, fixture/seed나 진단에서 발견한 특정 좌표에
맞춘 분기·출력 보정으로 통과시키지 않는다. 개발용 harness 변경은 별도 snapshot에만 둔다.

1. 원본 또는 직전 최선의 source/patch hash를 고정한다. 변경 범위와 수학적 동치를 적는다.
2. 영향받는 커널만 `cargo furiosa-opt compile ops::<kernel> --exact`로 검사하고 별도 schedule을 저장한다.
3. `cargo furiosa-opt build --release --bin test_kernels`로 공식 바이너리를 만든다.
4. 바이너리·fixture·소스·toolchain·schedule hash를 기록하고 Arena에 한 job을 제출한다.
5. 공식 세 커널, 세 입력 구성, 모든 출력이 통과해야 성능 후보로 인정한다.
6. 원본 대비 커널별 비율과 기하평균을 계산한다. 변동과 가까운 개선은 A/B 재측정한다.
7. 최종 조합은 동일 업로드 패키지를 독립 job으로 반복 측정한다. 다른 artifact의 결과를 섞지 않는다.
8. 통과한 후보를 `moa-submitter`로 평가하고 공식 점수·현재 3위와의 격차를 갱신한다.

예를 들어 이미 제출한 job의 재현 측정은 다음과 같이 수행한다.

```bash
python3 moa-dev/measure.py --jobs JOB_ID --reruns 2 \
  --label candidate-name --output docs/experiments/candidate-name.json \
  --provenance PATH_TO_SOURCE_AND_ARTIFACT_RECORD.json \
  --leaderboard-snapshot docs/experiments/leaderboard-20260921-initial.json
```

collector는 각 job의 실제 archive를 내려받아 실행 파일·fixture·entrypoint·manifest가
동일한지 확인한다. 잘못된/누락된 correctness·cycle 로그나 다른 artifact의 집계는 실패로 기록한다.
공식 점수 대신 계산하는 값은 `indicative_score`로만 표시한다.
원시 로그·대용량 schedule·실행 파일은 Git 제외 경로에 두고 source patch와 요약 결과는 저장소에 남긴다.

누적 순서나 정밀도가 바뀌는 후보는 별도 개발 harness에서 추가 입력 검사를 보강한다.
공식 fixture나 grader를 바꾸는 방식으로 후보를 통과시키지 않는다.

## 2라운드로 이어갈 작업

1라운드에서 만든 layout·fusion을 전체 모델에 적용할 때는 shared helper를 쓰는 full
attention·vision·audio 경로를 다시 검사한다. 실제 모델 경로는 `src/host/runtime.rs`가
각 device kernel과 KV state를 조율한다. 단일 커널 성능을 E2E token 속도로 환산하지 않는다.

2라운드 workload가 확정되면 모델 준비/초기화, prefill TTFT, decode TPOT, KV 메모리,
host-device 동기화, NPU 시간을 분리해 측정한다. FlashAttention 계열의 IO 관점은 이때
실제 attention 경로의 후보로 사용한다. 다른 대회의 재양자화·가지치기·CUDA Graph 최적화를
현재 세 커널의 해법으로 취급하지 않는다.

공식 채점 서버 인증, 과거 `BEEF` 항목의 소속·소스, 변경 중인 평가 기준은 별도 확인 항목이다.
이들 확인을 기다리는 동안 Arena의 소스 분석·정확성·성능 실험을 진행한다.

공식사이트의 제출중단 공지를 2026-09-21 11:04 UTC경 다시확인했다. 평가기준수정 중에는
공식제출이불가하고 Arena개발은가능하다고안내한다. 현재기록은공식순위가아니다.
[공식공지](https://micro2026-moa.github.io/leaderboard.html), [확인기록](experiments/official-submission-status-20260921.json).

FP8 FFN의E35full256변형은전체15개출력을통과했다. 단,811,219cycles여서현재채택하지않는다.
240→256유효slice,행interleave,실제zero tail,중간copy와lifetime이함께바뀌므로padding만이원인이라고단정하지않는다.
이통과경로의대형copy/전체zero-fill부터독립제거하고,메모리배치비용을다시측정한다.
[원인구분](experiments/e35-full256-native-cause-audit.md), [잔여병목](../moa-dev/experiments/candidates/ffn-parallel-epilogue/remaining-bottlenecks.md).
