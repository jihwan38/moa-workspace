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

다음 주요 검증은 FFN 후처리 병렬화, down column 분할·입력 전달 방식, O column 분할 수 조정이다.
공개 상위3위 목표에는 아직 못 미치며 공식 제출·순위 확인은 별도 인증 대기다.
공개 상위 기록은 9월21일 입력·평가 환경 변경 이전 제출이므로 같은 환경의 성능으로 단정하지 않는다.

## 조사에서 얻은 설계 근거

- [공식 평가 계약·현재 목표선](research/moa-rules-and-target.md)
- [RNGD 구조와 furiosa-opt 0.8.1의 실제 제약](research/rngd-furiosa-opt.md)
- [세 커널의 실제 호출·데이터 흐름과 정적 병목](research/kernel-analysis.md)
- [국내외 대회·수상 구현·논문에서 이전 가능한 기법](research/competition-techniques.md)

QKV의 정적 118,815 cycles 중 activation ring broadcast가 62,215다. O는 정적
195,823 중 broadcast 66,311, FP8 conversion 77,852다. FFN은 정적 1,918,472 중
down weight assembly copy 486,900, layout redistribution 256,500이 명확한 제거 대상이다.
이 값들은 compiler schedule의 구간이며 실제 절감량을 더해서 예상하지 않는다.

## 실험 순서와 판정

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

실제 compiler/SDK 제약도 실험표에 남긴다. 예를 들어 BF16 2개를 그대로 commit하면
4-byte packet이 되어 거절되므로 허용되는 padding/commit 경로가 필요하다. 또한 15-row
타일은 논리 축과 물리 padding을 동시에 맞춰야 하며 unsafe reshape로 permutation을 숨기지 않는다.

## 각 실험의 실행 절차

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
