# MOA 2026 작업 공간

대회 코드와 Docker 개발 도구를 하나의 Git 저장소에서 관리한다.
에디터에서는 이 폴더를 열고, 커널 코드는 `furiosa-opt-gemma4-12B/`에서 수정한다.

상위권을 목표로 한 조사 근거, 재현 가능한 기준 측정, 실험 순서는
[최적화 실행 계획](docs/OPTIMIZATION_PLAN.md)에 정리한다.

[현재 baseline 구성](baseline/README.md)에서 E134에 적용한 16개 구성요소의 코드·설정·의존 관계를
확인할 수 있다. [성능별 실험 분류](docs/experiments/optimization-results.md)는 빨라진 후보,
차이 미확인, 회귀, 정확도 실패를 비교 기준과 함께 정리한다.
새 조합은 [추가·제거·변경 기록 양식](docs/experiments/combination-template.md)을 사용한다.

현재 실행 소스는 **E134 / `baseline-e134` / `4f8fe7f`**다.
[Weighted native down](moa-dev/experiments/candidates/ffn-native-down-weighted-residual/README.md)을
검증해 채택했다. Down의 전체 FP8 weight DM 쓰기·읽기를 각각56.25MiB 제거했고,
양자화·방송·정밀도 복원 비용을 포함한 전체 FFN이 E128 대비 독립3회 모두 빨라졌다.
Paired 감소율 중앙값은 **4.4518%**다. QKV/O image는 유지한다.

초기 E133은 빨랐지만 작은 값 소실로 채택을 보류했다. E134는 잔차 저장과 native 합산의
배율을 함께 바꿨다. 표준9입력·실제down9case34560출력이 통과했고, 동일입력 전체FFN 반례에서
E133의1920개 허용오차 실패를 재현한 반면 E128/E134는3840개 모두 정답과 일치했다.
유한한 검증이며 임의BF16/전체모델 정확성 보장은 아니다. 실패한prototype과수리근거를보존한다.

E129–E132의 lowering실패와 실측회귀도 [성능별 분류](docs/experiments/optimization-results.md)에 남겼다.
[현재 작업](docs/experiments/current-work.md)과 [작업 규칙](AGENTS.md)에 채택·튜닝·연구 결정을 구분한다.
E134의 공식 제출·순위는 완료 후 별도 기록하며, Arena 이득을 공식점수로 취급하지 않는다.
아래는 과거 채택·실험 이력이다.

이전 **E104 / `baseline-e104` / `7f891f9`**는
[C07](moa-dev/experiments/candidates/c07-native-scale-dpe/README.md)의 materialized native
E5M2 up/gate를 채택했다. 동시 E79 대비 독립3회 FFN 중앙값은353154→335761(−4.93%)였다.
C13 v7은 기본 NPU PASS 후 E104 대비18.58% 느렸고, C14-A는 lowering 실패였다.
C14-B는 수정 후 실제 NPU PASS지만17.58% 회귀로 종료됐다. 컴파일 실패와 실측 회귀를 구분한다.

`opt/ffn-dataflow`의 후속 [메모리 배치 실험 E57–E64](docs/research/e55-memory-layout-audit.md)은
QKV·O·FFN 변경 후보 8개를 Arena에서 실행했다. 정확도 검사는 통과했지만 반복 가능한 성능
개선을 확보하지 못해 당시 runtime은 E55를 유지했다. 결과·실패 원인·재현 source를 함께 보존했다.

후속 [C02/C03 실험](moa-dev/experiments/candidates/c02-c03-e55/README.md)도 완료했다.
C02 F32 packet scale의 up/gate·down·결합은 단회 FFN +14.10%/+9.52%/+20.28%,
C03 cluster별 head 유지는 반복 QKV +1.56%로 모두 미채택이다. 기본 입력은 모두 통과했고,
별도로 C02 block 상쇄 진단과 C03 추가 입력·cache 경계도 통과했다.
소스·실패 경로·업로드 fingerprint를 보존했다.

이전 채택본 **E79**는 [C04 실험](moa-dev/experiments/candidates/c04-down-transfer/README.md)에서
2/4/8열·Main/Sub 방송·입력 gather를 결합한9개 후보를 NPU에서 비교했다.
8열+leader DMA/Main ring32 방송이 동시기 E74 대비 **369997 → 353136 cycles, −4.56%**로
독립3회 개선됐다. 기본45개·추가30개 비교와 projection 출력138240개를 통과했다.
`baseline-e79` / `2735b58`, [구성 변경표](docs/experiments/combinations/e79-down-broadcast.md).
공식 제출 **`df79b0b9`**는 **5.8470×**로 채점 완료됐다. 공식 FFN은**354648 cycles**다.
[채점 기록](docs/experiments/e79-official-submit-20260922.json), [리더보드 snapshot](docs/experiments/leaderboard-e79-20260922.json)의
BEEF는 **60팀 중41위**이며 새 제출이 반영됐다. 이전5.7805×보다 점수는 올랐지만 순위는 유지됐다.

이전 채택본은 **E74**다. [C05 실험](moa-dev/experiments/candidates/c05-streamed-decode/README.md)에서
전체 BF16 weight 대신 작은 F32 block dot을 저장하는 구조를 채택했다. 동시기 E55와
독립3회 비교해 FFN **377,804 → 369,246 cycles, −2.27%**다. 기본45개·추가30개 비교와
full-shape/cancellation 진단138,240개 출력이 통과했다. QKV/O image는 E55 그대로다.
`baseline-e74`는 source commit `9e2058e`이며 E55 태그·실험 원본도 보존한다.
공식 제출 **`01ba2ec2`**는 **5.7805×**로 완료됐다. 공식 FFN **369,335 cycles**와
[채점 기록](docs/experiments/e74-official-submit-20260922.json)은 Arena 측정과 별도로 보관한다.

E55/E74는 비교군으로 활용한다. 후보를 그 구조에서 시작하거나 가까운 변경만 탐색해야
한다는 제약은 없다. 다음 후보는 layout·decode·dot·scale·tile·통신을 함께 바꿀 수 있다.

아래는 QKV/O 기반이 된 **이전 E55**의 검증과 채택 당시 성능 기록이다.
동일 패키지3회에서 공식 입력3세트의15개 비교를 각각 통과했고, 별도 추가6세트의30개 비교도 통과했다.
QKV 상쇄 진단2개도 통과했고 비대상 KV cache8,380,416개 BF16 원소가 유지됐다.
같은 진단에서 E53은8,072개 출력이 허용오차를 벗어났다.
이 진단 입력은 현재 공식 QKV generator 지원 범위 밖이며, 공식 평가 실패와 구분한다.
[원본·E53·E54 비교와 E55 후속](docs/experiments/qkv-boundary-comparison.md).

아래는 이전 E55의 동일 패키지3회 실행에서 구한 중앙값이다.
E55는 job71027/71057/71072 모두15개 비교를 통과했다. 원본/E54 열은 이전 반복 기록이다.
전체 Gemma 추론 속도나 공식 순위를 뜻하지 않는다.

| 커널 | 원본 반복 중앙값 cycles | 이전 E54 반복 중앙값 cycles | E55 반복 중앙값 cycles |
|---|---:|---:|---:|
| QKV | 244,051 | 108,700 | 102,878 |
| Attention output | 404,222 | 50,970 | 51,249 |
| FFN | 4,264,297 | 382,268 | 380,339 |

E55는 원본 대비 QKV2.37×/O7.89×/FFN11.21×다. QKV cycles는 E54보다5.36% 낮다.
입력 근사가 있던 E53의95,621 cycles보다는7.59% 높으며, 관측된 상쇄 오류를 피하는 비용을 남긴다.
[반복 측정](docs/experiments/e55-fused-qkv-repeats.json)의 job 중앙값 범위는
QKV0.976%/O1.169%/FFN1.036%다. O/FFN은 E54와 같은 image이므로 cycle 차이를 소스 변경 효과로 해석하지 않는다.
[첫 screen](docs/experiments/e55-fused-qkv-screen.json),
[추가 입력 검사](docs/experiments/e55-fused-qkv-extra-correctness.json),
[실험 원장](docs/experiments/README.md)에 결과와 업로드 해시를 기록한다.

E55 QKV는 BF16 pre-RMS → full-H gather·ring32 복제 → BF16 dot → 같은 Main에서
완전한 F32 dot에 channel scale 곱하기 → 첫 BF16 변환 → row Switch → head RMS·RoPE·cache write 순서다.
E54의 별도 F32 재배치·scale pass를 합쳤으며 O/FFN device image는 E54와 같다.
O 경계8개 PASS는 같은 O image를 실행한 **기존 E53 job70810의 근거를 승계**하며,
새 E55 O 경계 실행이라고 세지 않는다. [O 경계 비교](docs/experiments/output-boundary-comparison.md).
[빌드 검사](docs/experiments/e55-source-build-audit.json)에서 독립 빌드와 live 빌드의 host가
byte-identical하며, 별도 추가 입력 빌드도 세 device image가 모두 일치함을 확인했다.
[소스 계약 검사](docs/experiments/e55-source-contract-audit.json)는 공개 signature와 공식 harness 보존을 확인한다.
현재 검증 범위를 넘어 임의 입력이나 모델 전체의 정확성을 보장하지 않는다.

9월 22일 공식 제출 **BEEF / `3ab72d66`의 채점이 완료됐고 E55는 5.6665×를 받았다.**
공식 cycles는 QKV **105,623**, O **51,662**, FFN **378,116**이다.
23개 업로드 소스가 검증본과 같고 서버의 skeleton contract도 통과했다.
[접수·채점 추적 기록](docs/experiments/e55-official-submit-retry-20260922.json)에 결과를 보존했다.
이 기록에는 공식 순위가 포함되지 않으며 최소3위 목표는 아직 미달성이다.
이전 [점검 중 거절 기록](docs/experiments/e55-official-submit-attempt-20260922.json)은 이력으로 보존한다.

```text
moa-workspace/
├── furiosa-opt-gemma4-12B/   # 공식 대회 코드와 빌드 결과
│   ├── src/ops.rs
│   ├── src/device/
│   └── scripts/rngd_test.sh
└── moa-dev/                 # Docker 설정, 실행 도구, 측정 기록
    ├── Dockerfile
    ├── moa
    └── README.md
```

## 개발 및 테스트

작업 공간의 루트에서 개발 셸을 실행한다.

```bash
./moa-dev/moa
```

Docker 내부의 `/workspace`는 호스트의 `furiosa-opt-gemma4-12B/`와 같은 파일을
보는 경로다. 개발 셸에서 다음 명령으로 수정한 코드를 빌드하고 Arena에서 검사한다.

```bash
RNGD_SUBMIT_TIMEOUT=70 ./scripts/rngd_test.sh
```

70초는 2026년 9월 21일 확인한 Arena 작업 실행 시간 상한이다.
도구 설치, 로그인, 정적 스케줄 확인, 공식 제출 명령은
[개발 환경 사용법](moa-dev/README.md)에 있다.

최적화할 코드와 변경 가능 범위는
[대회 README](furiosa-opt-gemma4-12B/README.md)를 따른다.
원본 코드로 통과한 Arena 작업 `69783`의 성능은
[기준 측정 결과](moa-dev/baseline-result-69783.json)에 기록되어 있다.

## Git 관리

Git 루트는 이 작업 공간이며, 두 하위 폴더에 별도 저장소나 서브모듈은 없다.
작업 원격 `origin`은 비공개 저장소
[jihwan38/moa-workspace](https://github.com/jihwan38/moa-workspace)다.
공식 저장소의 기존 커밋 이력을 유지했고, 공식 원격은 `upstream`으로 등록했다.
Git 명령은 이 폴더에서 실행하면 된다. 이 작업 공간의 `git push`와 `gh` 기본 대상은
작업 저장소 `jihwan38/moa-workspace`로 설정했다.

`moa-dev/state/`의 로그인 정보와 실행 로그는 Git에서 제외한다.
대회 폴더의 `target/`, `ref/`, `Cargo.lock`은 기존 대회 `.gitignore` 규칙을 따른다.
