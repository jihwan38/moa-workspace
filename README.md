# MOA 2026 작업 공간

대회 코드와 Docker 개발 도구를 하나의 Git 저장소에서 관리한다.
에디터에서는 이 폴더를 열고, 커널 코드는 `furiosa-opt-gemma4-12B/`에서 수정한다.

상위권을 목표로 한 조사 근거, 재현 가능한 기준 측정, 실험 순서는
[최적화 실행 계획](docs/OPTIMIZATION_PLAN.md)에 정리한다.

[현재 baseline 구성](baseline/README.md)에서 E55에 적용한 15개 방법의 코드·설정·의존 관계를
확인할 수 있다. [성능별 실험 분류](docs/experiments/optimization-results.md)는 빨라진 후보,
차이 미확인, 회귀, 정확도 실패를 비교 기준과 함께 정리한다.
새 조합은 [추가·제거·변경 기록 양식](docs/experiments/combination-template.md)을 사용한다.

`opt/ffn-dataflow`의 후속 [메모리 배치 실험 E57–E64](docs/research/e55-memory-layout-audit.md)은
QKV·O·FFN 변경 후보 8개를 Arena에서 실행했다. 정확도 검사는 통과했지만 반복 가능한 성능
개선을 확보하지 못해 runtime은 E55를 유지한다. 결과·실패 원인·재현 source를 함께 보존했다.

현재 `baseline`은 QKV와 O projection의 입력을 BF16으로 보존하는 **E55**다.
동일 패키지3회에서 공식 입력3세트의15개 비교를 각각 통과했고, 별도 추가6세트의30개 비교도 통과했다.
QKV 상쇄 진단2개도 통과했고 비대상 KV cache8,380,416개 BF16 원소가 유지됐다.
같은 진단에서 E53은8,072개 출력이 허용오차를 벗어났다.
이 진단 입력은 현재 공식 QKV generator 지원 범위 밖이며, 공식 평가 실패와 구분한다.
[원본·E53·E54 비교와 E55 후속](docs/experiments/qkv-boundary-comparison.md).

아래는 동일 패키지3회 실행의 job 중앙값들에서 구한 중앙값이다.
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

9월 22일 공식 제출이 재개돼 검증된 **E55가 BEEF / `3ab72d66`으로 접수됐다.**
23개 업로드 소스가 이전 검증본과 같고 서버의 skeleton contract도 통과했다.
[접수·채점 추적 기록](docs/experiments/e55-official-submit-retry-20260922.json)의 마지막 확인은
원격 빌드 진행 중이며, 아직 공식 점수·순위는 미확인이다. 최소3위 목표도 달성하지 못했다.
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
