# MOA 2026 작업 공간

대회 코드와 Docker 개발 도구를 하나의 Git 저장소에서 관리한다.
에디터에서는 이 폴더를 열고, 커널 코드는 `furiosa-opt-gemma4-12B/`에서 수정한다.

상위권을 목표로 한 조사 근거, 재현 가능한 기준 측정, 실험 순서는
[최적화 실행 계획](docs/OPTIMIZATION_PLAN.md)에 정리한다.

현재 `baseline`은 O projection의 입력을 BF16으로 보존하는 **E53**이다.
공식 입력3세트와 별도 추가6세트의 모든 비교, O 경계8개 검사를 통과했다.
이전 E47에서 작은 값이 FP8 근사로 사라지던 상쇄 입력도 원본과 같은 결과를 낸다.
[원본·E47·E53 경계 비교](docs/experiments/output-boundary-comparison.md).

아래는 같은 패키지3회 실행에서 얻은 job 중앙값들의 중앙값이다.
전체 Gemma 추론 속도나 공식 순위를 뜻하지 않는다.

| 커널 | 원본 반복 중앙값 cycles | E53 반복 중앙값 cycles | 원본 대비 |
|---|---:|---:|---:|
| QKV | 244,051 | 95,621 | 2.55× |
| Attention output | 404,222 | 50,882 | 7.94× |
| FFN | 4,264,297 | 377,880 | 11.28× |

[동일 패키지 반복 측정](docs/experiments/e53-bf16-output-repeats.json), [추가 입력 검사](docs/experiments/e53-bf16-output-extra-correctness.json),
[실험 원장](docs/experiments/README.md)에 소스·바이너리·업로드 해시를 기록한다.
E53은 O의 입력 근사와 scale 전 BF16 반올림을 제거했다. QKV/FFN device image는 E47과 같다.
같은 QKV image의 [KV cache 보존 검사](docs/experiments/e37-cache-preservation-correctness.json)에서
비대상 BF16 원소12,570,624개가 유지됐다.
[빌드 검사](docs/experiments/e53-source-build-audit.json)에서 정리한 소스로 측정 실행 파일을
byte-identical하게 재현했고, 별도 추가 입력 빌드도 세 device image가 모두 일치했다.
현재 검증 범위를 넘어 임의 입력이나 모델 전체의 정확성을 보장하지 않는다.

공식 제출 서버는 평가 기준 수정으로 중단 중이며 별도 제출기 인증도 아직 완료하지 않았다.
공식 점수·순위는 미확인이다. [공식 상태 확인](docs/experiments/official-submission-status-20260921.json).

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
