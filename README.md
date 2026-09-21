# MOA 2026 작업 공간

대회 코드와 Docker 개발 도구를 하나의 Git 저장소에서 관리한다.
에디터에서는 이 폴더를 열고, 커널 코드는 `furiosa-opt-gemma4-12B/`에서 수정한다.

상위권을 목표로 한 조사 근거, 재현 가능한 기준 측정, 실험 순서는
[최적화 실행 계획](docs/OPTIMIZATION_PLAN.md)에 정리한다.

현재 `baseline`의 검증된 E37은 세 커널을 같은 패키지로 3회 실행하고 추가 입력 6세트 검사를 통과했다.
아래는 각 job 중앙값의 중앙값이며, 전체 Gemma 추론 속도나 공식 순위를 뜻하지 않는다.

| 커널 | 원본 cycles | E37 cycles | 원본 대비 |
|---|---:|---:|---:|
| QKV | 244,051 | 94,195 | 2.59× |
| Attention output | 404,222 | 45,748 | 8.84× |
| FFN | 4,264,297 | 379,741 | 11.23× |

[반복 측정](docs/experiments/e37-combined-repeats.json), [추가 입력 검사](docs/experiments/e37-combined-extra-correctness.json),
[채택·실패 실험 원장](docs/experiments/README.md)에 소스/바이너리/업로드 패키지의 SHA256과 근거를 기록한다.
동일한 NPU 커널을 쓴 별도 [KV cache 보존 검사](docs/experiments/e37-cache-preservation-correctness.json)에서도
세 구성의 비대상 원소 12,570,624개가 모두 유지됐다.
빈 target의 [clean build](docs/experiments/e37-clean-build-audit.md)에서도 검증 실행 파일을 재현했다.
공식 제출 서버는 평가 기준 수정으로 중단 중이며, 별도 제출기 인증도 아직 완료하지 않았다.
[공식 상태 확인](docs/experiments/official-submission-status-20260921.json).

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
