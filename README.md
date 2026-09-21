# MOA 2026 작업 공간

대회 코드와 Docker 개발 도구를 하나의 Git 저장소에서 관리한다.
에디터에서는 이 폴더를 열고, 커널 코드는 `furiosa-opt-gemma4-12B/`에서 수정한다.

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
