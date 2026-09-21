# E18 fresh Docker 재현 확인

2026-09-21에 현재 `moa-dev/Dockerfile`을 **캐시 없이 새로 빌드**하고, 인증/호스트 mount 없는 임시 컨테이너에서 저장된 E18 소스와 Cargo.lock만으로 공식 `test_kernels`를 빌드했다. 결과는 **E18 NPU 제출 바이너리와 byte-identical**하다. Dockerfile 수정은 필요하지 않았다.

| 확인 항목 | 결과 |
| --- | --- |
| 새 image tag | `moa2026-repro-audit:0.8.1` |
| 새 image ID | `sha256:1bd28cb4a8a1d17d650c9415f575483b1a3abd9aa8c2c15f3203ae34f33474f6` |
| 공식 binary 크기 | 3,653,184 bytes |
| 기존/새 binary SHA256 | `dcc4b1b0225a21fcfcd727ad10380df17562dd6f93a09b97b7308cfb7096c45d` |
| 기존/새 ELF BuildID | `8e011ccf9ab345d66d7f3c1b33b0cbddb0dbf9d7` |
| Cargo.lock 전후 SHA256 | `aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09` |
| 전체 release build | exit 0, 1m13s |
| 입력 파일 변경 | 49개 검사, 변경 0개 |

상세 버전·해시·빌드 근거는 [repro-audit-e18.json](repro-audit-e18.json)에 있다. 기존 NPU 결과는 별도 [E18 측정](e18-qkv-row-interleaved.json)이다. 이번 감사에서는 새 바이너리를 실행하거나 NPU에 제출하지 않았다.

## 실제 재현 절차

Dockerfile과 그 유일한 `COPY` 입력인 `requirements-cpu.txt`를 원문 그대로 `/tmp/moa-repro-audit-context`에 복사했다. 다른 소스나 auth/state 폴더는 build context에 넣지 않았다.

```sh
docker build --no-cache --pull --progress plain \
  --iidfile IMAGE_ID_FILE \
  -t moa2026-repro-audit:0.8.1 /tmp/moa-repro-audit-context
```

새 이미지의 임시 컨테이너에는 mount가 없었다. `moa-dev/experiments/artifacts/e18-qkv-row-interleaved`의 `src`, `scripts`, `ref`, Cargo 파일과 toolchain 파일을 `/tmp/e18-repro`로 복사했다. 저장된 provenance의 소스 해시와 모든 복사 입력을 확인했다. 이전 target은 복사하지 않았고 빌드 전 target 부재를 확인했다. E18 lockfile은 `moa-dev/locks/gemma4-Cargo.lock`과도 같았으며 read-only 상태로 두었다.

`cargo furiosa-opt build --help`에서 `--locked` 지원을 확인한 뒤 다음을 실행했다.

```sh
CARGO_TARGET_DIR=/tmp/e18-repro/target \
FURIOSA_OPT_OUT_DIR=/tmp/e18-repro/target/furiosa-opt/kernel \
cargo furiosa-opt build --release --bin test_kernels --locked -j 8
```

세 공식 커널 QKV/O/FFN의 장치 코드와 host test binary가 모두 빌드됐다. 기존 unused `broadcast_sliding_heads`와 `scale_by_layer_gate` 경고 두 개만 있었다. 빌드 후 49개 입력과 lockfile 해시를 다시 확인하고 binary를 E18 캡처와 바이트 단위로 비교했다.

## 환경 확인과 제한

Rust nightly-2026-05-01, cargo-furiosa-opt 0.8.1, Arena 0.8.0, submitter 1.2.0, schedule-viewer 0.3.0을 확인했다. Python 3.10.12와 고정한 13개 Python 배포 버전이 전부 일치했다. torch 2.14.0+cpu / numpy 2.2.6 / safetensors 0.8.0의 실제 import와 CPU tensor 연산, `pip check`도 통과했다.

cargo-binstall, cargo-furiosa-opt, furiosa-opt-driver, Arena, schedule-viewer는 기존 실행 파일과 SHA까지 동일했다. 기록된 apt 패키지 버전도 package name과 architecture를 맞춰 비교했을 때 전부 일치했다. **moa-submitter만 같은 1.2.0 버전의 다른 실행 파일**이다. 기존 것은 소스 빌드였고 이번에는 cargo-binstall이 QuickInstall 바이너리를 받았다. 이 CLI는 공식 테스트 바이너리 빌드에 사용되지 않았다. schedule-viewer는 `--version`을 지원하지 않으므로 설치 패키지 기록과 동일 실행 파일 SHA로 버전을 확인했다.

Ubuntu 태그·apt 저장소·rustup installer·다운로드 서버까지 고정한 구성은 아니다. 이번 Ubuntu base digest와 설치 apt 버전을 기록했지만 **이미지 전체가 byte-identical하게 재생성된다는 뜻은 아니다**. 또한 golden fixture는 원본 E18 캡처에서 그대로 복사했고 reference generator는 다시 실행하지 않았다. 소스 기준은 root가 E18 commit `5aa5dbd`로 보고한 저장 스냅샷이며, 이 감사에서는 Git 명령을 사용하지 않고 실제 파일 바이트로 검증했다.

원본 `moa2026-dev:0.8.1` 이미지와 `moa2026-dev` 컨테이너의 ID·사용 이미지·시작 시간이 작업 전후 동일했다. 임시 감사 컨테이너만 제거했고 새 감사 이미지는 유지했다. Raw build logs, package/version manifest, binary와 해시는 Git-ignored `moa-dev/experiments/artifacts/repro-audit-e18/`에 보존했다.
