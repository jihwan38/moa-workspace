# Arena 모델 asset Rust inspector

2026-09-22. Arena job 71191의 `/usr/bin/env: python3: No such file or directory`를 피하기 위한 독립 CPU CLI다. 실행에는 Rust ELF 하나와 libc/libgcc_s만 필요하며 Python, Furiosa SDK, NPU 초기화를 사용하지 않는다. 기존 `fingerprint.py`, 모델 crate, 공식 harness는 변경하지 않았다. 이 문서는 로컬 빌드/CPU 검증까지만 기록하며 Arena 실제 결과는 root의 job 기록을 따른다.

- 실행 파일: `moa-dev/experiments/artifacts/arena-model-assets-rust-inspector/moa-model-asset-inspector` (ignored)
- 크기: **880,872 bytes**
- SHA256: **`1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11`**
- 내장 계약 SHA256: `8e2b9346ff1f4ee3d58581326c85bb89f2dca5c97cb6c5e2b1fd9c3260754055`
- 소스·빌드·검증 연결: [manifest.json](manifest.json), [dependency-audit.json](dependency-audit.json)

실행 예시는 아래와 같다. `RNGD_MODEL_DIR`은 원격 환경이 이미 제공하는 값을 사용한다.

```sh
./moa-model-asset-inspector --header-only
./moa-model-asset-inspector
```

인자 없이 실행하면 weight 전체 SHA를 최대 40초 동안 시도한다. `--header-only`는 weight의 첫 8-byte length와 JSON header만 읽고 tensor payload를 읽지 않는다. 두 모드 모두 metadata 네 파일 전체 SHA를 먼저 계산한다. basename은 `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `chat_template.json`으로 고정되어 있고, `config.json`이나 임의 파일을 탐색하지 않는다.

`src/host/load.rs::model_dir`와 동일하게 `RNGD_MODEL_DIR` 자체 → `Gemma4-12B-it` → `gemma-4-12b-it-NVFP4` 순서로 첫 `model.safetensors` 일반 파일을 고른다. 빈 env 값은 현재 디렉터리 의미다. 경로 선택 뒤 네 metadata 해시, weight header/계약 검사, 선택적인 weight 전체 해시, 종료 stat 검사를 수행한다. 출력은 JSON 한 개이며 selector 이름, 고정 basename, size/mtime, 해시, tensor 수, 검사 집계와 고정 오류 코드만 포함한다. 실제 tensor 이름·shape·값, header 본문, 환경 전체, 절대경로는 출력하지 않는다. 계약 실패 위치는 공개 소스에서 생성한 계약 배열의 index로만 표시한다.

Header length는 `1..=16 MiB`이고 파일 안에 들어가야 한다. 중첩 duplicate JSON key, NaN/Infinity, 깊이 초과는 거절한다. safetensors 0.4.5와 같은 dtype 크기·shape 곱·offset 길이·정렬 후 연속성·최종 파일 길이를 **추가 tensor까지** 검사한다. `__metadata__`는 string map 또는 null을 허용한다. 16-MiB cap, duplicate-key 거절과 이 inspector의 bounded-depth JSON visitor는 추가적인 보수적 parser 정책이다. 원래 loader의 모든 unknown-field 처리에도 동일한 depth 제한이 적용된다고 가정하지 않는다. `fingerprint.py`의 단순 descriptor-object 검사보다 강한 검사이며, nullable metadata 허용은 실제 SDK에 맞춘 차이다.

[부모 계약](../loader-header-contract.json)은 필수 **1,149 views / 9,304,785,568 data bytes**를 명시한다. `loader_contract_pass`는 이름·필요 shape·storage bytes·loader가 실제 강제하는 dtype 472개를 확인한다. `semantic_dtype_pass`는 여기에 나머지 677개의 의도 dtype까지 추가한다. 144개 global F32 scale은 loader처럼 정확 shape를 강제하지 않고 F32와 4 data bytes를 요구한다. safetensors 형식 검사 자체는 해당 shape의 곱이 1인지 확인한다.

두 pass는 **header의 필요조건**일 뿐이다. tensor 값, global scale의 finite/nonzero, reciprocal overflow, tokenizer 동작, `load_model` 성공, HBM 용량, 모델 inference 정확성을 증명하지 않는다. metadata 네 파일 역시 해시만 계산하고 내용의 의미를 검증하지 않는다.

파일마다 path-before, fd-open, fd-after, path-after의 device/inode/mode/size/mtime/ctime을 비교한다. 읽는 동안 변화나 파일 교체가 발견되면 이미 계산한 header/full SHA를 null로 무효화하고 계약 pass도 null로 바꾼다. 이 비교는 관측 가능한 stat 변화 검출이며 파일의 영구 불변성을 증명하지 않는다.

Weight 전체 해시는 첫 length/header를 포함한 **파일 전체**를 8-MiB chunk로 읽는다. 각 read/hash 전후 monotonic 40초를 검사하며 마지막 chunk 직후 시간이 지나도 `incomplete`, `sha256:null`이다. 부분 digest는 출력하지 않는다. OS read 자체를 중단하는 hard timeout은 아니다. Header와 metadata 처리 시간도 weight hash 예산과 별도다. `--header-only`는 `full_file.status=skipped`, `sha256:null`, `bytes_hashed=0`을 명시한다.

전체 status는 정상 `complete` / 의도적인 `header_only`, optional metadata 누락·weight hash 시간 초과는 `partial`이다. 이 셋은 exit 0이다. 파일/형식/계약 실패는 `error`와 exit 1이며, 잘못된 인자는 exit 2다. Header 계약 실패라도 파일이 안정적이면 올바르게 계산한 파일 identity SHA를 보존한다. 읽기 실패/파일 변경 시에는 보존하지 않는다.

## 재현과 검증

의존성은 cached `serde 1.0.229`, `serde_json 1.0.151`, `sha2 0.11.0`을 사용한다. 새 SHA 알고리즘을 구현하지 않았다. Cargo.lock의 registry package 21개 archive SHA와 풀린 소스 **1,077개 파일**을 archive member에 대조했다. 이 21개는 resolved set이며 전부 ELF에 링크됐다는 뜻은 아니다. [audit_dependencies.py](audit_dependencies.py)는 로컬 빌드 provenance 전용이고 Arena에는 필요 없다.

부모의 349-KB 계약을 Git에 두 번 보관하지 않으므로 이 하위 디렉터리에서 바로 `cargo build`하면 `include_bytes!` 입력이 없다. [build.sh](build.sh)가 부모 계약의 고정 SHA를 검증한 뒤 **새 isolated Docker scratch crate**에 정확히 복사한다. live crate와 `/workspace/target`은 사용하지 않는다.

```sh
sh moa-dev/experiments/candidates/arena-model-assets-e55/rust-inspector/build.sh \
  moa-dev/experiments/artifacts/arena-model-assets-rust-inspector-rebuild
```

실제 최초 frozen 빌드는 `/tmp/moa-arena-assets-rust`, 별도 `CARGO_TARGET_DIR=/tmp/moa-arena-assets-rust-target`, `CARGO_HOME=/home/moa/.cargo`에서 `cargo test --offline --release`와 `cargo build --release --offline --locked`로 수행했다. helper는 동일 source/lock에 `--locked`를 두 명령 모두 적용한다. 새 scratch 경로에서의 bit-for-bit ELF 재현은 별도 실행하지 않았으며 manifest는 실제 frozen 빌드 소스 해시를 기록한다.

- [cpu-tests.log](cpu-tests.log): release **11 tests PASS**. SHA known vectors, 전체 파일/hash 범위, selector 우선순위, 전체 1,149 계약, optional 누락, duplicate/비표준 JSON, dtype/shape/offset/extra tensor, stat 수정·rename, 40초 함수의 축소 deadline 전후, 마지막 chunk 뒤 digest 보류를 검사했다.
- [cpu-cli-tests.json](cpu-cli-tests.json): frozen ELF **5 checks PASS**. 실제 CLI 전체 1,149 header, metadata 네 SHA, 인자/env 오류와 민감 출력 부재, optional 누락, 작은 weight 전체 SHA 및 계약 실패를 확인했다. 전체 계약 CLI fixture는 logical 9,304,938,704B sparse 파일이지만 실제 할당은 **155,648B**였고 header-only로 실행한 뒤 삭제했다. 모델 데이터는 생성하거나 다운로드하지 않았다.
- 최초 컴파일에서 sha2 0.11 Output의 `LowerHex` 미지원만 발견해 digest bytes를 hex 문자열로 변환했다. 초기 1ms deadline test는 8-MiB buffer 할당 중에도 만료돼 100ms budget/200ms synthetic read로 수정했다. release 최종 테스트는 전부 통과했고, 이 조정은 해시 구현을 바꾸지 않았다.

동료의 독립 source review에서 blocker는 발견되지 않았다. semantic dtype pass를 loader pass까지 포함하도록 정렬했고, frozen 최종 `src/main.rs` SHA는 `d522f964bba50d396094bda0b9462ec5ed695c9bb50cc6412aca6def2fe87cee`다.
